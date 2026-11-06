"""In-RAM audio loudness leveling via ffmpeg.

Shared by transcription (``AI.transcribe``), the spoken answer (``AI.say``)
and the voice-note upload. Leveling
is a single static gain — no adaptive gain riding, so nothing pumps or wobbles:

1. measure the loudness (EBU R128 integrated, which gates out pauses) and true
   peak of the audio *excluding* the first and last ``EDGE_SECONDS`` — where the
   record-button thumps and phone handling live;
2. apply one gain to the whole file that brings that middle to
   ``TARGET_LUFS``, capped so the middle's peaks stay under ``PEAK_CEILING_DBTP``
   and at ``MAX_GAIN_DB``. The edges may clip; that's accepted.

``LEVEL_FILTER`` is an ffmpeg filter chain applied before both the measurement
and the gain. The result is re-encoded in memory — to a 16 kHz mono WAV for
transcription, and to a high-quality MP3 at the source's own sample rate and
channel count for voice notes. Callers fall back to the original bytes when
leveling is disabled or fails.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import wave

log = logging.getLogger("techno_optimism.audio")

SAMPLE_RATE = 16000
# Pre-filter: strip low-frequency rumble/thumps below the voice range.
DEFAULT_LEVEL_FILTER = "highpass=f=80"

EDGE_SECONDS = 1.5        # ignored at each end when measuring
TARGET_LUFS = -16.0       # speech loudness for the middle of the recording
PEAK_CEILING_DBTP = -1.0  # the middle's true peak never goes above this
MAX_GAIN_DB = 40.0        # don't blow a near-silent recording up into hiss


def level_filter_from_env() -> str:
    """The ffmpeg pre-filter from ``LEVEL_FILTER`` ("" disables leveling)."""
    return os.environ.get("LEVEL_FILTER", DEFAULT_LEVEL_FILTER)


# Speech-recognition input: 16 kHz mono 16-bit PCM.
_PCM_ARGS = ("-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le")

# Voice-note MP3: keep the source's sample rate and channels, LAME VBR V2
# ("standard", transparent for most listeners) at LAME's default algorithm
# quality. Higher compression_level values are faster but drop the
# psychoacoustic model, audibly degrading the result.
_MP3_ARGS = ("-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3")


async def level_audio(audio: bytes, level_filter: str) -> bytes | None:
    """Loudness-level ``audio`` to a 16 kHz mono WAV via ffmpeg.

    Returns the WAV bytes, or ``None`` when ``level_filter`` is empty, ffmpeg
    is missing, or it fails — the caller should then use the original audio.
    """
    pcm = await _ffmpeg_level(audio, level_filter, _PCM_ARGS)
    return pcm_to_wav(pcm) if pcm is not None else None


async def level_audio_mp3(
    audio: bytes, level_filter: str, edge_seconds: float = EDGE_SECONDS
) -> bytes | None:
    """Loudness-level ``audio`` to a VBR V2 MP3 via ffmpeg, keeping the
    source's sample rate and channel count. ``edge_seconds=0`` measures the
    whole file (for clean audio such as synthesized speech).

    Same fallback contract as :func:`level_audio`.
    """
    return await _ffmpeg_level(audio, level_filter, _MP3_ARGS, edge_seconds)


async def _ffmpeg_level(
    audio: bytes, level_filter: str, output_args: tuple[str, ...],
    edge_seconds: float = EDGE_SECONDS,
) -> bytes | None:
    """Apply a static leveling gain to ``audio`` (see the module docstring),
    encoded per ``output_args``. ``None`` if disabled or ffmpeg fails."""
    if not level_filter:
        return None
    try:
        # Measure the middle; a recording too short to have one is measured whole.
        e = edge_seconds
        stats = None
        if e > 0:
            # (areverse keeps the original timestamps; asetpts restarts them
            # from 0 so the second atrim cuts the reversed start, i.e. the end.)
            stats = await _measure(audio, f"{level_filter},atrim=start={e},areverse,"
                                          f"asetpts=N/SR/TB,atrim=start={e}")
        if stats is None:
            stats = await _measure(audio, level_filter)
        if stats is None:
            return None
        loudness, peak = stats
        gain = min(TARGET_LUFS - loudness, PEAK_CEILING_DBTP - peak, MAX_GAIN_DB)

        out = await _ffmpeg(
            audio,
            "-v", "error", "-i", "pipe:0",
            "-af", f"{level_filter},volume={gain:.2f}dB,asoftclip=type=hard",
            *output_args, "pipe:1",
        )
    except FileNotFoundError:
        log.warning("ffmpeg not found; using audio without leveling")
        return None
    if out is None:
        return None
    log.info("leveled audio: %.1f LUFS, %.1f dBTP, gain %+.1f dB; %d bytes in -> %d out",
             loudness, peak, gain, len(audio), len(out))
    return out


async def _measure(audio: bytes, chain: str) -> tuple[float, float] | None:
    """(integrated LUFS, true peak dBTP) of ``audio`` through ``chain``, or
    ``None`` if ffmpeg fails or there's no measurable audio (e.g. empty)."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-nostats", "-i", "pipe:0",
        "-af", f"{chain},loudnorm=print_format=json", "-f", "null", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate(input=audio)
    text = err.decode(errors="replace")
    if proc.returncode != 0:
        log.warning("ffmpeg loudness measurement failed (rc=%s): %s",
                    proc.returncode, text[-200:])
        return None
    try:
        stats = json.loads(text[text.rindex("{"):text.rindex("}") + 1])
        loudness, peak = float(stats["input_i"]), float(stats["input_tp"])
    except (ValueError, KeyError):
        return None
    if not (math.isfinite(loudness) and math.isfinite(peak)):
        return None
    return loudness, peak


async def _ffmpeg(audio: bytes, *args: str) -> bytes | None:
    """Run ffmpeg with ``audio`` on stdin; its stdout, or ``None`` on failure."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(input=audio)
    if proc.returncode != 0 or not out:
        log.warning("ffmpeg leveling failed (rc=%s): %s; using raw audio",
                    proc.returncode, err.decode(errors="replace")[:200])
        return None
    return out


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a valid in-memory WAV (a WAV streamed to an ffmpeg pipe
    has a bogus RIFF size header and gets read as truncated)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()
