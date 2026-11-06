"""In-RAM audio loudness leveling via ffmpeg.

Shared by transcription (``AI.transcribe``), the spoken answer (``AI.say``)
and the voice-note upload. Three stages:

1. ``LEVEL_FILTER`` — by default a high-pass for rumble and handling thumps,
   then a gentle ``speechnorm`` that evens out quiet and loud passages (at
   most ``SPEECHNORM_EXPANSION``x of lift, slow to change, so it doesn't
   pump);
2. one static gain for the whole file that brings it to ``TARGET_LUFS``
   (EBU R128 integrated, which gates out pauses), measured *excluding* the
   first and last ``EDGE_SECONDS`` — where the record-button thumps and phone
   handling live — and capped at ``MAX_GAIN_DB``;
3. a lookahead limiter at ``LIMIT_DBFS``. Isolated clicks mid-recording are
   squashed here instead of capping the gain: capping the gain at the loudest
   peak let a single click keep a whole recording ~20 dB too quiet.

Such a click would also dominate the loudness measurement, so the measurement
clamps outlier 400 ms blocks (:func:`robust_loudness`). The limiter takes a
little loudness off the speech's own loudest syllables too, so the gain is
settled by measuring the *limited* result and correcting it, up to
``GAIN_PASSES`` times.

The result is re-encoded in memory — to a 16 kHz mono WAV for transcription,
and to a high-quality MP3 at the source's own sample rate and channel count
for voice notes. Callers fall back to the original bytes when leveling is
disabled or fails.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import re
import wave

log = logging.getLogger("techno_optimism.audio")

SAMPLE_RATE = 16000
SPEECHNORM_EXPANSION = 6  # speechnorm's max lift of a quiet passage (~15.6 dB)
# Pre-filter: strip low-frequency rumble/thumps below the voice range, then
# even out quiet and loud passages.
DEFAULT_LEVEL_FILTER = (
    f"highpass=f=80,speechnorm=e={SPEECHNORM_EXPANSION}:r=0.0005:l=1"
)

EDGE_SECONDS = 1.5        # ignored at each end when measuring
TARGET_LUFS = -16.0       # speech loudness for the middle of the recording
LIMIT_DBFS = -1.0         # limiter ceiling; clicks above it are squashed
MAX_GAIN_DB = 40.0        # don't blow a near-silent recording up into hiss
GAIN_PASSES = 3           # max measurements to settle the gain
GAIN_TOLERANCE_DB = 0.5   # stop once the measured loudness is this close
BLOCK_CAP_PERCENTILE = 90  # measured blocks louder than this percentile ...
BLOCK_CAP_LU = 3.0         # ... + this many LU are clamped (clicks, bumps)


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

# Lookahead limiter: 5 ms attack catches clicks, 50 ms release recovers before
# the next syllable. level=false — its default auto-level would undo the gain.
_LIMITER = (f"alimiter=limit={10 ** (LIMIT_DBFS / 20):.4f}"
            ":attack=5:release=50:level=false:latency=true")


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
    # Measure the middle; a recording too short to have one is measured whole.
    # (areverse keeps the original timestamps; asetpts restarts them from 0 so
    # the second atrim cuts the reversed start, i.e. the end.)
    e = edge_seconds
    middle = (f",atrim=start={e},areverse,asetpts=N/SR/TB,atrim=start={e}"
              if e > 0 else "")
    try:
        gain, loudness = 0.0, None
        for _ in range(GAIN_PASSES):
            chain = f"{level_filter},volume={gain:.2f}dB,{_LIMITER}"
            measured = await _measure(audio, chain + middle) if middle else None
            if measured is None:
                middle = ""  # too short to have a middle: measure it whole
                measured = await _measure(audio, chain)
            if measured is None:
                return None
            if loudness is None:
                loudness = measured  # before any gain, for the log
            log.debug("gain %+.1f dB -> %.1f LUFS", gain, measured)
            new_gain = min(gain + TARGET_LUFS - measured, MAX_GAIN_DB)
            if abs(new_gain - gain) < GAIN_TOLERANCE_DB:
                break  # on target, or pinned at MAX_GAIN_DB
            gain = new_gain

        out = await _ffmpeg(
            audio,
            "-v", "error", "-i", "pipe:0",
            "-af", f"{level_filter},volume={gain:.2f}dB,{_LIMITER},"
                   "asoftclip=type=hard",
            *output_args, "pipe:1",
        )
    except FileNotFoundError:
        log.warning("ffmpeg not found; using audio without leveling")
        return None
    if out is None:
        return None
    log.info("leveled audio: %.1f LUFS, gain %+.1f dB; %d bytes in -> %d out",
             loudness, gain, len(audio), len(out))
    return out


async def _measure(audio: bytes, chain: str) -> float | None:
    """Click-robust integrated loudness (LUFS) of ``audio`` through ``chain``,
    or ``None`` if ffmpeg fails or there's no measurable audio (e.g. empty or
    silent). See :func:`robust_loudness`.

    Uses the ``ebur128`` meter's 400 ms momentary blocks: ``loudnorm``'s own
    ``input_i`` reads ~2 LU off the meter and even changes when the audio is
    reversed, which the middle-of-recording trim relies on."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-nostats", "-i", "pipe:0",
        "-af", f"{chain},ebur128=framelog=info", "-f", "null", "-",
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
    frames = text.rpartition("Summary:")[0]
    return robust_loudness([float(m) for m in _MOMENTARY.findall(frames)])


_MOMENTARY = re.compile(r"\bM:\s*(-?[\d.]+)")


def robust_loudness(blocks: list[float]) -> float | None:
    """EBU R128 integrated loudness from momentary (400 ms) block loudnesses,
    with outlier blocks first clamped to ``BLOCK_CAP_PERCENTILE`` of the
    blocks + ``BLOCK_CAP_LU``. A click or bump makes a few blocks far louder
    than the speech; unclamped, they dominate the energy mean and their
    relative gate can throw out the speech itself. ``None`` if every block is
    below the -70 LUFS absolute gate."""
    gated = sorted(b for b in blocks if b > -70)
    if not gated:
        return None
    cap = gated[min(len(gated) - 1, len(gated) * BLOCK_CAP_PERCENTILE // 100)]
    clamped = [min(b, cap + BLOCK_CAP_LU) for b in gated]

    def mean_lufs(values: list[float]) -> float:
        return 10 * math.log10(sum(10 ** (v / 10) for v in values) / len(values))

    relative_gate = mean_lufs(clamped) - 10
    return mean_lufs([b for b in clamped if b > relative_gate])


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
