"""In-RAM audio loudness leveling via ffmpeg.

Shared by transcription (``AI.transcribe``) and the voice-note upload: the
audio is piped through ffmpeg's ``LEVEL_FILTER`` and re-encoded, all in memory —
to a 16 kHz mono WAV for transcription, and to a high-quality MP3 at the
source's own sample rate and channel count for voice notes. Callers fall back to the original bytes when
leveling is disabled or fails.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave

log = logging.getLogger("techno_optimism.audio")

SAMPLE_RATE = 16000
DEFAULT_LEVEL_FILTER = "dynaudnorm=framelen=300:maxgain=20"


def level_filter_from_env() -> str:
    """The ffmpeg filter from ``LEVEL_FILTER`` ("" disables leveling)."""
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


async def level_audio_mp3(audio: bytes, level_filter: str) -> bytes | None:
    """Loudness-level ``audio`` to a VBR V2 MP3 via ffmpeg, keeping the
    source's sample rate and channel count.

    Same fallback contract as :func:`level_audio`.
    """
    return await _ffmpeg_level(audio, level_filter, _MP3_ARGS)


async def _ffmpeg_level(
    audio: bytes, level_filter: str, output_args: tuple[str, ...]
) -> bytes | None:
    """Pipe ``audio`` through ffmpeg's ``level_filter``, encoded per
    ``output_args``. ``None`` if disabled or ffmpeg fails."""
    if not level_filter:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-af", level_filter, *output_args, "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning("ffmpeg not found; using audio without leveling")
        return None

    out, err = await proc.communicate(input=audio)
    if proc.returncode != 0 or not out:
        log.warning("ffmpeg leveling failed (rc=%s): %s; using raw audio",
                    proc.returncode, err.decode(errors="replace")[:200])
        return None
    log.info("leveled audio: %d bytes in -> %d bytes out", len(audio), len(out))
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
