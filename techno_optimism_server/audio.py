"""In-RAM audio loudness leveling via ffmpeg.

Shared by transcription (``AI.transcribe``) and the voice-note upload: the
audio is piped through ffmpeg's ``LEVEL_FILTER`` and re-encoded to 16 kHz mono
16-bit WAV, all in memory. Callers fall back to the original bytes when
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


async def level_audio(audio: bytes, level_filter: str) -> bytes | None:
    """Loudness-level ``audio`` to a 16 kHz mono WAV via ffmpeg.

    Returns the WAV bytes, or ``None`` when ``level_filter`` is empty, ffmpeg
    is missing, or it fails — the caller should then use the original audio.
    """
    if not level_filter:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-af", level_filter, "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning("ffmpeg not found; using audio without leveling")
        return None

    pcm, err = await proc.communicate(input=audio)
    if proc.returncode != 0 or not pcm:
        log.warning("ffmpeg leveling failed (rc=%s): %s; using raw audio",
                    proc.returncode, err.decode(errors="replace")[:200])
        return None
    log.info("leveled audio: %d bytes in -> %d bytes wav", len(audio), len(pcm))
    return pcm_to_wav(pcm)


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
