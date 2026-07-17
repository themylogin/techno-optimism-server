"""Audio transcription via OpenAI's gpt-4o-transcribe model.

gpt-4o-transcribe transcribes multilingual audio in a single request, but
on an uneven recording -- a loud, clear intro followed by quieter and/or
other-language speech -- it tends to stop after the intro. Two cheap fixes,
applied together, make a single pass reliable (measured 100% vs ~17% on the
raw clip; neither fix alone is enough):

  1. A `prompt` telling it to transcribe the whole clip in every language.
  2. Uniform loudness leveling (ffmpeg `dynaudnorm`), which removes the
     loud-intro salience so the model treats the whole clip as one utterance.

If ffmpeg is unavailable we still send the prompt; that alone is unreliable
on degraded audio but fine for normal recordings.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave

from openai import AsyncOpenAI

log = logging.getLogger("techno_optimism.transcribe")

MODEL = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
PROMPT = os.environ.get(
    "TRANSCRIBE_PROMPT",
    "The recording may contain multiple languages. Transcribe all speech "
    "throughout the entire recording, in each language, and do not stop early.",
)
# ffmpeg leveling filter; set LEVEL_FILTER="" to disable preprocessing.
LEVEL_FILTER = os.environ.get("LEVEL_FILTER", "dynaudnorm=framelen=300:maxgain=20")
# A little sampling escapes the model's greedy "stop after the intro" decode;
# 0.3 measured most reliable (temp=0 truncates, high temp destabilizes).
TEMPERATURE = float(os.environ.get("TRANSCRIBE_TEMPERATURE", "0.3"))
SAMPLE_RATE = 16000

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a valid in-memory WAV. (A WAV streamed to an ffmpeg
    pipe has a bogus RIFF size header and gets read as truncated.)"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


async def _level(audio: bytes) -> tuple[bytes, str]:
    """Loudness-level `audio` to 16 kHz mono WAV via ffmpeg. Falls back to
    the original bytes (as mp3) if ffmpeg is missing/fails or leveling is off."""
    if not LEVEL_FILTER:
        return audio, "audio.mp3"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-af", LEVEL_FILTER, "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        log.warning("ffmpeg not found; sending audio without leveling")
        return audio, "audio.mp3"

    pcm, err = await proc.communicate(input=audio)
    if proc.returncode != 0 or not pcm:
        log.warning("ffmpeg leveling failed (rc=%s): %s; using raw audio",
                    proc.returncode, err.decode(errors="replace")[:200])
        return audio, "audio.mp3"
    log.info("leveled audio: %d bytes mp3 -> %d bytes wav", len(audio), len(pcm))
    return _pcm_to_wav(pcm), "audio.wav"


async def transcribe(audio: bytes) -> str:
    """Transcribe raw audio bytes (e.g. mp3) and return the recognized text."""
    payload, filename = await _level(audio)
    resp = await _get_client().audio.transcriptions.create(
        model=MODEL,
        file=(filename, payload),
        prompt=PROMPT,
        temperature=TEMPERATURE,
        response_format="text",
    )
    text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
    return text.strip()
