"""Text-to-speech via OpenAI. Returns MP3 audio bytes."""

from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI

log = logging.getLogger("techno_optimism.tts")

MODEL = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")
VOICE = os.environ.get("TTS_VOICE", "alloy")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


async def synthesize(text: str) -> bytes:
    """Synthesize `text` to speech and return MP3 bytes."""
    resp = await _get_client().audio.speech.create(
        model=MODEL,
        voice=VOICE,
        input=text,
        response_format="mp3",
    )
    audio = await resp.aread()
    log.info("synthesized %d chars -> %d bytes mp3", len(text), len(audio))
    return audio
