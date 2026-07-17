"""Answer a spoken question with a web-search-enabled reasoning model.

Streams the answer text as it is generated (so the caller can forward
progress) and returns the full, cleaned answer for text-to-speech.

Web-search results come back with inline citation markers delimited by
private-use-area characters (e.g. U+E200 ... U+E201). Those must be
stripped -- the answer is read aloud, so no markup or links.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Awaitable, Callable

from openai import AsyncOpenAI

log = logging.getLogger("techno_optimism.think")

MODEL = os.environ.get("THINK_MODEL", "gpt-5.6")

ANSWER_INSTRUCTION = (
    "Answer the question in a short and concise form, in the same language as "
    "the question. It will be read aloud, so use no formatting, markdown, or "
    "hyperlinks."
)

# Web-search citations are delimited by private-use-area glyphs; this matches
# any stray one left over (paired spans are removed incrementally below).
_STRAY_PUA = re.compile("[-]")


class _CitationStripper:
    """Incrementally removes citation spans from a stream of text deltas,
    holding back any span that is still open at a delta boundary."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = ""
        while True:
            i = self._buf.find("")
            if i == -1:
                out += self._buf
                self._buf = ""
                break
            close = self._buf.find("", i)
            if close == -1:
                out += self._buf[:i]      # citation still open; hold the rest
                self._buf = self._buf[i:]
                break
            out += self._buf[:i]
            self._buf = self._buf[close + 1:]
        return out

    def flush(self) -> str:
        out = _STRAY_PUA.sub("", self._buf)
        self._buf = ""
        return out


def _build_prompt(question: str, context: str | None) -> str:
    if context:
        return (
            "Here's the conversation context:\n\n"
            f"{context}\n\n"
            "the user asked the following question\n\n"
            f"{question}\n\n"
            f"{ANSWER_INSTRUCTION}"
        )
    return (
        "The user asked the following question:\n\n"
        f"{question}\n\n"
        f"{ANSWER_INSTRUCTION}"
    )


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


async def answer_stream(
    question: str,
    context: str | None,
    on_delta: Callable[[str], Awaitable[None]],
) -> str:
    """Stream an answer to `question` (optionally given conversation `context`),
    invoking `on_delta` with cleaned text chunks. Returns the full answer."""
    prompt = _build_prompt(question, context)
    stripper = _CitationStripper()
    parts: list[str] = []

    async with _get_client().responses.stream(
        model=MODEL,
        input=prompt,
        tools=[{"type": "web_search"}],
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                clean = stripper.feed(event.delta)
                if clean:
                    parts.append(clean)
                    await on_delta(clean)
        await stream.get_final_response()

    tail = stripper.flush()
    if tail:
        parts.append(tail)
        await on_delta(tail)

    answer = "".join(parts).strip()
    log.info("answer ready: %d chars", len(answer))
    return answer
