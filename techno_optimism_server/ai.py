"""Single AI interface over the OpenAI API.

Consolidates every model call the server makes:

    transcribe(audio)                      -> str      speech to text
    needs_context(question)                -> bool     does the question refer
                                                       to external context?
    ask(question, context, prev_id)        -> stream   yields Progress(text)
                                                       chunks then a final
                                                       Result(response_id, answer)
    say(text)                              -> bytes    text to speech (mp3)

Audio in is expected to be an encoded file (e.g. mp3). Answers are cleaned
for reading aloud: no markdown, citations, or links.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import wave
from dataclasses import dataclass
from typing import AsyncIterator

from openai import AsyncOpenAI

log = logging.getLogger("techno_optimism.ai")

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# ask() stream items
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    """The current thinking-display text. Always replaces what's on screen;
    `ask` assembles it from partial pieces before yielding."""
    text: str


@dataclass
class Result:
    """The finished answer plus the response id (for follow-up turns)."""
    response_id: str
    answer: str


# --------------------------------------------------------------------------- #
# Read-aloud cleaning
# --------------------------------------------------------------------------- #
# Web search adds citations as markdown links, sometimes wrapped in parens,
# and occasionally as private-use-area glyphs. All of it must go before TTS.
_CITE_PARENS = re.compile(r"\(\s*\[[^\]]+\]\(https?://[^)]+\)\s*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
_BARE_URL = re.compile(r"https?://\S+")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_UNDERLINE = re.compile(r"__([^_]+)__")
_CODE = re.compile(r"`([^`]+)`")
_PUA_SPAN = re.compile(".*?", re.S)
_PUA_STRAY = re.compile("[-]")
_WS_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def clean_for_speech(text: str) -> str:
    """Strip markdown, citations, and links so `text` reads cleanly aloud."""
    text = _PUA_SPAN.sub("", text)
    text = _CITE_PARENS.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BARE_URL.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _UNDERLINE.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _PUA_STRAY.sub("", text)
    text = _WS_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


class _SpeechStreamCleaner:
    """Cleans a stream of text deltas for read-aloud output.

    Emits text up to the first character that could start a markdown/citation
    construct ([, (, *, `, or a URL) and holds the rest back until flush, so
    no partial markup is ever emitted. The final flush cleans the remainder.
    """

    _OPENERS = "[(*`"

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        candidates = [self._buf.find(o) for o in self._OPENERS]
        http = self._buf.lower().find("http")
        candidates = [i for i in (*candidates, http) if i != -1]
        if not candidates:
            safe, self._buf = self._buf, ""
        else:
            cut = min(candidates)
            safe, self._buf = self._buf[:cut], self._buf[cut:]
        # `safe` has no markup openers, so plain glyph/whitespace cleaning only.
        safe = _PUA_SPAN.sub("", safe)
        safe = _PUA_STRAY.sub("", safe)
        return safe

    def flush(self) -> str:
        out = clean_for_speech(self._buf)
        self._buf = ""
        return out


# --------------------------------------------------------------------------- #
# AI interface
# --------------------------------------------------------------------------- #
class AI:
    """All model interactions, behind one object."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment

        # Transcription
        self.transcribe_model = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
        self.transcribe_prompt = os.environ.get(
            "TRANSCRIBE_PROMPT",
            "The recording may contain multiple languages. Transcribe all "
            "speech throughout the entire recording, in each language, and do "
            "not stop early.",
        )
        self.level_filter = os.environ.get(
            "LEVEL_FILTER", "dynaudnorm=framelen=300:maxgain=20"
        )
        self.transcribe_temperature = float(
            os.environ.get("TRANSCRIBE_TEMPERATURE", "0.3")
        )

        # Context classification
        self.classify_model = os.environ.get("CLASSIFY_MODEL", "gpt-4o-mini")

        # Answering
        self.think_model = os.environ.get("THINK_MODEL", "gpt-5.6")
        # Stream the model's reasoning summary as "thinking". "auto" asks for a
        # summary only when the model actually reasons; set "" to disable.
        self.reasoning_summary = os.environ.get("REASONING_SUMMARY", "auto") or None
        self.reasoning_effort = os.environ.get("REASONING_EFFORT") or None

        # Speech synthesis
        self.tts_model = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")
        self.tts_voice = os.environ.get("TTS_VOICE", "alloy")

    # -- transcription ----------------------------------------------------- #
    async def transcribe(self, audio: bytes) -> str:
        """Speech-to-text. Levels loudness first so quiet speech is heard."""
        payload, filename = await self._level(audio)
        resp = await self._client.audio.transcriptions.create(
            model=self.transcribe_model,
            file=(filename, payload),
            prompt=self.transcribe_prompt,
            temperature=self.transcribe_temperature,
            response_format="text",
        )
        text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
        return text.strip()

    async def _level(self, audio: bytes) -> tuple[bytes, str]:
        """Loudness-level to 16 kHz mono WAV via ffmpeg; fall back to raw mp3."""
        if not self.level_filter:
            return audio, "audio.mp3"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-v", "error", "-i", "pipe:0",
                "-af", self.level_filter, "-ar", str(SAMPLE_RATE), "-ac", "1",
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

    # -- context classification ------------------------------------------- #
    async def needs_context(self, question: str) -> bool:
        """True if the (already transcribed) question references external
        prior context the user just heard/saw."""
        resp = await self._client.chat.completions.create(
            model=self.classify_model,
            temperature=0,
            max_tokens=1,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"Question: {question}"},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        result = answer.startswith("y")
        log.info("context-check %r -> %s (%r)", question, result, answer)
        return result

    # -- answering --------------------------------------------------------- #
    async def ask(
        self,
        question: str,
        context: str | None = None,
        previous_response_id: str | None = None,
    ) -> AsyncIterator[Progress | Result]:
        """Answer `question` with a web-search reasoning model.

        Progress chunks carry a text view of what the model is doing: its
        *reasoning* as it thinks, and web-search activity ("Searching the
        web...", then the query). Many questions -- especially concise factual
        ones -- produce no reasoning; in that case Progress falls back to the
        answer text as it streams, so the caller always has something to show.
        The final Result carries the answer text (from output_text) for speech.
        """
        prompt = _build_answer_prompt(
            question, context, followup=bool(previous_response_id)
        )
        thinking_cleaner = _SpeechStreamCleaner()  # reasoning, for display
        answer_cleaner = _SpeechStreamCleaner()    # answer, for speech + fallback
        answer_parts: list[str] = []
        saw_reasoning = False
        display = ""      # current thinking line, assembled from partial pieces
        in_text = False   # currently streaming reasoning/answer text (vs a status line)

        def append_text(chunk: str) -> str:
            nonlocal display, in_text
            if not in_text:           # first text after a status line: start fresh
                display, in_text = "", True
            display += chunk
            return display

        kwargs: dict = dict(
            model=self.think_model,
            input=prompt,
            tools=[{"type": "web_search"}],
        )
        if self.reasoning_summary:
            kwargs["reasoning"] = {"summary": self.reasoning_summary}
            if self.reasoning_effort:
                kwargs["reasoning"]["effort"] = self.reasoning_effort
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        async with self._client.responses.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "response.reasoning_summary_text.delta":
                    saw_reasoning = True
                    chunk = thinking_cleaner.feed(event.delta)
                    if chunk:
                        yield Progress(append_text(chunk))
                elif event.type == "response.output_text.delta":
                    chunk = answer_cleaner.feed(event.delta)
                    if chunk:
                        answer_parts.append(chunk)
                        if not saw_reasoning:      # fallback: show the answer
                            yield Progress(append_text(chunk))
                elif event.type == "response.web_search_call.in_progress":
                    display, in_text = "Searching the web…", False
                    yield Progress(display)
                elif event.type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", None) == "web_search_call":
                        action = getattr(item, "action", None)
                        query = getattr(action, "query", None)
                        if query:
                            display = f"Searching the web… ({query})"
                            yield Progress(display)
                        in_text = False
            final = await stream.get_final_response()

        rtail = thinking_cleaner.flush()
        if rtail:
            yield Progress(append_text(rtail))
        atail = answer_cleaner.flush()
        if atail:
            answer_parts.append(atail)
            if not saw_reasoning:
                yield Progress(append_text(atail))

        answer = "".join(answer_parts).strip()
        log.info("answer ready: %d chars, reasoning=%s (response %s)",
                 len(answer), saw_reasoning, final.id)
        yield Result(response_id=final.id, answer=answer)

    # -- speech synthesis -------------------------------------------------- #
    async def say(self, text: str) -> bytes:
        """Text-to-speech. Returns MP3 bytes."""
        resp = await self._client.audio.speech.create(
            model=self.tts_model,
            voice=self.tts_voice,
            input=clean_for_speech(text),
            response_format="mp3",
        )
        audio = await resp.aread()
        log.info("synthesized %d chars -> %d bytes mp3", len(text), len(audio))
        return audio


# --------------------------------------------------------------------------- #
# Prompts & helpers
# --------------------------------------------------------------------------- #
_CLASSIFY_SYSTEM = (
    "You classify a single user question that was just spoken aloud.\n"
    "Decide: is the user referring to something external they likely just "
    "heard, saw, or read before asking -- something you (the assistant) were "
    "not part of and would need in order to answer well?\n"
    "- Answer 'yes' if the question points at outside context: another "
    "person's words or claims, 'this'/'that'/'it' with no antecedent, "
    '"what she said", "is he right", "what does this mean", etc.\n'
    "- Answer 'no' if the question is self-contained and can be answered on "
    "its own general knowledge.\n"
    "Reply with exactly one word: yes or no."
)

_ANSWER_INSTRUCTION = (
    "Answer the question in a short and concise form, in the same language as "
    "the question. It will be read aloud, so use no formatting, markdown, "
    "citations, sources, URLs, or hyperlinks."
)


def _build_answer_prompt(
    question: str, context: str | None, followup: bool = False
) -> str:
    # On a follow-up the model already holds the earlier turns (via
    # previous_response_id), so it's framed as a continuation rather than a
    # fresh question.
    if followup:
        lead = "This is a follow-up question in our ongoing conversation."
        if context:
            return (
                f"{lead} Here is some additional context the user is now "
                "referring to, on top of our earlier conversation:\n\n"
                f"{context}\n\n"
                "The user now asks:\n\n"
                f"{question}\n\n"
                f"{_ANSWER_INSTRUCTION}"
            )
        return (
            f"{lead}\n\n"
            "The user now asks:\n\n"
            f"{question}\n\n"
            f"{_ANSWER_INSTRUCTION}"
        )

    if context:
        return (
            "Here's the conversation context:\n\n"
            f"{context}\n\n"
            "the user asked the following question\n\n"
            f"{question}\n\n"
            f"{_ANSWER_INSTRUCTION}"
        )
    return (
        "The user asked the following question:\n\n"
        f"{question}\n\n"
        f"{_ANSWER_INSTRUCTION}"
    )


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a valid in-memory WAV (a WAV streamed to an ffmpeg pipe
    has a bogus RIFF size header and gets read as truncated)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()
