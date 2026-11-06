"""Single AI interface over the OpenAI API.

Consolidates every model call the server makes:

    transcribe(audio)                      -> str      speech to text
    needs_context(question)                -> bool     does the question refer
                                                       to external context?
    ask(question, context, prev_id)        -> stream   yields Progress(text)
                                                       chunks then a final
                                                       Result(response_id, answer)
    say(text)                              -> bytes    text to speech (leveled mp3)

Audio in is expected to be an encoded file (e.g. mp3). Answers are cleaned
for reading aloud: no markdown, citations, or links.

Every call is traced to Langfuse (when LANGFUSE_* keys are set): chat and
Responses calls through the Langfuse OpenAI wrapper, and the audio endpoints
(which the wrapper does not cover) as manual generations with the audio
attached as media.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import AsyncIterator

from langfuse import LangfuseMedia, get_client
from langfuse.openai import AsyncOpenAI  # drop-in wrapper: traces chat/responses calls

from techno_optimism_server.audio import level_audio, level_audio_mp3, level_filter_from_env

log = logging.getLogger("techno_optimism.ai")


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


class ProgressTracker:
    """Assembles the thinking-display text from partial pieces.

    Streaming reasoning/answer text is appended (`add_text`); a standalone
    status line such as a web search replaces the display (`status`) and starts
    a fresh text segment after it, so one segment never bleeds into the next.
    Each method returns the current full display string.
    """

    def __init__(self) -> None:
        self._display = ""
        self._in_text = False

    def status(self, text: str) -> str:
        """Set a standalone status line, replacing the display."""
        self._display = text
        self._in_text = False
        return self._display

    def add_text(self, chunk: str) -> str:
        """Append streaming text; starts fresh right after a status line."""
        if not self._in_text:
            self._display, self._in_text = "", True
        self._display += chunk
        return self._display


# --------------------------------------------------------------------------- #
# AI interface
# --------------------------------------------------------------------------- #
class AI:
    """All model interactions, behind one object."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
        # Start the Langfuse client (reads LANGFUSE_* from the environment) at
        # startup rather than lazily inside the first request.
        get_client()

        # Transcription
        self.transcribe_model = os.environ.get("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
        self.transcribe_prompt = os.environ.get(
            "TRANSCRIBE_PROMPT",
            "The recording may contain multiple languages. Transcribe all "
            "speech throughout the entire recording, in each language, and do "
            "not stop early.",
        )
        self.level_filter = level_filter_from_env()
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
    async def transcribe(self, audio: bytes, name: str = "transcribe-audio") -> str:
        """Speech-to-text. Levels loudness first so quiet speech is heard.

        `name` labels the trace step (question vs. context transcription).
        """
        payload, filename = await self._level(audio)
        content_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        with get_client().start_as_current_observation(
            as_type="generation",
            name=name,
            model=self.transcribe_model,
            model_parameters={"temperature": self.transcribe_temperature},
            # The audio the model actually heard (post-leveling).
            input={
                "audio": LangfuseMedia(content_bytes=payload, content_type=content_type),
                "prompt": self.transcribe_prompt,
            },
        ) as gen:
            resp = await self._client.audio.transcriptions.create(
                model=self.transcribe_model,
                file=(filename, payload),
                prompt=self.transcribe_prompt,
                temperature=self.transcribe_temperature,
                response_format="json",  # json (unlike text) carries token usage
            )
            text = (resp if isinstance(resp, str) else getattr(resp, "text", str(resp))).strip()
            gen.update(output=text, usage_details=_transcription_usage(resp))
        return text

    async def _level(self, audio: bytes) -> tuple[bytes, str]:
        """Loudness-level to 16 kHz mono WAV; fall back to the raw mp3."""
        wav = await level_audio(audio, self.level_filter)
        return (wav, "audio.wav") if wav is not None else (audio, "audio.mp3")

    # -- context classification ------------------------------------------- #
    async def needs_context(self, question: str) -> bool:
        """True if the (already transcribed) question references external
        prior context the user just heard/saw."""
        resp = await self._client.chat.completions.create(
            name="classify-needs-context",  # Langfuse observation name
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
        progress = ProgressTracker()

        kwargs: dict = dict(
            name="generate-answer",  # Langfuse observation name
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

        final = None
        # create(stream=True) rather than responses.stream(): it's the call the
        # Langfuse wrapper traces, and stream() rejects the `name` kwarg.
        stream = await self._client.responses.create(stream=True, **kwargs)
        async with stream:
            async for event in stream:
                if event.type == "response.reasoning_summary_text.delta":
                    saw_reasoning = True
                    chunk = thinking_cleaner.feed(event.delta)
                    if chunk:
                        yield Progress(progress.add_text(chunk))
                elif event.type == "response.output_text.delta":
                    chunk = answer_cleaner.feed(event.delta)
                    if chunk:
                        answer_parts.append(chunk)
                        if not saw_reasoning:      # fallback: show the answer
                            yield Progress(progress.add_text(chunk))
                elif event.type == "response.web_search_call.in_progress":
                    yield Progress(progress.status("Searching the web…"))
                elif event.type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", None) == "web_search_call":
                        action = getattr(item, "action", None)
                        query = getattr(action, "query", None)
                        if query:
                            yield Progress(progress.status(f"Searching the web… ({query})"))
                elif event.type == "response.completed":
                    final = event.response
                elif event.type in ("response.failed", "response.incomplete"):
                    raise RuntimeError(f"{event.type}: {event.response.error or event.response.incomplete_details}")
        if final is None:
            raise RuntimeError("response stream ended without response.completed")

        rtail = thinking_cleaner.flush()
        if rtail:
            yield Progress(progress.add_text(rtail))
        atail = answer_cleaner.flush()
        if atail:
            answer_parts.append(atail)
            if not saw_reasoning:
                yield Progress(progress.add_text(atail))

        answer = "".join(answer_parts).strip()
        log.info("answer ready: %d chars, reasoning=%s (response %s)",
                 len(answer), saw_reasoning, final.id)
        yield Result(response_id=final.id, answer=answer)

    # -- speech synthesis -------------------------------------------------- #
    async def say(self, text: str) -> bytes:
        """Text-to-speech, loudness-leveled. Returns MP3 bytes."""
        spoken = clean_for_speech(text)
        with get_client().start_as_current_observation(
            as_type="generation",
            name="synthesize-speech",
            model=self.tts_model,
            model_parameters={"voice": self.tts_voice},
            input=spoken,
        ) as gen:
            resp = await self._client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=spoken,
                response_format="mp3",
            )
            audio = await resp.aread()
            log.info("synthesized %d chars -> %d bytes mp3", len(text), len(audio))
            # Same loudness as everything else; TTS has no edge noise, so measure
            # the whole thing. Falls back to the raw TTS mp3.
            leveled = await level_audio_mp3(audio, self.level_filter, edge_seconds=0)
            out = leveled if leveled is not None else audio
            gen.update(output=LangfuseMedia(content_bytes=out, content_type="audio/mpeg"))
        return out


# --------------------------------------------------------------------------- #
# Prompts & helpers
# --------------------------------------------------------------------------- #
def _transcription_usage(resp) -> dict[str, int] | None:
    """Token usage from a json transcription response, in Langfuse's shape.

    gpt-4o-transcribe reports {input_tokens, output_tokens, input_token_details:
    {audio_tokens, text_tokens}}; whisper-1 reports duration instead (None).
    """
    usage = getattr(resp, "usage", None)
    if getattr(usage, "type", None) != "tokens":
        return None
    # Audio and text input are priced separately, so report them as separate
    # usage types (not also as an aggregate "input", which would double-count).
    token_details = getattr(usage, "input_token_details", None)
    audio = getattr(token_details, "audio_tokens", None)
    text = getattr(token_details, "text_tokens", None)
    if audio is None or text is None:
        return {"input": usage.input_tokens, "output": usage.output_tokens}
    return {"input_audio": audio, "input_text": text, "output": usage.output_tokens}


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

