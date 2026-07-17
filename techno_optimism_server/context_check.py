"""Decide whether a spoken question references external prior context.

Some questions are self-contained ("what is a polar bear") -- the assistant
can answer directly. Others reference something the user just heard or saw
("is it true what she's saying about polar bears?") -- to answer well the
assistant would need that surrounding context.

We ask a chat model to make that yes/no call.
"""

from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI

log = logging.getLogger("techno_optimism.context_check")

MODEL = os.environ.get("CLASSIFY_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
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

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


async def references_context(question: str) -> bool:
    """Return True if `question` references external prior context."""
    resp = await _get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip().lower()
    result = answer.startswith("y")
    log.info("context-check %r -> %s (%r)", question, result, answer)
    return result
