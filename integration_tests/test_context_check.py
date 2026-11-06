"""Tests for the references-external-context classifier.

These are integration tests: they call TypeSafe Jev, so they need
TYPESAFE_API_KEY (loaded from .env) and network access. Run with:

    python -m pytest tests/test_context_check.py -v
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

from techno_optimism_server.ai import AI

load_dotenv()

# One AI (and so one pooled TypeSafe HTTP client) for the module, so every test
# must share one event loop: pooled connections are bound to the loop that
# opened them.
pytestmark = pytest.mark.asyncio(loop_scope="module")

_ai = AI()


async def references_context(question: str) -> bool:
    return await _ai.needs_context(question)

# (question, expected) -- True means it references external prior context.
REFERENCES_CONTEXT = [
    "is it true what she's saying about polar bears?",
    "do you agree with him?",
    "wait, what did he just say?",
    "is that actually correct?",
    "can you explain what she meant by that?",
]

SELF_CONTAINED = [
    "what is a polar bear",
    "how tall is the Eiffel tower?",
    "what's the capital of France?",
    "explain how photosynthesis works",
    "what time is it in Tokyo?",
]


@pytest.mark.parametrize("question", REFERENCES_CONTEXT)
async def test_references_context_yes(question: str) -> None:
    assert await references_context(question) is True, question


@pytest.mark.parametrize("question", SELF_CONTAINED)
async def test_references_context_no(question: str) -> None:
    assert await references_context(question) is False, question
