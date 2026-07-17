"""Tests for the references-external-context classifier.

These are integration tests: they call the OpenAI chat API, so they need
OPENAI_API_KEY (loaded from .env) and network access. Run with:

    python -m pytest tests/test_context_check.py -v
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

from techno_optimism_server.context_check import references_context

load_dotenv()

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


@pytest.mark.asyncio
@pytest.mark.parametrize("question", REFERENCES_CONTEXT)
async def test_references_context_yes(question: str) -> None:
    assert await references_context(question) is True, question


@pytest.mark.asyncio
@pytest.mark.parametrize("question", SELF_CONTAINED)
async def test_references_context_no(question: str) -> None:
    assert await references_context(question) is False, question
