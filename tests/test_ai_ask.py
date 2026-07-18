"""Hermetic tests for AI.ask's hybrid thinking behavior.

The OpenAI streaming client is replaced with a fake that yields scripted
events, so we can assert exactly which Progress chunks ask() emits (reasoning
vs. answer fallback) without any network or key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from techno_optimism_server.ai import AI, Progress, Result


def ev(kind: str, delta: str):
    return SimpleNamespace(type=kind, delta=delta)


def reasoning(delta): return ev("response.reasoning_summary_text.delta", delta)
def answer(delta): return ev("response.output_text.delta", delta)
def search_start(): return SimpleNamespace(type="response.web_search_call.in_progress")


def search_done(query):
    action = SimpleNamespace(query=query)
    item = SimpleNamespace(type="web_search_call", action=action)
    return SimpleNamespace(type="response.output_item.done", item=item)


class _FakeStream:
    def __init__(self, events, final_id):
        self._events = events
        self._final_id = final_id

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for e in self._events:
            yield e

    async def get_final_response(self):
        return SimpleNamespace(id=self._final_id)


class _FakeResponses:
    def __init__(self, events, final_id):
        self._events = events
        self._final_id = final_id
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._events, self._final_id)


@pytest.fixture
def make_ai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _make(events, final_id="resp_out"):
        ai = AI()
        fake = _FakeResponses(events, final_id)
        ai._client = SimpleNamespace(responses=fake)
        return ai, fake

    return _make


async def collect(ai, **ask_kwargs):
    progress, result = [], None
    async for item in ai.ask("Q", **ask_kwargs):
        if isinstance(item, Progress):
            progress.append(item.text)
        elif isinstance(item, Result):
            result = item
    return progress, result


async def test_reasoning_streamed_as_thinking(make_ai):
    events = [
        reasoning("Weighing "), reasoning("the options."),
        answer("The answer "), answer("is 42."),
    ]
    ai, _ = make_ai(events)
    progress, result = await collect(ai)

    # thinking is the reasoning, NOT the answer
    assert progress == ["Weighing ", "the options."]
    assert result.answer == "The answer is 42."
    assert result.response_id == "resp_out"


async def test_answer_fallback_when_no_reasoning(make_ai):
    events = [answer("Mars has "), answer("two moons.")]
    ai, _ = make_ai(events)
    progress, result = await collect(ai)

    # no reasoning -> Progress falls back to the answer text
    assert progress == ["Mars has ", "two moons."]
    assert result.answer == "Mars has two moons."


async def test_reasoning_suppresses_answer_fallback(make_ai):
    # once reasoning is seen, the answer must not also be streamed as thinking
    events = [reasoning("Hmm. "), answer("Final answer here.")]
    ai, _ = make_ai(events)
    progress, result = await collect(ai)

    assert progress == ["Hmm. "]
    assert result.answer == "Final answer here."


async def test_thinking_and_answer_are_cleaned(make_ai):
    events = [
        reasoning("**Bold** reasoning "),
        answer("Neptune has 16 moons "),
        answer("([nasa.gov](https://nasa.gov/x))."),
    ]
    ai, _ = make_ai(events)
    progress, result = await collect(ai)

    joined = "".join(progress)
    assert "**" not in joined
    # citation/link stripped from the spoken answer
    assert "http" not in result.answer
    assert "](" not in result.answer
    assert result.answer.startswith("Neptune has 16 moons")


async def test_web_search_progress_streamed_as_text(make_ai):
    events = [
        search_start(),
        search_done("finance: BTC"),
        answer("Bitcoin is $64,175."),
    ]
    ai, _ = make_ai(events)
    progress, result = await collect(ai)

    joined = "".join(progress)
    assert "Searching the web" in joined      # gap-filling status
    assert "finance: BTC" in joined            # the actual query
    assert result.answer == "Bitcoin is $64,175."  # search text not in answer


async def test_non_web_search_output_item_done_ignored(make_ai):
    # output_item.done for a non-web_search item must not emit progress
    msg_item = SimpleNamespace(type="message", action=None)
    events = [
        SimpleNamespace(type="response.output_item.done", item=msg_item),
        answer("Hi."),
    ]
    ai, _ = make_ai(events)
    progress, result = await collect(ai)
    assert progress == ["Hi."]
    assert result.answer == "Hi."


async def test_request_includes_reasoning_and_web_search(make_ai):
    ai, fake = make_ai([answer("hi")])
    await collect(ai, previous_response_id="resp_prev")

    assert {"type": "web_search"} in fake.kwargs["tools"]
    assert fake.kwargs["reasoning"] == {"summary": "auto"}
    assert fake.kwargs["previous_response_id"] == "resp_prev"
