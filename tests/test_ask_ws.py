"""Branch-coverage tests for the ask_ws WebSocket handler.

The handler is driven through a real aiohttp test WebSocket; the AI boundary
is replaced with a FakeAI so every branch is deterministic and no network or
API key is needed. Storage writes to a per-test temp directory.

Conversation examples are the real Mars-moons exchanges used while building
the server.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from techno_optimism_server import handlers, storage
from techno_optimism_server.ai import Progress, Result
from techno_optimism_server.handlers import ask_ws, health

# --- real conversation examples ------------------------------------------- #
Q_NO_CTX = "How many moons does Mars have?"
A_NO_CTX = "Mars has two moons: Phobos and Deimos."
Q_CTX = "Is it true what he just said about Mars?"
CTX = "The speaker claimed that Mars has 17 moons and a breathable atmosphere."
A_CTX = ("No. Mars has two moons, Phobos and Deimos, and its thin, "
         "carbon-dioxide-rich atmosphere is not breathable.")

QUESTION_AUDIO = b"QUESTION_AUDIO_BYTES"
CONTEXT_AUDIO = b"CONTEXT_AUDIO_BYTES"
SPEECH = b"\xff\xf3ANSWER_MP3_BYTES"


# --- fake AI -------------------------------------------------------------- #
class FakeAI:
    """Stand-in for techno_optimism_server.ai.AI with scripted results."""

    def __init__(self, *, needs_context=False, transcripts=None, progress=(),
                 answer=A_NO_CTX, response_id="resp_test123", speech=SPEECH,
                 fail_in=None, fail_exc=None, unknown_items=(), chunk_delay=0.0):
        self._needs = needs_context
        self._transcripts = transcripts or {}
        self._progress = list(progress)
        self._answer = answer
        self._response_id = response_id
        self._speech = speech
        self._fail_in = fail_in
        self._fail_exc = fail_exc or RuntimeError("boom")
        self._unknown_items = list(unknown_items)
        self._chunk_delay = chunk_delay
        self.calls: list = []

    def _maybe_fail(self, where):
        if self._fail_in == where:
            raise self._fail_exc

    async def transcribe(self, audio: bytes) -> str:
        self.calls.append(("transcribe", audio))
        self._maybe_fail("transcribe")
        return self._transcripts.get(audio, "")

    async def needs_context(self, question: str) -> bool:
        self.calls.append(("needs_context", question))
        self._maybe_fail("needs_context")
        return self._needs

    async def ask(self, question, context=None, previous_response_id=None):
        self.calls.append(("ask", question, context, previous_response_id))
        self._maybe_fail("ask")
        for chunk in self._progress:
            if self._chunk_delay:
                await asyncio.sleep(self._chunk_delay)
            yield Progress(chunk)
        for item in self._unknown_items:
            yield item
        yield Result(self._response_id, self._answer)

    async def say(self, text: str) -> bytes:
        self.calls.append(("say", text))
        self._maybe_fail("say")
        return self._speech


# --- fixtures & helpers --------------------------------------------------- #
@pytest.fixture
async def make_client(tmp_path, monkeypatch):
    """Factory building a started TestClient wired to a given FakeAI."""
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    clients: list[TestClient] = []

    async def _make(ai, context_timeout=0.3, thinking_interval=1.0):
        monkeypatch.setattr(handlers, "CONTEXT_TIMEOUT", context_timeout)
        monkeypatch.setattr(handlers, "THINKING_INTERVAL", thinking_interval)
        app = web.Application()
        app["ai"] = ai
        app.add_routes([web.get("/v1/ask", ask_ws), web.get("/health", health)])
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        await c.close()


async def drain(ws):
    """Collect all (json_messages, binary_blobs) until the socket closes."""
    texts, blobs = [], []
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            texts.append(json.loads(msg.data))
        elif msg.type == WSMsgType.BINARY:
            blobs.append(msg.data)
        elif msg.type in (WSMsgType.CLOSED, WSMsgType.CLOSING, WSMsgType.ERROR):
            break
    return texts, blobs


def interaction_dir(base):
    """The single interaction directory created under `base`."""
    dirs = {p.parent for p in base.rglob("question.mp3")}
    assert len(dirs) == 1, f"expected one interaction dir, got {dirs}"
    return dirs.pop()


def load_json(d):
    return json.loads((d / "interaction.json").read_text(encoding="utf-8"))


# --- tests ---------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [
    "not json at all",
    "{}",                                   # object, no key
    '{"previous_response_id": ""}',         # empty id
    '{"previous_response_id": 5}',          # wrong type
    "[1, 2, 3]",                            # not an object
])
async def test_invalid_handshake_is_rejected(make_client, tmp_path, payload):
    client = await make_client(FakeAI())
    ws = await client.ws_connect("/v1/ask")
    await ws.send_str(payload)
    texts, blobs = await drain(ws)
    assert texts == [{"ok": False, "error": "invalid_handshake"}]
    assert blobs == []
    assert not list(tmp_path.rglob("question.mp3"))


async def test_non_binary_question_frame_is_rejected(make_client, tmp_path):
    # valid handshake, but the frame that should be the question is text
    client = await make_client(FakeAI())
    ws = await client.ws_connect("/v1/ask")
    await ws.send_str(json.dumps({"previous_response_id": "resp_prev"}))
    await ws.send_str("still not audio")
    texts, blobs = await drain(ws)
    assert texts == [{"ok": False, "error": "expected_binary_frame"}]
    assert blobs == []
    assert not list(tmp_path.rglob("question.mp3"))


async def test_previous_response_id_handshake(make_client, tmp_path):
    ai = FakeAI(needs_context=False, transcripts={QUESTION_AUDIO: Q_NO_CTX},
                response_id="resp_new456")
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_str(json.dumps({"previous_response_id": "resp_prev123"}))
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    assert {"msg": "uploaded"} in texts
    assert blobs == [SPEECH]
    # ask received the previous_response_id
    assert ("ask", Q_NO_CTX, None, "resp_prev123") in ai.calls
    # both ids persisted; previous via its own method, new via response_text
    data = load_json(interaction_dir(tmp_path))
    assert data["previous_response_id"] == "resp_prev123"
    assert data["response_id"] == "resp_new456"
    # done message carries the new id for the next turn
    done = [t for t in texts if t.get("msg") == "done"]
    assert done == [{"msg": "done", "response_id": "resp_new456"}]


async def test_done_message_precedes_final_blob(make_client):
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, progress=["Mars "],
                response_id="resp_final")
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)

    seen = []
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            seen.append(("text", json.loads(msg.data)))
        elif msg.type == WSMsgType.BINARY:
            seen.append(("blob", msg.data))
        elif msg.type in (WSMsgType.CLOSED, WSMsgType.CLOSING, WSMsgType.ERROR):
            break

    done_idx = next(i for i, s in enumerate(seen)
                    if s[0] == "text" and s[1].get("msg") == "done")
    blob_idx = next(i for i, s in enumerate(seen) if s[0] == "blob")
    assert done_idx < blob_idx
    assert seen[done_idx][1] == {"msg": "done", "response_id": "resp_final"}


async def test_no_context_happy_path(make_client, tmp_path):
    ai = FakeAI(needs_context=False,
                transcripts={QUESTION_AUDIO: Q_NO_CTX},
                progress=["Mars has ", "two moons: Phobos and Deimos."],
                answer=A_NO_CTX)
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    assert texts[0] == {"msg": "uploaded"}
    # thinking is now a tail-of-full-text ticker, not per-delta; the full
    # answer is short, so the (single, fast) frame carries all of it.
    thinking = [t["text"] for t in texts if t.get("msg") == "thinking"]
    assert thinking[-1] == "Mars has two moons: Phobos and Deimos."
    assert all(t.get("msg") != "need_context" for t in texts)
    assert blobs == [SPEECH]

    d = interaction_dir(tmp_path)
    assert (d / "question.mp3").read_bytes() == QUESTION_AUDIO
    assert (d / "response.mp3").read_bytes() == SPEECH
    assert not (d / "context.mp3").exists()
    data = load_json(d)
    assert data["question"] == Q_NO_CTX
    assert data["needs_context"] is False
    assert data["answer"] == A_NO_CTX
    assert data["response_id"] == "resp_test123"
    assert "context" not in data
    # the thinking texts actually sent to the client are saved as a list
    thinking_sent = [t["text"] for t in texts if t.get("msg") == "thinking"]
    assert data["progress"] == thinking_sent
    assert data["progress"] == ["Mars has two moons: Phobos and Deimos."]
    # ask was called with no context
    assert ("ask", Q_NO_CTX, None, None) in ai.calls


async def test_need_context_happy_path(make_client, tmp_path):
    ai = FakeAI(needs_context=True,
                transcripts={QUESTION_AUDIO: Q_CTX, CONTEXT_AUDIO: CTX},
                progress=["No. "], answer=A_CTX)
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")

    await ws.send_bytes(QUESTION_AUDIO)
    assert await ws.receive_json() == {"msg": "uploaded"}
    assert await ws.receive_json() == {"msg": "need_context"}
    await ws.send_bytes(CONTEXT_AUDIO)
    texts, blobs = await drain(ws)

    assert {"msg": "thinking", "text": "No. "} in texts
    assert blobs == [SPEECH]

    d = interaction_dir(tmp_path)
    assert (d / "context.mp3").read_bytes() == CONTEXT_AUDIO
    data = load_json(d)
    assert data["needs_context"] is True
    assert data["context"] == CTX
    assert data["answer"] == A_CTX
    assert ("ask", Q_CTX, CTX, None) in ai.calls


async def test_need_context_timeout(make_client, tmp_path):
    ai = FakeAI(needs_context=True, transcripts={QUESTION_AUDIO: Q_CTX})
    client = await make_client(ai, context_timeout=0.2)
    ws = await client.ws_connect("/v1/ask")

    await ws.send_bytes(QUESTION_AUDIO)
    assert await ws.receive_json() == {"msg": "uploaded"}
    assert await ws.receive_json() == {"msg": "need_context"}
    # never send the context blob -> server times out
    texts, blobs = await drain(ws)

    assert {"ok": False, "error": "context_not_received"} in texts
    assert blobs == []
    d = interaction_dir(tmp_path)
    data = load_json(d)
    assert data["needs_context"] is True
    assert "context" not in data
    assert not (d / "context.mp3").exists()


async def test_need_context_non_binary_reply(make_client, tmp_path):
    ai = FakeAI(needs_context=True, transcripts={QUESTION_AUDIO: Q_CTX})
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")

    await ws.send_bytes(QUESTION_AUDIO)
    assert await ws.receive_json() == {"msg": "uploaded"}
    assert await ws.receive_json() == {"msg": "need_context"}
    await ws.send_str("context typed as text, not audio")  # wrong frame type
    texts, blobs = await drain(ws)

    assert {"ok": False, "error": "context_not_received"} in texts
    assert blobs == []
    assert not (interaction_dir(tmp_path) / "context.mp3").exists()


@pytest.mark.parametrize("fail_in", ["transcribe", "needs_context", "ask", "say"])
async def test_processing_error_saves_traceback(make_client, tmp_path, fail_in):
    ai = FakeAI(needs_context=False,
                transcripts={QUESTION_AUDIO: Q_NO_CTX},
                fail_in=fail_in, fail_exc=ValueError("kaboom"))
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    assert {"msg": "uploaded"} in texts
    err = [t for t in texts if t.get("error") == "processing_failed"]
    assert err and "kaboom" in err[0]["detail"]
    assert blobs == []  # no answer audio on failure

    d = interaction_dir(tmp_path)
    assert (d / "question.mp3").read_bytes() == QUESTION_AUDIO
    tb = (d / "error.txt").read_text(encoding="utf-8")
    assert "Traceback" in tb
    assert "ValueError: kaboom" in tb
    # response audio never written on failure
    assert not (d / "response.mp3").exists()


async def test_error_before_transcript_text(make_client, tmp_path):
    # transcribe fails -> question.mp3 saved, but no question text in json
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX},
                fail_in="transcribe", fail_exc=RuntimeError("stt down"))
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    await drain(ws)

    d = interaction_dir(tmp_path)
    assert (d / "question.mp3").exists()
    assert (d / "error.txt").exists()
    assert not (d / "interaction.json").exists()  # nothing json-worthy yet


async def test_error_after_answer_before_tts(make_client, tmp_path):
    # say() fails -> answer + response_id already in json, but no response.mp3
    ai = FakeAI(needs_context=False,
                transcripts={QUESTION_AUDIO: Q_NO_CTX},
                answer=A_NO_CTX, fail_in="say")
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    assert blobs == []
    d = interaction_dir(tmp_path)
    data = load_json(d)
    assert data["answer"] == A_NO_CTX
    assert data["response_id"] == "resp_test123"
    assert not (d / "response.mp3").exists()
    assert (d / "error.txt").exists()


async def test_thinking_shows_full_text_when_fast(make_client):
    # Fast stream (no delay) finishes before a tick, so the guaranteed final
    # frame carries the whole accumulated text.
    chunks = ["Mars ", "has ", "two ", "moons."]
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, progress=chunks)
    client = await make_client(ai)  # default 1s interval; stream is instant
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    thinking = [t["text"] for t in texts if t.get("msg") == "thinking"]
    assert thinking == ["Mars has two moons."]
    assert blobs == [SPEECH]


async def test_thinking_tail_capped_at_100_chars(make_client):
    full = ("Neptune, the eighth planet, has sixteen known moons and the "
            "fastest winds in the solar system, reaching supersonic speeds.")
    assert len(full) > 100
    # one chunk so the whole thing arrives at once
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, progress=[full])
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    thinking = [t["text"] for t in texts if t.get("msg") == "thinking"]
    assert thinking[-1] == full[-100:]
    assert len(thinking[-1]) == 100
    assert blobs == [SPEECH]


async def test_thinking_ticks_periodically_with_growing_tail(make_client):
    # Slow stream + short interval -> multiple ticker frames, each a growing
    # suffix of the accumulated text.
    chunks = ["one ", "two ", "three ", "four ", "five "]
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, progress=chunks,
                chunk_delay=0.15)
    client = await make_client(ai, thinking_interval=0.05)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    thinking = [t["text"] for t in texts if t.get("msg") == "thinking"]
    assert len(thinking) >= 2                      # ticked more than once
    full = "".join(chunks)
    assert thinking[-1] == full[-100:]             # settles on the end
    # full text is < 100 chars, so each frame is the accumulation so far
    # (a growing prefix of the final text).
    for t in thinking:
        assert full.startswith(t)
    assert [len(t) for t in thinking] == sorted(len(t) for t in thinking)
    assert blobs == [SPEECH]


async def test_unknown_stream_item_is_ignored(make_client, tmp_path):
    # ask yields something that is neither Progress nor Result: it must be
    # silently skipped and the interaction still completes normally.
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX},
                progress=["Mars "], unknown_items=[object(), 42])
    client = await make_client(ai)
    ws = await client.ws_connect("/v1/ask")
    await ws.send_bytes(QUESTION_AUDIO)
    texts, blobs = await drain(ws)

    assert {"msg": "thinking", "text": "Mars "} in texts
    assert blobs == [SPEECH]  # still produced the answer
    data = load_json(interaction_dir(tmp_path))
    assert data["answer"] == A_NO_CTX


async def test_health_endpoint(make_client):
    client = await make_client(FakeAI())
    resp = await client.get("/health")
    assert resp.status == 200
    assert await resp.json() == {"status": "ok"}
