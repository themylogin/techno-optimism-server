"""Branch-coverage tests for the REST interaction handlers.

The handlers are driven through a real aiohttp test client; the AI boundary is
replaced with a FakeAI so every branch is deterministic and no network or API
key is needed. Storage writes to a per-test temp directory.

Conversation examples are the real Mars-moons exchanges used while building the
server.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from techno_optimism_server import handlers, storage
from techno_optimism_server.ai import Progress, Result
from techno_optimism_server.handlers import (
    create_interaction,
    get_answer_audio,
    get_interaction,
    health,
    upload_context,
)

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
            yield chunk if isinstance(chunk, Progress) else Progress(chunk)
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
        app["jobs"] = {}
        app.add_routes([
            web.get("/health", health),
            web.post("/v1/interactions", create_interaction),
            web.get("/v1/interactions/{id}", get_interaction),
            web.put("/v1/interactions/{id}/context", upload_context),
            web.get("/v1/interactions/{id}/answer.mp3", get_answer_audio),
        ])
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        await c.close()


async def create(client, audio=QUESTION_AUDIO, **params):
    """POST a question blob; return (id, first snapshot)."""
    resp = await client.post("/v1/interactions", data=audio, params=params)
    assert resp.status == 201, await resp.text()
    snap = await resp.json()
    return snap["id"], snap


async def poll(client, iid, until=lambda s: s["status"] in ("done", "error"),
               timeout=2.0):
    """Poll the status endpoint until `until(snapshot)`; return that snapshot."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    snap = None
    while True:
        resp = await client.get(f"/v1/interactions/{iid}")
        assert resp.status == 200
        snap = await resp.json()
        if until(snap):
            return snap
        if loop.time() > deadline:
            raise AssertionError(f"poll timed out; last snapshot: {snap}")
        await asyncio.sleep(0.01)


def interaction_dir(base):
    """The single interaction directory created under `base`."""
    dirs = {p.parent for p in base.rglob("question.mp3")}
    assert len(dirs) == 1, f"expected one interaction dir, got {dirs}"
    return dirs.pop()


def load_json(d):
    return json.loads((d / "interaction.json").read_text(encoding="utf-8"))


# --- tests ---------------------------------------------------------------- #
async def test_health_endpoint(make_client):
    client = await make_client(FakeAI())
    resp = await client.get("/health")
    assert resp.status == 200
    assert await resp.json() == {"status": "ok"}


async def test_create_returns_immediately(make_client):
    # POST returns 201 with an initial snapshot before any AI work is observable
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX})
    client = await make_client(ai)
    iid, snap = await create(client)
    assert iid
    assert snap["id"] == iid
    assert snap["status"] in ("transcribing", "thinking", "synthesizing", "done")


async def test_empty_body_rejected(make_client):
    client = await make_client(FakeAI())
    resp = await client.post("/v1/interactions", data=b"")
    assert resp.status == 400
    assert (await resp.json())["error"] == "empty_body"


async def test_no_context_happy_path(make_client, tmp_path):
    ai = FakeAI(needs_context=False,
                transcripts={QUESTION_AUDIO: Q_NO_CTX},
                # Progress items are full display snapshots (ask assembles them)
                progress=["Mars has ", "Mars has two moons: Phobos and Deimos."],
                answer=A_NO_CTX)
    client = await make_client(ai)
    iid, _ = await create(client)
    snap = await poll(client, iid)

    assert snap["status"] == "done"
    assert snap["question"] == Q_NO_CTX
    assert snap["answer_text"] == A_NO_CTX
    assert snap["response_id"] == "resp_test123"
    assert snap["answer_audio_url"] == f"/v1/interactions/{iid}/answer.mp3"
    # thinking settled on the final (full) snapshot text
    assert snap["thinking"] == "Mars has two moons: Phobos and Deimos."

    # answer audio is downloadable
    audio = await client.get(snap["answer_audio_url"])
    assert audio.status == 200
    assert await audio.read() == SPEECH

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
    # fast stream + 1s archive interval -> only the final tail archived
    assert data["progress"] == ["Mars has two moons: Phobos and Deimos."]
    assert ("ask", Q_NO_CTX, None, None) in ai.calls


async def test_need_context_happy_path(make_client, tmp_path):
    ai = FakeAI(needs_context=True,
                transcripts={QUESTION_AUDIO: Q_CTX, CONTEXT_AUDIO: CTX},
                progress=["No. "], answer=A_CTX)
    client = await make_client(ai)
    iid, _ = await create(client)

    # job parks in need_context until we upload the follow-up audio
    await poll(client, iid, until=lambda s: s["status"] == "need_context")
    resp = await client.put(f"/v1/interactions/{iid}/context", data=CONTEXT_AUDIO)
    assert resp.status == 200

    snap = await poll(client, iid)
    assert snap["status"] == "done"
    assert snap["answer_text"] == A_CTX

    audio = await client.get(snap["answer_audio_url"])
    assert await audio.read() == SPEECH

    d = interaction_dir(tmp_path)
    assert (d / "context.mp3").read_bytes() == CONTEXT_AUDIO
    data = load_json(d)
    assert data["needs_context"] is True
    assert data["context"] == CTX
    assert data["answer"] == A_CTX
    assert ("ask", Q_CTX, CTX, None) in ai.calls


async def test_context_upload_is_idempotent(make_client, tmp_path):
    # a retried context upload must not restart or corrupt the job
    ai = FakeAI(needs_context=True,
                transcripts={QUESTION_AUDIO: Q_CTX, CONTEXT_AUDIO: CTX},
                answer=A_CTX)
    client = await make_client(ai)
    iid, _ = await create(client)
    await poll(client, iid, until=lambda s: s["status"] == "need_context")

    r1 = await client.put(f"/v1/interactions/{iid}/context", data=CONTEXT_AUDIO)
    r2 = await client.put(f"/v1/interactions/{iid}/context", data=b"IGNORED_RETRY")
    assert r1.status == 200 and r2.status == 200

    snap = await poll(client, iid)
    assert snap["status"] == "done"
    # only the first upload counted; the retry's bytes were ignored
    assert (interaction_dir(tmp_path) / "context.mp3").read_bytes() == CONTEXT_AUDIO
    assert sum(1 for c in ai.calls if c[0] == "transcribe") == 2  # question + context


async def test_need_context_timeout(make_client, tmp_path):
    ai = FakeAI(needs_context=True, transcripts={QUESTION_AUDIO: Q_CTX})
    client = await make_client(ai, context_timeout=0.1)
    iid, _ = await create(client)

    snap = await poll(client, iid)  # never upload context -> times out to error
    assert snap["status"] == "error"
    assert snap["error"]["detail"] == "context_not_received"

    d = interaction_dir(tmp_path)
    data = load_json(d)
    assert data["needs_context"] is True
    assert "context" not in data
    assert not (d / "context.mp3").exists()


async def test_previous_response_id(make_client, tmp_path):
    ai = FakeAI(needs_context=False, transcripts={QUESTION_AUDIO: Q_NO_CTX},
                response_id="resp_new456")
    client = await make_client(ai)
    iid, _ = await create(client, previous_response_id="resp_prev123")
    snap = await poll(client, iid)

    assert snap["status"] == "done"
    assert snap["response_id"] == "resp_new456"
    assert ("ask", Q_NO_CTX, None, "resp_prev123") in ai.calls
    data = load_json(interaction_dir(tmp_path))
    assert data["previous_response_id"] == "resp_prev123"
    assert data["response_id"] == "resp_new456"


@pytest.mark.parametrize("fail_in", ["transcribe", "needs_context", "ask", "say"])
async def test_processing_error_saves_traceback(make_client, tmp_path, fail_in):
    ai = FakeAI(needs_context=False,
                transcripts={QUESTION_AUDIO: Q_NO_CTX},
                fail_in=fail_in, fail_exc=ValueError("kaboom"))
    client = await make_client(ai)
    iid, _ = await create(client)
    snap = await poll(client, iid)

    assert snap["status"] == "error"
    assert "kaboom" in snap["error"]["detail"]

    d = interaction_dir(tmp_path)
    assert (d / "question.mp3").read_bytes() == QUESTION_AUDIO
    tb = (d / "error.txt").read_text(encoding="utf-8")
    assert "Traceback" in tb
    assert "ValueError: kaboom" in tb
    assert not (d / "response.mp3").exists()
    # no answer audio to download
    audio = await client.get(f"/v1/interactions/{iid}/answer.mp3")
    assert audio.status == 404


async def test_error_before_transcript_text(make_client, tmp_path):
    # transcribe fails -> question.mp3 saved, but no interaction.json yet
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX},
                fail_in="transcribe", fail_exc=RuntimeError("stt down"))
    client = await make_client(ai)
    iid, _ = await create(client)
    await poll(client, iid)

    d = interaction_dir(tmp_path)
    assert (d / "question.mp3").exists()
    assert (d / "error.txt").exists()
    assert not (d / "interaction.json").exists()


async def test_error_after_answer_before_tts(make_client, tmp_path):
    # say() fails -> answer + response_id already in json, but no response.mp3
    ai = FakeAI(needs_context=False,
                transcripts={QUESTION_AUDIO: Q_NO_CTX},
                answer=A_NO_CTX, fail_in="say")
    client = await make_client(ai)
    iid, _ = await create(client)
    snap = await poll(client, iid)

    assert snap["status"] == "error"
    d = interaction_dir(tmp_path)
    data = load_json(d)
    assert data["answer"] == A_NO_CTX
    assert data["response_id"] == "resp_test123"
    assert not (d / "response.mp3").exists()
    assert (d / "error.txt").exists()


async def test_thinking_tail_capped_at_100_chars(make_client):
    full = ("Neptune, the eighth planet, has sixteen known moons and the "
            "fastest winds in the solar system, reaching supersonic speeds.")
    assert len(full) > 100
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, progress=[full])
    client = await make_client(ai)
    iid, _ = await create(client)
    snap = await poll(client, iid)

    # truncated: last 100 chars, prefixed with … to show the start was cut
    assert snap["thinking"] == "…" + full[-100:]
    assert len(snap["thinking"]) == 101


async def test_thinking_archived_periodically(make_client, tmp_path):
    # slow stream + short archive interval -> multiple archived snapshots
    snapshots = ["one ", "one two ", "one two three ", "one two three four ",
                 "one two three four five "]
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, progress=snapshots,
                chunk_delay=0.15)
    client = await make_client(ai, thinking_interval=0.05)
    iid, _ = await create(client)
    snap = await poll(client, iid, timeout=5.0)

    assert snap["status"] == "done"
    data = load_json(interaction_dir(tmp_path))
    progress = data["progress"]
    assert len(progress) >= 2                          # archived more than once
    assert progress[-1] == snapshots[-1]               # settled on the end
    # each archived frame is a growing prefix of the full text
    for p in progress:
        assert snapshots[-1].startswith(p)
    assert [len(p) for p in progress] == sorted(len(p) for p in progress)


async def test_unknown_stream_item_is_ignored(make_client, tmp_path):
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX},
                progress=["Mars "], unknown_items=[object(), 42])
    client = await make_client(ai)
    iid, _ = await create(client)
    snap = await poll(client, iid)

    assert snap["status"] == "done"
    data = load_json(interaction_dir(tmp_path))
    assert data["answer"] == A_NO_CTX


async def test_status_unknown_id_is_404(make_client):
    client = await make_client(FakeAI())
    resp = await client.get("/v1/interactions/nope")
    assert resp.status == 404
    assert (await resp.json())["error"] == "unknown_interaction"


async def test_context_unknown_id_is_404(make_client):
    client = await make_client(FakeAI())
    resp = await client.put("/v1/interactions/nope/context", data=CONTEXT_AUDIO)
    assert resp.status == 404


async def test_context_empty_body_rejected(make_client):
    ai = FakeAI(needs_context=True, transcripts={QUESTION_AUDIO: Q_CTX})
    client = await make_client(ai, context_timeout=5.0)
    iid, _ = await create(client)
    await poll(client, iid, until=lambda s: s["status"] == "need_context")
    resp = await client.put(f"/v1/interactions/{iid}/context", data=b"")
    assert resp.status == 400
    assert (await resp.json())["error"] == "empty_body"


async def test_answer_audio_unknown_id_is_404(make_client):
    client = await make_client(FakeAI())
    resp = await client.get("/v1/interactions/nope/answer.mp3")
    assert resp.status == 404
    assert (await resp.json())["error"] == "unknown_interaction"


async def test_answer_audio_404_before_ready(make_client):
    # a valid, still-in-flight job has no answer file yet
    ai = FakeAI(needs_context=True, transcripts={QUESTION_AUDIO: Q_CTX})
    client = await make_client(ai, context_timeout=5.0)
    iid, _ = await create(client)
    await poll(client, iid, until=lambda s: s["status"] == "need_context")

    resp = await client.get(f"/v1/interactions/{iid}/answer.mp3")
    assert resp.status == 404
    assert (await resp.json())["error"] == "answer_not_ready"


async def test_answer_audio_supports_range(make_client):
    ai = FakeAI(transcripts={QUESTION_AUDIO: Q_NO_CTX}, speech=SPEECH)
    client = await make_client(ai)
    iid, _ = await create(client)
    snap = await poll(client, iid)

    # a resumed download asks for the tail via Range
    resp = await client.get(snap["answer_audio_url"],
                            headers={"Range": "bytes=2-"})
    assert resp.status == 206
    assert await resp.read() == SPEECH[2:]
