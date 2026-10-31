"""Tests for the POST /voice-note proxy to the AI assistant.

A real local aiohttp server stands in for the assistant and records what it
receives, so the forwarded multipart body is checked end to end.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from techno_optimism_server import voice_note as vn

MP3 = b"\xff\xf3VOICE_NOTE_BYTES"
TIMESTAMP = "2026-10-30T15:11:58+01:00"


@pytest.fixture
async def assistant(monkeypatch):
    """A fake assistant; yields the dict recording its last request."""
    captured: dict = {"status": 200}

    async def handle(request: web.Request) -> web.Response:
        data = await request.post()
        captured["api_key"] = request.headers.get("X-API-Key")
        captured["timestamp"] = data.get("timestamp")
        captured["file"] = data["file"].file.read()
        captured["filename"] = data["file"].filename
        captured["content_type"] = data["file"].content_type
        return web.json_response({"ok": True}, status=captured["status"])

    app = web.Application()
    app.add_routes([web.post(vn.UPLOAD_PATH, handle)])
    server = TestServer(app)
    await server.start_server()

    monkeypatch.setenv("AI_ASSISTANT_URL", str(server.make_url("")))
    monkeypatch.setenv("AI_ASSISTANT_TOKEN", "assistant-token")
    yield captured
    await server.close()


@pytest.fixture
async def client():
    app = web.Application()
    app.add_routes([web.post("/voice-note", vn.create_voice_note)])
    c = TestClient(TestServer(app))
    await c.start_server()
    yield c
    await c.close()


async def _post(client, timestamp=TIMESTAMP, field="file", filename="note.mp3", file=MP3):
    form = FormData()
    if timestamp is not None:
        form.add_field("timestamp", timestamp)
    if file is not None:
        form.add_field(field, file, filename=filename, content_type="audio/mpeg")
    return await client.post("/voice-note", data=form)


async def test_upload_forwarded_to_assistant(client, assistant):
    resp = await _post(client)
    assert resp.status == 200, await resp.text()
    assert await resp.json() == {"ok": True}

    assert assistant["api_key"] == "assistant-token"
    assert assistant["timestamp"] == TIMESTAMP
    assert assistant["file"] == MP3
    assert assistant["filename"] == "voice-note.mp3"


async def test_epoch_timestamp_sent_as_iso(client, assistant):
    resp = await _post(client, timestamp="1200000000")
    assert resp.status == 200
    sent = datetime.fromisoformat(assistant["timestamp"])
    assert sent.tzinfo is not None
    assert sent.timestamp() == 1200000000


async def test_naive_iso_timestamp_gets_offset(client, assistant):
    resp = await _post(client, timestamp="2026-07-25T12:30:00")
    assert resp.status == 200
    assert datetime.fromisoformat(assistant["timestamp"]).tzinfo is not None


async def test_any_file_field_name_accepted(client, assistant):
    resp = await _post(client, field="audio")
    assert resp.status == 200
    assert assistant["file"] == MP3


async def test_ogg_extension_preserved(client, assistant):
    resp = await _post(client, filename="voice.ogg")
    assert resp.status == 200
    assert assistant["filename"] == "voice-note.ogg"
    assert assistant["content_type"] == "audio/ogg"


async def test_unknown_extension_falls_back_to_mp3(client, assistant):
    resp = await _post(client, filename="note.bin")
    assert resp.status == 200
    assert assistant["filename"] == "voice-note.mp3"
    assert assistant["content_type"] == "audio/mpeg"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"timestamp": None}, "missing_timestamp"),
        ({"timestamp": "not-a-date"}, "bad_timestamp"),
        ({"file": None}, "missing_file"),
        ({"file": b""}, "empty_file"),
    ],
)
async def test_bad_upload_rejected(client, kwargs, error):
    resp = await _post(client, **kwargs)
    assert resp.status == 400
    assert (await resp.json())["error"] == error


async def test_assistant_error_passed_through(client, assistant):
    assistant["status"] = 422
    resp = await _post(client)
    assert resp.status == 422
    assert await resp.json() == {"ok": True}


async def test_assistant_unreachable(client, monkeypatch):
    monkeypatch.setenv("AI_ASSISTANT_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("AI_ASSISTANT_TOKEN", "assistant-token")
    resp = await _post(client)
    assert resp.status == 502
    assert (await resp.json())["error"] == "assistant_unreachable"
