"""Tests for the POST /voice-note proxy to the AI assistant.

A real local aiohttp server stands in for the assistant and records what it
receives, so the forwarded multipart body is checked end to end.
"""

from __future__ import annotations

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


async def _post(client):
    form = FormData()
    form.add_field("timestamp", TIMESTAMP)
    form.add_field("file", MP3, filename="note.mp3", content_type="audio/mpeg")
    return await client.post("/voice-note", data=form)


async def test_upload_forwarded_to_assistant(client, assistant):
    resp = await _post(client)
    assert resp.status == 200, await resp.text()
    assert await resp.json() == {"ok": True}

    assert assistant["api_key"] == "assistant-token"
    assert assistant["timestamp"] == TIMESTAMP
    assert assistant["file"] == MP3
    assert assistant["filename"] == "note.mp3"


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
