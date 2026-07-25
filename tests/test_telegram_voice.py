"""Tests for the Telegram bot's voice-note upload path.

``_handle_voice`` downloads a voice message and POSTs it to the server's
``/voice-note``. A real local aiohttp server stands in for the REST server and
records the multipart request; a fake client supplies the audio and captures the
chat replies, so no Telegram API is touched.
"""

from __future__ import annotations

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from techno_optimism_server import telegram_bot as tb

VOICE = b"OggS\x00OGG_OPUS_BYTES"


class FakeClient:
    """Minimal stand-in exposing only what _handle_voice uses."""

    def __init__(self, session, audio=VOICE):
        self._session = session
        self._audio = audio
        self.messages: list[str] = []

    async def download_bytes(self, file_id):
        return self._audio

    async def send_message(self, chat_id, text, reply_markup=None):
        self.messages.append(text)
        return 1


@pytest.fixture
async def receiver(monkeypatch):
    """A local /voice-note server; yields (client, captured-request-record)."""
    captured: dict = {}

    async def handle(request: web.Request) -> web.Response:
        data = await request.post()
        file_field = next(
            (v for v in data.values() if isinstance(v, web.FileField)), None
        )
        captured["timestamp"] = data.get("timestamp")
        captured["auth"] = request.headers.get("X-Auth")
        captured["body"] = file_field.file.read() if file_field else None
        status = captured.get("status", 201)
        if status != 201:
            return web.json_response({"error": "boom"}, status=status)
        return web.json_response({"id": 1, "filename": "x.mp3"}, status=201)

    app = web.Application()
    app.add_routes([web.post("/voice-note", handle)])
    server = TestServer(app)
    await server.start_server()

    monkeypatch.setattr(tb, "SERVER_URL", str(server.make_url("")).rstrip("/"))
    monkeypatch.setattr(tb, "ACCESS_TOKEN", "test-token")

    session = ClientSession()
    yield FakeClient(session), captured
    await session.close()
    await server.close()


async def test_voice_note_posted_to_server(receiver):
    client, captured = receiver
    await tb._handle_voice(
        client, chat_id=99, voice={"file_id": "abc", "mime_type": "audio/ogg"},
        date=1200000000,
    )
    assert captured["timestamp"] == "1200000000"
    assert captured["auth"] == "test-token"
    assert captured["body"] == VOICE
    assert client.messages == ["Voice note saved."]


async def test_voice_note_server_error_reported(receiver):
    client, captured = receiver
    captured["status"] = 400
    await tb._handle_voice(
        client, chat_id=99, voice={"file_id": "abc"}, date=1200000000,
    )
    assert client.messages == ["Could not save voice note (HTTP 400)."]


async def test_voice_note_requires_access_token(monkeypatch):
    monkeypatch.setattr(tb, "ACCESS_TOKEN", None)
    client = FakeClient(session=None)
    await tb._handle_voice(
        client, chat_id=99, voice={"file_id": "abc"}, date=1200000000,
    )
    assert client.messages == ["Server is not configured to accept voice notes."]
