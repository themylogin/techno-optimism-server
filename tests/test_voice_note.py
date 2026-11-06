"""Tests for the POST /voice-note proxy to the AI assistant.

A real local aiohttp server stands in for the assistant and records what it
receives, so the forwarded multipart body is checked end to end.
"""

from __future__ import annotations

import io
import math
import shutil
import struct
import subprocess
import wave
from datetime import datetime

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from techno_optimism_server import voice_note as vn
from techno_optimism_server.audio import DEFAULT_LEVEL_FILTER, SAMPLE_RATE, level_audio

MP3 = b"\xff\xf3VOICE_NOTE_BYTES"
TIMESTAMP = "2026-10-30T15:11:58+01:00"


@pytest.fixture(autouse=True)
def no_leveling(monkeypatch):
    """Disable ffmpeg leveling so the forwarding tests see the bytes as sent;
    the leveling tests below re-enable it."""
    monkeypatch.setenv("LEVEL_FILTER", "")


def _quiet_wav(seconds: float = 10.0, amplitude: int = 300) -> bytes:
    """A quiet 440 Hz mono WAV, so leveling has something to boost."""
    rate = 22050
    frames = b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(int(rate * seconds))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def _peak(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        data = w.readframes(w.getnframes())
    return max(abs(v) for (v,) in struct.iter_unpack("<h", data))


def _decode_wav(audio: bytes) -> bytes:
    """Decode any audio to WAV with ffmpeg (to measure an MP3's loudness)."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
        input=audio, capture_output=True, check=True,
    ).stdout


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
    assert resp.status == 201, await resp.text()
    assert await resp.json() == {"filename": "voice-note.mp3"}

    assert assistant["api_key"] == "assistant-token"
    assert assistant["timestamp"] == TIMESTAMP
    assert assistant["file"] == MP3
    assert assistant["filename"] == "voice-note.mp3"


async def test_assistant_202_becomes_201(client, assistant):
    assistant["status"] = 202
    resp = await _post(client)
    assert resp.status == 201


async def test_epoch_timestamp_sent_as_iso(client, assistant):
    resp = await _post(client, timestamp="1200000000")
    assert resp.status == 201
    sent = datetime.fromisoformat(assistant["timestamp"])
    assert sent.tzinfo is not None
    assert sent.timestamp() == 1200000000


async def test_naive_iso_timestamp_gets_offset(client, assistant):
    resp = await _post(client, timestamp="2026-07-25T12:30:00")
    assert resp.status == 201
    assert datetime.fromisoformat(assistant["timestamp"]).tzinfo is not None


async def test_any_file_field_name_accepted(client, assistant):
    resp = await _post(client, field="audio")
    assert resp.status == 201
    assert assistant["file"] == MP3


async def test_ogg_extension_preserved(client, assistant):
    resp = await _post(client, filename="voice.ogg")
    assert resp.status == 201
    assert assistant["filename"] == "voice-note.ogg"
    assert assistant["content_type"] == "audio/ogg"


async def test_unknown_extension_falls_back_to_mp3(client, assistant):
    resp = await _post(client, filename="note.bin")
    assert resp.status == 201
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_audio_leveled_to_mp3(client, assistant, monkeypatch):
    monkeypatch.setenv("LEVEL_FILTER", DEFAULT_LEVEL_FILTER)
    quiet = _quiet_wav()
    resp = await _post(client, filename="note.wav", file=quiet)
    assert resp.status == 201, await resp.text()
    assert await resp.json() == {"filename": "voice-note.mp3"}

    assert assistant["filename"] == "voice-note.mp3"
    assert assistant["content_type"] == "audio/mpeg"
    sent = assistant["file"]
    assert sent[:3] == b"ID3" or sent[0] == 0xFF  # an MP3 stream
    decoded = _decode_wav(sent)
    with wave.open(io.BytesIO(decoded)) as w:  # source format kept
        assert (w.getnchannels(), w.getframerate()) == (1, 22050)
    assert _peak(decoded) > 2 * _peak(quiet)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_leveling_is_one_static_gain_set_by_the_middle():
    """A full-scale handling thump, 6 s of quiet "speech" that gets louder
    halfway, another thump. The thumps sit in the ignored edges, so the gain
    is set by the speech alone; and it's one gain for the whole file, so the
    speech is lifted from its very start and its own dynamics are kept."""
    rate = 16000

    def tone(seconds, freq, amplitude):
        return [amplitude * math.sin(2 * math.pi * freq * i / rate)
                for i in range(int(rate * seconds))]

    thump = tone(0.25, 50, 32000)
    samples = thump + tone(3.0, 300, 100) + tone(3.0, 300, 300) + thump
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(v)) for v in samples))

    out = await level_audio(buf.getvalue(), DEFAULT_LEVEL_FILTER)
    with wave.open(io.BytesIO(out)) as w:
        data = [v for (v,) in struct.iter_unpack("<h", w.readframes(w.getnframes()))]

    def rms_db(seq, start, end):
        chunk = seq[int(start * rate):int(end * rate)]
        return 20 * math.log10(math.sqrt(sum(v * v for v in chunk) / len(chunk)) / 32768)

    # Gain per 250 ms window of speech, skipping the first one: the clipped
    # thump's high-pass ringing spills into it.
    windows = [t / 4 for t in range(2, 24)]
    gains = [rms_db(data, t, t + 0.25) - rms_db(samples, t, t + 0.25) for t in windows]
    assert max(gains) - min(gains) < 0.5, gains  # static: no wobble, no ramp
    assert min(gains) > 20, gains  # quiet speech (-50/-40 dBFS) lifted a lot
    assert rms_db(data, 3.5, 6.0) > -25  # the louder half lands near target


async def test_undecodable_audio_forwarded_unchanged(client, assistant, monkeypatch):
    # ffmpeg can't decode these bytes (or isn't there): the original goes out.
    monkeypatch.setenv("LEVEL_FILTER", DEFAULT_LEVEL_FILTER)
    resp = await _post(client, filename="voice.ogg")
    assert resp.status == 201
    assert assistant["file"] == MP3
    assert assistant["filename"] == "voice-note.ogg"
