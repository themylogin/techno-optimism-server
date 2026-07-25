"""Tests for the POST /voice-note upload endpoint.

Driven through a real aiohttp test client against a temp SQLite database and a
temp voice-notes directory. No AI, network, or worker involved — the endpoint
only stores the file and inserts a row.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy import select

from techno_optimism_server import voice_note as vn
from techno_optimism_server.db import (
    Base,
    VoiceNote,
    make_engine,
    make_sessionmaker,
)

MP3 = b"\xff\xf3VOICE_NOTE_BYTES"
# 2026-07-25 12:30:00 local time.
TS = datetime(2026, 7, 25, 12, 30, 0)


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """A started test client with its own temp DB and voice-notes dir."""
    notes_dir = tmp_path / "voice-notes"
    monkeypatch.setattr(vn, "VOICE_NOTES_DIR", notes_dir)

    engine = make_engine(tmp_path / "db.sqlite3")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = make_sessionmaker(engine)

    app = web.Application()
    app[vn.DB_SESSIONMAKER_KEY] = sessionmaker
    app.add_routes([web.post("/voice-note", vn.create_voice_note)])

    c = TestClient(TestServer(app))
    await c.start_server()
    c._notes_dir = notes_dir
    c._sessionmaker = sessionmaker
    yield c
    await c.close()
    await engine.dispose()


async def _rows(client):
    async with client._sessionmaker() as session:
        result = await session.execute(select(VoiceNote))
        return result.scalars().all()


async def _post(client, *, timestamp, file=MP3, field="file", filename="note.mp3"):
    form = FormData()
    if timestamp is not None:
        form.add_field("timestamp", timestamp)
    if file is not None:
        form.add_field(field, file, filename=filename, content_type="audio/mpeg")
    return await client.post("/voice-note", data=form)


async def test_upload_stores_file_and_row(client):
    resp = await _post(client, timestamp="1200000000")  # 2008-01-10 21:20:00 UTC
    assert resp.status == 201, await resp.text()
    body = await resp.json()

    stored = client._notes_dir / body["filename"]
    assert stored.read_bytes() == MP3
    assert body["filename"].endswith(".mp3")

    rows = await _rows(client)
    assert len(rows) == 1
    row = rows[0]
    assert row.id == body["id"]
    assert row.filename == body["filename"]
    assert row.transcription is None
    assert row.transcription_retries == 0
    assert row.vikunja_id is None


async def test_iso_timestamp_drives_path(client):
    resp = await _post(client, timestamp="2026-07-25T12:30:00")
    assert resp.status == 201
    body = await resp.json()
    assert body["filename"] == "2026/07/25/12-30-00.mp3"


async def test_ogg_extension_preserved(client):
    resp = await _post(
        client, timestamp="2026-07-25T12:30:00", filename="voice.ogg"
    )
    assert resp.status == 201
    body = await resp.json()
    assert body["filename"] == "2026/07/25/12-30-00.ogg"
    assert (client._notes_dir / body["filename"]).read_bytes() == MP3


async def test_unknown_extension_falls_back_to_mp3(client):
    resp = await _post(
        client, timestamp="2026-07-25T12:30:00", filename="note.bin"
    )
    assert resp.status == 201
    assert (await resp.json())["filename"] == "2026/07/25/12-30-00.mp3"


async def test_epoch_timestamp_drives_path(client):
    # Pass an explicit epoch and check it lands where parse_timestamp maps it.
    epoch = TS.timestamp()
    resp = await _post(client, timestamp=str(epoch))
    assert resp.status == 201
    body = await resp.json()
    expected = TS.strftime("%Y/%m/%d/%H-%M-%S.mp3")
    assert body["filename"] == expected


async def test_collision_gets_unique_suffix(client):
    r1 = await _post(client, timestamp="2026-07-25T12:30:00")
    r2 = await _post(client, timestamp="2026-07-25T12:30:00")
    f1 = (await r1.json())["filename"]
    f2 = (await r2.json())["filename"]
    assert f1 == "2026/07/25/12-30-00.mp3"
    assert f2 != f1 and f2.startswith("2026/07/25/12-30-00-")
    # Both files exist on disk; neither clobbered the other.
    assert (client._notes_dir / f1).read_bytes() == MP3
    assert (client._notes_dir / f2).read_bytes() == MP3
    assert len(await _rows(client)) == 2


async def test_any_file_field_name_accepted(client):
    resp = await _post(client, timestamp="2026-07-25T12:30:00", field="audio")
    assert resp.status == 201


async def test_missing_timestamp_rejected(client):
    resp = await _post(client, timestamp=None)
    assert resp.status == 400
    assert (await resp.json())["error"] == "missing_timestamp"


async def test_bad_timestamp_rejected(client):
    resp = await _post(client, timestamp="not-a-date")
    assert resp.status == 400
    assert (await resp.json())["error"] == "bad_timestamp"


async def test_missing_file_rejected(client):
    resp = await _post(client, timestamp="2026-07-25T12:30:00", file=None)
    assert resp.status == 400
    assert (await resp.json())["error"] == "missing_file"


async def test_empty_file_rejected(client):
    resp = await _post(client, timestamp="2026-07-25T12:30:00", file=b"")
    assert resp.status == 400
    assert (await resp.json())["error"] == "empty_file"
