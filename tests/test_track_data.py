"""Tests for the POST /track-data append endpoint.

Driven through a real aiohttp test client against a temp tracks directory. The
endpoint only appends the uploaded bytes to ``log.json``, keeping the file
newline-terminated.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from techno_optimism_server import track_data as td

RECORD = b'{"lat": 52.1, "lon": 4.3}'


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """A started test client writing into a temp tracks dir."""
    tracks_dir = tmp_path / "tracks"
    monkeypatch.setattr(td, "TRACKS_DIR", tracks_dir)

    app = web.Application()
    app.add_routes([web.post("/track-data", td.post_track_data)])

    c = TestClient(TestServer(app))
    await c.start_server()
    c._log = tracks_dir / "log.json"
    yield c
    await c.close()


async def _post(client, data=RECORD, *, field="file", filename="track.json"):
    form = FormData()
    form.add_field(field, data, filename=filename)
    return await client.post("/track-data", data=form)


async def test_appends_and_terminates_with_newline(client):
    resp = await _post(client)
    assert resp.status == 201
    assert (await resp.json())["appended"] == len(RECORD)
    assert client._log.read_bytes() == RECORD + b"\n"


async def test_appends_after_previous_records(client):
    await _post(client)
    await _post(client, b'{"lat": 52.2, "lon": 4.4}')
    assert client._log.read_bytes() == (
        RECORD + b"\n" + b'{"lat": 52.2, "lon": 4.4}' + b"\n"
    )


async def test_existing_trailing_newline_not_doubled(client):
    await _post(client, RECORD + b"\n")
    await _post(client, RECORD + b"\n")
    assert client._log.read_bytes() == RECORD + b"\n" + RECORD + b"\n"


async def test_multiline_upload_appended_verbatim(client):
    await _post(client, b"a\nb\nc")
    assert client._log.read_bytes() == b"a\nb\nc\n"


async def test_missing_file_is_rejected(client):
    form = FormData()
    form.add_field("note", "no file here")
    resp = await client.post("/track-data", data=form)
    assert resp.status == 400
    assert (await resp.json())["error"] == "missing_file"
    assert not client._log.exists()


async def test_empty_file_is_rejected(client):
    resp = await _post(client, b"")
    assert resp.status == 400
    assert (await resp.json())["error"] == "empty_file"
    assert not client._log.exists()


async def test_field_name_does_not_matter(client):
    resp = await _post(client, field="payload", filename="whatever.bin")
    assert resp.status == 201
    assert client._log.read_bytes() == RECORD + b"\n"


async def test_concurrent_appends_keep_whole_records(client):
    """Racing uploads interleave whole lines, never partial ones."""
    records = [b"record-%02d" % i for i in range(20)]
    await asyncio.gather(*(_post(client, r) for r in records))
    lines = client._log.read_bytes().splitlines()
    assert sorted(lines) == sorted(records)
