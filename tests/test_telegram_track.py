"""Tests for archiving a loaded map under the tracks directory.

``_process_route`` writes ``{"id": "%Y/%m/%d/{id}", "waypoints": [...]}`` to
``static/route.json`` and archives the same JSON as ``tracks/{id}.json``. The
tile pipeline (download → render) is stubbed out, so nothing here touches the
network.
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from techno_optimism_server import telegram_bot as tb

POINTS = [(52.1, 4.3), (52.2, 4.4)]


class FakeClient:
    """Records the chat traffic ``_process_route`` produces."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_message(self, chat_id, text, reply_markup=None):
        self.messages.append(text)
        return 1

    async def edit_message(self, chat_id, message_id, text):
        self.messages.append(text)


@pytest.fixture
def route_env(tmp_path, monkeypatch):
    """Point the bot's outputs at tmp_path and stub the tile pipeline."""
    monkeypatch.setattr(tb, "ROUTE_JSON_PATH", tmp_path / "static" / "route.json")
    monkeypatch.setattr(tb, "TILES_ZIP_PATH", tmp_path / "static" / "tiles.zip")
    monkeypatch.setattr(tb, "TRACKS_DIR", tmp_path / "tracks")
    monkeypatch.setattr(tb, "MIN_EDIT_INTERVAL", 0)

    # One rendered tile on disk, laid out as {z}/{x}/{y}.jpg so parse_tile_path
    # can recover its coordinates.
    tile = tmp_path / "rendered" / "19" / "5" / "7.jpg"
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b"JPEG")

    async def fake_download_tiles(points, **kwargs):
        return [tile]

    async def fake_download_mapbox_tiles(tiles, zoom, **kwargs):
        return []

    async def fake_render_tiles(tiles, zoom, **kwargs):
        return [tile]

    monkeypatch.setattr(tb, "download_tiles", fake_download_tiles)
    monkeypatch.setattr(tb, "download_mapbox_tiles", fake_download_mapbox_tiles)
    monkeypatch.setattr(tb, "render_tiles", fake_render_tiles)
    return tmp_path


def test_new_track_ref_shape():
    ref = tb.new_track_ref(datetime(2026, 7, 28, 13, 45))
    assert re.fullmatch(r"2026/07/28/[A-Za-z0-9]{16}", ref)
    # Two refs for the same day differ in the id part.
    assert ref != tb.new_track_ref(datetime(2026, 7, 28, 13, 45))


async def test_route_json_carries_id_and_waypoints(route_env):
    client = FakeClient()
    await tb._process_route(client, chat_id=1, points=POINTS)

    route = json.loads((route_env / "static" / "route.json").read_text())
    assert route["waypoints"] == [[52.1, 4.3], [52.2, 4.4]]
    # The id names the archived copy of this very route.
    assert re.fullmatch(r"\d{4}/\d{2}/\d{2}/[A-Za-z0-9]{16}", route["id"])
    archived = Path(route_env / "tracks" / f"{route['id']}.json")
    assert archived.read_text() == (route_env / "static" / "route.json").read_text()
    assert client.messages[-1] == "Route successfully uploaded"


async def test_tiles_zip_holds_only_tiles(route_env):
    client = FakeClient()
    await tb._process_route(client, chat_id=1, points=POINTS)

    with zipfile.ZipFile(route_env / "static" / "tiles.zip") as zf:
        assert zf.namelist() == ["19/5/7.jpg"]


async def test_each_route_gets_its_own_track_file(route_env):
    client = FakeClient()
    await tb._process_route(client, chat_id=1, points=POINTS)
    await tb._process_route(client, chat_id=1, points=[(1.0, 2.0)])

    archived = sorted((route_env / "tracks").rglob("*.json"))
    assert len(archived) == 2
    # route.json is the most recent route; the earlier track file survives it.
    latest = json.loads((route_env / "static" / "route.json").read_text())
    assert latest["waypoints"] == [[1.0, 2.0]]
    assert Path(route_env / "tracks" / f"{latest['id']}.json") in archived
