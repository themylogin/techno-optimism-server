"""Tests for the live-location endpoint (POST/GET /location).

The handlers are driven through a real aiohttp test client. The 300s TTL is
shortened per test via monkeypatch so expiry can be observed without waiting.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from techno_optimism_server import location as location_mod
from techno_optimism_server.location import (
    LOCATION_KEY,
    get_location,
    new_holder,
    post_location,
)

BARCELONA = {"latitude": 41.4, "longitude": 2.1}


@pytest.fixture
async def make_client(monkeypatch):
    """Factory building a started TestClient for the /location routes."""
    clients: list[TestClient] = []

    async def _make(ttl=300.0):
        monkeypatch.setattr(location_mod, "LOCATION_TTL", ttl)
        app = web.Application()
        app[LOCATION_KEY] = new_holder()
        app.add_routes([
            web.post("/location", post_location),
            web.get("/location", get_location),
        ])
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        await c.close()


async def test_location_null_before_any_post(make_client):
    client = await make_client()
    resp = await client.get("/location")
    assert resp.status == 200
    assert await resp.json() is None


async def test_post_then_get_returns_location(make_client):
    client = await make_client()

    resp = await client.post("/location", json=BARCELONA)
    assert resp.status == 200
    body = await resp.json()
    assert body["latitude"] == BARCELONA["latitude"]
    assert body["longitude"] == BARCELONA["longitude"]
    assert body["ttl_seconds"] == 300.0

    resp = await client.get("/location")
    assert await resp.json() == BARCELONA


async def test_accuracy_stored_and_returned(make_client):
    client = await make_client()
    posted = {**BARCELONA, "accuracy": 12.5}

    resp = await client.post("/location", json=posted)
    assert (await resp.json())["accuracy"] == 12.5

    resp = await client.get("/location")
    assert await resp.json() == posted


async def test_accuracy_absent_omitted(make_client):
    # No accuracy posted -> the key is absent from both replies.
    client = await make_client()
    resp = await client.post("/location", json=BARCELONA)
    assert "accuracy" not in await resp.json()
    resp = await client.get("/location")
    body = await resp.json()
    assert body == BARCELONA and "accuracy" not in body


async def test_invalid_accuracy_rejected(make_client):
    client = await make_client()
    resp = await client.post("/location", json={**BARCELONA, "accuracy": "x"})
    assert resp.status == 400
    assert (await resp.json())["error"] == "invalid_location"


async def test_location_expires_after_ttl(make_client):
    # A tiny TTL lets us observe the lapse to null without a real 300s wait.
    client = await make_client(ttl=0.05)
    await client.post("/location", json=BARCELONA)

    assert await (await client.get("/location")).json() == BARCELONA
    await asyncio.sleep(0.06)
    assert await (await client.get("/location")).json() is None


async def test_post_resets_the_ttl(make_client):
    client = await make_client(ttl=0.1)
    await client.post("/location", json=BARCELONA)
    await asyncio.sleep(0.07)
    # Re-posting before expiry gives a fresh full TTL window.
    await client.post("/location", json=BARCELONA)
    await asyncio.sleep(0.07)
    assert await (await client.get("/location")).json() == BARCELONA


async def test_invalid_json_rejected(make_client):
    client = await make_client()
    resp = await client.post(
        "/location", data=b"not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status == 400
    assert (await resp.json())["error"] == "invalid_json"


@pytest.mark.parametrize(
    "body",
    [
        {"latitude": 41.4},                 # missing longitude
        {"longitude": 2.1},                 # missing latitude
        {"latitude": "x", "longitude": 2},  # non-numeric
        {},                                 # empty
    ],
)
async def test_invalid_location_rejected(make_client, body):
    client = await make_client()
    resp = await client.post("/location", json=body)
    assert resp.status == 400
    assert (await resp.json())["error"] == "invalid_location"
