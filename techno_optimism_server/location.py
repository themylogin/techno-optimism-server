"""Live-location endpoint.

The Telegram bot POSTs the walking-route *origin* here the moment the user
shares it; the mobile app polls ``GET /location`` to learn where the walk
starts.

The location lives in RAM only (single instance) and expires ``LOCATION_TTL``
seconds (default 300) after it was posted. Once it lapses, ``GET /location``
returns ``null`` again — exactly as it did before any location was set. Expiry
is lazy: it's evaluated on read, so no background task is needed.
"""

from __future__ import annotations

import asyncio
import logging
import os

from aiohttp import web

log = logging.getLogger("techno_optimism.location")

# Seconds a posted location stays live before GET /location returns null again.
LOCATION_TTL = float(os.environ.get("LOCATION_TTL", "300"))

# Application key holding a single-slot mutable holder: ``{"current": <loc>}``,
# where ``<loc>`` is a dict with latitude, longitude and expires_at, or None.
# A holder (mutated in place) rather than a bare value, because aiohttp forbids
# reassigning app[...] keys once the app has started. Initialized in create_app().
LOCATION_KEY = "location"


def new_holder() -> dict:
    """A fresh empty location holder, for create_app() and tests."""
    return {"current": None}


def _now() -> float:
    """Monotonic clock tied to the running event loop."""
    return asyncio.get_running_loop().time()


async def post_location(request: web.Request) -> web.Response:
    """POST /location — set the current live location (a walk's origin).

    Body: ``{"latitude": <float>, "longitude": <float>, "accuracy": <float>}``,
    where ``accuracy`` (radius in metres) is optional. The location is held in
    RAM and expires ``LOCATION_TTL`` seconds from now. Returns the stored
    location and how long it stays live.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - any malformed body is a 400
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        lat = float(body["latitude"])
        lon = float(body["longitude"])
        # accuracy (radius in metres) is optional; keep it None when absent.
        accuracy = body.get("accuracy")
        accuracy = float(accuracy) if accuracy is not None else None
    except (KeyError, TypeError, ValueError):
        return web.json_response({"error": "invalid_location"}, status=400)

    request.app[LOCATION_KEY]["current"] = {
        "latitude": lat,
        "longitude": lon,
        "accuracy": accuracy,
        "expires_at": _now() + LOCATION_TTL,
    }
    log.info(
        "location set to (%s, %s) ±%sm, live for %ss",
        lat, lon, accuracy, LOCATION_TTL,
    )
    reply = {"latitude": lat, "longitude": lon, "ttl_seconds": LOCATION_TTL}
    if accuracy is not None:
        reply["accuracy"] = accuracy
    return web.json_response(reply)


async def get_location(request: web.Request) -> web.Response:
    """GET /location — the current live location, or ``null`` once expired.

    Returns ``{"latitude": .., "longitude": ..}`` (plus ``"accuracy"`` when one
    was posted) while a posted location is still within its TTL; otherwise JSON
    ``null``. The stale entry is cleared here on read, so no timer is required.
    """
    holder = request.app[LOCATION_KEY]
    loc = holder["current"]
    if loc is None:
        return web.json_response(None)
    if _now() >= loc["expires_at"]:
        holder["current"] = None
        return web.json_response(None)
    out = {"latitude": loc["latitude"], "longitude": loc["longitude"]}
    if loc.get("accuracy") is not None:
        out["accuracy"] = loc["accuracy"]
    return web.json_response(out)
