"""Compute a walking route between two points via the Google Maps Routes API.

Given an origin and destination lat/lon, ``compute_walking_route`` asks the
Routes API (``directions/v2:computeRoutes``) for a ``WALK`` route and returns
its total distance, duration, and the polyline decoded into ``(lat, lon)``
points — the same shape ``parse_gpx`` produces, so the result can feed straight
into the tile-download flow.

The Routes API is billed separately from the Tile API but is reached with the
same API key (``GOOGLE_API_KEY``); the key just needs the Routes API enabled in
the Google Cloud console.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import aiohttp

log = logging.getLogger("techno_optimism.routes")

ROUTES_API = "https://routes.googleapis.com/directions/v2:computeRoutes"


@dataclass
class WalkingRoute:
    """A computed walking route ready to be turned into tiles."""

    distance_meters: int
    duration_seconds: int
    points: list[tuple[float, float]]


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into a list of ``(lat, lon)`` points."""
    points: list[tuple[float, float]] = []
    index = lat = lng = 0
    length = len(encoded)

    def _next_value(idx: int) -> tuple[int, int]:
        result = shift = 0
        while True:
            b = ord(encoded[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        # The low bit is the sign flag; the rest is a zig-zag encoded delta.
        delta = ~(result >> 1) if result & 1 else (result >> 1)
        return delta, idx

    while index < length:
        dlat, index = _next_value(index)
        lat += dlat
        dlng, index = _next_value(index)
        lng += dlng
        points.append((lat / 1e5, lng / 1e5))
    return points


def _parse_duration(duration: str) -> int:
    """Parse a Routes API duration string like ``"1234s"`` into seconds."""
    return int(float(duration.rstrip("s"))) if duration else 0


async def compute_walking_route(
    session: aiohttp.ClientSession,
    api_key: str,
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> WalkingRoute:
    """Ask the Routes API for a walking route between two ``(lat, lon)`` points."""
    body = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin[0], "longitude": origin[1]}
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination[0],
                    "longitude": destination[1],
                }
            }
        },
        "travelMode": "WALK",
        "polylineEncoding": "ENCODED_POLYLINE",
    }
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline"
        ),
    }
    async with session.post(ROUTES_API, json=body, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()

    routes = data.get("routes") or []
    if not routes:
        raise RuntimeError("Routes API returned no route between those points.")
    route = routes[0]

    encoded = route.get("polyline", {}).get("encodedPolyline", "")
    points = decode_polyline(encoded) if encoded else []
    if not points:
        raise RuntimeError("Routes API returned a route with no geometry.")

    return WalkingRoute(
        distance_meters=int(route.get("distanceMeters", 0)),
        duration_seconds=_parse_duration(route.get("duration", "")),
        points=points,
    )


def format_duration(seconds: int) -> str:
    """Human-readable duration, e.g. ``"1 h 23 min"`` or ``"12 min"``."""
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"
