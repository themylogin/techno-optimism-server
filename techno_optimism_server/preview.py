"""Render a satellite preview of a whole route.

Produces a single square (default 512×512) JPEG: Google satellite imagery with
the route drawn on top as a white polyline, at the deepest zoom that still fits
the entire route inside the frame. Used to show the user what they're about to
download before any tiles are fetched.

Web-Mercator math mirrors the slippy-map convention in :mod:`.tiles`; we reuse
that module's session + tile-fetch primitives so preview tiles land in the same
on-disk cache.
"""

from __future__ import annotations

import asyncio
import io
import math
import os
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw

from .tiles import _create_session, _fetch_tile

TILE_SIZE = 256
EARTH_RADIUS_M = 6_371_000.0


def route_length_m(points: list[tuple[float, float]]) -> float:
    """Total great-circle length of the polyline, in meters (haversine sum)."""
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        total += 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
    return total


def format_distance(meters: float) -> str:
    """Human-readable distance, e.g. ``"12.3 km"`` or ``"850 m"``."""
    return f"{meters / 1000:.2f} km" if meters >= 1000 else f"{round(meters)} m"


def _project(lat: float, lon: float) -> tuple[float, float]:
    """Project lat/lon to normalized Web-Mercator world coordinates in [0, 1]."""
    x = (lon + 180.0) / 360.0
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    y = 0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)
    return x, y


def _fit_zoom(dx: float, dy: float, usable: int, max_zoom: int) -> int:
    """Deepest zoom at which a route spanning ``dx``/``dy`` (world units) fits
    within ``usable`` pixels in both dimensions."""
    span = max(dx, dy)
    if span <= 0:  # a single point (or all points coincide)
        return max_zoom
    z = math.floor(math.log2(usable / (TILE_SIZE * span)))
    return max(0, min(int(z), max_zoom))


async def render_route_preview(
    points: list[tuple[float, float]],
    *,
    size: int = 512,
    padding: int = 32,
    map_type: str = "satellite",
    max_zoom: int = 20,
    line_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 5,
    dest: Path | None = None,
) -> bytes:
    """Render ``points`` as a white line over satellite imagery; return the JPEG.

    The frame is centered on the route's bounding box and zoomed so the whole
    route fits with ``padding`` px to spare. If ``dest`` is given the bytes are
    also written there.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    if not points:
        raise ValueError("no points to preview")

    world = [_project(lat, lon) for lat, lon in points]
    xs = [p[0] for p in world]
    ys = [p[1] for p in world]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    usable = size - 2 * padding
    zoom = _fit_zoom(maxx - minx, maxy - miny, usable, max_zoom)
    scale = TILE_SIZE * (1 << zoom)  # world-unit -> pixel at this zoom
    n = 1 << zoom

    # Pixel window: the size×size square centered on the route's bbox center.
    left = (minx + maxx) / 2 * scale - size / 2
    top = (miny + maxy) / 2 * scale - size / 2

    tx0, tx1 = math.floor(left / TILE_SIZE), math.floor((left + size) / TILE_SIZE)
    ty0 = max(0, math.floor(top / TILE_SIZE))
    ty1 = min(n - 1, math.floor((top + size) / TILE_SIZE))

    async with aiohttp.ClientSession() as session:
        token = await _create_session(session, api_key, map_type)

        async def fetch(tx: int, ty: int):
            # Wrap x around the antimeridian; y is already clamped in range.
            path = await _fetch_tile(session, token, api_key, zoom, tx % n, ty, map_type)
            return tx, ty, path

        coords = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
        tiles = await asyncio.gather(*(fetch(tx, ty) for tx, ty in coords))

    canvas = Image.new("RGB", (size, size))
    for tx, ty, path in tiles:
        img = Image.open(path).convert("RGB")
        canvas.paste(img, (int(tx * TILE_SIZE - left), int(ty * TILE_SIZE - top)))

    draw = ImageDraw.Draw(canvas)
    pts = [(p[0] * scale - left, p[1] * scale - top) for p in world]
    if len(pts) >= 2:
        draw.line(pts, fill=line_color, width=line_width, joint="curve")
    else:  # degenerate route: mark the single point
        (px, py), r = pts[0], line_width * 2
        draw.ellipse([px - r, py - r, px + r, py + r], fill=line_color)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data
