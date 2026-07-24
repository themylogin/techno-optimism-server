"""Download Mapbox satellite raster tiles covering a GPX track.

Satellite imagery comes from Mapbox: Google stopped serving satellite (and 3D)
Map Tiles to EEA-registered accounts, so its ``mapType=satellite`` 403s there.
Mapbox serves 256px satellite raster from ``/v4/mapbox.satellite/{z}/{x}/{y}`` on
the standard slippy-map scheme, authenticated with a single access token (no
session step). This is the same v4 endpoint the vector overlay already uses.

Tiles are cached on disk under ``cache/tiles/{map_type}/{z}/{x}/{y}`` so a second
run is a no-op for already-fetched tiles (the volume ``./cache:/app/cache``
mounts this directory in the container). ``map_type`` is always ``satellite``;
it survives as the cache namespace shared with the vector-render step.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import aiohttp

log = logging.getLogger("techno_optimism.tiles")

# 256px satellite raster, same v4 endpoint used for the vector overlay.
MAPBOX_TILE_API = "https://api.mapbox.com/v4"
MAPBOX_SATELLITE_TILESET = "mapbox.satellite"
CACHE_DIR = Path(os.environ.get("TILE_CACHE_DIR", "cache")) / "tiles"

# GPX namespace used to find <trkpt> elements regardless of the file's prefix.
_GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert a WGS84 lat/lon to the (x, y) of the containing slippy-map tile."""
    n = 1 << zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # Clamp to valid range (points exactly at the poles/antimeridian).
    x = min(max(x, 0), n - 1)
    y = min(max(y, 0), n - 1)
    return x, y


def parse_gpx(source) -> list[tuple[float, float]]:
    """Parse a GPX file into a list of ``(lat, lon)`` track points.

    ``source`` may be a path or any file-like object accepted by
    ``ElementTree.parse``.
    """
    root = ET.parse(source).getroot()
    elems = root.findall(".//gpx:trkpt", _GPX_NS)
    if not elems:  # fall back to prefix-agnostic search for odd namespaces
        elems = [e for e in root.iter() if e.tag.endswith("trkpt")]

    points: list[tuple[float, float]] = []
    for pt in elems:
        lat, lon = pt.get("lat"), pt.get("lon")
        if lat is None or lon is None:
            continue
        points.append((float(lat), float(lon)))
    return points


def tiles_for_points(
    points: list[tuple[float, float]], zoom: int
) -> list[tuple[int, int]]:
    """Every unique tile (x, y) at ``zoom`` touched by one of ``points``.

    Order is deterministic (sorted) so a "first N tiles" slice is stable across
    invocations.
    """
    seen: set[tuple[int, int]] = set()
    for lat, lon in points:
        seen.add(latlon_to_tile(lat, lon, zoom))
    return sorted(seen)


def with_neighbors(
    tiles: list[tuple[int, int]], zoom: int
) -> list[tuple[int, int]]:
    """Expand a tile list to also include the 8 neighbors of each tile.

    Neighbors are clamped to the valid tile range for ``zoom`` and the result is
    deduped and sorted, so a route hugging a tile edge still gets full coverage
    of the adjacent tiles.
    """
    n = 1 << zoom
    expanded: set[tuple[int, int]] = set()
    for x, y in tiles:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    expanded.add((nx, ny))
    return sorted(expanded)


async def _get_tile_bytes(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
    zoom: int,
    x: int,
    y: int,
    retries: int,
) -> tuple[bytes, str]:
    """GET a tile, retrying transient failures with a short exponential backoff.

    Returns ``(body, content_type)``. Raises on a non-transient status or after
    the retries are exhausted.
    """
    for attempt in range(retries):
        async with session.get(url, params=params) as resp:
            if resp.status in (404, 429, 500, 502, 503, 504) and attempt < retries - 1:
                delay = 0.5 * (2**attempt)
                log.warning(
                    "tile %d/%d/%d got %d, retrying in %.1fs",
                    zoom, x, y, resp.status, delay,
                )
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return await resp.read(), resp.headers.get("Content-Type", "")
    raise RuntimeError("unreachable: retry loop exited without returning")


async def _fetch_tile(
    session: aiohttp.ClientSession,
    zoom: int,
    x: int,
    y: int,
    map_type: str = "satellite",
    retries: int = 4,
) -> Path:
    """Download one Mapbox satellite tile into the cache, skipping if cached."""
    token = os.environ.get("MAPBOX_TOKEN")
    if not token:
        raise RuntimeError("MAPBOX_TOKEN is not set")

    tile_dir = CACHE_DIR / map_type / str(zoom) / str(x)
    # Mapbox satellite is JPEG; a tile is cached if a file exists under any
    # image extension (older caches may hold PNGs).
    for cached in (tile_dir / f"{y}.jpg", tile_dir / f"{y}.png"):
        if cached.exists():
            log.debug("tile %d/%d/%d cached", zoom, x, y)
            return cached

    url = f"{MAPBOX_TILE_API}/{MAPBOX_SATELLITE_TILESET}/{zoom}/{x}/{y}.jpg"
    data, content_type = await _get_tile_bytes(
        session, url, {"access_token": token}, zoom, x, y, retries
    )

    ext = "jpg" if "jpeg" in content_type or "jpg" in content_type else "png"
    dest = tile_dir / f"{y}.{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dest.write_bytes, data)
    log.info("downloaded tile %d/%d/%d (%d bytes)", zoom, x, y, len(data))
    return dest


async def download_tiles(
    points: list[tuple[float, float]],
    zoom: int = 14,
    limit: int | None = None,
    concurrency: int = 8,
    map_type: str = "satellite",
    include_neighbors: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Download every zoom-``zoom`` tile crossed by ``points`` into the cache.

    Args:
        points: the route as a list of ``(lat, lon)`` pairs.
        zoom: tile zoom level (default 14).
        limit: if set, only download the first ``limit`` tiles (for testing).
        concurrency: max simultaneous tile requests.
        map_type: cache namespace for the fetched tiles (always ``satellite``).
        include_neighbors: also download the 8 neighboring tiles of each tile.
        progress: optional callback invoked as ``progress(done, total)`` after
            each tile finishes (both cached-hit and freshly downloaded count).

    Returns the list of cached tile paths.
    """
    if not os.environ.get("MAPBOX_TOKEN"):
        raise RuntimeError("MAPBOX_TOKEN is not set")

    tiles = tiles_for_points(points, zoom)
    if include_neighbors:
        tiles = with_neighbors(tiles, zoom)
    if limit is not None:
        tiles = tiles[:limit]
    total = len(tiles)
    log.info("route covers %d %s tile(s) at zoom %d", total, map_type, zoom)

    sem = asyncio.Semaphore(concurrency)
    done = 0
    async with aiohttp.ClientSession() as session:

        async def worker(x: int, y: int) -> Path:
            nonlocal done
            async with sem:
                path = await _fetch_tile(session, zoom, x, y, map_type)
            # asyncio is single-threaded, so this increment needs no lock.
            done += 1
            if progress is not None:
                progress(done, total)
            return path

        return await asyncio.gather(*(worker(x, y) for x, y in tiles))
