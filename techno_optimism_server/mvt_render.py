"""Render the Mapbox vector *style* over a Google raster tile, in pure Python.

The configured Mapbox style (``MAPBOX_STYLE_URL``) is a deliberately tiny
overlay — a transparent background with three ``line`` layers pulled from the
``road`` source-layer of ``mapbox-streets-v8`` (walking ``path``s, ``steps``,
and ``pedestrian`` ways), all drawn in red. There is nothing else to it: no
fills, no labels, no icons. So rather than drive a full Mapbox GL renderer
(WebGL, a headless browser, glyphs, sprites…), we fetch the one vector tile,
decode it, and stroke those three layers onto the Google tile with Pillow.

Only the handful of style constructs the overlay actually uses is supported:
``interpolate``/``exponential`` line widths, ``step`` dash arrays, and the
per-layer feature filters — all evaluated at the fixed render zoom.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import math
import os
from pathlib import Path
from typing import Callable

import aiohttp
import mapbox_vector_tile
from PIL import Image, ImageDraw

from .tiles import CACHE_DIR

log = logging.getLogger("techno_optimism.mvt_render")

# The style's line layers all read from this Mapbox tileset. It maxes out at
# zoom 16, so a request for a deeper tile is served by over-zooming its z16
# ancestor; we do the same and transform the ancestor's coordinates down.
VECTOR_TILESET = "mapbox.mapbox-streets-v8"
VECTOR_MAXZOOM = 16
V4_TILE_URL = "https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}.vector.pbf"

# Where downloaded vector tiles and rendered (satellite + overlay) tiles live,
# alongside the raster map-type dirs (cache/tiles/satellite/...). Rendered tiles
# are namespaced by "generation" so a style/renderer change can be rolled out
# under a fresh dir without invalidating the old ones — bump this constant when
# the style or the renderer changes.
MAPBOX_CACHE = CACHE_DIR / "mapbox"
RENDERED_CACHE = CACHE_DIR / "rendered"
RENDER_GENERATION = "gen01"

RED = (255, 0, 0)


# --- minimal Mapbox style-expression evaluation (at a fixed zoom) ------------
#
# Every expression in the overlay bottoms out in ``["zoom"]``, so once the
# render zoom is fixed each one collapses to a constant. We only implement the
# two families the style uses: exponential interpolation and zoom steps.

def _interpolate_exponential(base: float, zoom: float, stops: list) -> float:
    """Evaluate ``["interpolate", ["exponential", base], ["zoom"], *stops]``.

    ``stops`` is a flat ``[z0, v0, z1, v1, ...]`` list. Below/above the stop
    range the value is clamped to the nearest segment and extrapolated along
    it, matching Mapbox GL.
    """
    pts = list(zip(stops[0::2], stops[1::2]))
    if zoom <= pts[0][0]:
        lo, hi = pts[0], pts[1] if len(pts) > 1 else pts[0]
    elif zoom >= pts[-1][0]:
        lo, hi = (pts[-2], pts[-1]) if len(pts) > 1 else (pts[-1], pts[-1])
    else:
        lo, hi = pts[0], pts[-1]
        for a, b in zip(pts, pts[1:]):
            if a[0] <= zoom <= b[0]:
                lo, hi = a, b
                break
    (z0, v0), (z1, v1) = lo, hi
    if z1 == z0:
        return v0
    if base == 1:
        t = (zoom - z0) / (z1 - z0)
    else:
        t = (base ** (zoom - z0) - 1) / (base ** (z1 - z0) - 1)
    return v0 + (v1 - v0) * t


def _step(zoom: float, base, stops: list):
    """Evaluate ``["step", ["zoom"], base, z1, v1, z2, v2, ...]``."""
    value = base
    for z, v in zip(stops[0::2], stops[1::2]):
        if zoom >= z:
            value = v
        else:
            break
    return value


# --- the three line layers, as plain predicates + evaluated paint -----------
#
# Filters are transcribed from the style JSON and reduced to the render zoom.
# All three share ``structure in {none, ford}`` and ``LineString`` geometry.

def _structure_ok(p: dict) -> bool:
    return p.get("structure", "none") in ("none", "ford")


def _layer_specs(zoom: float) -> list[dict]:
    """Build the draw spec (predicate, width, dash) for each layer at ``zoom``."""
    return [
        {
            "id": "road-pedestrian",
            "match": lambda p: p.get("class") == "pedestrian"
            and _structure_ok(p)
            and p.get("layer", 0) >= 0,
            "width": _interpolate_exponential(1.5, zoom, [14, 0.5, 18, 12]),
            "dash": _step(zoom, [2, 0.3], [15, [1, 0.3], 16, [1, 0.3], 17, [1, 0.25]]),
        },
        {
            "id": "road-path",
            # zoom >= 16: everything on class==path except steps (steps has its
            # own layer below).
            "match": lambda p: p.get("class") == "path"
            and p.get("type") != "steps"
            and _structure_ok(p),
            "width": _interpolate_exponential(1.5, zoom, [13, 0.5, 14, 1, 15, 1, 18, 4]),
            "dash": _step(zoom, [4, 0.3], [15, [1.75, 0.3], 16, [1, 0.3], 17, [1, 0.25]]),
        },
        {
            "id": "road-steps",
            "match": lambda p: p.get("type") == "steps" and _structure_ok(p),
            "width": _interpolate_exponential(1.5, zoom, [15, 1, 16, 1.6, 18, 6]),
            "dash": _step(zoom, [1, 0], [15, [1.75, 1], 16, [1, 0.75], 17, [0.3, 0.3]]),
        },
    ]


# --- geometry ----------------------------------------------------------------

def _iter_linestrings(geometry: dict):
    """Yield each ring of a (Multi)LineString feature as a list of (x, y)."""
    t, coords = geometry["type"], geometry["coordinates"]
    if t == "LineString":
        yield coords
    elif t == "MultiLineString":
        yield from coords


def _draw_dashed(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    width: float,
    dash: list[float] | None,
) -> None:
    """Stroke a polyline, honouring a ``[dash, gap]`` pattern (in line-widths)."""
    w = max(1, round(width))
    if not dash or dash[1] <= 0:
        draw.line(pts, fill=RED, width=w, joint="curve")
        return

    on_len = max(1.0, dash[0] * width)
    off_len = max(1.0, dash[1] * width)
    pattern = [on_len, off_len]
    pi, remaining, drawing = 0, pattern[0], True
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        dx, dy = (x1 - x0) / seg, (y1 - y0) / seg
        pos = 0.0
        while pos < seg:
            take = min(remaining, seg - pos)
            if drawing:
                ax, ay = x0 + dx * pos, y0 + dy * pos
                bx, by = x0 + dx * (pos + take), y0 + dy * (pos + take)
                draw.line([(ax, ay), (bx, by)], fill=RED, width=w)
            pos += take
            remaining -= take
            if remaining <= 1e-6:
                drawing = not drawing
                pi = (pi + 1) % len(pattern)
                remaining = pattern[pi]


# --- coordinates -------------------------------------------------------------

def _source_coord(zoom: int, x: int, y: int) -> tuple[int, int, int]:
    """The (z, x, y) vector tile that covers ``zoom/x/y``.

    ``mapbox-streets-v8`` stops at :data:`VECTOR_MAXZOOM`, so a deeper tile is
    served by over-zooming its ancestor at that zoom.
    """
    src_z = min(zoom, VECTOR_MAXZOOM)
    shift = zoom - src_z
    return src_z, x >> shift, y >> shift


def parse_tile_path(path: Path) -> tuple[int, int, int]:
    """Recover ``(z, x, y)`` from a cached tile path ``.../{z}/{x}/{y}.ext``."""
    return int(path.parent.parent.name), int(path.parent.name), int(path.stem)


# --- download vector tiles ---------------------------------------------------

async def _fetch_vector_tile(
    session: aiohttp.ClientSession, token: str, z: int, x: int, y: int
) -> bytes:
    url = V4_TILE_URL.format(tileset=VECTOR_TILESET, z=z, x=x, y=y)
    async with session.get(url, params={"access_token": token}) as resp:
        resp.raise_for_status()
        data = await resp.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


async def download_mapbox_tiles(
    tiles: list[tuple[int, int]],
    zoom: int,
    concurrency: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Download the vector tile for each ``(x, y)`` into ``cache/tiles/mapbox``.

    Stored decompressed at ``mapbox/{zoom}/{x}/{y}.pbf`` — 1:1 with the raster
    tiles, so the render step reads one raster + one vector tile per output. The
    over-zoomed ancestor shared by many deep tiles is fetched only once.
    """
    token = os.environ.get("MAPBOX_TOKEN")
    if not token:
        raise RuntimeError("MAPBOX_TOKEN is not set")

    total = len(tiles)
    log.info("downloading %d vector tile(s) at zoom %d", total, zoom)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    # Cache in-flight ancestor fetches so shared source tiles hit the network once.
    source_cache: dict[tuple[int, int, int], asyncio.Task[bytes]] = {}

    async with aiohttp.ClientSession() as session:

        async def worker(x: int, y: int) -> Path:
            nonlocal done
            dest = MAPBOX_CACHE / str(zoom) / str(x) / f"{y}.pbf"
            if not dest.exists():
                src = _source_coord(zoom, x, y)
                async with sem:
                    task = source_cache.get(src)
                    if task is None:
                        task = asyncio.create_task(
                            _fetch_vector_tile(session, token, *src)
                        )
                        source_cache[src] = task
                    data = await task
                dest.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(dest.write_bytes, data)
            done += 1
            if progress is not None:
                progress(done, total)
            return dest

        return await asyncio.gather(*(worker(x, y) for x, y in tiles))


# --- compose + render --------------------------------------------------------

def _compose(base_path: Path, pbf: bytes, zoom: int, x: int, y: int, scale: int) -> Image.Image:
    """Stroke the style's red path overlay from ``pbf`` onto the raster tile."""
    size = 256 * scale
    base = Image.open(base_path).convert("RGB").resize((size, size), Image.LANCZOS)

    tile = mapbox_vector_tile.decode(pbf, default_options={"y_coord_down": True})
    road = tile.get("road")
    if not road or not road["features"]:
        return base

    extent = road["extent"]
    # The tile occupies a sub-square of its (possibly over-zoomed) source tile;
    # map that sub-square's extent coordinates onto the output pixels.
    src_z, src_x, src_y = _source_coord(zoom, x, y)
    span = 1 << (zoom - src_z)
    off_x, off_y = x - src_x * span, y - src_y * span

    def project(px: float, py: float) -> tuple[float, float]:
        return (
            (px / extent * span - off_x) * size,
            (py / extent * span - off_y) * size,
        )

    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Draw layers bottom-up (pedestrian, path, steps) as the style orders them.
    for spec in _layer_specs(zoom):
        width = spec["width"] * scale
        for feat in road["features"]:
            if feat["geometry"]["type"] not in ("LineString", "MultiLineString"):
                continue
            if not spec["match"](feat["properties"]):
                continue
            for ring in _iter_linestrings(feat["geometry"]):
                pts = [project(px, py) for px, py in ring]
                if len(pts) >= 2:
                    _draw_dashed(draw, pts, width, spec["dash"])

    base.paste(overlay, (0, 0), overlay)
    return base


def _render_one(
    zoom: int, x: int, y: int, dest: Path, map_type: str, scale: int
) -> Path:
    """Blocking compose+write for a single tile (call via ``to_thread``)."""
    base_path = CACHE_DIR / map_type / str(zoom) / str(x) / f"{y}.jpg"
    if not base_path.exists():
        raise FileNotFoundError(f"base tile not cached: {base_path}")
    pbf_path = MAPBOX_CACHE / str(zoom) / str(x) / f"{y}.pbf"
    if not pbf_path.exists():
        raise FileNotFoundError(f"vector tile not cached: {pbf_path}")

    img = _compose(base_path, pbf_path.read_bytes(), zoom, x, y, scale)
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    dest.write_bytes(buf.getvalue())
    return dest


async def render_tiles(
    tiles: list[tuple[int, int]],
    zoom: int,
    generation: str = RENDER_GENERATION,
    map_type: str = "satellite",
    scale: int = 2,
    concurrency: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Render each ``(x, y)`` (cached raster + cached vector) to
    ``cache/tiles/rendered/{generation}/{zoom}/{x}/{y}.jpg``."""
    total = len(tiles)
    log.info(
        "rendering %d tile(s) at zoom %d into generation %r", total, zoom, generation
    )
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def worker(x: int, y: int) -> Path:
        nonlocal done
        dest = RENDERED_CACHE / generation / str(zoom) / str(x) / f"{y}.jpg"
        # Rendering is deterministic for a given generation, so a tile already
        # on disk is reused rather than composed again.
        if not dest.exists():
            async with sem:
                await asyncio.to_thread(
                    _render_one, zoom, x, y, dest, map_type, scale
                )
        done += 1
        if progress is not None:
            progress(done, total)
        return dest

    return await asyncio.gather(*(worker(x, y) for x, y in tiles))


async def render_tile(
    zoom: int,
    x: int,
    y: int,
    dest: Path,
    map_type: str = "satellite",
    scale: int = 2,
) -> Path:
    """Render a single tile to an arbitrary ``dest`` (used for one-off tests).

    Downloads the vector tile into the cache first if it isn't there yet.
    """
    await download_mapbox_tiles([(x, y)], zoom)
    return await asyncio.to_thread(
        _render_one, zoom, x, y, dest, map_type, scale
    )
