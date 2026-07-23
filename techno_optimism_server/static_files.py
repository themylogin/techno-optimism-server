"""Static file serving with a content digest.

aiohttp's built-in ``add_static`` answers GET (and HEAD) and honours ``Range``,
but it can't attach a custom header. Clients here want to know whether the
``route.json`` / ``tiles.zip`` they hold is still current without downloading the
whole (multi-MB) blob, so every static response carries ``X-SHA1``: the SHA-1 of
the file's bytes. A client issues a cheap ``HEAD`` and re-downloads only when the
digest changed.

"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from aiohttp import web

_CHUNK = 1024 * 1024  # 1 MiB read window while hashing


def _sha1(path: Path) -> str:
    """SHA-1 of the file's bytes."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def make_static_handler(static_dir: Path):
    """Build a handler serving files under ``static_dir`` with an ``X-SHA1`` header.

    Handles GET (via FileResponse, so ``Range`` still works) and HEAD (FileResponse
    emits headers with no body). The ``X-SHA1`` header is present on both.
    """
    root = static_dir.resolve()

    async def handler(request: web.Request) -> web.StreamResponse:
        rel = request.match_info["filename"]
        target = (root / rel).resolve()

        # Confine to the static root; reject traversal (../) escapes.
        if target != root and root not in target.parents:
            return web.json_response({"error": "forbidden"}, status=403)
        if not target.is_file():
            return web.json_response({"error": "not_found"}, status=404)

        # Hashing reads the whole file; run it off the event loop so a large
        # blob (tiles.zip) doesn't stall every other connection.
        digest = await asyncio.to_thread(_sha1, target)
        return web.FileResponse(target, headers={"X-SHA1": digest})

    return handler
