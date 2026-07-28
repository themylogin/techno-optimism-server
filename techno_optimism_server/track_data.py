"""Track-data append endpoint.

    POST /track-data   multipart: a file whose contents are appended

The uploaded bytes are appended verbatim to ``tracks/log.json`` (the same volume
the bot archives loaded maps into), followed by a newline when they don't
already end with one — so the log stays a newline-terminated stream of records
and the next append always starts on its own line.

The write goes out in a single ``O_APPEND`` call, which the kernel serializes,
so concurrent uploads interleave whole records rather than shredding each other.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiohttp import web

log = logging.getLogger("techno_optimism.track_data")

# Loaded maps (see telegram_bot.new_track_ref) and this log share one volume,
# mounted as ./tracks:/app/tracks.
TRACKS_DIR = Path(os.environ.get("TRACKS_DIR", "tracks"))
LOG_NAME = "log.json"


def append_to_log(data: bytes, base: Path | None = None) -> Path:
    """Append ``data`` to the tracks log, keeping it newline-terminated.

    Blocking (creates the directory and writes), so call it off the event loop.
    Returns the log's path.
    """
    path = (base or TRACKS_DIR) / LOG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data.endswith(b"\n"):
        data += b"\n"
    # One write in append mode: the offset lookup and the write are atomic, so
    # two uploads racing here cannot overwrite one another.
    with path.open("ab") as fh:
        fh.write(data)
    return path


async def post_track_data(request: web.Request) -> web.Response:
    """POST /track-data — append the uploaded file to tracks/log.json."""
    reader = await request.post()

    # The payload can arrive under any field name; take the first uploaded file.
    file_field = next(
        (v for v in reader.values() if isinstance(v, web.FileField)), None
    )
    if file_field is None:
        return web.json_response({"error": "missing_file"}, status=400)
    data = file_field.file.read()
    if not data:
        return web.json_response({"error": "empty_file"}, status=400)

    path = await asyncio.to_thread(append_to_log, data)
    log.info("appended %d bytes to %s", len(data), path)
    return web.json_response({"appended": len(data)}, status=201)
