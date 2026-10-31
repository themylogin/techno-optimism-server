"""Voice-note upload endpoint.

    POST /voice-note   multipart: an audio file + a `timestamp` field

Clients may use any file field name and send `timestamp` as a Unix epoch or an
ISO-8601 string. The upload is normalized to what the AI assistant's
``POST /v1/upload/voice-note`` expects — a `file` field and an ISO-8601
`timestamp` with a UTC offset — and forwarded with ``AI_ASSISTANT_TOKEN``. Any
success from the assistant becomes ``201 {"filename"}``; its errors are returned
to the caller as they are. Nothing is stored here.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from datetime import datetime

import aiohttp
from aiohttp import web

log = logging.getLogger("techno_optimism.voice_note")

UPLOAD_PATH = "/v1/upload/voice-note"

# Audio extensions preserved from the upload (e.g. .ogg). Anything else — or a
# missing extension — is sent as .mp3.
_ALLOWED_EXTS = frozenset(
    {".mp3", ".ogg", ".oga", ".opus", ".m4a", ".wav", ".webm", ".aac", ".flac"}
)


def _extension(filename: str | None) -> str:
    """The forwarded file's extension, taken from the upload's name (default .mp3)."""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _ALLOWED_EXTS else ".mp3"


def parse_timestamp(raw: str) -> datetime:
    """Parse the `timestamp` field into a timezone-aware ``datetime``.

    Accepts a Unix epoch (seconds, int or float) or an ISO-8601 string, so
    clients can send whichever is convenient; an ISO string without an offset is
    taken as local time. Raises ``ValueError`` otherwise.
    """
    raw = raw.strip()
    try:
        return datetime.fromtimestamp(float(raw)).astimezone()
    except (ValueError, OverflowError, OSError):
        pass
    # datetime.fromisoformat handles a trailing 'Z' only from Python 3.11+.
    when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return when if when.tzinfo else when.astimezone()


async def create_voice_note(request: web.Request) -> web.Response:
    """POST /voice-note — normalize the upload and forward it to the AI assistant."""
    reader = await request.post()

    raw_ts = reader.get("timestamp")
    if not isinstance(raw_ts, str) or not raw_ts.strip():
        return web.json_response({"error": "missing_timestamp"}, status=400)
    try:
        when = parse_timestamp(raw_ts)
    except ValueError:
        return web.json_response({"error": "bad_timestamp"}, status=400)

    # The audio can arrive under any field name; take the first uploaded file.
    file_field = next(
        (v for v in reader.values() if isinstance(v, web.FileField)), None
    )
    if file_field is None:
        return web.json_response({"error": "missing_file"}, status=400)
    audio = file_field.file.read()
    if not audio:
        return web.json_response({"error": "empty_file"}, status=400)

    filename = f"voice-note{_extension(file_field.filename)}"
    form = aiohttp.FormData()
    form.add_field("timestamp", when.isoformat())
    form.add_field(
        "file", audio,
        filename=filename,
        content_type=mimetypes.guess_type(filename)[0] or file_field.content_type,
    )

    url = os.environ["AI_ASSISTANT_URL"].rstrip("/") + UPLOAD_PATH
    headers = {"X-API-Key": os.environ["AI_ASSISTANT_TOKEN"]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form, headers=headers) as resp:
                body = await resp.read()
                if resp.ok:
                    log.info("forwarded voice note (%d bytes)", len(audio))
                    return web.json_response({"filename": filename}, status=201)
                log.warning(
                    "assistant rejected voice note with %s: %r",
                    resp.status, body[:1000],
                )
                return web.Response(
                    status=resp.status,
                    body=body,
                    content_type=resp.content_type,
                    charset=resp.charset,
                )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        log.warning("forwarding voice note to %s failed: %s", url, exc)
        return web.json_response({"error": "assistant_unreachable"}, status=502)
