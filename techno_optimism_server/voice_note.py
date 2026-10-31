"""Voice-note upload endpoint.

    POST /voice-note   multipart: an audio file + a `timestamp` field

A thin proxy: the request body goes unchanged to the AI assistant's
``POST /v1/upload/voice-note`` (authenticated with ``AI_ASSISTANT_TOKEN``) and
the assistant's response is returned to the caller. Nothing is stored here.
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from aiohttp import web

log = logging.getLogger("techno_optimism.voice_note")

UPLOAD_PATH = "/v1/upload/voice-note"


async def create_voice_note(request: web.Request) -> web.Response:
    """POST /voice-note — forward the upload to the AI assistant."""
    url = os.environ["AI_ASSISTANT_URL"].rstrip("/") + UPLOAD_PATH
    token = os.environ["AI_ASSISTANT_TOKEN"]

    body = await request.read()
    # The raw header, not request.content_type: it carries the multipart boundary.
    headers = {"X-API-Key": token, "Content-Type": request.headers["Content-Type"]}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=body, headers=headers) as resp:
                log.info("forwarded voice note (%d bytes): %s", len(body), resp.status)
                return web.Response(
                    status=resp.status,
                    body=await resp.read(),
                    content_type=resp.content_type,
                    charset=resp.charset,
                )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        log.warning("forwarding voice note to %s failed: %s", url, exc)
        return web.json_response({"error": "assistant_unreachable"}, status=502)
