"""Request handlers."""

from __future__ import annotations

import logging
from uuid import uuid4

from aiohttp import WSMsgType, web

from techno_optimism_server.transcribe import transcribe

log = logging.getLogger("techno_optimism.handlers")

# Max size of a single WebSocket message, in bytes.
WS_MAX_MSG_SIZE = 32 * 1024 * 1024


async def health(request: web.Request) -> web.Response:
    """Liveness probe."""
    return web.json_response({"status": "ok"})


async def ask_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for one-shot audio transcription.

    Protocol:
        1. Client connects and sends a single binary frame containing an
           audio file (e.g. mp3).
        2. Server transcribes it with the `gpt-4o-transcribe` model.
        3. Server sends the result back as one JSON text frame.
        4. Server closes the connection.
    """
    ws = web.WebSocketResponse(max_msg_size=WS_MAX_MSG_SIZE)
    await ws.prepare(request)

    conn_id = uuid4().hex[:8]
    log.info("[%s] client connected from %s", conn_id, request.remote)

    try:
        msg = await ws.receive()

        if msg.type != WSMsgType.BINARY:
            log.warning("[%s] expected binary frame, got %s", conn_id, msg.type.name)
            await ws.send_json({"ok": False, "error": "expected_binary_frame"})
            return ws

        audio: bytes = msg.data
        log.info("[%s] received audio blob: %d bytes; transcribing", conn_id, len(audio))

        try:
            text = await transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - surface failure to the client
            log.exception("[%s] transcription failed", conn_id)
            await ws.send_json({"ok": False, "error": "transcription_failed",
                                "detail": str(exc)})
            return ws

        log.info("[%s] transcription ok: %d chars", conn_id, len(text))
        await ws.send_json({"ok": True, "text": text})

    finally:
        await ws.close()
        log.info("[%s] connection closed", conn_id)

    return ws
