"""Request handlers."""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from datetime import datetime
from uuid import uuid4

from aiohttp import WSMsgType, web

from techno_optimism_server.ai import Progress, Result
from techno_optimism_server.storage import Storage, interaction_dir

log = logging.getLogger("techno_optimism.handlers")

# Max size of a single WebSocket message, in bytes.
WS_MAX_MSG_SIZE = 32 * 1024 * 1024
# How long to wait for the follow-up context blob after need_context, seconds.
CONTEXT_TIMEOUT = float(os.environ.get("CONTEXT_TIMEOUT", "60"))


async def health(request: web.Request) -> web.Response:
    """Liveness probe."""
    return web.json_response({"status": "ok"})


async def ask_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint: spoken question in, spoken answer out.

    Protocol:
        1. Client sends a binary frame with the question audio (e.g. mp3).
        2. Server acks: {"msg": "uploaded"}.
        3. Server transcribes it and decides whether it references external
           context the user just heard/saw.
        4. If it does, server sends {"msg": "need_context"} and the client
           sends a second binary frame with the surrounding-context audio,
           which the server transcribes.
        5. Server answers with a web-search-enabled reasoning model, streaming
           the answer as {"msg": "thinking", "text": "<chunk>"} frames.
        6. Server synthesizes the final answer to speech and sends it as one
           binary frame, then closes.
    """
    ai = request.app["ai"]
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

        started = datetime.now()
        audio: bytes = msg.data
        log.info("[%s] received question blob: %d bytes", conn_id, len(audio))
        storage = Storage(interaction_dir(started, conn_id))
        await storage.save_question(audio)
        await ws.send_json({"msg": "uploaded"})

        try:
            question = await ai.transcribe(audio)
            await storage.save_question_text(question)
            log.info("[%s] question: %r", conn_id, question)

            needs_context = await ai.needs_context(question)
            await storage.save_needs_context(needs_context)

            context = None
            if needs_context:
                await ws.send_json({"msg": "need_context"})
                ctx_audio = await _receive_context_blob(ws, conn_id)
                if ctx_audio is None:
                    await ws.send_json({"ok": False, "error": "context_not_received"})
                    return ws
                await storage.save_context(ctx_audio)
                context = await ai.transcribe(ctx_audio)
                await storage.save_context_text(context)
                log.info("[%s] context: %r", conn_id, context)

            answer = ""
            async for item in ai.ask(question, context):
                if isinstance(item, Progress):
                    await ws.send_json({"msg": "thinking", "text": item.text})
                elif isinstance(item, Result):
                    answer = item.answer
                    await storage.save_response_text(item.answer, item.response_id)

            speech = await ai.say(answer)
            await ws.send_bytes(speech)
            await storage.save_response(speech)
            log.info("[%s] sent answer audio: %d bytes", conn_id, len(speech))

        except Exception as exc:  # noqa: BLE001 - surface failure to the client
            log.exception("[%s] processing failed", conn_id)
            await storage.save_error(traceback.format_exc())
            await ws.send_json({"ok": False, "error": "processing_failed",
                                "detail": str(exc)})
            return ws

    finally:
        await ws.close()
        log.info("[%s] connection closed", conn_id)

    return ws


async def _receive_context_blob(
    ws: web.WebSocketResponse, conn_id: str
) -> bytes | None:
    """Wait for the follow-up context audio frame; return its bytes or None."""
    try:
        msg = await ws.receive(timeout=CONTEXT_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("[%s] timed out waiting for context blob", conn_id)
        return None
    if msg.type != WSMsgType.BINARY:
        log.warning("[%s] expected context binary frame, got %s", conn_id, msg.type.name)
        return None
    log.info("[%s] received context blob: %d bytes", conn_id, len(msg.data))
    return msg.data
