"""Request handlers."""

from __future__ import annotations

import asyncio
import json
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
# The model's reasoning streams in tiny deltas; rather than forward each one,
# we send the tail of the accumulated reasoning on a steady cadence so the
# client can show a smooth "thinking" ticker of its thought process.
THINKING_INTERVAL = float(os.environ.get("THINKING_INTERVAL", "1.0"))
THINKING_TAIL = int(os.environ.get("THINKING_TAIL", "100"))


async def health(request: web.Request) -> web.Response:
    """Liveness probe."""
    return web.json_response({"status": "ok"})


async def ask_ws(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint: spoken question in, spoken answer out.

    Protocol:
        0. Optionally, the client's first frame is text
           {"previous_response_id": "..."} to continue a prior conversation;
           the question blob then follows.
        1. Client sends a binary frame with the question audio (e.g. mp3).
        2. Server acks: {"msg": "uploaded"}.
        3. Server transcribes it and decides whether it references external
           context the user just heard/saw.
        4. If it does, server sends {"msg": "need_context"} and the client
           sends a second binary frame with the surrounding-context audio,
           which the server transcribes.
        5. Server answers with a web-search-enabled reasoning model. About once
           a second it sends {"msg": "thinking", "text": "<tail>"} where <tail>
           is the last ~100 chars of the model's reasoning so far (a ticker).
           Simple questions may produce no reasoning, hence no such frames.
        6. Server sends {"msg": "done", "response_id": "..."} (for chaining),
           then the synthesized answer as one binary frame, then closes.
    """
    ai = request.app["ai"]
    ws = web.WebSocketResponse(max_msg_size=WS_MAX_MSG_SIZE)
    await ws.prepare(request)

    conn_id = uuid4().hex[:8]
    log.info("[%s] client connected from %s", conn_id, request.remote)

    try:
        msg = await ws.receive()

        # Optional first frame: {"previous_response_id": "..."} to continue a
        # prior conversation. The question blob follows it.
        previous_response_id = None
        if msg.type == WSMsgType.TEXT:
            previous_response_id = _parse_handshake(msg.data)
            if previous_response_id is None:
                log.warning("[%s] invalid handshake frame", conn_id)
                await ws.send_json({"ok": False, "error": "invalid_handshake"})
                return ws
            log.info("[%s] continuing from %s", conn_id, previous_response_id)
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
        if previous_response_id:
            await storage.save_previous_response_id(previous_response_id)
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
            response_id = ""
            thinking = {"full": "", "last": None}

            async def _thinking_ticker() -> None:
                # Every THINKING_INTERVAL, push the tail of the text so far.
                while True:
                    await asyncio.sleep(THINKING_INTERVAL)
                    tail = thinking["full"][-THINKING_TAIL:]
                    if tail and tail != thinking["last"]:
                        thinking["last"] = tail
                        await ws.send_json({"msg": "thinking", "text": tail})

            ticker = asyncio.create_task(_thinking_ticker())
            try:
                async for item in ai.ask(question, context,
                                         previous_response_id=previous_response_id):
                    if isinstance(item, Progress):
                        thinking["full"] += item.text
                    elif isinstance(item, Result):
                        answer = item.answer
                        response_id = item.response_id
                        await storage.save_response_text(item.answer, item.response_id)
            finally:
                ticker.cancel()
                try:
                    await ticker
                except (asyncio.CancelledError, Exception):
                    pass  # ticker is best-effort display

            # Guarantee a final frame (covers answers faster than one tick).
            final_tail = thinking["full"][-THINKING_TAIL:]
            if final_tail and final_tail != thinking["last"]:
                await ws.send_json({"msg": "thinking", "text": final_tail})

            speech = await ai.say(answer)
            # Send the response id before the audio so the client can chain
            # the next turn with it.
            await ws.send_json({"msg": "done", "response_id": response_id})
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


def _parse_handshake(raw: str) -> str | None:
    """Extract a non-empty previous_response_id from a handshake frame, or
    None if the frame is not a valid {"previous_response_id": str}."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    prev = data.get("previous_response_id")
    return prev if isinstance(prev, str) and prev else None


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
