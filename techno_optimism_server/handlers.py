"""Request handlers.

The ask flow is a background job the client drives over plain REST, so any
request can be safely retried over a flaky link:

    POST /v1/interactions              upload question audio -> {id, status}
    GET  /v1/interactions/{id}         poll the status snapshot
    PUT  /v1/interactions/{id}/context upload the follow-up context audio
    GET  /v1/interactions/{id}/answer.mp3   download the answer (Range-enabled)

Every call returns immediately. All model IO happens in a per-job asyncio task
whose progress the client learns by polling. Job state lives in RAM only
(single instance, short-lived interactions); the answer audio is served from
the interaction directory on disk and stays there indefinitely.
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from aiohttp import web

from techno_optimism_server import storage as storage_mod
from techno_optimism_server.ai import Progress, Result
from techno_optimism_server.storage import Storage, interaction_dir

log = logging.getLogger("techno_optimism.handlers")

# How long to wait for the follow-up context upload after need_context, seconds.
CONTEXT_TIMEOUT = float(os.environ.get("CONTEXT_TIMEOUT", "60"))
# The model's reasoning streams in tiny deltas; the client sees only the latest
# tail (via polling). We also archive that tail to interaction.json on a steady
# cadence rather than on every delta, to bound disk churn and the archive size.
THINKING_INTERVAL = float(os.environ.get("THINKING_INTERVAL", "1.0"))
THINKING_TAIL = int(os.environ.get("THINKING_TAIL", "100"))

# Lifecycle states a job moves through.
TRANSCRIBING = "transcribing"
NEED_CONTEXT = "need_context"
THINKING = "thinking"
SYNTHESIZING = "synthesizing"
DONE = "done"
ERROR = "error"


@dataclass
class Job:
    """One ask in flight, held in the app's in-RAM registry.

    The background task mutates these fields as it goes; every HTTP handler
    reads them to build the status snapshot the client polls.
    """

    id: str
    storage: Storage
    previous_response_id: str | None = None
    status: str = TRANSCRIBING
    question: str | None = None
    thinking: str | None = None
    response_id: str | None = None
    answer_text: str | None = None
    error: str | None = None
    context_audio: bytes | None = None
    context_arrived: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


async def health(request: web.Request) -> web.Response:
    """Liveness probe."""
    return web.json_response({"status": "ok"})


# --------------------------------------------------------------------------- #
# REST handlers
# --------------------------------------------------------------------------- #
async def create_interaction(request: web.Request) -> web.Response:
    """POST /v1/interactions — upload question audio, start the background job.

    The audio is the raw request body. `?previous_response_id=` optionally
    continues a prior conversation. Returns 201 with the initial snapshot
    immediately; the id names the interaction directory on disk.
    """
    ai = request.app["ai"]
    jobs = request.app["jobs"]

    audio = await request.read()
    if not audio:
        return web.json_response({"error": "empty_body"}, status=400)

    previous_response_id = request.query.get("previous_response_id") or None

    started = datetime.now()
    conn_id = uuid4().hex[:8]
    directory = interaction_dir(started, conn_id)
    interaction_id = str(directory.relative_to(storage_mod.BASE_DIR)).replace(
        os.sep, "-"
    )
    log.info("[%s] question blob: %d bytes", interaction_id, len(audio))

    store = Storage(directory)
    await store.save_question(audio)
    if previous_response_id:
        await store.save_previous_response_id(previous_response_id)
        log.info("[%s] continuing from %s", interaction_id, previous_response_id)

    job = Job(id=interaction_id, storage=store,
              previous_response_id=previous_response_id)
    jobs[interaction_id] = job
    job.task = asyncio.create_task(_run(job, ai, audio))

    return web.json_response(_snapshot(job), status=201)


async def get_interaction(request: web.Request) -> web.Response:
    """GET /v1/interactions/{id} — the current status snapshot."""
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        return web.json_response({"error": "unknown_interaction"}, status=404)
    return web.json_response(_snapshot(job))


async def upload_context(request: web.Request) -> web.Response:
    """PUT /v1/interactions/{id}/context — supply the follow-up context audio.

    Idempotent: only the first upload unblocks the waiting job; later retries
    just return the current snapshot.
    """
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        return web.json_response({"error": "unknown_interaction"}, status=404)

    audio = await request.read()
    if not audio:
        return web.json_response({"error": "empty_body"}, status=400)

    if not job.context_arrived.is_set():
        log.info("[%s] context blob: %d bytes", job.id, len(audio))
        job.context_audio = audio
        job.context_arrived.set()

    return web.json_response(_snapshot(job))


async def get_answer_audio(request: web.Request) -> web.StreamResponse:
    """GET /v1/interactions/{id}/answer.mp3 — the synthesized answer.

    Served straight off disk via FileResponse, which honours Range requests so
    a download dropped mid-transfer resumes instead of restarting.
    """
    job = request.app["jobs"].get(request.match_info["id"])
    if job is None:
        return web.json_response({"error": "unknown_interaction"}, status=404)

    audio_path = job.storage.path / "response.mp3"
    if not audio_path.exists():
        return web.json_response({"error": "answer_not_ready"}, status=404)
    return web.FileResponse(audio_path)


# --------------------------------------------------------------------------- #
# Background job
# --------------------------------------------------------------------------- #
async def _run(job: Job, ai, audio: bytes) -> None:
    """Drive one interaction end to end, updating `job` as it progresses."""
    store = job.storage
    try:
        question = await ai.transcribe(audio)
        job.question = question
        await store.save_question_text(question)
        log.info("[%s] question: %r", job.id, question)

        needs_context = await ai.needs_context(question)
        await store.save_needs_context(needs_context)

        context = None
        if needs_context:
            job.status = NEED_CONTEXT
            try:
                await asyncio.wait_for(job.context_arrived.wait(), CONTEXT_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("[%s] timed out waiting for context", job.id)
                await store.save_error("context not received in time")
                job.error = "context_not_received"
                job.status = ERROR
                return
            await store.save_context(job.context_audio)
            context = await ai.transcribe(job.context_audio)
            await store.save_context_text(context)
            log.info("[%s] context: %r", job.id, context)

        job.status = THINKING
        answer, response_id = await _think(job, ai, question, context)
        job.answer_text = answer
        job.response_id = response_id

        job.status = SYNTHESIZING
        speech = await ai.say(answer)
        await store.save_response(speech)
        log.info("[%s] answer audio: %d bytes", job.id, len(speech))
        job.status = DONE

    except Exception as exc:  # noqa: BLE001 - surface failure via the snapshot
        log.exception("[%s] processing failed", job.id)
        await store.save_error(traceback.format_exc())
        job.error = str(exc)
        job.status = ERROR


async def _think(job: Job, ai, question: str, context: str | None):
    """Stream the answer, exposing the latest thinking tail on `job.thinking`.

    Returns (answer, response_id). The tail is archived to interaction.json on
    a THINKING_INTERVAL cadence (not per delta) to keep disk writes bounded.
    """
    store = job.storage
    answer = ""
    response_id = ""
    archived: str | None = None

    async def _archive_ticker() -> None:
        nonlocal archived
        while True:
            await asyncio.sleep(THINKING_INTERVAL)
            if job.thinking and job.thinking != archived:
                await store.add_progress(job.thinking)
                archived = job.thinking

    ticker = asyncio.create_task(_archive_ticker())
    try:
        async for item in ai.ask(question, context,
                                 previous_response_id=job.previous_response_id):
            if isinstance(item, Progress):
                job.thinking = _thinking_tail(item.text)
            elif isinstance(item, Result):
                answer = item.answer
                response_id = item.response_id
                await store.save_response_text(item.answer, item.response_id)
    finally:
        ticker.cancel()
        try:
            await ticker
        except (asyncio.CancelledError, Exception):
            pass  # archive ticker is best-effort

    # Guarantee the final tail is archived (covers streams faster than a tick).
    if job.thinking and job.thinking != archived:
        await store.add_progress(job.thinking)

    return answer, response_id


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _snapshot(job: Job) -> dict:
    """The client-facing status of a job. Fields appear as they become known."""
    snap: dict = {"id": job.id, "status": job.status}
    if job.question is not None:
        snap["question"] = job.question
    if job.thinking is not None:
        snap["thinking"] = job.thinking
    if job.status == DONE:
        snap["response_id"] = job.response_id
        snap["answer_text"] = job.answer_text
        snap["answer_audio_url"] = f"/v1/interactions/{job.id}/answer.mp3"
    if job.status == ERROR:
        snap["error"] = {"detail": job.error}
    return snap


def _thinking_tail(text: str) -> str:
    """The last THINKING_TAIL chars, prefixed with … when the start is cut off."""
    if len(text) <= THINKING_TAIL:
        return text
    return "…" + text[-THINKING_TAIL:]
