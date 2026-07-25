"""Background worker: transcribe voice notes, then publish them to Vikunja.

Runs as its own process (``python -m techno_optimism_server.worker``). Each loop
iteration does two passes over the ``voice_note`` table and then sleeps:

1. Transcription — every note without a transcription is (re)transcribed the
   same way the assistant does (``AI.transcribe``). A failure bumps the retry
   counter and stamps the time; the next attempt is held off with exponential
   backoff (first retry after 10s, doubling each time) until ``MAX_RETRIES`` is
   reached, after which the note is left permanently failed.

2. Publication — every note that has finished transcription (succeeded, or
   given up) and isn't in Vikunja yet is pushed as a task: the transcription as
   the title (or "<transcription failed>"), the original mp3 as an attachment.
   A failure just leaves the note for the next loop to retry.

Both passes tolerate partial failure: one bad note never blocks the others.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from sqlalchemy import or_, select

from techno_optimism_server.ai import AI
from techno_optimism_server.db import VoiceNote, apply_migrations, make_engine, make_sessionmaker
from techno_optimism_server.voice_note import VOICE_NOTES_DIR

log = logging.getLogger("techno_optimism.worker")

# Give up transcribing after this many failed attempts; the note is then
# published with a "<transcription failed>" title so it isn't lost.
MAX_RETRIES = int(os.environ.get("TRANSCRIBE_MAX_RETRIES", "10"))
# First retry after this many seconds, doubling each subsequent attempt.
BASE_BACKOFF = float(os.environ.get("TRANSCRIBE_BASE_BACKOFF", "10"))
# Pause between loop iterations.
LOOP_SLEEP = float(os.environ.get("WORKER_SLEEP", "10"))

TRANSCRIPTION_FAILED_TITLE = "<transcription failed>"


def _backoff_due(note: VoiceNote, now: datetime) -> bool:
    """Whether enough time has passed since the last failed attempt to retry."""
    if note.transcription_last_retry is None:
        return True
    delay = BASE_BACKOFF * (2 ** (note.transcription_retries - 1))
    return now >= note.transcription_last_retry + timedelta(seconds=delay)


async def transcribe_pending(sessionmaker, ai: AI, base: Path) -> None:
    """Attempt transcription for every not-yet-transcribed, not-exhausted note."""
    async with sessionmaker() as session:
        result = await session.execute(
            select(VoiceNote).where(
                VoiceNote.transcription.is_(None),
                VoiceNote.transcription_retries < MAX_RETRIES,
            )
        )
        notes = result.scalars().all()

    now = datetime.now()
    for note in notes:
        if not _backoff_due(note, now):
            continue
        await _transcribe_one(sessionmaker, ai, base, note.id)


async def _transcribe_one(sessionmaker, ai: AI, base: Path, note_id: int) -> None:
    async with sessionmaker() as session:
        note = await session.get(VoiceNote, note_id)
        if note is None or note.transcription is not None:
            return
        try:
            audio = (base / note.filename).read_bytes()
            text = await ai.transcribe(audio)
            note.transcription = text
            note.transcription_error = None
            log.info("transcribed note %d: %r", note.id, text)
        except Exception as exc:  # noqa: BLE001 - record and back off
            note.transcription_retries += 1
            note.transcription_error = traceback.format_exc()
            note.transcription_last_retry = datetime.now()
            gave_up = note.transcription_retries >= MAX_RETRIES
            log.warning(
                "transcription of note %d failed (attempt %d/%d%s): %s",
                note.id, note.transcription_retries, MAX_RETRIES,
                ", giving up" if gave_up else "", exc,
            )
        await session.commit()


async def publish_pending(
    sessionmaker, http: aiohttp.ClientSession, base: Path,
    vikunja_url: str, token: str, project_id: int,
) -> None:
    """Push every finished-but-unpublished note to Vikunja as a task."""
    async with sessionmaker() as session:
        result = await session.execute(
            select(VoiceNote).where(
                VoiceNote.vikunja_id.is_(None),
                or_(
                    VoiceNote.transcription.is_not(None),
                    VoiceNote.transcription_retries >= MAX_RETRIES,
                ),
            )
        )
        note_ids = [n.id for n in result.scalars().all()]

    for note_id in note_ids:
        await _publish_one(
            sessionmaker, http, base, vikunja_url, token, project_id, note_id
        )


async def _publish_one(
    sessionmaker, http: aiohttp.ClientSession, base: Path,
    vikunja_url: str, token: str, project_id: int, note_id: int,
) -> None:
    async with sessionmaker() as session:
        note = await session.get(VoiceNote, note_id)
        if note is None or note.vikunja_id is not None:
            return
        title = note.transcription or TRANSCRIPTION_FAILED_TITLE
        filename, audio = note.filename, (base / note.filename).read_bytes()
    try:
        task_id = await _create_task(http, vikunja_url, token, project_id, title)
        await _upload_attachment(http, vikunja_url, token, task_id, filename, audio)
    except Exception as exc:  # noqa: BLE001 - retry on the next loop
        log.warning("publishing note %d to Vikunja failed: %s", note_id, exc)
        return
    async with sessionmaker() as session:
        note = await session.get(VoiceNote, note_id)
        if note is not None:
            note.vikunja_id = task_id
            await session.commit()
    log.info("published note %d as Vikunja task %d", note_id, task_id)


# --------------------------------------------------------------------------- #
# Vikunja REST client
# --------------------------------------------------------------------------- #
async def _create_task(
    http: aiohttp.ClientSession, base_url: str, token: str,
    project_id: int, title: str,
) -> int:
    url = f"{base_url.rstrip('/')}/api/v1/projects/{project_id}/tasks"
    headers = {"Authorization": f"Bearer {token}"}
    async with http.put(url, json={"title": title}, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["id"]


async def _upload_attachment(
    http: aiohttp.ClientSession, base_url: str, token: str,
    task_id: int, filename: str, audio: bytes,
) -> None:
    url = f"{base_url.rstrip('/')}/api/v1/tasks/{task_id}/attachments"
    headers = {"Authorization": f"Bearer {token}"}
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    form = aiohttp.FormData()
    form.add_field(
        "files", audio,
        filename=os.path.basename(filename), content_type=content_type,
    )
    async with http.put(url, data=form, headers=headers) as resp:
        resp.raise_for_status()


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #
async def run_worker() -> None:
    apply_migrations()

    vikunja_url = os.environ["VIKUNJA_URL"]
    token = os.environ["VIKUNJA_API_TOKEN"]
    project_id = int(os.environ["VIKUNJA_PROJECT_ID"])

    ai = AI()
    engine = make_engine()
    sessionmaker = make_sessionmaker(engine)
    base = VOICE_NOTES_DIR

    log.info("worker started (sleep=%ss, max_retries=%d)", LOOP_SLEEP, MAX_RETRIES)
    try:
        async with aiohttp.ClientSession() as http:
            while True:
                try:
                    await transcribe_pending(sessionmaker, ai, base)
                    await publish_pending(
                        sessionmaker, http, base, vikunja_url, token, project_id
                    )
                except Exception:  # noqa: BLE001 - never let the loop die
                    log.exception("worker loop iteration failed")
                await asyncio.sleep(LOOP_SLEEP)
    finally:
        await engine.dispose()


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
