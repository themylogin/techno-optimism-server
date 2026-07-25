"""Voice-note upload endpoint.

    POST /voice-note   multipart: an mp3 file + a `timestamp` field

The audio is written under the voice-notes directory at
``%Y/%m/%d/%H-%M-%S.mp3`` (derived from `timestamp`), and a ``voice_note`` row
is inserted. Everything after that — transcription and publishing to Vikunja —
is done asynchronously by the worker (see ``worker.py``), so the upload returns
as soon as the bytes are on disk and the row exists.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from aiohttp import web

from techno_optimism_server.db import VoiceNote

log = logging.getLogger("techno_optimism.voice_note")

# Where the .mp3 files live. Kept alongside the SQLite file (see db.DB_PATH) so
# a single mounted volume covers both.
VOICE_NOTES_DIR = Path(os.environ.get("VOICE_NOTES_DIR", "voice-notes"))

DB_SESSIONMAKER_KEY = "db_sessionmaker"

# Audio extensions preserved from the upload (e.g. Telegram voice notes arrive
# as .ogg). Anything else — or a missing extension — is stored as .mp3.
_ALLOWED_EXTS = frozenset(
    {".mp3", ".ogg", ".oga", ".opus", ".m4a", ".wav", ".webm", ".aac", ".flac"}
)


def _extension(filename: str | None) -> str:
    """The stored file's extension, taken from the upload's name (default .mp3)."""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _ALLOWED_EXTS else ".mp3"


def parse_timestamp(raw: str) -> datetime:
    """Parse the `timestamp` field into a local ``datetime``.

    Accepts a Unix epoch (seconds, int or float) or an ISO-8601 string, so
    clients can send whichever is convenient. Raises ``ValueError`` otherwise.
    """
    raw = raw.strip()
    try:
        return datetime.fromtimestamp(float(raw))
    except (ValueError, OverflowError, OSError):
        pass
    # datetime.fromisoformat handles a trailing 'Z' only from Python 3.11+.
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def audio_path(when: datetime, ext: str = ".mp3", base: Path | None = None) -> Path:
    """Absolute path for a note recorded at `when`: base/%Y/%m/%d/%H-%M-%S{ext}.

    If that second is already taken, a short unique suffix is inserted before
    the extension so no upload ever clobbers another.
    """
    base = base or VOICE_NOTES_DIR
    target = base / when.strftime(f"%Y/%m/%d/%H-%M-%S{ext}")
    if target.exists():
        target = target.with_name(f"{target.stem}-{uuid4().hex[:8]}{ext}")
    return target


async def create_voice_note(request: web.Request) -> web.Response:
    """POST /voice-note — store the mp3 and insert its row."""
    reader = await request.post()

    raw_ts = reader.get("timestamp")
    if not isinstance(raw_ts, str) or not raw_ts.strip():
        return web.json_response({"error": "missing_timestamp"}, status=400)
    try:
        when = parse_timestamp(raw_ts)
    except ValueError:
        return web.json_response({"error": "bad_timestamp"}, status=400)

    # The mp3 can arrive under any field name; take the first uploaded file.
    file_field = next(
        (v for v in reader.values() if isinstance(v, web.FileField)), None
    )
    if file_field is None:
        return web.json_response({"error": "missing_file"}, status=400)
    audio = file_field.file.read()
    if not audio:
        return web.json_response({"error": "empty_file"}, status=400)

    path = audio_path(when, _extension(file_field.filename))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    relname = path.relative_to(VOICE_NOTES_DIR).as_posix()
    log.info("stored voice note %s (%d bytes)", relname, len(audio))

    sessionmaker = request.app[DB_SESSIONMAKER_KEY]
    async with sessionmaker() as session:
        note = VoiceNote(filename=relname)
        session.add(note)
        await session.commit()
        note_id = note.id

    return web.json_response({"id": note_id, "filename": relname}, status=201)
