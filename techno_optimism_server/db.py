"""SQLite persistence for voice notes.

One table, ``voice_note``, tracks each uploaded recording through two
independent pipelines the worker drives (see ``worker.py``): transcription and
then publication to Vikunja. The server only ever inserts rows; the worker
reads and updates them. They are separate processes sharing one SQLite file, so
WAL mode + a busy timeout are enabled to keep concurrent access from erroring.

Schema changes go through Alembic (``migrations/``). ``apply_migrations`` runs
the pending migrations at process startup so both the server and the worker
converge on ``head`` before doing anything else.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, String, Text, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

log = logging.getLogger("techno_optimism.db")

# SQLite file holding the voice_note table. Defaults under the voice-notes
# directory so a single mounted volume covers both the audio and the database.
DB_PATH = Path(os.environ.get("DB_PATH", "voice-notes/db.sqlite3"))


def sync_url(path: Path | None = None) -> str:
    """Blocking SQLite URL — used by Alembic migrations."""
    return f"sqlite:///{(path or DB_PATH).as_posix()}"


def async_url(path: Path | None = None) -> str:
    """Async SQLite URL — used by the server and worker at runtime."""
    return f"sqlite+aiosqlite:///{(path or DB_PATH).as_posix()}"


class Base(DeclarativeBase):
    pass


class VoiceNote(Base):
    """One uploaded voice note and the state of its two pipelines.

    ``transcription`` stays NULL until the worker succeeds; each failure bumps
    ``transcription_retries`` and stamps ``transcription_last_retry`` so the
    worker can back off exponentially and give up after a bounded number of
    tries. ``vikunja_id`` stays NULL until the note is published as a task.
    """

    __tablename__ = "voice_note"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Path to the .mp3 relative to the voice-notes directory, e.g.
    # "2026/07/25/12-30-00.mp3".
    filename: Mapped[str] = mapped_column(String, nullable=False)
    transcription: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcription_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcription_retries: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    transcription_last_retry: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    vikunja_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record) -> None:
    """WAL + a busy timeout so the server and worker can share the file."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def make_engine(path: Path | None = None) -> AsyncEngine:
    """Async engine for the runtime processes."""
    return create_async_engine(async_url(path), future=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


def apply_migrations(path: Path | None = None) -> None:
    """Upgrade the database to ``head``.

    Called at startup by both the server and the worker; running twice is a
    no-op. Building the Alembic ``Config`` in code means no alembic.ini needs to
    ship in the image. A short retry rides out the rare case of both processes
    racing to create the schema on first boot.
    """
    from alembic import command
    from alembic.config import Config

    target = path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.set_main_option(
        "script_location", str(Path(__file__).parent / "migrations")
    )
    cfg.set_main_option("sqlalchemy.url", sync_url(target))

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            command.upgrade(cfg, "head")
            return
        except Exception as exc:  # noqa: BLE001 - retry transient lock races
            last_exc = exc
            log.warning("migration attempt %d failed: %s", attempt + 1, exc)
            time.sleep(0.5)
    raise RuntimeError(f"could not apply migrations: {last_exc}")
