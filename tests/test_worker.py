"""Tests for the transcription + Vikunja publication worker.

The two passes are exercised directly (``transcribe_pending`` /
``publish_pending``) against a temp SQLite database and temp audio files. The AI
boundary is a FakeAI; the Vikunja HTTP calls are monkeypatched so no network is
touched.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from techno_optimism_server import worker as w
from techno_optimism_server.db import Base, VoiceNote, make_engine, make_sessionmaker

AUDIO = b"\xff\xf3AUDIO"


class FakeAI:
    def __init__(self, result="hello world", exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    async def transcribe(self, audio):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
async def db(tmp_path):
    """(sessionmaker, base_dir) backed by a fresh temp database."""
    base = tmp_path / "voice-notes"
    base.mkdir()
    engine = make_engine(tmp_path / "db.sqlite3")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield make_sessionmaker(engine), base
    await engine.dispose()


async def _add_note(sessionmaker, base, name="2026/07/25/12-30-00.mp3", **fields):
    """Write an mp3 on disk and insert its row; return the note id."""
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(AUDIO)
    async with sessionmaker() as session:
        note = VoiceNote(filename=name, **fields)
        session.add(note)
        await session.commit()
        return note.id


async def _get(sessionmaker, note_id):
    async with sessionmaker() as session:
        return await session.get(VoiceNote, note_id)


# --- backoff unit --------------------------------------------------------- #
def test_backoff_due_never_tried():
    note = VoiceNote(filename="x", transcription_retries=0)
    assert w._backoff_due(note, datetime.now()) is True


def test_backoff_due_respects_exponential_delay():
    now = datetime(2026, 7, 25, 12, 0, 0)
    note = VoiceNote(
        filename="x",
        transcription_retries=1,  # delay = BASE_BACKOFF * 2**0 = 10s
        transcription_last_retry=now,
    )
    assert w._backoff_due(note, now + timedelta(seconds=9)) is False
    assert w._backoff_due(note, now + timedelta(seconds=10)) is True

    note.transcription_retries = 3  # delay = 10 * 2**2 = 40s
    note.transcription_last_retry = now
    assert w._backoff_due(note, now + timedelta(seconds=39)) is False
    assert w._backoff_due(note, now + timedelta(seconds=40)) is True


# --- transcription pass --------------------------------------------------- #
async def test_transcribe_success(db):
    sessionmaker, base = db
    nid = await _add_note(sessionmaker, base)
    ai = FakeAI(result="a grocery list")

    await w.transcribe_pending(sessionmaker, ai, base)

    note = await _get(sessionmaker, nid)
    assert note.transcription == "a grocery list"
    assert note.transcription_error is None
    assert note.transcription_retries == 0


async def test_transcribe_failure_records_retry_and_backs_off(db):
    sessionmaker, base = db
    nid = await _add_note(sessionmaker, base)
    ai = FakeAI(exc=RuntimeError("stt down"))

    await w.transcribe_pending(sessionmaker, ai, base)
    note = await _get(sessionmaker, nid)
    assert note.transcription is None
    assert note.transcription_retries == 1
    assert "stt down" in note.transcription_error
    assert note.transcription_last_retry is not None
    assert ai.calls == 1

    # Immediately re-running must not attempt again: the 10s backoff isn't due.
    await w.transcribe_pending(sessionmaker, ai, base)
    assert ai.calls == 1
    assert (await _get(sessionmaker, nid)).transcription_retries == 1


async def test_transcribe_retries_once_backoff_elapsed(db):
    sessionmaker, base = db
    # A note that failed once, long enough ago that the retry is now due.
    nid = await _add_note(
        sessionmaker, base,
        transcription_retries=1,
        transcription_last_retry=datetime.now() - timedelta(hours=1),
    )
    ai = FakeAI(result="second time lucky")
    await w.transcribe_pending(sessionmaker, ai, base)
    assert (await _get(sessionmaker, nid)).transcription == "second time lucky"


async def test_transcribe_gives_up_at_max_retries(db):
    sessionmaker, base = db
    nid = await _add_note(
        sessionmaker, base,
        transcription_retries=w.MAX_RETRIES - 1,
        transcription_last_retry=datetime.now() - timedelta(hours=1),
    )
    ai = FakeAI(exc=RuntimeError("still broken"))
    await w.transcribe_pending(sessionmaker, ai, base)
    note = await _get(sessionmaker, nid)
    assert note.transcription_retries == w.MAX_RETRIES
    assert note.transcription is None

    # Exhausted notes are never attempted again.
    await w.transcribe_pending(sessionmaker, ai, base)
    assert ai.calls == 1


# --- publication pass ----------------------------------------------------- #
@pytest.fixture
def fake_vikunja(monkeypatch):
    """Capture Vikunja calls; return the recorder."""
    calls = {"tasks": [], "attachments": [], "next_id": 100, "fail": False}

    async def _create_task(http, base_url, token, project_id, title):
        if calls["fail"]:
            raise RuntimeError("vikunja down")
        calls["tasks"].append((base_url, token, project_id, title))
        calls["next_id"] += 1
        return calls["next_id"]

    async def _upload(http, base_url, token, task_id, filename, audio):
        calls["attachments"].append((task_id, filename, audio))

    monkeypatch.setattr(w, "_create_task", _create_task)
    monkeypatch.setattr(w, "_upload_attachment", _upload)
    return calls


async def _publish(sessionmaker, base):
    await w.publish_pending(
        sessionmaker, http=None, base=base,
        vikunja_url="http://vikunja.test", token="tok", project_id=7,
    )


async def test_publish_transcribed_note(db, fake_vikunja):
    sessionmaker, base = db
    nid = await _add_note(sessionmaker, base, transcription="buy milk")

    await _publish(sessionmaker, base)

    note = await _get(sessionmaker, nid)
    assert note.vikunja_id == 101
    assert fake_vikunja["tasks"] == [("http://vikunja.test", "tok", 7, "buy milk")]
    task_id, filename, audio = fake_vikunja["attachments"][0]
    assert task_id == 101
    assert filename == "2026/07/25/12-30-00.mp3"
    assert audio == AUDIO


async def test_publish_failed_transcription_uses_placeholder_title(db, fake_vikunja):
    sessionmaker, base = db
    nid = await _add_note(
        sessionmaker, base, transcription_retries=w.MAX_RETRIES
    )
    await _publish(sessionmaker, base)
    note = await _get(sessionmaker, nid)
    assert note.vikunja_id == 101
    assert fake_vikunja["tasks"][0][3] == w.TRANSCRIPTION_FAILED_TITLE


async def test_publish_skips_untranscribed_note(db, fake_vikunja):
    sessionmaker, base = db
    nid = await _add_note(sessionmaker, base, transcription_retries=2)
    await _publish(sessionmaker, base)
    assert fake_vikunja["tasks"] == []
    assert (await _get(sessionmaker, nid)).vikunja_id is None


async def test_publish_failure_leaves_note_for_retry(db, fake_vikunja):
    sessionmaker, base = db
    fake_vikunja["fail"] = True
    nid = await _add_note(sessionmaker, base, transcription="buy milk")
    await _publish(sessionmaker, base)
    assert (await _get(sessionmaker, nid)).vikunja_id is None

    # Next loop succeeds and publishes it.
    fake_vikunja["fail"] = False
    await _publish(sessionmaker, base)
    assert (await _get(sessionmaker, nid)).vikunja_id == 101


async def test_publish_skips_already_published(db, fake_vikunja):
    sessionmaker, base = db
    await _add_note(sessionmaker, base, transcription="done", vikunja_id=42)
    await _publish(sessionmaker, base)
    assert fake_vikunja["tasks"] == []
