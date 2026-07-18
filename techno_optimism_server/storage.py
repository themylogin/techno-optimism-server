"""Per-interaction storage.

One `Storage` object owns one interaction directory and writes each piece of
data the moment it arrives, so a crash mid-session still leaves everything
received up to that point on disk. Layout:

    <path>/
        question.mp3        raw question audio
        context.mp3         raw context audio (only when context was used)
        response.mp3        synthesized answer audio
        interaction.json    {"previous_response_id", "question",
                             "needs_context", "context", "answer",
                             "response_id"} as they become known
        error.txt           traceback, if the session raised

All methods are async and push blocking file I/O to a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

log = logging.getLogger("techno_optimism.storage")

BASE_DIR = Path(os.environ.get("INTERACTIONS_DIR", "interactions"))


def interaction_dir(when: datetime, conn_id: str) -> Path:
    """Directory for an interaction: interactions/%Y/%m/%d/%H-%M-%S, suffixed
    with the connection id if that second already has one."""
    target = BASE_DIR / when.strftime("%Y/%m/%d/%H-%M-%S")
    if target.exists():
        target = target.parent / f"{target.name}-{conn_id}"
    return target


class Storage:
    def __init__(self, path: str | Path) -> None:
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._data: dict = {}
        self._lock = asyncio.Lock()

    # -- audio blobs ------------------------------------------------------- #
    async def save_question(self, audio: bytes) -> None:
        await asyncio.to_thread(self._write_bytes, "question.mp3", audio)

    async def save_context(self, audio: bytes) -> None:
        await asyncio.to_thread(self._write_bytes, "context.mp3", audio)

    async def save_response(self, audio: bytes) -> None:
        await asyncio.to_thread(self._write_bytes, "response.mp3", audio)

    # -- json fields ------------------------------------------------------- #
    async def save_previous_response_id(self, previous_response_id: str) -> None:
        await self._set(previous_response_id=previous_response_id)

    async def save_question_text(self, text: str) -> None:
        await self._set(question=text)

    async def save_needs_context(self, needs_context: bool) -> None:
        await self._set(needs_context=needs_context)

    async def save_context_text(self, text: str) -> None:
        await self._set(context=text)

    async def save_response_text(self, text: str, response_id: str) -> None:
        await self._set(answer=text, response_id=response_id)

    # -- error ------------------------------------------------------------- #
    async def save_error(self, traceback_text: str) -> None:
        await asyncio.to_thread(self._write_text, "error.txt", traceback_text)

    # -- internals --------------------------------------------------------- #
    async def _set(self, **fields: object) -> None:
        async with self._lock:
            self._data.update(fields)
            payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        await asyncio.to_thread(self._write_text, "interaction.json", payload)

    def _write_bytes(self, name: str, data: bytes) -> None:
        (self._dir / name).write_bytes(data)

    def _write_text(self, name: str, data: str) -> None:
        (self._dir / name).write_text(data, encoding="utf-8")
