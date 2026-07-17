"""Persist each interaction to disk.

Layout, one directory per interaction:

    interactions/%Y/%m/%d/%H-%M-%S/
        question.mp3        raw question audio
        context.mp3         raw context audio (only when context was used)
        response.mp3        synthesized answer audio
        interaction.json    {"question", "context", "answer"} transcripts/answer

Two interactions in the same second would share a directory, so on collision
the directory name is suffixed with the connection id to keep them separate.
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


async def save_interaction(
    when: datetime,
    conn_id: str,
    question_audio: bytes,
    context_audio: bytes | None,
    response_audio: bytes,
    question: str,
    context: str | None,
    answer: str,
) -> None:
    """Write the interaction's audio and JSON. Runs file I/O off the loop."""
    await asyncio.to_thread(
        _save_sync, when, conn_id, question_audio, context_audio, response_audio,
        question, context, answer,
    )


def _save_sync(
    when: datetime,
    conn_id: str,
    question_audio: bytes,
    context_audio: bytes | None,
    response_audio: bytes,
    question: str,
    context: str | None,
    answer: str,
) -> None:
    target = BASE_DIR / when.strftime("%Y/%m/%d/%H-%M-%S")
    if target.exists():
        target = target.parent / f"{target.name}-{conn_id}"
    target.mkdir(parents=True, exist_ok=True)

    (target / "question.mp3").write_bytes(question_audio)
    if context_audio is not None:
        (target / "context.mp3").write_bytes(context_audio)
    (target / "response.mp3").write_bytes(response_audio)

    payload = {"question": question, "context": context, "answer": answer}
    (target / "interaction.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("saved interaction to %s", target)
