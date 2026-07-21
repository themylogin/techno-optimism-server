"""Telegram bot that turns an uploaded GPX route into cached map tiles.

Send the bot a ``.gpx`` document and it:

    1. parses it and writes the track as ``static/route.json`` (a JSON list of
       ``[lat, lon]`` pairs),
    2. downloads every map tile the route crosses (plus neighbors), editing a
       single reply about once a second with live ``done/total`` progress,
    3. packages this route's tiles into ``static/tiles.zip``,
    4. finishes with "Route successfully uploaded".

The ``static/`` directory is served by the REST server under ``/static``.

It talks to the Telegram Bot API directly over aiohttp long-polling, so it
needs no extra dependencies beyond what the server already uses.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from techno_optimism_server.tiles import CACHE_DIR, download_tiles, parse_gpx

log = logging.getLogger("techno_optimism.telegram")

# Load .env before reading the config constants below, so values like
# TELEGRAM_ALLOWED_USER_ID are visible at import time (not just at __main__).
load_dotenv()

# Served assets live in the static directory (mounted as its own volume and
# exposed by the server under /static). Tiles are cached separately in CACHE_DIR.
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "static"))
ROUTE_JSON_PATH = STATIC_DIR / "route.json"
TILES_ZIP_PATH = STATIC_DIR / "tiles.zip"

# Tile parameters for an uploaded route; overridable via the environment.
TILE_ZOOM = int(os.environ.get("TILE_ZOOM", 19))
TILE_MAP_TYPE = os.environ.get("TILE_MAP_TYPE", "satellite")
TILE_INCLUDE_NEIGHBORS = os.environ.get("TILE_INCLUDE_NEIGHBORS", "1") != "0"

# Only this Telegram user id may use the bot. Unset -> nobody is authorized yet;
# the bot then replies with each sender's id so it can be added to .env.
ALLOWED_USER_ID = os.environ.get("TELEGRAM_ALLOWED_USER_ID")

# Never edit the status message more often than this (seconds), to respect
# Telegram's edit rate limits.
MIN_EDIT_INTERVAL = 1.0


class TelegramClient:
    """Thin async wrapper over the handful of Bot API methods we need."""

    def __init__(self, token: str, session: aiohttp.ClientSession) -> None:
        self._api = f"https://api.telegram.org/bot{token}"
        self._file_api = f"https://api.telegram.org/file/bot{token}"
        self._session = session

    async def _call(self, method: str, **params) -> dict:
        async with self._session.post(f"{self._api}/{method}", json=params) as resp:
            data = await resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {data}")
        return data["result"]

    async def get_updates(self, offset: int, timeout: int = 30) -> list[dict]:
        # Long-poll; use a client read timeout a bit longer than the poll.
        try:
            async with self._session.get(
                f"{self._api}/getUpdates",
                params={"offset": offset, "timeout": timeout},
                timeout=aiohttp.ClientTimeout(total=timeout + 10),
            ) as resp:
                data = await resp.json()
        except asyncio.TimeoutError:
            return []
        if not data.get("ok"):
            raise RuntimeError(f"Telegram getUpdates failed: {data}")
        return data["result"]

    async def send_message(self, chat_id: int, text: str) -> int:
        result = await self._call("sendMessage", chat_id=chat_id, text=text)
        return result["message_id"]

    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        try:
            await self._call(
                "editMessageText",
                chat_id=chat_id,
                message_id=message_id,
                text=text,
            )
        except RuntimeError as exc:
            # Editing to identical text raises "message is not modified" — ignore.
            if "not modified" not in str(exc):
                raise

    async def download_bytes(self, file_id: str) -> bytes:
        """Fetch a Telegram file's raw bytes without touching disk."""
        info = await self._call("getFile", file_id=file_id)
        file_path = info["file_path"]
        async with self._session.get(f"{self._file_api}/{file_path}") as resp:
            resp.raise_for_status()
            return await resp.read()


def _zip_tiles(paths: list[Path], dest: Path) -> None:
    """Package exactly this route's tiles into ``dest`` (a zip file).

    Only the given tile paths are added — the tiles the route touches — not the
    whole cache. Each entry keeps its ``tiles/<map_type>/<z>/<x>/<y>`` layout so
    the archive is self-describing. Blocking, so call via ``asyncio.to_thread``.
    """
    # Tiles live under CACHE_DIR (cache/tiles/...); arcnames stay cache-relative
    # (tiles/...) even though the zip itself is written into static/.
    base = CACHE_DIR.parent
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in paths:
            zf.write(path, arcname=path.relative_to(base).as_posix())


async def _handle_document(client: TelegramClient, chat_id: int, doc: dict) -> None:
    """Download the GPX, fetch its tiles, and report progress in one message."""
    name = doc.get("file_name", "route.gpx")
    if not name.lower().endswith(".gpx"):
        await client.send_message(chat_id, "Please send a .gpx file.")
        return

    gpx_bytes = await client.download_bytes(doc["file_id"])
    points = parse_gpx(io.BytesIO(gpx_bytes))
    if not points:
        await client.send_message(chat_id, "No track points found in that GPX.")
        return

    # Persist the route as a JSON list of [lat, lon] pairs under static/.
    ROUTE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(
        ROUTE_JSON_PATH.write_text,
        json.dumps([[lat, lon] for lat, lon in points]),
        "utf-8",
    )
    log.info("wrote %d route points to %s", len(points), ROUTE_JSON_PATH)

    status_id = await client.send_message(chat_id, "Route received. Preparing tiles…")

    # Throttled status editor: coalesces to at most one edit per
    # MIN_EDIT_INTERVAL and skips edits that would not change the text.
    last_text = "Route received. Preparing tiles…"
    last_edit = time.monotonic()

    async def set_status(text: str) -> None:
        nonlocal last_text, last_edit
        if text == last_text:
            return
        wait = MIN_EDIT_INTERVAL - (time.monotonic() - last_edit)
        if wait > 0:
            await asyncio.sleep(wait)
        await client.edit_message(chat_id, status_id, text)
        last_text, last_edit = text, time.monotonic()

    # A tiny shared state the progress callback writes and the ticker reads.
    state = {"done": 0, "total": None}

    def on_progress(done: int, total: int) -> None:
        state["done"], state["total"] = done, total

    download = asyncio.create_task(
        download_tiles(
            points,
            zoom=TILE_ZOOM,
            map_type=TILE_MAP_TYPE,
            include_neighbors=TILE_INCLUDE_NEIGHBORS,
            progress=on_progress,
        )
    )

    # Refresh the status message about once a second until the download finishes.
    while not download.done():
        await asyncio.sleep(MIN_EDIT_INTERVAL)
        total = state["total"]
        text = (
            f"Downloading tiles… {state['done']}/{total}"
            if total is not None
            else "Preparing tiles…"
        )
        await set_status(text)

    try:
        paths = await download
    except Exception as exc:  # surface failures to the user instead of hanging
        log.exception("tile download failed")
        await set_status(f"Tile download failed: {exc}")
        return

    await set_status(f"Downloaded {len(paths)}/{len(paths)} tiles. Packaging…")
    await asyncio.to_thread(_zip_tiles, paths, TILES_ZIP_PATH)
    log.info("packaged %d tiles into %s", len(paths), TILES_ZIP_PATH)

    await set_status(
        f"Downloaded {len(paths)} tiles and packaged them into tiles.zip."
    )
    await client.send_message(chat_id, "Route successfully uploaded")


async def _handle_update(client: TelegramClient, update: dict) -> None:
    message = update.get("message") or update.get("channel_post")
    if not message:
        return
    chat_id = message["chat"]["id"]

    sender = message.get("from") or {}
    user_id = sender.get("id")
    log.info(
        "message from user id=%s username=%s", user_id, sender.get("username")
    )
    if ALLOWED_USER_ID is None or str(user_id) != ALLOWED_USER_ID:
        await client.send_message(
            chat_id,
            f"Not authorized. Your Telegram user id is {user_id}.\n"
            "Set TELEGRAM_ALLOWED_USER_ID to this value to enable access.",
        )
        return

    if "document" in message:
        await _handle_document(client, chat_id, message["document"])
    elif "text" in message:
        await client.send_message(chat_id, "Send me a .gpx route file to begin.")


async def run_bot() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    async with aiohttp.ClientSession() as session:
        client = TelegramClient(token, session)
        log.info("telegram bot polling for updates")
        offset = 0
        while True:
            try:
                updates = await client.get_updates(offset)
            except Exception:
                log.exception("getUpdates error; backing off")
                await asyncio.sleep(3)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    await _handle_update(client, update)
                except Exception:
                    log.exception("error handling update %s", update.get("update_id"))


if __name__ == "__main__":
    # .env is already loaded at import time (see top of module).
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run_bot())
