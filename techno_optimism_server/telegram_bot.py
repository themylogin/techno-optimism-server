"""Telegram bot that turns an uploaded GPX route into cached map tiles.

Send the bot a ``.gpx`` document and it:

    1. parses it and writes the track as ``static/route.json`` (a JSON list of
       ``[lat, lon]`` pairs),
    2. downloads every map tile the route crosses (plus neighbors), editing a
       single reply about once a second with live ``done/total`` progress,
    3. packages this route's tiles into ``static/tiles.zip``,
    4. finishes with "Route successfully uploaded".

The ``static/`` directory is served by the REST server under ``/static``.

Alternatively, share two locations with the bot: the first is the origin (the
bot replies "Now send the destination location."), the second the destination.
The bot then computes a walking route with the Google Maps Routes API, reports
its distance and duration with a Yes/No confirmation, and — on Yes — feeds that
route through the same tile pipeline as an uploaded GPX. ``/reset`` clears this
per-chat state at any point.

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

from techno_optimism_server.routes import (
    compute_walking_route,
    format_duration,
)
from techno_optimism_server.mvt_render import (
    download_mapbox_tiles,
    parse_tile_path,
    render_tiles,
)
from techno_optimism_server.preview import (
    format_distance,
    render_route_preview,
    route_length_m,
)
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

# Per-chat state for the two-location walking-route flow. A chat is in exactly
# one of these states at a time:
#   • absent            — idle; a location starts a new route as the origin.
#   • {"origin": ...}   — origin received; the next location is the destination.
#   • {"points": ...}   — route computed; awaiting the Yes/No confirmation.
# Cleared by /reset, by answering the confirmation, or on any error.
_route_state: dict[int, dict] = {}

# Inline-keyboard callback payloads for the confirmation prompt.
CONFIRM_YES = "route_confirm_yes"
CONFIRM_NO = "route_confirm_no"

# The bot's command menu (the button beside the message input), registered on
# startup via setMyCommands.
BOT_COMMANDS = [
    {"command": "reset", "description": "Reset the current route state"},
]


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

    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict | None = None
    ) -> int:
        params = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        result = await self._call("sendMessage", **params)
        return result["message_id"]

    async def send_photo(
        self,
        chat_id: int,
        photo: bytes,
        caption: str | None = None,
        reply_markup: dict | None = None,
    ) -> int:
        """Upload a JPEG as a photo message (multipart, so no URL needed)."""
        form = aiohttp.FormData()
        form.add_field("chat_id", str(chat_id))
        if caption is not None:
            form.add_field("caption", caption)
        if reply_markup is not None:
            form.add_field("reply_markup", json.dumps(reply_markup))
        form.add_field(
            "photo", photo, filename="preview.jpg", content_type="image/jpeg"
        )
        async with self._session.post(f"{self._api}/sendPhoto", data=form) as resp:
            data = await resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram sendPhoto failed: {data}")
        return data["result"]["message_id"]

    async def answer_callback_query(self, callback_query_id: str) -> None:
        await self._call("answerCallbackQuery", callback_query_id=callback_query_id)

    async def set_my_commands(self, commands: list[dict]) -> None:
        """Register the bot's command menu (the button beside the input box)."""
        await self._call("setMyCommands", commands=commands)

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


def _zip_tiles(entries: list[tuple[Path, str]], dest: Path) -> None:
    """Package exactly this route's tiles into ``dest`` (a zip file).

    ``entries`` is a list of ``(source_path, arcname)`` — only the tiles the
    route touches, not the whole cache. Rendered tiles live under
    ``tiles/rendered/<gen>/...`` on disk but are archived under the flat
    ``{z}/{x}/{y}.jpg`` layout the mobile app reads. Blocking, so call via
    ``asyncio.to_thread``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname=arcname)


async def _send_route_preview(
    client: TelegramClient,
    chat_id: int,
    points: list[tuple[float, float]],
    caption: str | None = None,
    reply_markup: dict | None = None,
) -> None:
    """Reply with a satellite preview of the route (white line on imagery).

    If the render fails we don't want to strand the flow, so fall back to a
    plain text message carrying the same caption/keyboard.
    """
    try:
        image = await render_route_preview(points)
    except Exception:  # noqa: BLE001 - preview is best-effort
        log.exception("route preview render failed")
        if reply_markup is not None or caption is not None:
            await client.send_message(
                chat_id, caption or "Route ready.", reply_markup=reply_markup
            )
        return
    await client.send_photo(chat_id, image, caption=caption, reply_markup=reply_markup)


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

    # Show the route on satellite first, then go straight to downloading tiles.
    await _send_route_preview(
        client, chat_id, points, caption=f"Route: {format_distance(route_length_m(points))}"
    )
    await _process_route(client, chat_id, points)


async def _process_route(
    client: TelegramClient, chat_id: int, points: list[tuple[float, float]]
) -> None:
    """Persist a route's points, download its tiles, and report progress.

    Shared by the GPX-upload path and the confirmed walking-route path — both
    end up here with a list of ``(lat, lon)`` points.
    """
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

    # A tiny shared state each progress callback writes and the ticker reads.
    state = {"done": 0, "total": None}

    def on_progress(done: int, total: int) -> None:
        state["done"], state["total"] = done, total

    async def run_phase(label: str, coro):
        """Run ``coro`` while editing the status ~once/sec with its progress."""
        state["done"], state["total"] = 0, None
        task = asyncio.create_task(coro)
        while not task.done():
            await asyncio.sleep(MIN_EDIT_INTERVAL)
            total = state["total"]
            await set_status(
                f"{label}… {state['done']}/{total}" if total else f"{label}…"
            )
        return await task

    try:
        # 1. Google raster tiles for the route (plus neighbors).
        paths = await run_phase(
            "Downloading satellite tiles",
            download_tiles(
                points,
                zoom=TILE_ZOOM,
                map_type=TILE_MAP_TYPE,
                include_neighbors=TILE_INCLUDE_NEIGHBORS,
                progress=on_progress,
            ),
        )
        tiles = [(x, y) for _z, x, y in map(parse_tile_path, paths)]

        # 2. The matching Mapbox vector tiles for the same coordinates.
        await run_phase(
            "Downloading vector tiles",
            download_mapbox_tiles(tiles, TILE_ZOOM, progress=on_progress),
        )

        # 3. Render each raster tile with the vector style overlaid on top.
        rendered = await run_phase(
            "Rendering tiles",
            render_tiles(
                tiles, TILE_ZOOM, map_type=TILE_MAP_TYPE, progress=on_progress
            ),
        )
    except Exception as exc:  # surface failures to the user instead of hanging
        log.exception("tile pipeline failed")
        await set_status(f"Tile processing failed: {exc}")
        return

    # The mobile app reads tiles from the zip by a flat {z}/{x}/{y}.jpg path.
    entries = [
        (path, f"{z}/{x}/{y}.jpg")
        for path, (z, x, y) in zip(rendered, map(parse_tile_path, rendered))
    ]
    await set_status(f"Rendered {len(rendered)} tiles. Packaging…")
    await asyncio.to_thread(_zip_tiles, entries, TILES_ZIP_PATH)
    log.info("packaged %d rendered tiles into %s", len(rendered), TILES_ZIP_PATH)

    await set_status(
        f"Downloaded and rendered {len(rendered)} tiles into tiles.zip."
    )
    await client.send_message(chat_id, "Route successfully uploaded")


async def _handle_location(client: TelegramClient, chat_id: int, loc: dict) -> None:
    """Drive the two-location walking-route flow one location at a time.

    First location becomes the origin; the second triggers a Routes API lookup
    whose distance/duration is reported with a Yes/No confirmation.
    """
    point = (loc["latitude"], loc["longitude"])
    state = _route_state.get(chat_id)

    # No origin yet (or we were mid-confirmation): treat this as a fresh origin.
    if not state or "origin" not in state:
        _route_state[chat_id] = {"origin": point}
        await client.send_message(chat_id, "Now send the destination location.")
        return

    # We have an origin — this location is the destination. Compute the route.
    origin = state["origin"]
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        _route_state.pop(chat_id, None)
        await client.send_message(chat_id, "GOOGLE_API_KEY is not set.")
        return

    await client.send_message(chat_id, "Computing walking route…")
    try:
        route = await compute_walking_route(client._session, api_key, origin, point)
    except Exception as exc:
        log.exception("walking-route computation failed")
        _route_state.pop(chat_id, None)
        await client.send_message(chat_id, f"Could not compute a route: {exc}")
        return

    # Hold the geometry pending the user's confirmation. Show the route on a
    # satellite preview first, then ask whether to use it.
    _route_state[chat_id] = {"points": route.points}
    km = route.distance_meters / 1000
    await _send_route_preview(
        client,
        chat_id,
        route.points,
        caption=(
            f"Walking route: {km:.2f} km, "
            f"{format_duration(route.duration_seconds)}.\nUse this route?"
        ),
        reply_markup={
            "inline_keyboard": [
                [
                    {"text": "Yes", "callback_data": CONFIRM_YES},
                    {"text": "No", "callback_data": CONFIRM_NO},
                ]
            ]
        },
    )


async def _handle_callback(client: TelegramClient, callback: dict) -> None:
    """Handle the Yes/No answer to a walking-route confirmation."""
    await client.answer_callback_query(callback["id"])
    message = callback.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    if chat_id is None:
        return

    data = callback.get("data")
    state = _route_state.get(chat_id)

    if data == CONFIRM_NO:
        _route_state.pop(chat_id, None)
        await client.send_message(chat_id, "Route discarded. Send a location to start over.")
        return

    if data == CONFIRM_YES:
        if not state or "points" not in state:
            await client.send_message(chat_id, "No pending route. Send a location to start.")
            return
        points = state["points"]
        _route_state.pop(chat_id, None)
        # Confirmed: treat exactly like an uploaded GPX route.
        await _process_route(client, chat_id, points)


async def _handle_update(client: TelegramClient, update: dict) -> None:
    if "callback_query" in update:
        callback = update["callback_query"]
        sender = callback.get("from") or {}
        if ALLOWED_USER_ID is None or str(sender.get("id")) != ALLOWED_USER_ID:
            await client.answer_callback_query(callback["id"])
            return
        await _handle_callback(client, callback)
        return

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

    text = message.get("text", "")
    command = text.strip().split()[0].split("@")[0] if text.strip() else ""
    if command == "/reset":
        _route_state.pop(chat_id, None)
        await client.send_message(chat_id, "State reset.")
    elif "document" in message:
        await _handle_document(client, chat_id, message["document"])
    elif "location" in message:
        await _handle_location(client, chat_id, message["location"])
    elif text:
        await client.send_message(
            chat_id,
            "Send me a .gpx route file, or share a location to build a walking "
            "route.",
        )


async def run_bot() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    async with aiohttp.ClientSession() as session:
        client = TelegramClient(token, session)
        try:
            await client.set_my_commands(BOT_COMMANDS)
        except Exception:
            # A missing menu is not fatal — log and keep serving updates.
            log.exception("failed to register bot commands")
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
