"""Asyncio REST server.

Exposes:
    GET  /health                          -> liveness probe
    POST /v1/interactions                 -> upload question audio, start a job
    GET  /v1/interactions/{id}            -> poll the job's status snapshot
    PUT  /v1/interactions/{id}/context    -> upload the follow-up context audio
    GET  /v1/interactions/{id}/answer.mp3 -> download the answer audio (Range)
    GET  /static/{file}                   -> serve static assets (route.json,
                                             tiles.zip) from the static volume

Every endpoint except /health requires an `X-Auth: {ACCESS_TOKEN}` header
(see techno_optimism_server.auth). The check is a middleware, so all routes —
including any added later — are guarded by default.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

from techno_optimism_server.ai import AI
from techno_optimism_server.auth import auth_middleware
from techno_optimism_server.handlers import (
    create_interaction,
    get_answer_audio,
    get_interaction,
    health,
    upload_context,
)
from techno_optimism_server.static_files import make_static_handler

load_dotenv()  # load OPENAI_API_KEY, LOG_LEVEL, etc. from .env if present

log = logging.getLogger("techno_optimism.server")


def create_app() -> web.Application:
    """Build and configure the aiohttp application."""
    app = web.Application(
        # Allow reasonably large audio blobs in a request body.
        client_max_size=int(os.environ.get("MAX_BLOB_BYTES", 16 * 1024 * 1024)),
        # Guards every route (except /health) behind the X-Auth token. Applies
        # to any route added below, so new endpoints are protected by default.
        middlewares=[auth_middleware],
    )
    app["ai"] = AI()
    app["jobs"] = {}  # id -> Job, in-RAM registry of interactions
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/v1/interactions", create_interaction),
            web.get("/v1/interactions/{id}", get_interaction),
            web.put("/v1/interactions/{id}/context", upload_context),
            web.get("/v1/interactions/{id}/answer.mp3", get_answer_audio),
        ]
    )

    # Serve the static volume (route.json, tiles.zip) under /static. The bot
    # writes here; ensure the directory exists so add_static doesn't error.
    static_dir = Path(os.environ.get("STATIC_DIR", "static"))
    static_dir.mkdir(parents=True, exist_ok=True)
    # Custom static handler: serves GET/HEAD and adds an X-SHA1 content digest so
    # clients can cheaply check (via HEAD) whether their copy is still current.
    static_handler = make_static_handler(static_dir)
    # add_get registers HEAD too (allow_head defaults to True).
    app.router.add_get("/static/{filename:.+}", static_handler)
    return app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    log.info("Starting server on %s:%s", host, port)
    web.run_app(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
