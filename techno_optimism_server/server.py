"""Asyncio WebSocket server.

Exposes:
    GET  /health      -> liveness probe
    WS   /v1/ask      -> client connects and streams binary blobs
"""

from __future__ import annotations

import logging
import os

from aiohttp import web
from dotenv import load_dotenv

from techno_optimism_server.ai import AI
from techno_optimism_server.handlers import ask_ws, health

load_dotenv()  # load OPENAI_API_KEY, LOG_LEVEL, etc. from .env if present

log = logging.getLogger("techno_optimism.server")


def create_app() -> web.Application:
    """Build and configure the aiohttp application."""
    app = web.Application(
        # Allow reasonably large binary blobs over the WS connection.
        client_max_size=int(os.environ.get("MAX_BLOB_BYTES", 16 * 1024 * 1024)),
    )
    app["ai"] = AI()
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/v1/ask", ask_ws),
        ]
    )
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
