"""Access-token authentication.

Every request must carry `X-Auth: {ACCESS_TOKEN}` (the token from .env) or it is
rejected with 401. The check is a middleware, so it guards *all* routes by
default — any endpoint added later is protected automatically, with no per-handler
opt-in. The only exemption is the liveness probe, listed in ``PUBLIC_PATHS``.
"""

from __future__ import annotations

import hmac
import logging
import os

from aiohttp import web

log = logging.getLogger("techno_optimism.auth")

# Paths reachable without the token. Keep this list tiny and explicit: anything
# not here requires X-Auth.
PUBLIC_PATHS = frozenset({"/health"})

AUTH_HEADER = "X-Auth"


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Reject any request lacking a valid ``X-Auth`` header.

    Fails closed: if ``ACCESS_TOKEN`` is unset the server rejects everything but
    the public paths, rather than silently running unauthenticated.
    """
    if request.path in PUBLIC_PATHS:
        return await handler(request)

    expected = os.environ.get("ACCESS_TOKEN")
    if not expected:
        log.error("ACCESS_TOKEN is not configured; rejecting %s", request.path)
        return web.json_response({"error": "unauthorized"}, status=401)

    provided = request.headers.get(AUTH_HEADER, "")
    # Constant-time comparison to avoid leaking the token via timing.
    if not hmac.compare_digest(provided, expected):
        return web.json_response({"error": "unauthorized"}, status=401)

    return await handler(request)
