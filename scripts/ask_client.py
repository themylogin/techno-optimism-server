"""Test client for the /v1/ask transcription endpoint.

Uploads an audio file as a single binary blob and prints the JSON the
server returns.

Usage:
    python scripts/ask_client.py                 # uploads ./file.blob
    python scripts/ask_client.py path/to/a.mp3   # uploads the given file
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import aiohttp

URL = os.environ.get("ASK_URL", "ws://127.0.0.1:8080/v1/ask")


async def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "file.blob"
    with open(path, "rb") as f:
        payload = f.read()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL) as ws:
            await ws.send_bytes(payload)
            print(f"uploaded {len(payload)} bytes from {path}", file=sys.stderr)

            msg = await ws.receive()
            if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                data = msg.data if isinstance(msg.data, str) else msg.data.decode()
                try:
                    print(json.dumps(json.loads(data), indent=2, ensure_ascii=False))
                except json.JSONDecodeError:
                    print(data)
            else:
                print(f"unexpected reply: {msg.type} {msg.data!r}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
