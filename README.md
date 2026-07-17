# techno-optimism-server

A Python `asyncio` server built on [aiohttp](https://docs.aiohttp.org/).

## Endpoints

| Method | Path        | Description                                        |
|--------|-------------|----------------------------------------------------|
| GET    | `/health`   | Liveness probe, returns `{"status": "ok"}`.        |
| WS     | `/v1/ask`   | Client connects and streams **binary** blobs.      |

### `/v1/ask` protocol

1. Client opens a WebSocket to `/v1/ask`.
2. Client sends one or more **binary** frames, each a self-contained blob.
3. For every binary frame the server processes the blob (see
   `techno_optimism_server/handlers.py::handle_blob`) and replies with a binary frame.
4. Text frames are rejected with `{"ok": false, "error": "expected_binary_frame"}`.

The default `handle_blob` echoes the payload — replace it with the real logic.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m techno_optimism_server.server
```

Configuration via environment variables:

| Var             | Default   | Meaning                            |
|-----------------|-----------|------------------------------------|
| `HOST`          | `0.0.0.0` | Bind address                       |
| `PORT`          | `8080`    | Bind port                          |
| `LOG_LEVEL`     | `INFO`    | Logging level                      |
| `MAX_BLOB_BYTES`| `16MiB`   | Max request/message size           |

## Try it

```bash
# in one terminal
python -m techno_optimism_server.server

# in another
python scripts/ask_client.py
```
