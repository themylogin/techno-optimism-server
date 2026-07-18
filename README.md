# techno-optimism-server

A Python `asyncio` server built on [aiohttp](https://docs.aiohttp.org/).

## Endpoints

| Method | Path        | Description                                        |
|--------|-------------|----------------------------------------------------|
| GET    | `/health`   | Liveness probe, returns `{"status": "ok"}`.        |
| WS     | `/v1/ask`   | Client connects and streams **binary** blobs.      |

### `/v1/ask` protocol

0. *(optional)* To continue a prior conversation, the client's first frame is
   text `{"previous_response_id": "resp_..."}`; the question blob then follows.
1. Client sends one **binary** frame containing an audio file (e.g. mp3) — a
   spoken question.
2. Server acks: `{"msg": "uploaded"}`.
3. Server transcribes the audio (`gpt-4o-transcribe`) and decides whether the
   question references external context the user just heard/saw.
4. If it does, server sends `{"msg": "need_context"}` and the client sends a
   second **binary** frame with the surrounding-context audio, which the
   server transcribes.
5. Server answers with a web-search-enabled reasoning model, streaming the
   answer as `{"msg": "thinking", "text": "<chunk>"}` frames.
6. Server sends `{"msg": "done", "response_id": "resp_..."}` (pass this back as
   `previous_response_id` next turn), then the synthesized answer as one
   **binary** frame, then closes.

Errors: a malformed handshake → `{"ok": false, "error": "invalid_handshake"}`;
a non-binary question frame → `{"ok": false, "error": "expected_binary_frame"}`;
no context blob in time → `{"ok": false, "error": "context_not_received"}`;
any processing failure → `{"ok": false, "error": "processing_failed", ...}`.

## Docker

```bash
# .env must contain OPENAI_API_KEY (and optionally LOG_LEVEL, etc.)
HOST_PORT=52066 docker compose up --build
```

The server always listens on 8080 inside the container; it is published to
`127.0.0.1:$HOST_PORT` on the host (localhost only; default 8080).
`./interactions` is mounted so saved interactions land on the host. ffmpeg is
in the image.

## Tests

Two suites:

- `tests/` — hermetic unit tests (AI and storage mocked, no network/key).
  `pytest` runs these by default.
- `integration_tests/` — live tests that call the OpenAI API; need
  `OPENAI_API_KEY` (from `.env`) and network.

```bash
pip install -r requirements-dev.txt

# unit tests with branch coverage
python -m pytest tests/ --cov=techno_optimism_server --cov-branch --cov-report=term-missing

# integration tests (hits the API)
python -m pytest integration_tests/ -v
```

CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs the unit
tests on every push/PR, and the integration tests only when an `OPENAI_API_KEY`
repository secret is configured.

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
