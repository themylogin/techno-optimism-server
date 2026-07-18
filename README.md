# techno-optimism-server

A Python `asyncio` server built on [aiohttp](https://docs.aiohttp.org/).

## Endpoints

| Method | Path        | Description                                        |
|--------|-------------|----------------------------------------------------|
| GET    | `/health`   | Liveness probe, returns `{"status": "ok"}`.        |
| WS     | `/v1/ask`   | Client connects and streams **binary** blobs.      |

### `/v1/ask` protocol

1. Client opens a WebSocket to `/v1/ask` and sends one **binary** frame
   containing an audio file (e.g. mp3) — a spoken question.
2. Server immediately acks: `{"msg": "uploaded"}`.
3. Server transcribes the audio (`gpt-4o-transcribe`).
4. Server asks a chat model whether the question references external context
   the user just heard/saw, then sends the routing decision:
   - `{"msg": "need_context"}` — the question refers to outside context
     (e.g. *"is it true what she's saying about polar bears?"*).
   - `{"msg": "thinking", "text": "Thinking..."}` — self-contained question
     (e.g. *"what is a polar bear"*).
5. Server closes the connection.

A non-binary first frame is rejected with
`{"ok": false, "error": "expected_binary_frame"}`; processing errors return
`{"ok": false, "error": "processing_failed", ...}`.

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
