# techno-optimism-server

A Python `asyncio` server built on [aiohttp](https://docs.aiohttp.org/).

## Endpoints

A spoken question is handled as a background **interaction** the client creates,
then polls. Every request returns immediately, so any call can be safely retried
over an unreliable connection.

| Method | Path                             | Description                                       |
|--------|----------------------------------|---------------------------------------------------|
| GET    | `/health`                        | Liveness probe, returns `{"status": "ok"}`.       |
| POST   | `/v1/interactions`               | Upload question audio; starts a job.              |
| GET    | `/v1/interactions/{id}`          | Poll the job's status snapshot.                   |
| PUT    | `/v1/interactions/{id}/context`  | Upload the follow-up context audio.               |
| GET    | `/v1/interactions/{id}/answer.mp3` | Download the answer audio (supports `Range`).   |
| POST   | `/location`                      | Set the live walk origin `{latitude, longitude}` (held in RAM, expires after `LOCATION_TTL`, default 300s). |
| GET    | `/location`                      | The live location `{latitude, longitude}`, or `null` once expired. |
| GET/HEAD | `/static/{file}`               | Serve static assets (`route.json`, `tiles.zip`). |
| POST   | `/voice-note`                    | Forward a voice note to the AI assistant (multipart: an audio file + a `timestamp` field). |
| POST   | `/track-data`                    | Append an uploaded file's contents to `tracks/log.json` (multipart: any single file field). |

Static responses (both `GET` and `HEAD`) carry an `X-SHA1` header with the SHA-1
of the file's bytes, so a client can `HEAD` a file and skip the download when its
digest is unchanged:

```bash
curl -sI -H "X-Auth: $ACCESS_TOKEN" http://localhost:8080/static/tiles.zip | grep -i x-sha1
```

### Authentication

Every endpoint **except `/health`** requires an `X-Auth` header whose value equals
`ACCESS_TOKEN` from the environment (`.env`); missing or wrong tokens get `401`.
The check is a middleware, so it guards all routes by default — any endpoint
added later is protected automatically. If `ACCESS_TOKEN` is unset the server
fails closed and rejects every protected request.

```bash
curl -H "X-Auth: $ACCESS_TOKEN" http://localhost:8080/v1/interactions/$id
```

### Flow

1. **POST `/v1/interactions`** with the raw audio file (e.g. mp3) as the request
   body — a spoken question. Optionally append `?previous_response_id=resp_...`
   to continue a prior conversation. Returns `201` with the initial snapshot,
   including the interaction `id`.
2. **Poll GET `/v1/interactions/{id}`**. The snapshot's `status` moves through
   `transcribing → thinking → synthesizing → done` (or `error`):

   ```json
   {
     "id": "2026-07-20-14-30-05-ab12cd34",
     "status": "done",
     "question": "How many moons does Mars have?",
     "thinking": "…latest tail of the model's reasoning",
     "response_id": "resp_...",
     "answer_text": "Mars has two moons: Phobos and Deimos.",
     "answer_audio_url": "/v1/interactions/{id}/answer.mp3"
   }
   ```

   The server transcribes the audio (`gpt-4o-transcribe`) and decides whether the
   question references external context the user just heard/saw. If it does, the
   status becomes `need_context` and the job waits.
3. **On `need_context`, PUT `/v1/interactions/{id}/context`** with the
   surrounding-context audio as the body. This unblocks the job; the upload is
   idempotent, so it can be retried. If no context arrives within
   `CONTEXT_TIMEOUT` the job ends in `error`.
4. **When `status` is `done`, GET the `answer_audio_url`** to download the
   synthesized answer. It is served off disk with `Range` support, so a dropped
   download resumes rather than restarting. Pass `response_id` back as
   `previous_response_id` on the next turn to chain the conversation.

Job state lives in RAM (single instance; interactions are short-lived), so a
`GET` for an unknown or pre-restart id returns `404 unknown_interaction`. Other
errors: an empty upload body → `400 empty_body`; a processing failure → `error`
status with the detail in `error.detail`.

## Voice notes

**POST `/voice-note`** forwards an upload to the AI assistant. It takes a
`multipart/form-data` body with the audio file (any field name) and a `timestamp`
(Unix epoch seconds or an ISO-8601 string). The audio is loudness-leveled in RAM
with ffmpeg (`LEVEL_FILTER`, the same pass transcription uses) and re-encoded to a
VBR V2 MP3 (LAME's transparent "standard" preset, original sample rate and
channels) sent as `voice-note.mp3`; if ffmpeg is missing or can't decode it,
the original is forwarded instead. The upload is sent as a `file`
field and an ISO-8601 `timestamp` with a UTC offset, sent to
`$AI_ASSISTANT_URL/v1/upload/voice-note` with `AI_ASSISTANT_TOKEN` as its
`X-API-Key`. Any success from the assistant returns `201 {"filename"}`; its errors
are returned as-is. Nothing is stored here. Validation errors: `missing_timestamp`, `bad_timestamp`,
`missing_file`, `empty_file` (all `400`). If the assistant can't be reached the
call returns `502 assistant_unreachable`.

```bash
curl -H "X-Auth: $ACCESS_TOKEN" \
     -F timestamp=2026-10-30T15:11:58+01:00 -F file=@note.mp3 \
     http://localhost:8080/voice-note
```

## Docker

```bash
# .env must contain OPENAI_API_KEY (and optionally LOG_LEVEL, etc.)
HOST_PORT=52066 docker compose up --build
```

The server always listens on 8080 inside the container; it is published to
`127.0.0.1:$HOST_PORT` on the host (localhost only; default 8080).
`./interactions` is mounted so saved interactions land on the host. ffmpeg is
in the image. Voice-note uploads need `AI_ASSISTANT_URL` and `AI_ASSISTANT_TOKEN`
in `.env`.

Both the `server` and `telegram-bot` services mount `./tracks` (`TRACKS_DIR`):
every loaded map — an uploaded GPX or a confirmed Google walking route — is
archived there as `%Y/%m/%d/{id}.json` (`{id}` is 16 random ASCII
letters/digits), holding the same JSON as `static/route.json`:

```json
{"id": "2026/07/28/a1B2c3D4e5F6g7H8", "waypoints": [[52.1, 4.3], [52.2, 4.4]]}
```

`route.json` is overwritten by the next upload, so its `id` says which archived
track the currently served route (and `tiles.zip`) was built from.
`POST /track-data` appends to `tracks/log.json` in the same volume, always
leaving the file newline-terminated so each upload lands on its own line.

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

| Var                | Default   | Meaning                                         |
|--------------------|-----------|-------------------------------------------------|
| `HOST`             | `0.0.0.0` | Bind address                                    |
| `PORT`             | `8080`    | Bind port                                       |
| `LOG_LEVEL`        | `INFO`    | Logging level                                   |
| `MAX_BLOB_BYTES`   | `16MiB`   | Max request body size                           |
| `CONTEXT_TIMEOUT`  | `60`      | Seconds to wait for the context upload          |
| `THINKING_INTERVAL`| `1.0`     | Cadence for archiving the thinking tail to disk |
| `THINKING_TAIL`    | `100`     | Max chars of reasoning shown in `thinking`      |
| `LEVEL_FILTER`     | `dynaudnorm=framelen=300:maxgain=20` | ffmpeg loudness filter for transcription and voice notes (`""` disables) |
| `AI_ASSISTANT_URL` / `AI_ASSISTANT_TOKEN` | — | Voice-note upload target and its API key |

## Try it

```bash
# in one terminal
python -m techno_optimism_server.server

# in another — ask a question and poll for the answer ($ACCESS_TOKEN from .env)
id=$(curl -s -H "X-Auth: $ACCESS_TOKEN" --data-binary @question.mp3 \
       http://localhost:8080/v1/interactions | jq -r .id)

curl -s -H "X-Auth: $ACCESS_TOKEN" http://localhost:8080/v1/interactions/$id | jq   # poll until "done"

curl -s -H "X-Auth: $ACCESS_TOKEN" http://localhost:8080/v1/interactions/$id/answer.mp3 -o answer.mp3
```
