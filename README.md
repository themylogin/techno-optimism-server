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
| POST   | `/voice-note`                    | Upload a voice note (multipart: an mp3 file + a `timestamp` field). |
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

A separate, fire-and-forget pipeline for recorded voice notes.

**POST `/voice-note`** takes a `multipart/form-data` body with two fields: the
audio file (any field name) and a `timestamp` (Unix epoch seconds or an ISO-8601
string). The audio is written under the voice-notes directory at
`%Y/%m/%d/%H-%M-%S.<ext>` (derived from `timestamp`, with a unique suffix on
collision). The extension follows the upload's filename — e.g. Telegram voice
notes are `.ogg`, and anything without a recognised audio extension is stored as
`.mp3`. A row is inserted into the `voice_note` SQLite table, and the call
returns `201 {"id", "filename"}`. Validation errors: `missing_timestamp`,
`bad_timestamp`, `missing_file`, `empty_file` (all `400`).

```bash
curl -H "X-Auth: $ACCESS_TOKEN" \
     -F timestamp=$(date +%s) -F file=@note.mp3 \
     http://localhost:8080/voice-note
```

A **worker** process (`python -m techno_optimism_server.worker`) then drives each
note through two stages on a loop (sleeping `WORKER_SLEEP`, default 10s, between
iterations):

1. **Transcription** — untranscribed notes are transcribed the same way the
   assistant transcribes questions. A failure is recorded (`transcription_error`,
   `transcription_retries`, `transcription_last_retry`) and retried with
   exponential backoff — first retry after `TRANSCRIBE_BASE_BACKOFF` (10s),
   doubling each time — until `TRANSCRIBE_MAX_RETRIES` (10), after which the note
   is left permanently failed.
2. **Publication** — each note that has finished transcription (succeeded, or
   given up) and isn't in Vikunja yet is created as a task via `VIKUNJA_URL` /
   `VIKUNJA_API_TOKEN` / `VIKUNJA_PROJECT_ID`: the transcription becomes the task
   title (or `<transcription failed>`), and the original mp3 is attached. A
   failure just leaves the note to be retried on the next loop; `vikunja_id` is
   stored on success.

The schema lives in Alembic migrations (`techno_optimism_server/migrations/`),
applied automatically at startup by both the server and the worker.

## Docker

```bash
# .env must contain OPENAI_API_KEY (and optionally LOG_LEVEL, etc.)
HOST_PORT=52066 docker compose up --build
```

The server always listens on 8080 inside the container; it is published to
`127.0.0.1:$HOST_PORT` on the host (localhost only; default 8080).
`./interactions` is mounted so saved interactions land on the host. ffmpeg is
in the image. The `worker` service shares the `./voice-notes` volume (mp3s + the
SQLite database) with the server and needs `VIKUNJA_URL`, `VIKUNJA_API_TOKEN`,
and `VIKUNJA_PROJECT_ID` in `.env`.

Both the `server` and `telegram-bot` services mount `./tracks` (`TRACKS_DIR`):
every loaded map —
an uploaded GPX or a confirmed Google walking route — is archived there as
`%Y/%m/%d/{id}.json` (`{id}` is 16 random ASCII letters/digits), holding the
same `[[lat, lon], …]` JSON as `static/route.json`. The generated
`static/tiles.zip` carries an extra `id` member whose contents are that track's
`%Y/%m/%d/{id}` reference, so a downloaded tile pack can be traced back to the
route it was built for. `POST /track-data` appends to `tracks/log.json` in the
same volume, always leaving the file newline-terminated so each upload lands on
its own line.

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
| `VOICE_NOTES_DIR`  | `voice-notes` | Directory for uploaded voice-note mp3s      |
| `DB_PATH`          | `voice-notes/db.sqlite3` | SQLite file for the `voice_note` table |
| `WORKER_SLEEP`     | `10`      | Worker loop pause, seconds (worker only)        |
| `TRANSCRIBE_BASE_BACKOFF` | `10` | First-retry backoff, seconds; doubles (worker) |
| `TRANSCRIBE_MAX_RETRIES` | `10` | Transcription attempts before giving up (worker) |
| `VIKUNJA_URL` / `VIKUNJA_API_TOKEN` / `VIKUNJA_PROJECT_ID` | — | Vikunja target (worker only) |

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
