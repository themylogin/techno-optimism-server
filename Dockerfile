FROM python:3.13-slim

# ffmpeg is required for audio loudness leveling in transcription.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY techno_optimism_server ./techno_optimism_server

# The server reads HOST/PORT from the environment (see server.py). 0.0.0.0 so
# it is reachable from outside the container; PORT is overridable at runtime.
ENV HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["python", "-m", "techno_optimism_server.server"]
