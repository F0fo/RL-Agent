# CPU-only image — Battleship is a tiny model (~65k params); GPU isn't worth the
# image bloat. Final image is ~700MB (mostly CPU torch).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install CPU torch from the dedicated index (much smaller than the default wheel).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Web-only deps (avoid pulling matplotlib/imageio/etc. that are dev-only).
COPY requirements.web.txt /tmp/requirements.web.txt
RUN pip install --no-cache-dir -r /tmp/requirements.web.txt

# Application code + the v2 sonar checkpoints
COPY config.py ./
COPY battleship/ ./battleship/
COPY web/ ./web/
COPY checkpoints_sonar_v2/ ./checkpoints_sonar_v2/

# Railway sets $PORT at runtime; default to 8080 for local docker run.
ENV PORT=8080
EXPOSE 8080

# Shell form so $PORT expands at runtime. `python -m uvicorn` (instead of bare
# `uvicorn`) puts the cwd on sys.path so the `web.server` import resolves.
CMD python -m uvicorn web.server:app --host 0.0.0.0 --port ${PORT}
