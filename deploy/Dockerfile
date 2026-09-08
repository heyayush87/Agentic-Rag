# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build. The builder compiles wheels; the runtime image carries
# only what is needed to serve, which keeps the attack surface and the image
# small.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install ".[groq,local,search,api,ui]"


# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production \
    LOG_FORMAT=json \
    CHROMA_DIR=/var/lib/retailiq/chroma

# Never run as root.
RUN useradd --create-home --uid 1000 retailiq

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=retailiq:retailiq src ./src
COPY --chown=retailiq:retailiq data ./data

RUN mkdir -p /var/lib/retailiq/chroma && chown -R retailiq:retailiq /var/lib/retailiq

USER retailiq
EXPOSE 8000

# Reports "degraded" until the index is built, so an orchestrator can hold
# traffic back rather than routing users into errors.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "retailiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
