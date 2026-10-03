# ==============================================================================
# Knowledge Fabric - Multi-Stage Production Dockerfile
# ==============================================================================

# --- Stage 1: Build & Packaging ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir .

# --- Stage 2: Minimal Hardened Runtime ---
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Knowledge Fabric" \
      org.opencontainers.image.description="MCP-native evidence retrieval and knowledge governance platform" \
      org.opencontainers.image.vendor="Knowledge Fabric Contributors" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install minimal runtime dependencies (libpq for postgres, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create secure non-root system user and group (UID/GID 10001)
RUN groupadd -g 10001 fabric && \
    useradd -u 10001 -g fabric -m -d /home/fabric -s /bin/bash fabric

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=fabric:fabric /opt/venv /opt/venv

# Copy configuration and runtime assets
COPY --chown=fabric:fabric config ./config
COPY --chown=fabric:fabric db/schema ./db/schema
COPY --chown=fabric:fabric src ./src
COPY --chown=fabric:fabric scripts ./scripts

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KF_UI_HOST="0.0.0.0" \
    KF_UI_PORT=8080

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

ENTRYPOINT ["knowledge-fabric-ui"]
