# Stage 1: Build wheel artifact
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip build wheel

# Copy source and build files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build standalone wheel
RUN python -m build --wheel --outdir /wheels

# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Configure non-buffering Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    MICROLLM_CACHE_DIR=/app/.cache

# Create unprivileged system group and user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -M appuser

# Install package from wheel built in stage 1
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Ensure cache directory exists and is owned by unprivileged user
RUN mkdir -p /app/.cache && \
    chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

# Expose default HTTP server port
EXPOSE 8000

# Container healthcheck via PSL urllib against /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

# Default entrypoint and arguments for MicroLLM System-1 server
ENTRYPOINT ["microllm", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
