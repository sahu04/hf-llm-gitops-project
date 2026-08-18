# --- Build stage: install dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install CPU-only torch first (much smaller than the default GPU build)
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --target=/build/deps \
    -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# --- Final stage: minimal runtime image ---
FROM python:3.11-slim

# Create a non-root user (security best practice — never run containers as root)
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY main.py .

# Cache directory for Hugging Face model downloads, owned by the non-root user
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app

ENV HF_HOME=/app/.cache \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# Basic container-level healthcheck (Kubernetes probes are the primary mechanism — see k8s/deployment.yaml)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
