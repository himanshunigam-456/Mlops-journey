# syntax=docker/dockerfile:1.7

# ──────────── builder stage ────────────
# Installs deps + builds wheel. Heavy (~2 GB) — discarded after copying.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Cache-friendly layer order: project metadata first, source last
COPY pyproject.toml /build/pyproject.toml
COPY src /build/src

# Install runtime deps + the package itself into /install (an isolated prefix)
RUN pip install --upgrade pip && \
    pip install --prefix=/install \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" \
      "mlflow>=2.16,<3.0" "scikit-learn>=1.5" "xgboost>=2.1" \
      "pandas>=2.2" "numpy>=2.0" "boto3>=1.34" && \
    pip install --prefix=/install /build

# ──────────── runtime stage ────────────
# Slim runtime — only what's needed to run uvicorn + the model.
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    PORT=8000

# Non-root user — security baseline. Containers running as root in K8s
# are flagged by every security scanner and most admission controllers.
RUN useradd -m -u 1001 -s /bin/bash app

WORKDIR /home/app

# Copy only the install tree from the builder layer. /install/lib goes to
# /usr/local/lib, /install/bin to /usr/local/bin — drop-in for the system Python.
COPY --from=builder /install /usr/local

USER app

EXPOSE 8000

# Container-level healthcheck — hits our /healthz endpoint.
# Use Python urllib instead of curl because the slim image doesn't ship curl.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" \
  || exit 1

CMD ["uvicorn", "credit_risk.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
