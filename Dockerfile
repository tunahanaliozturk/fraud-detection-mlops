# syntax=docker/dockerfile:1
# The scoring service only. Train and export on the host first (see the README), then mount the exported model and
# the transaction history read-only: the image holds code, never data or a model.
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv

FROM python:3.14-slim@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2 AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.14-slim@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2
# LightGBM's Linux wheel links against the system OpenMP runtime, which the slim image does not ship.
# Nothing is installed at run time, so pip leaves the image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && apt-get upgrade --yes --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall --yes --quiet pip \
    && useradd --system --uid 10001 --home-dir /app fraud
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba
USER fraud
EXPOSE 8000
# The slim image has no curl or wget, so the health check is a Python one-liner.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]
ENTRYPOINT ["fraud"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model-dir", "/app/models/champion", "--history", "/app/data/transactions.parquet"]
