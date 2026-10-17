FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    HF_HOME=/data/hf \
    APP_VECTORS_DIR=/data/vectors \
    APP_FRONTEND_DIR=/app/frontend/dist
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
WORKDIR /app
COPY pyproject.toml ./
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; print('\n'.join(p['dependencies'] + p['optional-dependencies']['embed']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt
COPY README.md ./
COPY arxiv_rag ./arxiv_rag
RUN pip install --no-deps .
COPY scripts ./scripts
COPY --from=frontend /build/dist ./frontend/dist
RUN useradd --uid 1000 --create-home app && mkdir -p /data && chown app:app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
CMD ["uvicorn", "arxiv_rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
