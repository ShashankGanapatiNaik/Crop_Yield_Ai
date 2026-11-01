# Build context is the repository root (the API imports the shared `ml` package).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY backend backend
COPY ml/__init__.py ml/__init__.py
COPY ml/src ml/src
COPY ml/artifacts ml/artifacts
COPY ml/reports/eval ml/reports/eval
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=5 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
