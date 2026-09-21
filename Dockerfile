FROM python:3.11-slim

WORKDIR /app

# Kept minimal on purpose — add build-essential etc. only if a later
# dependency (e.g. a specific embedding library) needs compiling.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Cloud Run injects $PORT and expects the container to bind to it; local
# `docker run` without -e PORT falls back to 8000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
