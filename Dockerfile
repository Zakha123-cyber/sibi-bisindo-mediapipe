FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib MODEL_BUNDLE_DIR=/models OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libportaudio2 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-features.txt requirements-api.txt requirements-service.txt ./
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-service.txt
RUN useradd --uid 10001 --create-home app
COPY src/ ./src/
COPY configs/ ./configs/
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=3)"
CMD ["python","-m","uvicorn","sibi_bisindo.service.app:create_app","--factory","--host","0.0.0.0","--port","8000","--workers","1","--ws-max-size","2000000","--ws-max-queue","1","--limit-concurrency","64","--no-access-log"]
