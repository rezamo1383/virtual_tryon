# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG INSTALL_PREPROCESSING=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-preprocessing.txt ./
RUN if [ "$INSTALL_PREPROCESSING" = "true" ]; then \
        python -m pip install -r requirements-preprocessing.txt; \
    else \
        python -m pip install -r requirements.txt; \
    fi

COPY app ./app
COPY api.py cli.py main.py config.py ./
COPY config/tenants.example.json ./config/tenants.example.json

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p inputs outputs temp logs models config \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
