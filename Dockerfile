# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG INSTALL_PREPROCESSING=true
ARG INSTALL_POSE=true
ARG INSTALL_BACKGROUND_REMOVAL=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN set -eux; \
    apt-get -o Acquire::Retries=5 update; \
    for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
            libgl1 \
            libglib2.0-0 \
            libgomp1; then \
            break; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        sleep 5; \
    done; \
    rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    install_requirements() { \
        requirement_file="$1"; \
        for attempt in 1 2 3; do \
            if python -m pip install --prefer-binary \
                -r "$requirement_file"; then \
                return 0; \
            fi; \
            if [ "$attempt" -eq 3 ]; then return 1; fi; \
            sleep 5; \
        done; \
    }; \
    if [ "$INSTALL_PREPROCESSING" = "true" ]; then \
        requirements="requirements-preprocessing-core.txt"; \
    else \
        requirements="requirements.txt"; \
    fi; \
    install_requirements "$requirements"; \
    if [ "$INSTALL_PREPROCESSING" = "true" ] \
        && [ "$INSTALL_POSE" = "true" ]; then \
        install_requirements requirements-pose.txt; \
    fi; \
    if [ "$INSTALL_PREPROCESSING" = "true" ] \
        && [ "$INSTALL_BACKGROUND_REMOVAL" = "true" ]; then \
        install_requirements requirements-background-removal.txt; \
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

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
