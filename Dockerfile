# Production image. uv installs from the lockfile; gunicorn (gthread, 1 worker)
# serves the Fast Dash app + its Flask-SocketIO chat stream. The analyst spawns
# short-lived subprocesses (the code sandbox), which gthread handles fine.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra prod

COPY analyst ./analyst
COPY app.py README.md ./
RUN uv sync --frozen --no-dev --extra prod

EXPOSE 8080

CMD ["gunicorn", "app:server", \
     "--worker-class", "gthread", "--workers", "1", "--threads", "8", \
     "--bind", "0.0.0.0:8080", "--timeout", "120", "--access-logfile", "-"]
