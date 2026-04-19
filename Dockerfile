# syntax=docker/dockerfile:1.7

# --- build stage: resolve deps into a venv using uv ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# --- runtime stage: slim image with just the venv ---
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

RUN useradd --system --no-create-home --uid 1001 app
USER app

EXPOSE 8787

CMD ["uvicorn", "proxy.main:app", "--host", "0.0.0.0", "--port", "8787"]
