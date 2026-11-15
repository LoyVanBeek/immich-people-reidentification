# Multi-stage build for the Pi 5 (ARM64) — build with:
#   docker buildx build --platform linux/arm64 -t reid-batch .
# The image is pure Python: no compiled wheels, so the same Dockerfile builds
# natively on any architecture.
#
# Runtime needs no network beyond the Immich API.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm
RUN groupadd --system reid \
    && useradd --system --gid reid --create-home --shell /usr/sbin/nologin reid \
    && mkdir -p /data /etc/reid \
    && chown reid:reid /data /etc/reid
COPY --from=builder --chown=reid:reid /app/.venv /app/.venv
USER reid
ENV PATH="/app/.venv/bin:$PATH"
VOLUME /data
ENTRYPOINT ["reid-batch"]
CMD ["--config", "/etc/reid/reid.toml"]
