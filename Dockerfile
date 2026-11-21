# Stage 1: Dependency Builder
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
RUN python -m venv /opt/venv

COPY requirements.txt ./
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Shared Runtime Base (OSMesa + Non-root user)
FROM python:3.12-slim-bookworm AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa

# Stage 3: Development & Test Runner
FROM runtime-base AS dev

WORKDIR /app
COPY --chown=1000:1000 . .

USER 1000:1000
EXPOSE 8080
CMD ["python", "run_web_server.py"]

# Stage 4: Production Runtime
FROM runtime-base AS production

WORKDIR /app
COPY --chown=1000:1000 assets/ ./assets/
COPY --chown=1000:1000 scene_builder.py pick_and_place_verify.py messages.py sim_plugin.py viewer_plugin.py run_web_server.py scene.xml ./

USER 1000:1000
EXPOSE 8080
ENTRYPOINT ["python", "run_web_server.py"]
