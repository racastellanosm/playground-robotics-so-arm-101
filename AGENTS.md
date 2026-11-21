# AGENTS.md — AI Agent Guidelines

This repository strictly follows the **Zero Host Dependencies** philosophy.

## Golden Rules for Agents

1. **Do not assume or use host runtimes**:
   - Never run `python`, `pip`, `uv`, or `pytest` directly on the host machine.
   - All code execution, testing, rendering, and web server operations **must** go through `Makefile` targets (which orchestrate Docker and Docker Compose).
2. **Orchestration via `Makefile`**:
   - Build images: `make build`
   - Run all simulation tests and base verification: `make tests` (supports `SEED=42`, `TRIALS=5`, `JSON_OUT=tests/sim_suite_report.json`)
   - Run base verification and renders only: `make verify`
   - Start MuJoCo Studio WebViewer: `make start` (detached) or `make run` (foreground)
   - Inspect logs: `make logs`
   - Stop services: `make stop`
   - Full cleanup: `make clean`
3. **Docker Topology**:
   - Multi-stage `Dockerfile` with `MUJOCO_GL=osmesa` for headless Linux offscreen rendering and non-root execution (`1000:1000`).
   - `docker-compose.yml` defines the `sim` service (`so-arm-101:prod`) for production.
   - `docker-compose.dev.yml` defines `so-arm-101:dev` with `entrypoint: []`, `./:/app` volume mount, and `STUDIO_HOT_RELOAD=all` for development and ephemeral test runs (`docker compose ... run --rm --no-deps sim ...`).
   - **Important**: Keep `so-arm-101:dev` and `so-arm-101:prod` image tags separate so the production `ENTRYPOINT` does not overwrite the development image.
4. **MuJoCo Scene Conventions (`scene_builder.py`)**:
   - The tabletop surface is at `z = 0.0`.
   - The SO-ARM101 arm (`assets/so101/so101.xml`) is mounted at `(0, 0, 0)` with `so101_pinch_site` centered between the gripper jaws.
   - Keep visual geoms (`group=2`, `contype=0`, `conaffinity=0`, `mass=0`) separate from collision geoms (`group=3`).
   - If you modify `scene_builder.py` or `assets/so101/so101.xml` and update `scene.xml`, preserve `meshdir="assets/so101/assets"` in `<compiler>`.
5. **Mandatory Verification After Changes**:
   - After any change to the scene, controller, or tests, run `make tests` and confirm that both `tests/sim_suite.py` and `pick_and_place_verify.py` exit with code `0`.
