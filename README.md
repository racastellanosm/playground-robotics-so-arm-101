# SO-ARM101 MuJoCo Simulation & Pick-and-Place

**MuJoCo** simulation of the **SO-ARM101** robotic arm (`mujoco_menagerie` `robotstudio_so101`) mounted on a tabletop, featuring reachability checks, pick-and-place manipulation, physical validation, a seed-configurable simulation test suite, and an interactive web viewer (**MuJoCo Studio**).

The entire environment is **100% containerized** following the *Zero Host Dependencies* principle: the host machine only requires **Docker (with Compose)** and **GNU Make**.

---

## Project Structure

- `assets/so101/`: Official `so101.xml` model and STL meshes from `mujoco_menagerie`.
- `scene_builder.py`: Procedural `mujoco.MjSpec` builder (`build_scene_spec`) assembling the table (`work_table`), `so101` arm, red cube (`red_cube`), blue cylinder (`blue_cylinder`), green tray (`green_tray`), and optional transit obstacles.
- `pick_and_place_verify.py`: Inverse kinematics validation (`solve_site_ik`), physical reachability check for both objects, red cube pick-and-place into the green tray, and offscreen rendering into `renders/`.
- `tests/sim_suite.py`: Repeatable, seed-configurable simulation test suite (`--seed`, `--trials`, `--json-out`):
  1. Randomized red cube position trials (`position_variation`).
  2. Gripper friction and closure slip detection (`grip_friction_and_closure`).
  3. Transit obstacle avoidance and collision detection (`obstacle_avoidance`).
  4. Out-of-reach target handling and joint limit safety (`out_of_reach_target`).
  5. Mid-transit drop injection, failure detection, and autonomous retry (`mid_transit_drop_and_retry`).
- `run_web_server.py`, `sim_plugin.py`, `viewer_plugin.py`, `messages.py`: **MuJoCo Studio** web server and interactive plugins (`http://localhost:8080`).
- `scene.xml`: Baked MJCF scene compatible with direct loading in MuJoCo.
- `Dockerfile`, `docker-compose.yml`, `docker-compose.dev.yml`, `Makefile`: Container infrastructure and orchestration.

---

## Host Prerequisites

- **Docker** + **Docker Compose**
- **GNU Make**

---

## Quick Start (`Makefile`)

```bash
# Display all available commands
make help

# Build dev and production Docker images
make build

# Run the simulation test suite and base verification with renders
make tests

# Start the MuJoCo Studio WebViewer server (http://localhost:8080)
make run
```

### Available Targets

| Command | Description |
| :--- | :--- |
| `make help` | Displays the self-documented help menu |
| `make build` | Builds `so-arm-101:dev` and `so-arm-101:prod` Docker images |
| `make install` | Builds the development container with all dependencies |
| `make tests` / `make test` | Runs `tests/sim_suite.py` and `pick_and_place_verify.py` in an ephemeral container |
| `make verify` | Runs `pick_and_place_verify.py` and generates renders in `renders/` |
| `make run` / `make dev` | Starts the MuJoCo Studio WebViewer server in the foreground (`STUDIO_PORT=8080`) |
| `make start` | Starts the WebViewer server in the background (`-d`) |
| `make stop` | Stops background containers |
| `make logs` | Streams live logs from the `sim` container |
| `make shell` | Opens an interactive `bash` shell inside the container |
| `make clean` | Stops containers and removes networks and volumes |

### Configurable Parameters for `make tests`

```bash
make tests SEED=123 TRIALS=3 JSON_OUT=tests/sim_suite_report.json
```
