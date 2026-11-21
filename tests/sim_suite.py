"""Repeatable, seed-configurable MuJoCo simulation test suite for SO-ARM101."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
  sys.path.insert(0, str(PROJECT_DIR))

import pick_and_place_verify as ppv
import scene_builder as sb


@dataclass
class CaseResult:
  """Structured outcome for a single simulation test case."""

  scenario: str
  case_id: str
  passed: bool
  task_succeeded: bool
  sim_time_s: float
  wall_time_s: float
  failure_reason: str | None = None
  details: dict[str, object] = field(default_factory=dict)


def _count_obstacle_contacts(
    data: mujoco.MjData, obs_geom_id: int | None, table_geom_id: int
) -> int:
  """Counts contacts involving the transit obstacle (excluding static table)."""
  if obs_geom_id is None:
    return 0
  hits = 0
  for i in range(data.ncon):
    c = data.contact[i]
    if (c.geom1 == obs_geom_id and c.geom2 != table_geom_id) or (
        c.geom2 == obs_geom_id and c.geom1 != table_geom_id
    ):
      hits += 1
  return hits


def _step_move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_arm_goal: np.ndarray,
    gripper_goal: float,
    duration_s: float,
    *,
    obs_geom_id: int | None = None,
    table_geom_id: int = -1,
    site_id: int = -1,
    inject_drop_frac: float | None = None,
    monitor_retention: bool = False,
) -> tuple[int, bool, bool]:
  """Steps a cosine-interpolated move while monitoring collisions and drops."""
  start_ctrl = data.ctrl.copy()
  target_ctrl = np.array([*q_arm_goal, gripper_goal], dtype=np.float64)
  steps = max(1, int(round(duration_s / model.opt.timestep)))
  obs_hits = 0
  dropped = False
  drop_injected = False

  for i in range(1, steps + 1):
    frac = i / steps
    alpha = 0.5 * (1.0 - np.cos(np.pi * frac))
    ctrl = start_ctrl + alpha * (target_ctrl - start_ctrl)
    if inject_drop_frac is not None and frac >= inject_drop_frac:
      ctrl[5] = sb.GRIPPER_OPEN_CTRL
      target_ctrl[5] = sb.GRIPPER_OPEN_CTRL
      drop_injected = True
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    obs_hits += _count_obstacle_contacts(data, obs_geom_id, table_geom_id)
    if monitor_retention and site_id >= 0:
      cube_pos = data.body("red_cube").xpos
      pinch_pos = data.site_xpos[site_id]
      if float(np.linalg.norm(cube_pos - pinch_pos)) > 0.042 or (
          drop_injected and float(cube_pos[2]) < 0.048
      ):
        dropped = True
        break

  return obs_hits, dropped, drop_injected


def _hold_and_monitor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration_s: float,
    *,
    obs_geom_id: int | None = None,
    table_geom_id: int = -1,
) -> int:
  """Holds current actuator targets for duration_s while counting obstacle hits."""
  steps = max(1, int(round(duration_s / model.opt.timestep)))
  obs_hits = 0
  for _ in range(steps):
    mujoco.mj_step(model, data)
    obs_hits += _count_obstacle_contacts(data, obs_geom_id, table_geom_id)
  return obs_hits


def _solve_pick_place_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_xyz: np.ndarray,
    tray_xyz: np.ndarray,
    transit_z: float,
) -> dict[str, np.ndarray]:
  """Solves top-down IK for cube pick and tray drop."""
  qpos_idx, dof_idx, q_min, q_max = ppv.get_arm_indices(model)
  site_id = int(model.site(sb.PINCH_SITE_NAME).id)
  ctx_qpos = data.qpos.copy()

  targets = (
      ("cube_hover", [cube_xyz[0], cube_xyz[1], ppv.HOVER_Z]),
      ("cube_grasp", [cube_xyz[0], cube_xyz[1], cube_xyz[2]]),
      ("tray_hover", [tray_xyz[0], tray_xyz[1], transit_z]),
      ("tray_drop", [tray_xyz[0], tray_xyz[1], ppv.DROP_Z]),
  )
  solved: dict[str, np.ndarray] = {}
  seed = np.array(sb.HOME_ARM_QPOS, dtype=np.float64)
  for key, pos in targets:
    q, _, _ = ppv.solve_site_ik(
        model,
        site_id,
        qpos_idx,
        dof_idx,
        q_min,
        q_max,
        pos,
        q_seed=seed,
        context_qpos=ctx_qpos,
    )
    solved[key] = q
    seed = q
  return solved


def _cube_in_tray(data: mujoco.MjData) -> tuple[bool, float]:
  """Checks whether red_cube is resting stably inside green_tray."""
  cube_pos = data.body("red_cube").xpos
  tray_pos = data.body("green_tray").xpos
  cube_speed = float(np.linalg.norm(data.joint("red_cube_free").qvel[:3]))
  lim = sb.GREEN_TRAY_INNER_HALF_XY - sb.RED_CUBE_HALF_SIZE
  in_xy = (
      abs(float(cube_pos[0] - tray_pos[0])) <= lim
      and abs(float(cube_pos[1] - tray_pos[1])) <= lim
  )
  in_z = 0.015 <= float(cube_pos[2]) <= 0.030
  return bool(in_xy and in_z and cube_speed < 0.02), cube_speed


def run_cycle(
    *,
    cube_pos: tuple[float, float, float] = sb.RED_CUBE_POS,
    cube_friction: tuple[float, float, float] = (1.2, 0.02, 0.005),
    gripper_friction: tuple[float, float, float] = (1.2, 0.02, 0.005),
    gripper_closed_ctrl: float = sb.GRIPPER_CLOSED_CTRL,
    obstacle: dict[str, object] | None = None,
    transit_z: float = ppv.HOVER_Z,
    inject_mid_drop: bool = False,
    allow_retry: bool = False,
) -> dict[str, object]:
  """Runs one full pick-and-place cycle and returns physics diagnostics."""
  spec = sb.build_scene_spec(
      cube_pos=cube_pos,
      cube_friction=cube_friction,
      gripper_friction=gripper_friction,
      obstacle=obstacle,
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  ppv.reset_to_home(model, data)

  site_id = int(model.site(sb.PINCH_SITE_NAME).id)
  table_geom_id = int(model.geom("table_top_col").id)
  obs_geom_id = (
      int(model.geom("transit_obstacle_col").id)
      if obstacle is not None
      else None
  )

  obs_hits = _hold_and_monitor(
      model, data, 0.30, obs_geom_id=obs_geom_id, table_geom_id=table_geom_id
  )
  tray_xyz = data.body("green_tray").xpos.copy()
  q_home = np.array(sb.HOME_ARM_QPOS, dtype=np.float64)

  retries = 0
  drop_detected = False
  slip_detected = False
  failure_reason: str | None = None

  for attempt in range(2 if allow_retry else 1):
    cube_xyz = data.body("red_cube").xpos.copy()
    try:
      ik = _solve_pick_place_ik(model, data, cube_xyz, tray_xyz, transit_z)
    except RuntimeError:
      failure_reason = "ik_unreachable"
      break

    # Pick sequence.
    h, _, _ = _step_move(
        model, data, ik["cube_hover"], sb.GRIPPER_OPEN_CTRL, 0.7,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    h, _, _ = _step_move(
        model, data, ik["cube_grasp"], sb.GRIPPER_OPEN_CTRL, 0.5,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    h, _, _ = _step_move(
        model, data, ik["cube_grasp"], gripper_closed_ctrl, 0.40,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    obs_hits += _hold_and_monitor(
        model, data, 0.30, obs_geom_id=obs_geom_id, table_geom_id=table_geom_id
    )

    # Lift and verify grasp retention (slip detection).
    h, _, _ = _step_move(
        model, data, ik["cube_hover"], gripper_closed_ctrl, 0.7,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    obs_hits += _hold_and_monitor(
        model, data, 0.20, obs_geom_id=obs_geom_id, table_geom_id=table_geom_id
    )

    cube_after_lift = data.body("red_cube").xpos.copy()
    pinch_after_lift = data.site_xpos[site_id].copy()
    grasp_dist = float(np.linalg.norm(cube_after_lift - pinch_after_lift))
    if float(cube_after_lift[2]) < 0.050 or grasp_dist > 0.038:
      slip_detected = True
      failure_reason = "grasp_slip"
      break

    # Transit toward tray (with optional fault injection on first attempt).
    drop_frac = 0.25 if (inject_mid_drop and attempt == 0) else None
    h, dropped, _ = _step_move(
        model,
        data,
        ik["tray_hover"],
        gripper_closed_ctrl,
        0.8,
        obs_geom_id=obs_geom_id,
        table_geom_id=table_geom_id,
        site_id=site_id,
        inject_drop_frac=drop_frac,
        monitor_retention=True,
    )
    obs_hits += h

    if dropped:
      drop_detected = True
      if allow_retry and attempt == 0:
        retries += 1
        h, _, _ = _step_move(
            model, data, q_home, sb.GRIPPER_OPEN_CTRL, 0.6,
            obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
        )
        obs_hits += h
        obs_hits += _hold_and_monitor(
            model, data, 0.60, obs_geom_id=obs_geom_id, table_geom_id=table_geom_id
        )
        continue
      failure_reason = "mid_transit_drop"
      break

    # Lower into tray and release.
    h, _, _ = _step_move(
        model, data, ik["tray_drop"], gripper_closed_ctrl, 0.5,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    h, _, _ = _step_move(
        model, data, ik["tray_drop"], sb.GRIPPER_OPEN_CTRL, 0.35,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    obs_hits += _hold_and_monitor(
        model, data, 0.25, obs_geom_id=obs_geom_id, table_geom_id=table_geom_id
    )
    h, _, _ = _step_move(
        model, data, ik["tray_hover"], sb.GRIPPER_OPEN_CTRL, 0.5,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    h, _, _ = _step_move(
        model, data, q_home, sb.GRIPPER_OPEN_CTRL, 0.7,
        obs_geom_id=obs_geom_id, table_geom_id=table_geom_id,
    )
    obs_hits += h
    obs_hits += _hold_and_monitor(
        model, data, 0.80, obs_geom_id=obs_geom_id, table_geom_id=table_geom_id
    )
    break

  in_tray, cube_speed = _cube_in_tray(data)
  if obs_hits > 0:
    failure_reason = "obstacle_collision"
  elif not in_tray and failure_reason is None:
    failure_reason = "missed_tray"

  return {
      "task_succeeded": bool(in_tray and obs_hits == 0),
      "sim_time_s": float(data.time),
      "failure_reason": failure_reason,
      "obstacle_contacts": obs_hits,
      "slip_detected": slip_detected,
      "drop_detected": drop_detected,
      "retries": retries,
      "cube_final_xyz": np.round(data.body("red_cube").xpos, 4).tolist(),
      "cube_speed_mps": round(cube_speed, 5),
  }


def run_position_variation(seed: int, trials: int) -> list[CaseResult]:
  """Scenario 1: randomized reachable cube positions with fixed RNG seed."""
  rng = np.random.default_rng(seed)
  results: list[CaseResult] = []
  for idx in range(trials):
    x = float(rng.uniform(0.175, 0.240))
    y = float(rng.uniform(-0.140, -0.085))
    cube_pos = (round(x, 4), round(y, 4), sb.RED_CUBE_POS[2])
    t0 = time.perf_counter()
    diag = run_cycle(cube_pos=cube_pos)
    wall_s = time.perf_counter() - t0
    ok = bool(diag["task_succeeded"])
    results.append(
        CaseResult(
            scenario="position_variation",
            case_id=f"trial_{idx + 1}",
            passed=ok,
            task_succeeded=ok,
            sim_time_s=float(diag["sim_time_s"]),
            wall_time_s=wall_s,
            failure_reason=diag["failure_reason"],  # type: ignore[arg-type]
            details={"cube_pos": list(cube_pos), **diag},
        )
    )
  return results


def run_grip_friction_tests() -> list[CaseResult]:
  """Scenario 2: nominal grip vs low-friction and underclosed slip detection."""
  configs = (
      (
          "nominal_grip",
          (1.2, 0.02, 0.005),
          (1.2, 0.02, 0.005),
          sb.GRIPPER_CLOSED_CTRL,
          True,
      ),
      (
          "low_friction_slip",
          (0.01, 0.0001, 0.0001),
          (0.01, 0.0001, 0.0001),
          0.20,
          False,
      ),
      (
          "underclosed_gripper_slip",
          (1.2, 0.02, 0.005),
          (1.2, 0.02, 0.005),
          0.35,
          False,
      ),
  )
  results: list[CaseResult] = []
  for case_id, c_fric, g_fric, g_ctrl, expect_success in configs:
    t0 = time.perf_counter()
    diag = run_cycle(
        cube_friction=c_fric,
        gripper_friction=g_fric,
        gripper_closed_ctrl=g_ctrl,
    )
    wall_s = time.perf_counter() - t0
    if expect_success:
      passed = bool(diag["task_succeeded"] and not diag["slip_detected"])
    else:
      passed = bool(
          diag["slip_detected"] and diag["failure_reason"] == "grasp_slip"
      )
    results.append(
        CaseResult(
            scenario="grip_friction_and_closure",
            case_id=case_id,
            passed=passed,
            task_succeeded=bool(diag["task_succeeded"]),
            sim_time_s=float(diag["sim_time_s"]),
            wall_time_s=wall_s,
            failure_reason=diag["failure_reason"],  # type: ignore[arg-type]
            details=diag,
        )
    )
  return results


def run_obstacle_tests() -> list[CaseResult]:
  """Scenario 3: static obstacle between cube and tray (safe transit vs hit)."""
  obstacle = {"pos": (0.23, -0.068, 0.028), "size": (0.018, 0.008, 0.028)}
  cases = (
      ("obstacle_safe_transit", 0.090, True),
      ("obstacle_low_transit_check", 0.032, False),
  )
  results: list[CaseResult] = []
  for case_id, transit_z, expect_clear in cases:
    t0 = time.perf_counter()
    diag = run_cycle(obstacle=obstacle, transit_z=transit_z)
    wall_s = time.perf_counter() - t0
    if expect_clear:
      passed = bool(
          diag["task_succeeded"] and int(diag["obstacle_contacts"]) == 0
      )
    else:
      passed = bool(
          int(diag["obstacle_contacts"]) > 0
          and diag["failure_reason"] == "obstacle_collision"
      )
    results.append(
        CaseResult(
            scenario="obstacle_avoidance",
            case_id=case_id,
            passed=passed,
            task_succeeded=bool(diag["task_succeeded"]),
            sim_time_s=float(diag["sim_time_s"]),
            wall_time_s=wall_s,
            failure_reason=diag["failure_reason"],  # type: ignore[arg-type]
            details=diag,
        )
    )
  return results


def run_out_of_reach_tests() -> list[CaseResult]:
  """Scenario 4: unreachable targets verify safe error handling and limits."""
  spec = sb.build_scene_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  ppv.reset_to_home(model, data)
  qpos_idx, dof_idx, q_min, q_max = ppv.get_arm_indices(model)
  site_id = int(model.site(sb.PINCH_SITE_NAME).id)

  targets = (
      ("far_target", (0.55, 0.0, 0.015)),
      ("behind_base_target", (-0.30, 0.0, 0.015)),
  )
  results: list[CaseResult] = []
  for case_id, target in targets:
    t0 = time.perf_counter()
    rejected = False
    err_msg = ""
    try:
      ppv.solve_site_ik(
          model, site_id, qpos_idx, dof_idx, q_min, q_max, target,
          context_qpos=data.qpos.copy(),
      )
    except RuntimeError as exc:
      rejected = True
      err_msg = str(exc)
    wall_s = time.perf_counter() - t0
    q_cur = data.qpos[qpos_idx]
    within_limits = bool(
        np.all(q_cur >= q_min - 1e-6) and np.all(q_cur <= q_max + 1e-6)
    )
    results.append(
        CaseResult(
            scenario="out_of_reach_target",
            case_id=case_id,
            passed=bool(rejected and within_limits),
            task_succeeded=False,
            sim_time_s=0.0,
            wall_time_s=wall_s,
            failure_reason="out_of_reach" if rejected else "unexpected_ik_pass",
            details={"target": list(target), "error": err_msg},
        )
    )
  return results


def run_drop_and_retry_test() -> list[CaseResult]:
  """Scenario 5: mid-transit drop injection, failure detection, and retry."""
  t0 = time.perf_counter()
  diag = run_cycle(inject_mid_drop=True, allow_retry=True)
  wall_s = time.perf_counter() - t0
  passed = bool(
      diag["drop_detected"]
      and int(diag["retries"]) == 1
      and diag["task_succeeded"]
  )
  return [
      CaseResult(
          scenario="mid_transit_drop_and_retry",
          case_id="drop_recovery",
          passed=passed,
          task_succeeded=bool(diag["task_succeeded"]),
          sim_time_s=float(diag["sim_time_s"]),
          wall_time_s=wall_s,
          failure_reason=diag["failure_reason"],  # type: ignore[arg-type]
          details=diag,
      )
  ]


def _print_summary(results: list[CaseResult], seed: int) -> dict[str, object]:
  """Prints a structured console summary and returns aggregate metrics."""
  total = len(results)
  passed_count = sum(1 for r in results if r.passed)
  task_ok_count = sum(1 for r in results if r.task_succeeded)
  avg_sim_s = float(np.mean([r.sim_time_s for r in results]))
  avg_wall_s = float(np.mean([r.wall_time_s for r in results]))

  failure_breakdown: dict[str, int] = {}
  for r in results:
    if r.failure_reason:
      failure_breakdown[r.failure_reason] = (
          failure_breakdown.get(r.failure_reason, 0) + 1
      )

  print(f"\n=== SO-ARM101 Simulation Test Suite (seed={seed}) ===")
  header = (
      f"{'Scenario':<28} {'Case':<26} {'Status':<6} "
      f"{'Task':<6} {'Sim(s)':>7} {'Wall(s)':>8}  Failure Reason"
  )
  print(header)
  print("-" * len(header))
  for r in results:
    status = "PASS" if r.passed else "FAIL"
    task = "OK" if r.task_succeeded else "FAIL"
    reason = r.failure_reason or "-"
    print(
        f"{r.scenario:<28} {r.case_id:<26} {status:<6} "
        f"{task:<6} {r.sim_time_s:7.2f} {r.wall_time_s:8.3f}  {reason}"
    )

  summary = {
      "seed": seed,
      "total_cases": total,
      "passed_cases": passed_count,
      "pass_rate": round(passed_count / max(1, total), 4),
      "task_success_count": task_ok_count,
      "avg_sim_time_s": round(avg_sim_s, 3),
      "avg_wall_time_s": round(avg_wall_s, 3),
      "failure_breakdown": failure_breakdown,
  }
  print("\n--- Summary ---")
  print(f"  Pass rate         : {passed_count}/{total} ({summary['pass_rate'] * 100:.1f}%)")
  print(f"  Avg cycle sim/wall: {avg_sim_s:.2f} s / {avg_wall_s:.3f} s")
  print(f"  Failure reasons   : {failure_breakdown or 'none'}")
  return summary


def main() -> int:
  parser = argparse.ArgumentParser(
      description="SO-ARM101 MuJoCo simulation test suite."
  )
  parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
  parser.add_argument(
      "--trials",
      type=int,
      default=5,
      help="Number of randomized cube position trials.",
  )
  parser.add_argument(
      "--json-out",
      type=Path,
      default=None,
      help="Optional path to write JSON report.",
  )
  args = parser.parse_args()

  results: list[CaseResult] = []
  results.extend(run_position_variation(args.seed, args.trials))
  results.extend(run_grip_friction_tests())
  results.extend(run_obstacle_tests())
  results.extend(run_out_of_reach_tests())
  results.extend(run_drop_and_retry_test())

  summary = _print_summary(results, args.seed)
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  JSON report       : {args.json_out}")

  return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
  raise SystemExit(main())
