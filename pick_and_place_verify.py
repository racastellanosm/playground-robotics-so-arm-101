"""Reachability validation, pick-and-place verification, and rendering for SO-ARM101."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import scene_builder as sb

APPROACH_AXIS_EE = np.array([1.0, 0.0, 0.0], dtype=np.float64)
APPROACH_DIR_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)
HOVER_Z = 0.085
DROP_Z = 0.032


def get_arm_indices(
    model: mujoco.MjModel,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
  """Returns arm qpos indices, dof indices, and lower/upper joint bounds."""
  joints = [model.joint(name) for name in sb.ARM_JOINT_NAMES]
  qpos_idx = [int(j.qposadr[0]) for j in joints]
  dof_idx = [int(j.dofadr[0]) for j in joints]
  q_min = np.array([float(j.range[0]) for j in joints], dtype=np.float64)
  q_max = np.array([float(j.range[1]) for j in joints], dtype=np.float64)
  return qpos_idx, dof_idx, q_min, q_max


def solve_site_ik(
    model: mujoco.MjModel,
    site_id: int,
    qpos_indices: Sequence[int],
    dof_indices: Sequence[int],
    q_min: np.ndarray,
    q_max: np.ndarray,
    pos_world: Sequence[float] | np.ndarray,
    *,
    q_seed: Sequence[float] | np.ndarray | None = None,
    q_posture: Sequence[float] | np.ndarray | None = None,
    context_qpos: np.ndarray | None = None,
    approach_axis_ee: np.ndarray = APPROACH_AXIS_EE,
    approach_dir_world: np.ndarray = APPROACH_DIR_DOWN,
    pos_tol_m: float = 1.5e-3,
    ori_tol_rad: float = 2e-2,
    max_iters: int = 250,
    damping: float = 2e-2,
) -> tuple[np.ndarray, float, float]:
  """Damped least-squares IK for the 5-DOF SO-101 arm with limit clamping."""
  data = mujoco.MjData(model)
  pos_des = np.asarray(pos_world, dtype=np.float64)[:3]
  ax_ee = np.asarray(approach_axis_ee, dtype=np.float64)[:3]
  ax_ee /= float(np.linalg.norm(ax_ee))
  a_des = np.asarray(approach_dir_world, dtype=np.float64)[:3]
  a_des /= float(np.linalg.norm(a_des))

  n_dof = len(dof_indices)
  seed0 = (
      np.array(sb.HOME_ARM_QPOS, dtype=np.float64)
      if q_seed is None
      else np.asarray(q_seed, dtype=np.float64).copy()
  )
  q_nom = seed0 if q_posture is None else np.asarray(q_posture, dtype=np.float64)

  ba, bb = np.zeros(n_dof), np.zeros(n_dof)
  ba[1], ba[2], bb[1], bb[2] = 1.0, -1.6, -1.0, 1.6
  seeds = [
      seed0,
      q_nom,
      0.5 * (q_min + q_max),
      ba,
      bb,
      np.array([0.0, -0.3, 0.5, 1.1, 0.0]),
      np.array([0.0, 0.0, 0.47, 1.18, 0.0]),
  ]
  seeds += [np.array([pan, *q_nom[1:]]) for pan in (-0.6, 0.6)]

  jacp = np.zeros((3, model.nv), dtype=np.float64)
  jacr = np.zeros((3, model.nv), dtype=np.float64)
  q_idx = list(qpos_indices)
  eye_d = np.eye(n_dof, dtype=np.float64)

  best_q = seed0.copy()
  best_pos_err = float("inf")
  best_ori_err = float("inf")

  for s in seeds:
    q = np.clip(s.copy(), q_min, q_max)
    for _ in range(max_iters):
      data.qpos[:] = model.qpos0 if context_qpos is None else context_qpos
      data.qpos[q_idx] = q
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)

      r_cur = data.site_xmat[site_id].reshape(3, 3)
      e_pos = pos_des - data.site_xpos[site_id]
      e_rot = np.cross(r_cur @ ax_ee, a_des)

      p_err = float(np.linalg.norm(e_pos))
      r_err = float(np.linalg.norm(e_rot))
      if p_err < best_pos_err:
        best_pos_err = p_err
        best_ori_err = r_err
        best_q = q.copy()
      if p_err <= pos_tol_m and r_err <= ori_tol_rad:
        return q, p_err, r_err

      mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
      jp = jacp[:, dof_indices]
      jr = jacr[:, dof_indices]
      j_mat = np.vstack([jp, 0.35 * jr])
      err = np.concatenate([e_pos, 0.35 * e_rot])
      jjt = j_mat @ j_mat.T + (damping**2) * np.eye(j_mat.shape[0])
      j_pinv = j_mat.T @ np.linalg.solve(jjt, np.eye(j_mat.shape[0]))
      dq = j_pinv @ err + 0.05 * ((eye_d - j_pinv @ j_mat) @ (q_nom - q))
      step = float(np.max(np.abs(dq)))
      if step > 0.15:
        dq *= 0.15 / step
      q = np.clip(q + dq, q_min, q_max)

  if best_pos_err > 5e-3:
    raise RuntimeError(
        f"solve_site_ik failed for {pos_des.tolist()}: "
        f"pos_err={best_pos_err:.4f} m, ori_err={best_ori_err:.4f} rad"
    )
  return best_q, best_pos_err, best_ori_err


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Resets simulation state so the arm starts at HOME_ARM_QPOS with open jaws."""
  mujoco.mj_resetData(model, data)
  qpos_idx, _, _, _ = get_arm_indices(model)
  grip_q_idx = int(model.joint(sb.GRIPPER_JOINT_NAME).qposadr[0])
  data.qpos[qpos_idx] = sb.HOME_ARM_QPOS
  data.qpos[grip_q_idx] = sb.GRIPPER_OPEN_CTRL
  data.ctrl[:5] = sb.HOME_ARM_QPOS
  data.ctrl[5] = sb.GRIPPER_OPEN_CTRL
  mujoco.mj_forward(model, data)


def validate_reachability(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, tuple[np.ndarray, float, float]]:
  """Verifies initial settling and IK reachability of objects and target tray."""
  reset_to_home(model, data)
  settle_steps = int(round(0.5 / model.opt.timestep))
  for _ in range(settle_steps):
    mujoco.mj_step(model, data)

  cube_vel = float(np.linalg.norm(data.joint("red_cube_free").qvel[:3]))
  cyl_vel = float(np.linalg.norm(data.joint("blue_cylinder_free").qvel[:3]))
  assert cube_vel < 0.01, f"Red cube failed to settle: speed={cube_vel:.4f} m/s"
  assert cyl_vel < 0.01, f"Blue cylinder failed to settle: speed={cyl_vel:.4f} m/s"

  qpos_idx, dof_idx, q_min, q_max = get_arm_indices(model)
  site_id = int(model.site(sb.PINCH_SITE_NAME).id)
  ctx_qpos = data.qpos.copy()

  cube_xyz = data.body("red_cube").xpos.copy()
  cyl_xyz = data.body("blue_cylinder").xpos.copy()
  tray_xyz = data.body("green_tray").xpos.copy()

  targets = {
      "blue_cylinder_hover": np.array([cyl_xyz[0], cyl_xyz[1], HOVER_Z]),
      "blue_cylinder_grasp": np.array([cyl_xyz[0], cyl_xyz[1], cyl_xyz[2]]),
      "red_cube_hover": np.array([cube_xyz[0], cube_xyz[1], HOVER_Z]),
      "red_cube_grasp": np.array([cube_xyz[0], cube_xyz[1], cube_xyz[2]]),
      "green_tray_hover": np.array([tray_xyz[0], tray_xyz[1], HOVER_Z]),
      "green_tray_drop": np.array([tray_xyz[0], tray_xyz[1], DROP_Z]),
  }

  print("\n=== Reachability Validation (IK) ===")
  solutions: dict[str, tuple[np.ndarray, float, float]] = {}
  seed = np.array(sb.HOME_ARM_QPOS, dtype=np.float64)
  for label, pos in targets.items():
    q, p_err, r_err = solve_site_ik(
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
    assert p_err < 0.005, f"{label} pos_err={p_err:.4f} m >= 0.005 m"
    assert r_err < 0.05, f"{label} ori_err={r_err:.4f} rad >= 0.05 rad"
    assert np.all(q >= q_min - 1e-6) and np.all(q <= q_max + 1e-6)
    solutions[label] = (q, p_err, r_err)
    seed = q
    print(
        f"  {label:<22s} target={np.round(pos, 3)}  "
        f"pos_err={p_err * 1000:.2f} mm  ori_err={r_err:.4f} rad"
    )
  return solutions


def _step_cosine_move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_arm_goal: np.ndarray,
    gripper_goal: float,
    duration_s: float,
) -> None:
  """Smoothly moves position actuators from current ctrl to goal over duration_s."""
  start_ctrl = data.ctrl.copy()
  target_ctrl = np.array([*q_arm_goal, gripper_goal], dtype=np.float64)
  steps = max(1, int(round(duration_s / model.opt.timestep)))
  for i in range(1, steps + 1):
    alpha = 0.5 * (1.0 - np.cos(np.pi * (i / steps)))
    data.ctrl[:] = start_ctrl + alpha * (target_ctrl - start_ctrl)
    mujoco.mj_step(model, data)


def _hold(model: mujoco.MjModel, data: mujoco.MjData, duration_s: float) -> None:
  """Holds current actuator targets for duration_s."""
  steps = max(1, int(round(duration_s / model.opt.timestep)))
  for _ in range(steps):
    mujoco.mj_step(model, data)


def _render_view(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: str,
    out_path: Path,
) -> np.ndarray:
  """Renders an RGB frame from the specified camera and saves it to PNG."""
  opt = mujoco.MjvOption()
  opt.geomgroup[2] = 1
  opt.geomgroup[3] = 0
  opt.sitegroup[4] = 0
  renderer.update_scene(data, camera=camera, scene_option=opt)
  rgb = renderer.render()
  Image.fromarray(rgb).save(out_path)
  return rgb


def _save_summary_grid(
    panels: list[tuple[str, np.ndarray]], out_path: Path
) -> None:
  """Stitches 4 RGB frames into a labeled 2x2 verification sheet."""
  tile_w, tile_h = 640, 360
  header_h = 34
  canvas = Image.new("RGB", (tile_w * 2, (tile_h + header_h) * 2), (24, 28, 34))
  draw = ImageDraw.Draw(canvas)
  for idx, (label, rgb) in enumerate(panels):
    col = idx % 2
    row = idx // 2
    x0 = col * tile_w
    y0 = row * (tile_h + header_h)
    tile = Image.fromarray(rgb).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
    canvas.paste(tile, (x0, y0 + header_h))
    draw.text((x0 + 14, y0 + 9), label, fill=(235, 240, 245))
  canvas.save(out_path)


def run_pick_and_place(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_solutions: dict[str, tuple[np.ndarray, float, float]],
    *,
    render_dir: Path = Path("renders"),
) -> dict[str, object]:
  """Executes reach check on both objects and red-cube pick-and-place into tray."""
  render_dir.mkdir(parents=True, exist_ok=True)
  reset_to_home(model, data)
  _hold(model, data, 0.25)

  site_id = int(model.site(sb.PINCH_SITE_NAME).id)
  cyl_initial = data.body("blue_cylinder").xpos.copy()

  renderer = mujoco.Renderer(model, height=720, width=1280)
  panels: list[tuple[str, np.ndarray]] = []
  try:
    panels.append((
        "1. Initial Scene (SO-ARM101 Ready)",
        _render_view(
            renderer, data, "overview_cam", render_dir / "01_initial_scene.png"
        ),
    ))

    # 1. Reach check on blue cylinder (hover -> touch-free approach -> hover).
    q_cyl_hover = ik_solutions["blue_cylinder_hover"][0]
    q_cyl_grasp = ik_solutions["blue_cylinder_grasp"][0]
    _step_cosine_move(model, data, q_cyl_hover, sb.GRIPPER_OPEN_CTRL, 0.8)
    _step_cosine_move(model, data, q_cyl_grasp, sb.GRIPPER_OPEN_CTRL, 0.6)
    _hold(model, data, 0.15)
    cyl_reach_err = float(
        np.linalg.norm(data.site_xpos[site_id] - cyl_initial)
    )
    assert cyl_reach_err < 0.015, (
        f"Physical reach to blue_cylinder failed: err={cyl_reach_err:.4f} m"
    )
    _step_cosine_move(model, data, q_cyl_hover, sb.GRIPPER_OPEN_CTRL, 0.6)

    # 2. Approach and grasp red cube.
    q_cube_hover = ik_solutions["red_cube_hover"][0]
    q_cube_grasp = ik_solutions["red_cube_grasp"][0]
    _step_cosine_move(model, data, q_cube_hover, sb.GRIPPER_OPEN_CTRL, 0.9)
    _step_cosine_move(model, data, q_cube_grasp, sb.GRIPPER_OPEN_CTRL, 0.6)
    _hold(model, data, 0.15)
    cube_reach_err = float(
        np.linalg.norm(data.site_xpos[site_id] - data.body("red_cube").xpos)
    )
    assert cube_reach_err < 0.015, (
        f"Physical reach to red_cube failed: err={cube_reach_err:.4f} m"
    )

    _step_cosine_move(model, data, q_cube_grasp, sb.GRIPPER_CLOSED_CTRL, 0.45)
    _hold(model, data, 0.40)
    panels.append((
        "2. Grasp Red Cube",
        _render_view(
            renderer, data, "overview_cam", render_dir / "02_grasp_red_cube.png"
        ),
    ))

    # 3. Lift red cube above obstacles and tray rim.
    _step_cosine_move(model, data, q_cube_hover, sb.GRIPPER_CLOSED_CTRL, 0.8)
    _hold(model, data, 0.25)
    lifted_z = float(data.body("red_cube").xpos[2])
    assert lifted_z > 0.06, f"Red cube was not lifted cleanly: z={lifted_z:.4f} m"
    panels.append((
        f"3. Lift Red Cube (z={lifted_z:.3f} m)",
        _render_view(
            renderer, data, "overview_cam", render_dir / "03_lift_red_cube.png"
        ),
    ))

    # 4. Transport and release inside green tray.
    q_tray_hover = ik_solutions["green_tray_hover"][0]
    q_tray_drop = ik_solutions["green_tray_drop"][0]
    _step_cosine_move(model, data, q_tray_hover, sb.GRIPPER_CLOSED_CTRL, 0.9)
    _step_cosine_move(model, data, q_tray_drop, sb.GRIPPER_CLOSED_CTRL, 0.6)
    _hold(model, data, 0.15)
    _step_cosine_move(model, data, q_tray_drop, sb.GRIPPER_OPEN_CTRL, 0.40)
    _hold(model, data, 0.30)

    # 5. Retreat and settle.
    _step_cosine_move(model, data, q_tray_hover, sb.GRIPPER_OPEN_CTRL, 0.6)
    _step_cosine_move(
        model,
        data,
        np.array(sb.HOME_ARM_QPOS, dtype=np.float64),
        sb.GRIPPER_OPEN_CTRL,
        0.8,
    )
    _hold(model, data, 1.0)
    panels.append((
        "4. Placed Inside Green Tray",
        _render_view(
            renderer, data, "overview_cam", render_dir / "04_placed_in_tray.png"
        ),
    ))
    _render_view(renderer, data, "top_cam", render_dir / "05_top_final.png")
  finally:
    renderer.close()

  _save_summary_grid(panels, render_dir / "pick_and_place_summary.png")

  cube_final = data.body("red_cube").xpos.copy()
  cube_speed = float(np.linalg.norm(data.joint("red_cube_free").qvel[:3]))
  cyl_final = data.body("blue_cylinder").xpos.copy()
  cyl_drift = float(np.linalg.norm(cyl_final - cyl_initial))

  tray_center = np.array(sb.GREEN_TRAY_POS, dtype=np.float64)
  margin_lim = sb.GREEN_TRAY_INNER_HALF_XY - sb.RED_CUBE_HALF_SIZE
  dx = abs(float(cube_final[0] - tray_center[0]))
  dy = abs(float(cube_final[1] - tray_center[1]))

  assert dx <= margin_lim, (
      f"Red cube X outside tray interior: dx={dx:.4f} > {margin_lim:.4f}"
  )
  assert dy <= margin_lim, (
      f"Red cube Y outside tray interior: dy={dy:.4f} > {margin_lim:.4f}"
  )
  assert 0.015 <= cube_final[2] <= 0.030, (
      f"Red cube Z not resting on tray floor: z={cube_final[2]:.4f}"
  )
  assert cube_speed < 0.02, f"Red cube still moving: speed={cube_speed:.4f} m/s"
  assert cyl_drift < 0.01, (
      f"Blue cylinder disturbed during task: drift={cyl_drift:.4f} m"
  )

  return {
      "cyl_reach_err_m": cyl_reach_err,
      "cube_reach_err_m": cube_reach_err,
      "lifted_z_m": lifted_z,
      "cube_final_xyz": cube_final.tolist(),
      "cube_speed_mps": cube_speed,
      "cyl_drift_m": cyl_drift,
  }


def main() -> None:
  """Builds scene, validates reachability, runs pick-and-place, and renders."""
  spec = sb.build_scene_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  ik_solutions = validate_reachability(model, data)
  summary = run_pick_and_place(
      model, data, ik_solutions, render_dir=sb.PROJECT_DIR / "renders"
  )
  print("\n=== Pick-and-Place Simulation Verified ===")
  print(f"  Blue cylinder reach err : {summary['cyl_reach_err_m'] * 1000:.2f} mm")
  print(f"  Red cube reach err      : {summary['cube_reach_err_m'] * 1000:.2f} mm")
  print(f"  Red cube peak lift Z    : {summary['lifted_z_m']:.4f} m")
  print(f"  Red cube final XYZ      : {np.round(summary['cube_final_xyz'], 4)}")
  print(f"  Red cube final speed    : {summary['cube_speed_mps']:.5f} m/s")
  print(f"  Blue cylinder drift     : {summary['cyl_drift_m'] * 1000:.2f} mm")
  print(f"  Summary render saved to : {sb.PROJECT_DIR / 'renders' / 'pick_and_place_summary.png'}")


if __name__ == "__main__":
  main()
