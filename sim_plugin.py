"""Simulation-side Studio plugin for the SO-ARM101 pick-and-place scene."""

from __future__ import annotations

import os
import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

import pick_and_place_verify as ppv
import scene_builder as sb

SCENE_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scene.xml"
)


class SimPlugin(plugin.SimPlugin):
  """Builds the SO-ARM101 tabletop scene and drives the pick-and-place demo."""

  spec: mujoco.MjSpec | None = None
  phase: str = "Ready"
  _needs_home_init: bool = True
  _pending_action: str | None = None
  _running: bool = False
  _stages: list[tuple[str, np.ndarray, float, float]] = []
  _stage_idx: int = 0
  _stage_elapsed: float = 0.0
  _start_ctrl: np.ndarray | None = None

  def _build_stages(
      self, model: mujoco.MjModel, data: mujoco.MjData
  ) -> list[tuple[str, np.ndarray, float, float]]:
    """Computes IK waypoints for reaching both objects and placing red cube."""
    qpos_idx, dof_idx, q_min, q_max = ppv.get_arm_indices(model)
    site_id = int(model.site(sb.PINCH_SITE_NAME).id)
    ctx_qpos = data.qpos.copy()

    cube_xyz = data.body("red_cube").xpos.copy()
    cyl_xyz = data.body("blue_cylinder").xpos.copy()
    tray_xyz = data.body("green_tray").xpos.copy()

    targets = (
        ("blue_cyl_hover", [cyl_xyz[0], cyl_xyz[1], ppv.HOVER_Z]),
        ("blue_cyl_reach", [cyl_xyz[0], cyl_xyz[1], cyl_xyz[2]]),
        ("red_cube_hover", [cube_xyz[0], cube_xyz[1], ppv.HOVER_Z]),
        ("red_cube_grasp", [cube_xyz[0], cube_xyz[1], cube_xyz[2]]),
        ("green_tray_hover", [tray_xyz[0], tray_xyz[1], ppv.HOVER_Z]),
        ("green_tray_drop", [tray_xyz[0], tray_xyz[1], ppv.DROP_Z]),
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

    q_home = np.array(sb.HOME_ARM_QPOS, dtype=np.float64)
    g_open = sb.GRIPPER_OPEN_CTRL
    g_closed = sb.GRIPPER_CLOSED_CTRL
    return [
        ("Reach blue cylinder (hover)", solved["blue_cyl_hover"], g_open, 0.8),
        ("Reach blue cylinder (check)", solved["blue_cyl_reach"], g_open, 0.6),
        ("Retract from blue cylinder", solved["blue_cyl_hover"], g_open, 0.6),
        ("Approach red cube (hover)", solved["red_cube_hover"], g_open, 0.9),
        ("Descend to red cube", solved["red_cube_grasp"], g_open, 0.6),
        ("Close gripper on red cube", solved["red_cube_grasp"], g_closed, 0.45),
        ("Hold grasp", solved["red_cube_grasp"], g_closed, 0.35),
        ("Lift red cube", solved["red_cube_hover"], g_closed, 0.8),
        ("Move over green tray", solved["green_tray_hover"], g_closed, 0.9),
        ("Lower into green tray", solved["green_tray_drop"], g_closed, 0.6),
        ("Release red cube", solved["green_tray_drop"], g_open, 0.40),
        ("Retract above tray", solved["green_tray_hover"], g_open, 0.6),
        ("Return home", q_home, g_open, 0.8),
    ]

  @messages.handler
  def on_load_model(self, event: messages.LoadModelEvent) -> bool:
    if event.path:
      return False
    self.spec = sb.build_scene_spec()
    self.handle.set_model(
        self.spec,
        watch=[
            str(sb.SO101_XML_PATH),
            str(sb.PROJECT_DIR / "scene_builder.py"),
        ],
    )
    return True

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    if self._needs_home_init:
      self._needs_home_init = False
      ppv.reset_to_home(event.model, event.data)
      self.phase = "Ready"
      self._running = False

  @messages.handler
  def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    if self in event.plugins:
      self.handle.reload()

  @messages.handler
  def on_demo_command(self, event: user_messages.DemoCommand) -> None:
    self._pending_action = event.action

  @messages.handler
  def on_bake(self, event: user_messages.BakeEvent) -> None:
    del event
    if self.spec is None:
      return
    with open(SCENE_XML, "w", encoding="utf-8") as f:
      f.write(self.spec.to_xml())
    self.handle.dispatch(messages.LoadModelEvent(path=SCENE_XML))

  @messages.handler
  def on_pre_step(self, event: messages.PreStepEvent) -> None:
    model, data = event.model, event.data
    if self._pending_action is not None:
      action = self._pending_action
      self._pending_action = None
      ppv.reset_to_home(model, data)
      self._stage_idx = 0
      self._stage_elapsed = 0.0
      self._start_ctrl = data.ctrl.copy()
      if action == "run":
        self._stages = self._build_stages(model, data)
        self._running = True
        self.phase = self._stages[0][0]
      else:
        self._stages = []
        self._running = False
        self.phase = "Ready"

    if not self._running or not self._stages:
      return

    label, q_goal, g_goal, duration_s = self._stages[self._stage_idx]
    self.phase = label
    if self._start_ctrl is None:
      self._start_ctrl = data.ctrl.copy()

    self._stage_elapsed += float(model.opt.timestep)
    frac = min(1.0, self._stage_elapsed / max(duration_s, 1e-4))
    alpha = 0.5 * (1.0 - np.cos(np.pi * frac))
    target = np.array([*q_goal, g_goal], dtype=np.float64)
    data.ctrl[:] = self._start_ctrl + alpha * (target - self._start_ctrl)

    if frac >= 1.0:
      self._stage_idx += 1
      self._stage_elapsed = 0.0
      self._start_ctrl = target.copy()
      if self._stage_idx >= len(self._stages):
        self._running = False
        self.phase = "Complete"

  @messages.handler
  def on_post_step(self, event: messages.PostStepEvent) -> None:
    data = event.data
    cube_pos = data.body("red_cube").xpos.copy()
    tray_pos = data.body("green_tray").xpos.copy()
    lim = sb.GREEN_TRAY_INNER_HALF_XY - sb.RED_CUBE_HALF_SIZE
    in_tray = (
        abs(float(cube_pos[0] - tray_pos[0])) <= lim
        and abs(float(cube_pos[1] - tray_pos[1])) <= lim
        and 0.015 <= float(cube_pos[2]) <= 0.030
    )
    self.handle.send_to_viewer(
        user_messages.DemoTelemetry(
            phase=self.phase,
            sim_time=float(data.time),
            cube_z=float(cube_pos[2]),
            cube_in_tray=in_tray,
        )
    )
