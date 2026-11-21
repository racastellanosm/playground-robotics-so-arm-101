"""Shared message definitions between sim_plugin.py and viewer_plugin.py."""

import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class DemoCommand(messages.Event):
  """UI -> sim: trigger or reset the SO-ARM101 pick-and-place sequence."""

  action: str = "run"  # "run" or "reset"


@dataclasses.dataclass(frozen=True)
class BakeEvent(messages.Event):
  """UI -> sim: write the current scene to scene.xml and switch to it."""


@dataclasses.dataclass(frozen=True)
class DemoTelemetry(messages.Snapshot):
  """sim -> UI: latest-wins status of the pick-and-place task."""

  phase: str = "Ready"
  sim_time: float = 0.0
  cube_z: float = 0.015
  cube_in_tray: bool = False
