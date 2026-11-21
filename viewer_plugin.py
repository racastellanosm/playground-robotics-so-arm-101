"""Viewer-side Dear ImGui Studio plugin for the SO-ARM101 pick-and-place demo."""

from __future__ import annotations

import os
import pathlib

import messages as user_messages
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins

imgui = app_plugins.imgui
WORKSPACE_DIR = pathlib.Path(os.path.abspath(__file__)).parent


class ViewerPlugin(plugin.ViewerPlugin):
  """Interactive controls and telemetry for SO-ARM101 pick-and-place."""

  telemetry: user_messages.DemoTelemetry = user_messages.DemoTelemetry()
  window_open: bool = True

  @messages.handler
  def on_telemetry(self, snapshot: user_messages.DemoTelemetry) -> None:
    self.telemetry = snapshot

  @messages.handler
  def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
    del event
    if not self.window_open:
      return
    expanded, self.window_open = imgui.Begin(
        "SO-ARM101 Pick & Place###So101Panel", self.window_open
    )
    try:
      if not expanded:
        return
      imgui.TextUnformatted(f"Workspace: {WORKSPACE_DIR.name}")
      imgui.Separator()

      if imgui.Button("Run Pick & Place"):
        self.viewer.send_to_sim(user_messages.DemoCommand(action="run"))
      imgui.SameLine()
      if imgui.Button("Reset"):
        self.viewer.send_to_sim(user_messages.DemoCommand(action="reset"))

      t = self.telemetry
      imgui.TextUnformatted(f"Phase: {t.phase}")
      imgui.TextUnformatted(f"Sim time: {t.sim_time:.2f} s")
      imgui.TextUnformatted(f"Red cube Z: {t.cube_z:.4f} m")
      tray_txt = "YES" if t.cube_in_tray else "NO"
      imgui.TextUnformatted(f"In green tray: {tray_txt}")

      imgui.Separator()
      if imgui.Button("Bake to scene.xml"):
        self.viewer.send_to_sim(user_messages.BakeEvent())
      imgui.SetItemTooltip(
          "Write the current Python MjSpec scene to scene.xml in the workspace."
      )
    finally:
      imgui.End()
