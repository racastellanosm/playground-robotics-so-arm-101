"""Standalone MuJoCo Studio WebViewer server for the SO-ARM101 simulation."""

from __future__ import annotations

import os
from mujoco.studio import app as studio_app
from mujoco.studio import launch_web
from mujoco.studio import viewer_protocol

import sim_plugin
import viewer_plugin


def main() -> None:
  """Starts the MuJoCo Studio WebViewer server with SO-ARM101 plugins."""
  port = int(os.environ.get("STUDIO_PORT", "8080"))
  hot_reload = os.environ.get("STUDIO_HOT_RELOAD", "")
  config = viewer_protocol.ViewerConfig(
      title="SO-ARM101 MuJoCo Studio",
      gfx="web",
      http_port=port,
      open_browser=False,
      hot_reload=hot_reload,
  )
  launch_web.run(
      config,
      plugins=[
          studio_app.App(),
          viewer_plugin.ViewerPlugin(),
          sim_plugin.SimPlugin(),
          *studio_app.SIM_PLUGINS,
      ],
  )


if __name__ == "__main__":
  main()
