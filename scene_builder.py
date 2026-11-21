"""Procedural MuJoCo MjSpec builder for the SO-ARM101 tabletop pick-and-place scene."""

from __future__ import annotations

from pathlib import Path
import mujoco

PROJECT_DIR = Path(__file__).resolve().parent
SO101_XML_PATH = PROJECT_DIR / "assets" / "so101" / "so101.xml"

ROBOT_PREFIX = "so101_"
PINCH_SITE_NAME = f"{ROBOT_PREFIX}pinch_site"
ARM_JOINT_NAMES = (
    f"{ROBOT_PREFIX}shoulder_pan",
    f"{ROBOT_PREFIX}shoulder_lift",
    f"{ROBOT_PREFIX}elbow_flex",
    f"{ROBOT_PREFIX}wrist_flex",
    f"{ROBOT_PREFIX}wrist_roll",
)
GRIPPER_JOINT_NAME = f"{ROBOT_PREFIX}gripper"
ARM_ACTUATOR_NAMES = ARM_JOINT_NAMES
GRIPPER_ACTUATOR_NAME = GRIPPER_JOINT_NAME

GRIPPER_OPEN_CTRL = 0.65
GRIPPER_CLOSED_CTRL = 0.02
HOME_ARM_QPOS = (0.0, -0.78, 0.88, 0.68, 0.0)

TABLE_TOP_Z = 0.0
FLOOR_Z = -0.75

RED_CUBE_HALF_SIZE = 0.015
RED_CUBE_POS = (0.20, -0.11, TABLE_TOP_Z + RED_CUBE_HALF_SIZE)

BLUE_CYLINDER_RADIUS = 0.015
BLUE_CYLINDER_HALF_HEIGHT = 0.020
BLUE_CYLINDER_POS = (0.20, 0.11, TABLE_TOP_Z + BLUE_CYLINDER_HALF_HEIGHT)

GREEN_TRAY_POS = (0.26, 0.00, TABLE_TOP_Z)
GREEN_TRAY_BOTTOM_THICKNESS = 0.004
GREEN_TRAY_WALL_THICKNESS = 0.006
GREEN_TRAY_WALL_HEIGHT = 0.018
GREEN_TRAY_INNER_HALF_XY = 0.049


def _add_scene_defaults(
    spec: mujoco.MjSpec,
) -> tuple[mujoco._specs.MjsDefault, mujoco._specs.MjsDefault]:
  """Defines non-colliding visual and semi-transparent collision default classes."""
  scene_def = spec.add_default("scene_objects", None)

  vis_def = spec.add_default("scene_visual", scene_def)
  vis_def.geom.group = 2
  vis_def.geom.contype = 0
  vis_def.geom.conaffinity = 0
  vis_def.geom.density = 0.0
  vis_def.geom.mass = 0.0

  col_def = spec.add_default("scene_collision", scene_def)
  col_def.geom.group = 3
  col_def.geom.contype = 1
  col_def.geom.conaffinity = 1
  col_def.geom.condim = 4
  col_def.geom.friction = [1.0, 0.02, 0.005]
  col_def.geom.solref = [0.005, 1.0]
  col_def.geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
  return vis_def, col_def


def _add_environment_assets(spec: mujoco.MjSpec) -> None:
  """Adds skybox, studio lighting, floor, materials, and cameras."""
  spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
  spec.visual.headlight.ambient = [0.20, 0.20, 0.20]
  spec.visual.headlight.specular = [0.10, 0.10, 0.10]
  spec.visual.rgba.haze = [0.15, 0.22, 0.30, 1.0]
  spec.visual.global_.azimuth = 135.0
  spec.visual.global_.elevation = -28.0
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 720

  spec.add_texture(
      type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
      rgb1=[0.30, 0.45, 0.65],
      rgb2=[0.06, 0.09, 0.14],
      width=512,
      height=3072,
  )
  spec.add_texture(
      name="floor_checker",
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      mark=mujoco.mjtMark.mjMARK_EDGE,
      rgb1=[0.20, 0.24, 0.28],
      rgb2=[0.14, 0.17, 0.21],
      markrgb=[0.35, 0.40, 0.45],
      width=300,
      height=300,
  )
  floor_mat = spec.add_material(
      name="floor_material",
      texrepeat=[8.0, 8.0],
      texuniform=True,
      reflectance=0.08,
  )
  floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_checker"

  spec.add_material(
      name="table_wood_material",
      rgba=[0.36, 0.28, 0.21, 1.0],
      roughness=0.70,
      metallic=0.0,
      specular=0.10,
  )
  spec.add_material(
      name="table_leg_material",
      rgba=[0.20, 0.22, 0.25, 1.0],
      roughness=0.45,
      metallic=0.3,
  )
  spec.add_material(
      name="red_cube_material",
      rgba=[0.88, 0.18, 0.16, 1.0],
      roughness=0.45,
      metallic=0.0,
  )
  spec.add_material(
      name="blue_cylinder_material",
      rgba=[0.16, 0.42, 0.88, 1.0],
      roughness=0.42,
      metallic=0.0,
  )
  spec.add_material(
      name="green_tray_material",
      rgba=[0.16, 0.66, 0.30, 1.0],
      roughness=0.55,
      metallic=0.0,
  )
  spec.add_material(
      name="obstacle_material",
      rgba=[0.92, 0.52, 0.14, 1.0],
      roughness=0.50,
      metallic=0.0,
  )

  world = spec.worldbody
  world.add_light(
      name="key_light",
      pos=[0.6, -0.6, 1.6],
      dir=[-0.3, 0.3, -1.0],
      diffuse=[0.65, 0.65, 0.62],
      specular=[0.18, 0.18, 0.18],
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      castshadow=True,
  )
  world.add_light(
      name="fill_light",
      pos=[-0.5, 0.7, 1.4],
      dir=[0.3, -0.3, -1.0],
      diffuse=[0.28, 0.30, 0.34],
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      castshadow=False,
  )
  world.add_geom(
      name="room_floor",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      pos=[0.0, 0.0, FLOOR_Z],
      size=[0.0, 0.0, 0.05],
      material="floor_material",
  )

  world.add_camera(
      name="overview_cam",
      pos=[0.52, -0.44, 0.40],
      xyaxes=[0.65, 0.76, 0.0, -0.45, 0.38, 0.81],
      fovy=45.0,
  )
  world.add_camera(
      name="top_cam",
      pos=[0.18, 0.0, 0.62],
      xyaxes=[0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
      fovy=48.0,
  )


def _add_table(
    spec: mujoco.MjSpec,
    vis_def: mujoco._specs.MjsDefault,
    col_def: mujoco._specs.MjsDefault,
) -> None:
  """Adds a static tabletop centered under the robot workspace with top at z=0."""
  table = spec.worldbody.add_body(name="work_table", pos=[0.0, 0.0, 0.0])
  slab_cx = 0.20
  slab_hx, slab_hy, slab_hz = 0.35, 0.42, 0.02
  slab_pos = [slab_cx, 0.0, TABLE_TOP_Z - slab_hz]

  table.add_geom(
      name="table_top_visual",
      default=vis_def,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=slab_pos,
      size=[slab_hx, slab_hy, slab_hz],
      material="table_wood_material",
  )
  table.add_geom(
      name="table_top_col",
      default=col_def,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=slab_pos,
      size=[slab_hx, slab_hy, slab_hz],
      mass=12.0,
      friction=[0.8, 0.02, 0.005],
      rgba=[0.20, 0.75, 0.35, 0.45],
  )

  leg_height = abs(FLOOR_Z) - 2.0 * slab_hz
  leg_hz = 0.5 * leg_height
  leg_z = FLOOR_Z + leg_hz
  for idx, (lx, ly) in enumerate((
      (slab_cx - slab_hx + 0.04, -slab_hy + 0.04),
      (slab_cx + slab_hx - 0.04, -slab_hy + 0.04),
      (slab_cx - slab_hx + 0.04, slab_hy - 0.04),
      (slab_cx + slab_hx - 0.04, slab_hy - 0.04),
  )):
    table.add_geom(
        name=f"table_leg_{idx}_visual",
        default=vis_def,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[lx, ly, leg_z],
        size=[0.025, 0.025, leg_hz],
        material="table_leg_material",
    )
    table.add_geom(
        name=f"table_leg_{idx}_col",
        default=col_def,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[lx, ly, leg_z],
        size=[0.025, 0.025, leg_hz],
        mass=2.0,
        rgba=[0.20, 0.75, 0.35, 0.40],
    )


def _attach_so101_robot(
    spec: mujoco.MjSpec,
    *,
    gripper_friction: tuple[float, float, float] = (1.2, 0.02, 0.005),
) -> None:
  """Attaches the Menagerie SO-101 arm onto the table and configures grasp pads."""
  robot_spec = mujoco.MjSpec.from_file(str(SO101_XML_PATH))
  robot_spec.option.timestep = spec.option.timestep
  robot_spec.option.iterations = spec.option.iterations
  robot_spec.option.ls_iterations = spec.option.ls_iterations
  robot_spec.option.integrator = spec.option.integrator
  robot_spec.option.cone = spec.option.cone
  robot_spec.option.impratio = spec.option.impratio
  robot_spec.option.noslip_iterations = spec.option.noslip_iterations

  # Tune jaw pad friction for reliable grasping and add a centered pinch site
  # with clearance from the fixed jaw during vertical descent.
  gripper_body = robot_spec.body("gripper")
  gripper_body.add_site(
      name="pinch_site",
      group=4,
      pos=[0.014, 0.0, -0.096],
      quat=[1.0, 0.0, 1.0, 0.0],
      size=[0.005, 0.005, 0.005],
      rgba=[0.95, 0.25, 0.25, 0.45],
  )

  for geom in robot_spec.geoms:
    if geom.classname and geom.classname.name in (
        "collision_gripper",
        "collision_gripper_mesh",
    ):
      geom.condim = 4
      geom.friction = list(gripper_friction)
      geom.solref = [0.005, 1.0]
      geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]

  mount_frame = spec.worldbody.add_frame(
      name="so101_mount",
      pos=[0.0, 0.0, TABLE_TOP_Z],
      quat=[1.0, 0.0, 0.0, 0.0],
  )
  spec.attach(robot_spec, frame=mount_frame, prefix=ROBOT_PREFIX)


def _add_red_cube(
    spec: mujoco.MjSpec,
    vis_def: mujoco._specs.MjsDefault,
    col_def: mujoco._specs.MjsDefault,
    pos: tuple[float, float, float],
    friction: tuple[float, float, float] = (1.2, 0.02, 0.005),
) -> None:
  """Adds the movable red cube inside the arm's reachable workspace."""
  cube = spec.worldbody.add_body(name="red_cube", pos=list(pos))
  cube.add_freejoint(name="red_cube_free")
  half = [RED_CUBE_HALF_SIZE, RED_CUBE_HALF_SIZE, RED_CUBE_HALF_SIZE]
  cube.add_geom(
      name="red_cube_visual",
      default=vis_def,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      material="red_cube_material",
  )
  cube.add_geom(
      name="red_cube_col",
      default=col_def,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      mass=0.035,
      condim=4,
      friction=list(friction),
      rgba=[0.90, 0.25, 0.20, 0.50],
  )


def _add_blue_cylinder(
    spec: mujoco.MjSpec,
    vis_def: mujoco._specs.MjsDefault,
    col_def: mujoco._specs.MjsDefault,
    pos: tuple[float, float, float],
) -> None:
  """Adds the movable blue cylinder inside the arm's reachable workspace."""
  cyl = spec.worldbody.add_body(name="blue_cylinder", pos=list(pos))
  cyl.add_freejoint(name="blue_cylinder_free")
  size = [BLUE_CYLINDER_RADIUS, BLUE_CYLINDER_HALF_HEIGHT, 0.0]
  cyl.add_geom(
      name="blue_cylinder_visual",
      default=vis_def,
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=size,
      material="blue_cylinder_material",
  )
  cyl.add_geom(
      name="blue_cylinder_col",
      default=col_def,
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=size,
      mass=0.040,
      condim=6,
      friction=[1.0, 0.02, 0.005],
      rgba=[0.20, 0.50, 0.95, 0.50],
  )


def _add_green_tray(
    spec: mujoco.MjSpec,
    vis_def: mujoco._specs.MjsDefault,
    col_def: mujoco._specs.MjsDefault,
    pos: tuple[float, float, float],
) -> None:
  """Adds a hollow open-top green target tray resting on the table."""
  tray = spec.worldbody.add_body(name="green_tray", pos=list(pos))
  bt = GREEN_TRAY_BOTTOM_THICKNESS
  wt = GREEN_TRAY_WALL_THICKNESS
  wh = GREEN_TRAY_WALL_HEIGHT
  ih = GREEN_TRAY_INNER_HALF_XY
  oh = ih + wt

  wall_hz = 0.5 * (wh - bt)
  wall_z = bt + wall_hz
  parts = (
      ("bottom", [0.0, 0.0, 0.5 * bt], [oh, oh, 0.5 * bt], 0.16,
       [0.20, 0.75, 0.35, 0.50]),
      ("wall_front", [ih + 0.5 * wt, 0.0, wall_z], [0.5 * wt, oh, wall_hz],
       0.04, [0.25, 0.55, 0.90, 0.45]),
      ("wall_back", [-(ih + 0.5 * wt), 0.0, wall_z], [0.5 * wt, oh, wall_hz],
       0.04, [0.25, 0.55, 0.90, 0.45]),
      ("wall_left", [0.0, ih + 0.5 * wt, wall_z], [ih, 0.5 * wt, wall_hz],
       0.04, [0.25, 0.55, 0.90, 0.45]),
      ("wall_right", [0.0, -(ih + 0.5 * wt), wall_z], [ih, 0.5 * wt, wall_hz],
       0.04, [0.25, 0.55, 0.90, 0.45]),
  )
  for suffix, p, s, mass, col_rgba in parts:
    tray.add_geom(
        name=f"green_tray_{suffix}_visual",
        default=vis_def,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=p,
        size=s,
        material="green_tray_material",
    )
    tray.add_geom(
        name=f"green_tray_{suffix}_col",
        default=col_def,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=p,
        size=s,
        mass=mass,
        friction=[0.9, 0.02, 0.005],
        rgba=col_rgba,
    )

  tray.add_site(
      name="green_tray_target_site",
      group=4,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[0.0, 0.0, bt + 0.5 * (wh - bt)],
      size=[ih, ih, 0.5 * (wh - bt)],
      rgba=[0.15, 0.80, 0.85, 0.30],
  )


def _add_obstacle(
    spec: mujoco.MjSpec,
    vis_def: mujoco._specs.MjsDefault,
    col_def: mujoco._specs.MjsDefault,
    obstacle: dict[str, object],
) -> None:
  """Adds a static box obstacle on the tabletop for transit clearance checks."""
  pos = [float(v) for v in obstacle["pos"]]
  size = [float(v) for v in obstacle["size"]]
  body = spec.worldbody.add_body(name="transit_obstacle", pos=pos)
  body.add_geom(
      name="transit_obstacle_visual",
      default=vis_def,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      material="obstacle_material",
  )
  body.add_geom(
      name="transit_obstacle_col",
      default=col_def,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      mass=0.25,
      rgba=[0.95, 0.55, 0.15, 0.50],
  )


def build_scene_spec(
    *,
    cube_pos: tuple[float, float, float] = RED_CUBE_POS,
    cylinder_pos: tuple[float, float, float] = BLUE_CYLINDER_POS,
    tray_pos: tuple[float, float, float] = GREEN_TRAY_POS,
    cube_friction: tuple[float, float, float] = (1.2, 0.02, 0.005),
    gripper_friction: tuple[float, float, float] = (1.2, 0.02, 0.005),
    obstacle: dict[str, object] | None = None,
) -> mujoco.MjSpec:
  """Builds the complete SO-ARM101 tabletop pick-and-place MjSpec."""
  spec = mujoco.MjSpec()
  spec.modelname = "so101_tabletop_pick_and_place"
  spec.compiler.degree = False
  spec.compiler.autolimits = True
  spec.compiler.meshdir = "assets/so101/assets"

  spec.option.timestep = 0.002
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10.0
  spec.option.noslip_iterations = 3

  vis_def, col_def = _add_scene_defaults(spec)
  _add_environment_assets(spec)
  _add_table(spec, vis_def, col_def)
  _attach_so101_robot(spec, gripper_friction=gripper_friction)
  _add_red_cube(spec, vis_def, col_def, cube_pos, friction=cube_friction)
  _add_blue_cylinder(spec, vis_def, col_def, cylinder_pos)
  _add_green_tray(spec, vis_def, col_def, tray_pos)
  if obstacle is not None:
    _add_obstacle(spec, vis_def, col_def, obstacle)
  return spec
