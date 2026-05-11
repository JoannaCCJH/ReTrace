"""Hardcoded constants for the sim/ package.

GRID_XY: 400 grid start positions matching TraceGen's inference-time grid.
See test_helpers.prepare_grid_queries_with_depth / create_uniform_grid_points.
Mismatch would silently scramble trace selection.
"""
import numpy as np

GRID_SIZE = 20
TRAJECTORY_HORIZON = 32

# Image size used at inference: matches cfg/train.yaml model.vision_encoder.image_size.
# The training grid is created in pixel coords as linspace(0, H-1, GRID_SIZE) then
# normalized by H -> linspace(0, (IMAGE_SIZE - 1) / IMAGE_SIZE, GRID_SIZE).
IMAGE_SIZE = 384

_linspace = np.linspace(0.0, (IMAGE_SIZE - 1) / IMAGE_SIZE, GRID_SIZE, dtype=np.float64)
_xx, _yy = np.meshgrid(_linspace, _linspace, indexing="xy")
GRID_XY = np.stack([_xx.ravel(), _yy.ravel()], axis=-1).astype(np.float32)

TRACE_MOTION_THRESHOLD = 0.002
TRACE_WARN_DISTANCE = 0.15

# Close-gate is xy-separated from z so it fires for top-down grasps where the
# EE physically contacts the object from above and can't descend further. The
# old 3D-radius gate was structurally unreachable: with xy~2 cm and z~7 cm
# (gripper resting on the mug rim), 3D = 7.3 cm, never < 4 cm.
GRIPPER_CLOSE_XY_RADIUS = 0.04
# z_above range = ee_z - obj_z. Allow a small negative slack for contact-
# induced overshoot, and up to 10 cm above to permit grasps before the
# gripper actually touches the object. Marginal grasps are OK -- the
# post-grasp hold+lift in the rollout loop firms up the grip before lateral
# motion starts. Tightening this without loosening dz_scale leaves a window
# where the trace's xy/z minima don't co-occur and the gate never fires.
GRIPPER_CLOSE_Z_MIN = -0.02
GRIPPER_CLOSE_Z_MAX = 0.08
# After the close-gate fires, hold the EE at the grasp pose for HOLD_STEPS so
# the gripper has time to finish closing before any lateral shear. Subsequent
# dense_targets are shifted backward by HOLD_STEPS so the trajectory's shape is
# preserved -- only the last HOLD_STEPS planned steps fall off the horizon end.
POST_GRASP_HOLD_STEPS = 10
# Tightened from 0.12 to 0.05: with 12 cm slack the EE could satisfy the
# basket gate while sweeping past the goal en route to a trace endpoint
# beyond it (release-on-pass-by drop). 5 cm forces the EE to actually be
# over the goal region before the gripper opens; we observed min |xy|
# < 4 cm achievable on the white-mug task, so this should still trigger.
GOAL_XY_RADIUS = 0.05
GOAL_Z_CLEARANCE = 0.08
# Height (above the goal box top) at which the descend-before-release mode
# parks the EE before the gripper opens. 5 cm = short freefall, mug lands in
# region. Used only when --placement_mode descend.
GOAL_Z_PLACE_OFFSET = 0.05
# Object-to-goal-center xy distance below which the descend latch fires. Set
# wider than the region's half-extents so the descent begins *before* the mug
# enters the region's xy footprint, giving the OSC controller (5 cm/step max)
# 10-20 steps to lower the EE from carry altitude to placement altitude.
# Tighter than this and release fires before the EE has time to descend;
# wider and the EE descends prematurely while still en route.
PLACE_LATCH_RADIUS = 0.15

EPISODE_HORIZON = 500
