"""Two-state gripper policy from privileged sim state."""
import numpy as np

from sim.constants import (
    GOAL_XY_RADIUS,
    GOAL_Z_CLEARANCE,
    GRIPPER_CLOSE_XY_RADIUS,
    GRIPPER_CLOSE_Z_MAX,
    GRIPPER_CLOSE_Z_MIN,
)

OPEN = "OPEN"
CLOSED_ON_OBJECT = "CLOSED_ON_OBJECT"
OPEN_AT_GOAL = "OPEN_AT_GOAL"


class GripperOracle:
    """Three-phase gripper state machine.

    Phase 1 (OPEN): approach object. Close when the gripper is over the object
        in xy (< GRIPPER_CLOSE_XY_RADIUS) AND vertically aligned within
        [GRIPPER_CLOSE_Z_MIN, GRIPPER_CLOSE_Z_MAX] (ee_z - obj_z). Splitting xy
        from z handles top-down grasps where a single 3D radius is structurally
        unreachable (gripper resting on the object has xy ~ 0 but z ~ 7 cm).

    Phase 2 (CLOSED_ON_OBJECT): carry object toward goal. Two release paths:
      - Closest-approach: while the object is inside the region's xy footprint
        AND above its top face, track the running min xy distance to goal
        center; release when the next sample is farther (we just passed the
        most-centered point). Drops the object as close to center as the
        trace allows.
      - Exit-fallback: if the object was ever inside-and-above but is no
        longer, release immediately. Catches the case where the trace passes
        through the region with monotonically decreasing xy_dist (so the
        closest-approach trigger never fires before exiting).

      Two gating regimes for inside/above:
        - Region goal (goal_half_extents provided): inside = box footprint
          (|Δx| < half_x and |Δy| < half_y); above = obj.z > goal.z + half_z
          + GOAL_Z_CLEARANCE.
        - Body goal (goal_half_extents = None): legacy circle gate (xy_dist
          < GOAL_XY_RADIUS, obj.z > goal.z + GOAL_Z_CLEARANCE).

    Phase 3 (OPEN_AT_GOAL): release and stay open.

    Returns +1 for close, -1 for open (robosuite/LIBERO convention).
    """

    def __init__(self):
        self.state = OPEN
        # Closest-approach tracker: running min xy_dist *while* the object is
        # inside-and-above. Release on the next sample farther than this.
        self._min_xy_dist = float("inf")
        # Exit-fallback flag: the object was inside-and-above on some prior
        # step. If we ever see inside-and-above flip back to False, release --
        # the trace already swept past the optimum, no point holding on.
        self._touched_region = False
        # Per-trial diagnostics (read by rollout.py after the loop).
        self.diag = {
            "ever_inside_xy": False,
            "ever_above": False,
            "ever_inside_and_above": False,
            "min_xy_dist_inside_above": float("inf"),
            "released_via": None,  # 'closest_approach' | 'exit_fallback' | None
        }

    def step(self, ee_pos: np.ndarray, object_pos: np.ndarray,
             goal_pos: np.ndarray,
             goal_half_extents: np.ndarray | None = None) -> int:
        if self.state == OPEN:
            xy = np.linalg.norm(ee_pos[:2] - object_pos[:2])
            z_above = ee_pos[2] - object_pos[2]
            if (xy < GRIPPER_CLOSE_XY_RADIUS
                    and GRIPPER_CLOSE_Z_MIN <= z_above <= GRIPPER_CLOSE_Z_MAX):
                self.state = CLOSED_ON_OBJECT
        elif self.state == CLOSED_ON_OBJECT:
            if goal_half_extents is not None:
                inside_xy = (
                    abs(object_pos[0] - goal_pos[0]) < float(goal_half_extents[0])
                    and abs(object_pos[1] - goal_pos[1]) < float(goal_half_extents[1])
                )
                z_top = float(goal_pos[2]) + float(goal_half_extents[2])
                above = (object_pos[2] - z_top) > GOAL_Z_CLEARANCE
            else:
                inside_xy = (np.linalg.norm(object_pos[:2] - goal_pos[:2])
                             < GOAL_XY_RADIUS)
                above = (object_pos[2] - goal_pos[2]) > GOAL_Z_CLEARANCE

            if inside_xy:
                self.diag["ever_inside_xy"] = True
            if above:
                self.diag["ever_above"] = True

            xy_dist = float(np.linalg.norm(object_pos[:2] - goal_pos[:2]))
            if inside_xy and above:
                self.diag["ever_inside_and_above"] = True
                if xy_dist > self._min_xy_dist:
                    self.state = OPEN_AT_GOAL
                    self.diag["released_via"] = "closest_approach"
                else:
                    self._min_xy_dist = xy_dist
                    self.diag["min_xy_dist_inside_above"] = xy_dist
                self._touched_region = True
            elif self._touched_region:
                # Were inside-above; now we're not. Release at the boundary --
                # we already passed the optimum.
                self.state = OPEN_AT_GOAL
                self.diag["released_via"] = "exit_fallback"
        return 1 if self.state == CLOSED_ON_OBJECT else -1
