"""Per-trial rollout loop + failure-mode classifier.

Public API:
    - `run_trial(env, predict_fn, seed, ...)`: original end-to-end entry
      point. Wraps build_initial_state + run_segment + finalize_trial.
    - `build_initial_state(...) / run_segment(...) / finalize_trial(...)`:
      segmented interface used by trigger/collect.py to fork rollouts at
      arbitrary t. run_segment can resume from any t and accepts an
      explicit replan schedule and an on_step callback (used by the
      collect driver to inject perturbations and capture snapshots).
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim.constants import (
    GOAL_XY_RADIUS,
    GOAL_Z_PLACE_OFFSET,
    GRID_XY,
    PLACE_LATCH_RADIUS,
    POST_GRASP_HOLD_STEPS,
)
from sim.gripper_oracle import GripperOracle
from sim.interp import linear_interp
from sim.trace_lifter import (
    lift_all_traces_to_camera_frame,
    lift_to_world,
    project_ee_to_pixel,
    project_world_to_pixel,
    reconstruct_absolute_trace,
)
from sim.trace_selector import select_top_k_traces, select_trace

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    success: bool
    steps: int
    failure_mode: str
    diag: dict = field(default_factory=dict)
    initial_rgb: np.ndarray = None


@dataclass
class RolloutState:
    """All per-step loop state needed to (a) continue stepping a rollout
    from any t, and (b) deep-copy/restore the rollout for counterfactual
    branching. Anything the inner loop in `run_segment` reads or writes
    lives here; nothing else does.
    """
    t: int                           # next step index to execute
    obs: Any                         # last observation (LiberoObs or stub)
    dense_targets: np.ndarray        # [horizon, 3], world frame
    sel_diag: dict                   # diag from initial plan_from_obs
    oracle: GripperOracle
    closed_at_t: int | None
    opened_at_t: int | None
    ee_at_close: np.ndarray | None
    obj_at_close: np.ndarray | None
    ee_at_open: np.ndarray | None
    basket_at_open: np.ndarray | None
    placing_latched: bool
    placing_started_at_t: int | None
    success: bool
    steps: int                       # final step count (set on termination)
    terminated: bool                 # True once success/done fired
    final_obs: Any
    initial_rgb: np.ndarray          # obs.rgb at t=0 (for TrialResult)
    logs: dict[str, list[float]]     # obj_z, ee_err, ee_obj, ee_obj_xy,
                                     #   ee_obj_z, ee_basket_xy, ee_basket_z
    rgb_frames: list[np.ndarray] | None  # None when viz_dir is None


def classify_failure_mode(success: bool,
                          object_z_trajectory: np.ndarray,
                          object_xy_final: np.ndarray,
                          basket_xy: np.ndarray,
                          ee_target_errors: np.ndarray,
                          lift_threshold_m: float = 0.02,
                          drop_threshold_m: float = 0.10,
                          basket_margin_m: float = GOAL_XY_RADIUS,
                          ee_err_threshold_m: float = 0.05,
                          ee_err_frac: float = 0.5) -> str:
    if success:
        return "success"
    # Check EE reachability first: if the robot never got near its targets,
    # no amount of downstream logic matters.
    if (ee_target_errors > ee_err_threshold_m).mean() > ee_err_frac:
        return "ee_unreachable"
    z0 = float(object_z_trajectory[0])
    z_max = float(object_z_trajectory.max())
    z_end = float(object_z_trajectory[-1])
    lifted = (z_max - z0) > lift_threshold_m
    dropped = lifted and (z_max - z_end) > drop_threshold_m
    at_basket = np.linalg.norm(object_xy_final - basket_xy) < basket_margin_m + 0.08
    if not lifted:
        return "grasp_miss"
    if dropped and not at_basket:
        return "dropped_in_transit"
    if not at_basket:
        return "missed_basket"
    return "timeout"


OSC_POS_SCALE = 0.05  # robosuite OSC_POSE output_max for position (meters/step).


def build_osc_action(target_pos: np.ndarray, gripper: int,
                     current_ee_pos: np.ndarray,
                     pos_scale: float = OSC_POS_SCALE) -> np.ndarray:
    """[dx, dy, dz, 0, 0, 0, gripper] in robosuite OSC_POSE convention.

    The OSC_POSE controller expects action components in [-1, 1] that get
    scaled to +/- pos_scale meters per step. A raw meter-delta would be
    squashed through the controller's affine input->output mapping; divide
    by pos_scale and clip so a commanded 1-step move saturates the limit.
    """
    delta = (target_pos - current_ee_pos).astype(np.float32) / pos_scale
    delta = np.clip(delta, -1.0, 1.0)
    return np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, float(gripper)],
                    dtype=np.float32)


def _inside_goal_xy(object_pos: np.ndarray, goal_pos: np.ndarray,
                    goal_half_extents: np.ndarray | None) -> bool:
    """Mirror of GripperOracle's inside_xy check, for the rollout-side
    placement override. Returns True iff the object's xy is inside the goal
    region's footprint (box for site goals, circle for body goals).
    """
    if goal_half_extents is not None:
        return bool(
            abs(object_pos[0] - goal_pos[0]) < float(goal_half_extents[0])
            and abs(object_pos[1] - goal_pos[1]) < float(goal_half_extents[1])
        )
    return bool(np.linalg.norm(object_pos[:2] - goal_pos[:2]) < GOAL_XY_RADIUS)


def plan_from_obs(obs, predict_fn: Callable, *,
                  dz_scale: float, n_steps: int,
                  viz_dir: Path | None) -> tuple[np.ndarray, dict]:
    """Run predict -> lift -> top-K -> world -> median -> anchor -> interp.

    Pure function of the current obs. Returns (dense_targets [n_steps, 3],
    sel_diag dict). When viz_dir is set, writes pred_traces.png /
    first_depth.png / selected_traces.png / world_trace_overlay.png /
    traces_3d.npz / combined_trace_world.npz there.

    obs.rgb / obs.depth are passed raw (OpenGL bottom-up). predict_fn's
    prepare_model_inputs handles the top-down flip internally -- callers must
    not pre-flip.
    """
    rgb_t, depth_t, pred = predict_fn(obs.rgb, obs.depth, obs.language)

    if viz_dir is not None:
        from sim.visualize_pred import (
            reconstruct_absolute_traces,
            save_depth_viz,
            save_overlay,
        )
        rgb_input = (rgb_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Hp, Wp = rgb_input.shape[:2]
        traces_px = reconstruct_absolute_traces(pred, img_h=Hp, img_w=Wp)
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(rgb_input, traces_px, viz_dir / "pred_traces.png")
        if depth_t is not None:
            save_depth_viz(depth_t[0].cpu().numpy(), viz_dir / "first_depth.png")

    # Lift every grid trace to camera-frame 2.5D (px_x, px_y, z_metric) using
    # the same recipe as test_example.py: cumsum xy with grid prepend, sample
    # z0 from depth at each trace's start pixel, cumsum [z0, dz...] along time.
    # Skip the sensored/predicted depth-ratio rescale -- LIBERO obs.depth is
    # already metric meters via get_real_depth_map. Falls back to flipped+
    # resized obs.depth when the model wasn't conditioned on depth (depth_t is
    # None) so this step is independent of use_depth.
    S = rgb_t.shape[-1]
    if depth_t is not None:
        depth_for_lift = depth_t[0].cpu().numpy()
    else:
        import torch
        import torchvision.transforms.functional as TF
        obs_depth_top = np.ascontiguousarray(obs.depth[::-1])
        depth_for_lift = TF.resize(
            torch.from_numpy(obs_depth_top).float()[None, None],
            [S, S], antialias=True,
        )[0, 0].numpy()
    traces_3d = lift_all_traces_to_camera_frame(
        pred, depth_for_lift, S, S, dz_scale=dz_scale,
    )
    logger.info(
        "traces_3d (dz_scale=%g) shape=%s, px_x=[%.1f,%.1f], px_y=[%.1f,%.1f], "
        "z=[%.4f,%.4f]",
        dz_scale, traces_3d.shape,
        float(traces_3d[..., 0].min()), float(traces_3d[..., 0].max()),
        float(traces_3d[..., 1].min()), float(traces_3d[..., 1].max()),
        float(traces_3d[..., 2].min()), float(traces_3d[..., 2].max()),
    )
    if viz_dir is not None:
        np.savez(viz_dir / "traces_3d.npz", traces_3d=traces_3d)

    H, W = obs.rgb.shape[:2]
    ee_pixel, ee_z_cam = project_ee_to_pixel(
        obs.ee_pose[:3], obs.K, obs.cam_to_world, img_shape=(H, W)
    )

    # Top-K trace selection on the lifted 3D bank, anchored at the EE pixel.
    # ee_pixel is normalized [0,1] in the (H, W) frame of obs.rgb; traces_3d
    # lives in the model-input pixel frame (S x S), so scale by S.
    ee_pixel_px = (ee_pixel * np.array([S, S], dtype=np.float32)).astype(np.float32)
    selected_3d, selected_idx, sel_diag = select_top_k_traces(
        traces_3d, ee_pixel_px, img_size=S, k=5, motion_threshold=0.1,
    )
    logger.info("top-K selection: %s", sel_diag)
    if viz_dir is not None:
        from sim.viz import save_top_k_traces_overlay
        save_top_k_traces_overlay(
            rgb_input, traces_3d[..., :2], selected_idx, ee_pixel_px,
            viz_dir / "selected_traces.png", motion_threshold=0.1,
        )

    # Lift each selected trace from 2.5D (S-frame px, S-frame px, metric z)
    # to world frame, then median-combine across the K samples. obs.K is built
    # for obs.rgb's (H, W) frame, so scale the pixel coords down from S before
    # back-projection. Median (per-timestep) is robust to a single outlier
    # trace; switch to mean / k-medoids later if multimodality matters.
    px_to_obs = np.array([W / S, H / S], dtype=np.float32)
    selected_world = np.empty_like(selected_3d)  # [K, 1+T, 3]
    for ki in range(selected_3d.shape[0]):
        trace_px_obs = selected_3d[ki, :, :2] * px_to_obs[None, :]
        selected_world[ki] = lift_to_world(
            trace_px_obs, obs.depth, obs.K, obs.cam_to_world,
            z_override=selected_3d[ki, :, 2],
        )
    trace_world = np.median(selected_world, axis=0).astype(np.float32)
    logger.info(
        "world combined: start=%s end=%s; K endpoint spread (std)=%.4f m",
        trace_world[0].tolist(), trace_world[-1].tolist(),
        float(np.linalg.norm(selected_world[:, -1].std(axis=0))),
    )
    if viz_dir is not None:
        np.savez(viz_dir / "combined_trace_world.npz",
                 selected_world=selected_world, combined=trace_world)

    # Convention sanity check + back-projection viz. If the world frame really
    # is the robot's, then (a) trace_world[0] should be near obs.ee_pose[:3]
    # since the closest grid cell sits on the gripper, and (b) projecting
    # trace_world back through obs.K / obs.cam_to_world should land on the
    # original 2.5D pixel path. Both are logged for inspection.
    trace_norm_px, _ = project_world_to_pixel(
        trace_world, obs.K, obs.cam_to_world, img_shape=(H, W)
    )
    trace_back_px = trace_norm_px * np.array([S, S], dtype=np.float32)
    ee_world_dist = float(np.linalg.norm(trace_world[0] - obs.ee_pose[:3]))
    ee_pixel_back_dist = float(np.linalg.norm(trace_back_px[0] - ee_pixel_px))
    logger.info(
        "world frame sanity: trace_world[0]=%s vs ee_pose=%s (|dx|=%.4f m); "
        "round-trip start px=%s vs ee_pixel_px=%s (|dx|=%.1f px)",
        np.round(trace_world[0], 4).tolist(),
        np.round(obs.ee_pose[:3], 4).tolist(),
        ee_world_dist,
        np.round(trace_back_px[0], 1).tolist(),
        np.round(ee_pixel_px, 1).tolist(),
        ee_pixel_back_dist,
    )
    if viz_dir is not None:
        from sim.viz import save_world_trace_overlay
        save_world_trace_overlay(
            rgb_input, trace_back_px, ee_pixel_px,
            viz_dir / "world_trace_overlay.png",
        )

    # Anchor the dense targets at the actual EE pose: trace_world[0] is the
    # closest grid cell's lifted position, which can sit a few cm from the
    # gripper, and feeding it as dense_targets[0] yanks the EE on the first
    # OSC step. Prepending obs.ee_pose[:3] makes the trajectory flow smoothly
    # from the real start (linear_interp tolerates the near-duplicate when the
    # grid cell already sits on the EE).
    trace_world_anchored = np.concatenate(
        [obs.ee_pose[:3][None, :].astype(np.float32), trace_world], axis=0
    )
    dense_targets = linear_interp(trace_world_anchored, N=n_steps)
    return dense_targets, sel_diag


def _validate_viz_replans(viz_replans: str) -> None:
    if viz_replans not in ("first", "all", "none"):
        raise ValueError(
            f"viz_replans must be 'first', 'all', or 'none', got {viz_replans!r}"
        )


def _validate_placement_mode(placement_mode: str) -> None:
    if placement_mode not in ("trace", "descend"):
        raise ValueError(
            f"placement_mode must be 'trace' or 'descend', got {placement_mode!r}"
        )


def _plan_viz_dir(t: int, viz_dir: Path | None, viz_replans: str) -> Path | None:
    if viz_dir is None or viz_replans == "none":
        return None
    if viz_replans == "first" and t != 0:
        return None
    return viz_dir / f"replan_{t:03d}"


def build_initial_state(env, predict_fn: Callable, seed: int, *,
                        dz_scale: float = 1.0,
                        placement_mode: str = "trace",
                        viz_replans: str = "all",
                        viz_dir: Path | None = None) -> RolloutState:
    """Reset the env, plan once at t=0, return a RolloutState ready for
    `run_segment` to step forward."""
    _validate_placement_mode(placement_mode)
    _validate_viz_replans(viz_replans)
    obs = env.reset(seed=seed)
    dense_targets, sel_diag = plan_from_obs(
        obs, predict_fn,
        dz_scale=dz_scale, n_steps=env.horizon,
        viz_dir=_plan_viz_dir(0, viz_dir, viz_replans),
    )
    rgb_frames = ([np.ascontiguousarray(obs.rgb[::-1])]
                  if viz_dir is not None else None)
    return RolloutState(
        t=0,
        obs=obs,
        dense_targets=dense_targets,
        sel_diag=sel_diag,
        oracle=GripperOracle(),
        closed_at_t=None,
        opened_at_t=None,
        ee_at_close=None,
        obj_at_close=None,
        ee_at_open=None,
        basket_at_open=None,
        placing_latched=False,
        placing_started_at_t=None,
        success=False,
        steps=env.horizon,
        terminated=False,
        final_obs=obs,
        initial_rgb=obs.rgb,
        logs={
            "obj_z": [], "ee_err": [],
            "ee_obj": [], "ee_obj_xy": [], "ee_obj_z": [],
            "ee_basket_xy": [], "ee_basket_z": [],
        },
        rgb_frames=rgb_frames,
    )


def run_segment(env, state: RolloutState, predict_fn: Callable, *,
                t_end: int,
                replan_at: list[int] | None = None,
                replan_decider: Callable[[RolloutState, int], bool] | None = None,
                dz_scale: float = 1.0,
                placement_mode: str = "trace",
                viz_replans: str = "all",
                viz_dir: Path | None = None,
                on_step: Callable[[RolloutState, int], None] | None = None
                ) -> RolloutState:
    """Step the rollout forward from state.t to t_end (exclusive).

    replan_at:      sorted iterable of step indices where to call
                    plan_from_obs and overwrite dense_targets[t:env.horizon].
                    (Does NOT include t=0 -- that's done by
                    build_initial_state.)
    replan_decider: optional `f(state, t) -> bool`. Queried at every step
                    AFTER the replan_at check; when it returns True a replan
                    is issued. The two sources are unioned -- a step fires
                    plan_from_obs at most once even if both flag it.
    on_step:        optional callback `on_step(state, t)` invoked at the top
                    of each iteration BEFORE any logic. Used by the
                    data-collection driver to inject perturbations and
                    capture snapshots.
    """
    _validate_placement_mode(placement_mode)
    _validate_viz_replans(viz_replans)
    if state.terminated:
        return state
    replan_set = set(replan_at or [])

    for t in range(state.t, t_end):
        if on_step is not None:
            on_step(state, t)
        obs = state.obs

        # Replan trigger: schedule (replan_set) and/or learned decider. Both
        # collapse into the same plan_from_obs call -- scheduling and
        # runtime triggers shouldn't double-replan on the same step.
        should_replan = (t in replan_set) or (
            replan_decider is not None and replan_decider(state, t)
        )
        if should_replan:
            new_targets, _ = plan_from_obs(
                obs, predict_fn,
                dz_scale=dz_scale, n_steps=env.horizon - t,
                viz_dir=_plan_viz_dir(t, viz_dir, viz_replans),
            )
            state.dense_targets[t:env.horizon] = new_targets
            logger.info(
                "replan at t=%d: rewrote dense_targets[%d:%d] (%d steps)",
                t, t, env.horizon, env.horizon - t,
            )

        prev_state = state.oracle.state
        gripper = state.oracle.step(obs.ee_pose[:3], obs.object_pos,
                                    obs.goal_pos, obs.goal_half_extents)
        if prev_state != state.oracle.state \
                and state.oracle.state == "CLOSED_ON_OBJECT":
            state.closed_at_t = t
            state.ee_at_close = obs.ee_pose[:3].copy()
            state.obj_at_close = obs.object_pos.copy()
        if prev_state != state.oracle.state \
                and state.oracle.state == "OPEN_AT_GOAL":
            state.opened_at_t = t
            state.ee_at_open = obs.ee_pose[:3].copy()
            state.basket_at_open = obs.goal_pos.copy()
        # Post-grasp override: hold the EE at the grasp pose for HOLD_STEPS so
        # the gripper can finish closing before lateral motion shears the
        # marginal grip. After the buffer, resume dense_targets shifted
        # backward by HOLD_STEPS so the trajectory's shape is preserved --
        # only the last HOLD_STEPS planned steps fall off the horizon end.
        if state.closed_at_t is not None and t > state.closed_at_t:
            delta = t - state.closed_at_t
            if delta <= POST_GRASP_HOLD_STEPS:
                target = state.ee_at_close
            else:
                idx = max(state.closed_at_t, t - POST_GRASP_HOLD_STEPS)
                target = state.dense_targets[
                    min(idx, len(state.dense_targets) - 1)
                ]
        else:
            target = state.dense_targets[t]
        # Descend-before-release override (latched): once the object first
        # enters the goal region's xy footprint with the gripper still closed,
        # set placing_latched=True and clamp target.z to (goal_z + half_z +
        # GOAL_Z_PLACE_OFFSET) for the rest of the carry phase. The trace
        # still drives target.xy. The latch matters because the trace can
        # graze the region for just 1-2 steps before drifting back out, and
        # the OSC controller (5 cm/step max) cannot descend the EE
        # meaningfully in that window if we only override while currently
        # inside.
        if (placement_mode == "descend"
                and state.oracle.state == "CLOSED_ON_OBJECT"
                and not state.placing_latched):
            xy_to_goal = float(np.linalg.norm(
                obs.object_pos[:2] - obs.goal_pos[:2]
            ))
            if xy_to_goal < PLACE_LATCH_RADIUS:
                state.placing_latched = True
                state.placing_started_at_t = t
                logger.info("placement: descend latched at t=%d "
                            "(object_xy=%s, goal_xy=%s, |xy|=%.3f m)", t,
                            np.round(obs.object_pos[:2], 3).tolist(),
                            np.round(obs.goal_pos[:2], 3).tolist(),
                            xy_to_goal)
        if placement_mode == "descend" and state.placing_latched \
                and state.oracle.state == "CLOSED_ON_OBJECT":
            half_z = (float(obs.goal_half_extents[2])
                      if obs.goal_half_extents is not None else 0.0)
            place_z = float(obs.goal_pos[2]) + half_z + GOAL_Z_PLACE_OFFSET
            target = np.array([target[0], target[1], place_z], dtype=np.float32)
        action = build_osc_action(target, gripper, obs.ee_pose[:3])
        obs, _, done, info = env.step(action)
        state.obs = obs
        state.logs["obj_z"].append(float(obs.object_pos[2]))
        state.logs["ee_err"].append(float(np.linalg.norm(target - obs.ee_pose[:3])))
        state.logs["ee_obj"].append(float(np.linalg.norm(obs.ee_pose[:3] - obs.object_pos)))
        state.logs["ee_obj_xy"].append(float(np.linalg.norm(obs.ee_pose[:2] - obs.object_pos[:2])))
        state.logs["ee_obj_z"].append(float(obs.ee_pose[2] - obs.object_pos[2]))
        state.logs["ee_basket_xy"].append(float(np.linalg.norm(obs.ee_pose[:2] - obs.goal_pos[:2])))
        state.logs["ee_basket_z"].append(float(obs.ee_pose[2] - obs.goal_pos[2]))
        if state.rgb_frames is not None:
            state.rgb_frames.append(np.ascontiguousarray(obs.rgb[::-1]))
        state.final_obs = obs
        state.t = t + 1
        if info.get("success", False):
            state.success = True
            state.steps = t
            state.terminated = True
            break
        if done:
            # Env signaled termination without success -- stop stepping and
            # let classify_failure_mode decide why.
            state.steps = t
            state.terminated = True
            break

    return state


def finalize_trial(state: RolloutState, viz_dir: Path | None = None) -> TrialResult:
    """Build TrialResult from a (possibly terminated) RolloutState. Writes
    rollout.mp4 to viz_dir if rgb_frames were captured."""
    if viz_dir is not None and state.rgb_frames:
        from sim.viz import save_rollout_video
        save_rollout_video(state.rgb_frames, viz_dir / "rollout", fps=30)

    ee_obj_arr = np.asarray(state.logs["ee_obj"], dtype=np.float32)
    ee_obj_xy_arr = np.asarray(state.logs["ee_obj_xy"], dtype=np.float32)
    ee_obj_z_arr = np.asarray(state.logs["ee_obj_z"], dtype=np.float32)
    ee_basket_xy_arr = np.asarray(state.logs["ee_basket_xy"], dtype=np.float32)
    ee_basket_z_arr = np.asarray(state.logs["ee_basket_z"], dtype=np.float32)
    t_min_obj = int(ee_obj_arr.argmin()) if ee_obj_arr.size else -1
    logger.info(
        "ee-vs-object: |min 3d|=%.3f m at t=%d; min |xy|=%.3f m; "
        "z(ee-obj) span=[%.3f, %.3f] m; closed_at_t=%s; "
        "ee-vs-basket: min |xy|=%.3f m; z(ee-basket) span=[%.3f, %.3f] m; "
        "opened_at_t=%s",
        float(ee_obj_arr.min()) if ee_obj_arr.size else float("nan"),
        t_min_obj,
        float(ee_obj_xy_arr.min()) if ee_obj_xy_arr.size else float("nan"),
        float(ee_obj_z_arr.min()) if ee_obj_z_arr.size else float("nan"),
        float(ee_obj_z_arr.max()) if ee_obj_z_arr.size else float("nan"),
        state.closed_at_t,
        float(ee_basket_xy_arr.min()) if ee_basket_xy_arr.size else float("nan"),
        float(ee_basket_z_arr.min()) if ee_basket_z_arr.size else float("nan"),
        float(ee_basket_z_arr.max()) if ee_basket_z_arr.size else float("nan"),
        state.opened_at_t,
    )
    if state.closed_at_t is not None:
        logger.info(
            "grasp event: t=%d ee=%s obj=%s |ee-obj xy|=%.3f m, z_above=%.3f m",
            state.closed_at_t,
            np.round(state.ee_at_close, 3).tolist(),
            np.round(state.obj_at_close, 3).tolist(),
            float(np.linalg.norm(state.ee_at_close[:2] - state.obj_at_close[:2])),
            float(state.ee_at_close[2] - state.obj_at_close[2]),
        )
    if state.opened_at_t is not None:
        logger.info(
            "release event: t=%d ee=%s basket=%s |ee-basket xy|=%.3f m, "
            "z_above=%.3f m",
            state.opened_at_t,
            np.round(state.ee_at_open, 3).tolist(),
            np.round(state.basket_at_open, 3).tolist(),
            float(np.linalg.norm(state.ee_at_open[:2] - state.basket_at_open[:2])),
            float(state.ee_at_open[2] - state.basket_at_open[2]),
        )
    logger.info(
        "release gate diag: %s; min_xy_inside_above=%s",
        {k: v for k, v in state.oracle.diag.items()
         if k != "min_xy_dist_inside_above"},
        f"{state.oracle.diag['min_xy_dist_inside_above']:.3f} m"
        if state.oracle.diag["min_xy_dist_inside_above"] != float("inf")
        else "inf",
    )

    failure_mode = classify_failure_mode(
        success=state.success,
        object_z_trajectory=np.asarray(state.logs["obj_z"], dtype=np.float32),
        object_xy_final=state.final_obs.object_pos[:2],
        basket_xy=state.final_obs.goal_pos[:2],
        ee_target_errors=np.asarray(state.logs["ee_err"], dtype=np.float32),
    )
    return TrialResult(success=state.success, steps=state.steps,
                       failure_mode=failure_mode, diag=state.sel_diag,
                       initial_rgb=state.initial_rgb)


def run_trial(env, predict_fn: Callable, seed: int,
              dz_scale: float = 1.0,
              placement_mode: str = "trace",
              replan_freq: int = 0,
              replan_decider: Callable[[RolloutState, int], bool] | None = None,
              viz_replans: str = "all",
              viz_dir: Path | None = None) -> TrialResult:
    """Run one trial end-to-end. Thin wrapper over
    build_initial_state + run_segment + finalize_trial.

    placement_mode:  'trace' (target.z follows dense_targets) or 'descend'
                     (target.z is clamped to placement altitude once the
                     object enters the goal region).
    replan_freq:     0 = one-shot at t=0 (legacy). N>0 = replan at t in
                     {N, 2N, ...} (strictly less than env.horizon).
    replan_decider:  optional runtime decider `f(state, t) -> bool` for
                     learned-trigger replanning. Caller is responsible for
                     deciding how this composes with replan_freq.
    viz_replans:     'first' / 'all' / 'none'. Per-plan artifact policy.
    viz_dir:         per-trial directory. None disables viz.
    """
    state = build_initial_state(env, predict_fn, seed,
                                dz_scale=dz_scale,
                                placement_mode=placement_mode,
                                viz_replans=viz_replans,
                                viz_dir=viz_dir)
    if replan_freq > 0:
        replan_at = list(range(replan_freq, env.horizon, replan_freq))
    else:
        replan_at = []
    state = run_segment(env, state, predict_fn,
                        t_end=env.horizon,
                        replan_at=replan_at,
                        replan_decider=replan_decider,
                        dz_scale=dz_scale,
                        placement_mode=placement_mode,
                        viz_replans=viz_replans,
                        viz_dir=viz_dir)
    return finalize_trial(state, viz_dir=viz_dir)
