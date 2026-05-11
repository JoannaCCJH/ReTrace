"""Per-trial rollout loop + failure-mode classifier."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


def run_trial(env, predict_fn: Callable, seed: int,
              dz_scale: float = 1.0,
              placement_mode: str = "trace",
              replan_freq: int = 0,
              viz_replans: str = "all",
              viz_dir: Path | None = None) -> TrialResult:
    """Run one trial. Returns TrialResult including artifacts.

    predict_fn: (rgb, depth, language) -> (rgb_t [3,S,S], depth_t [1,S,S]|None,
                                           pred_trace [400, 32, 3] numpy, relative).
    dz_scale:   multiplier on the model's predicted dz before cumsum, to
                convert training-monocular units into metric meters.
    placement_mode: how the carry-phase target.z is computed once the object
                enters the goal region's xy footprint.
                  - 'trace':   target.z follows dense_targets verbatim (the
                               model + linear-interp anchor decide altitude).
                  - 'descend': target.z is overridden to (goal.z + half_z +
                               GOAL_Z_PLACE_OFFSET) so the EE actively descends
                               to a low placement altitude before the oracle
                               opens the gripper. Drops mug from ~5 cm above
                               the box top -> short freefall, lands in region.
    replan_freq: replan every N env steps. 0 (default) = one-shot at t=0
                (legacy open-loop behavior). When >0, the predict -> lift ->
                top-K -> world -> anchor -> interp pipeline reruns at t in
                {N, 2N, ...} (strictly less than env.horizon) using the
                current obs, and dense_targets[t:env.horizon] is overwritten
                with a fresh plan interpolated to env.horizon - t steps.
                Each replan anchors at the CURRENT obs.ee_pose. The post-grasp
                hold + descend latch sit on top of dense_targets, so they
                still take precedence when active; replans during those
                overrides are wasted compute but harmless.
    viz_replans: per-replan artifact policy, one of 'first', 'all', 'none'.
                'all' (default) writes pred_traces.png / selected_traces.png /
                world_trace_overlay.png / traces_3d.npz / combined_trace_world.npz
                under viz_dir/replan_TTT/ for every plan (t=0 plus each replan).
                'first' writes them only for the t=0 plan. 'none' suppresses
                per-plan artifacts entirely. rollout.mp4 and initial_rgb.png
                are unaffected -- they're trial-level and still go in viz_dir.
    viz_dir:    per-trial directory. When set, per-plan artifacts go in
                viz_dir/replan_TTT/ (subject to viz_replans), rollout.mp4 goes
                in viz_dir.
    """
    if placement_mode not in ("trace", "descend"):
        raise ValueError(
            f"placement_mode must be 'trace' or 'descend', got {placement_mode!r}"
        )
    if viz_replans not in ("first", "all", "none"):
        raise ValueError(
            f"viz_replans must be 'first', 'all', or 'none', got {viz_replans!r}"
        )
    obs = env.reset(seed=seed)
    obs0 = obs

    def _plan_viz_dir(t: int) -> Path | None:
        if viz_dir is None or viz_replans == "none":
            return None
        if viz_replans == "first" and t != 0:
            return None
        return viz_dir / f"replan_{t:03d}"

    dense_targets, sel_diag = plan_from_obs(
        obs, predict_fn,
        dz_scale=dz_scale, n_steps=env.horizon,
        viz_dir=_plan_viz_dir(0),
    )

    oracle = GripperOracle()
    obj_z_log, ee_err_log = [], []
    # Per-step EE-vs-target geometry for threshold calibration. Lets you see
    # whether grasp_miss is "trace never got close" (large min distances) vs
    # "EE was close but oracle threshold didn't fire" (small min, close gate
    # never triggered) vs "EE-pos is offset from grip TCP" (z_min biased).
    ee_obj_log, ee_obj_xy_log, ee_obj_z_log = [], [], []
    ee_basket_xy_log, ee_basket_z_log = [], []
    closed_at_t: int | None = None
    opened_at_t: int | None = None
    ee_at_close: np.ndarray | None = None  # EE world pos when grasp fired
    ee_at_open: np.ndarray | None = None   # EE world pos when release fired
    obj_at_close: np.ndarray | None = None # mug world pos at grasp time
    basket_at_open: np.ndarray | None = None  # basket world xy at release time
    # Capture flipped (top-down, display orientation) rgb each step for the
    # rollout.mp4. Includes the reset frame as t=0.
    rgb_frames = ([np.ascontiguousarray(obs.rgb[::-1])]
                  if viz_dir is not None else None)
    success = False
    steps = env.horizon
    final_obs = obs
    # Latched in 'descend' mode the first step the mug enters the goal region's
    # xy footprint. Once latched, target.z stays clamped to placement altitude
    # for the rest of the carry phase, even if the trace's xy drifts the mug
    # back out of the region. Without this latch, the trace can graze the
    # region for 1-2 steps and the OSC controller (5 cm/step max) can't
    # descend meaningfully before the override stops firing.
    placing_latched = False
    placing_started_at_t: int | None = None
    for t in range(env.horizon):
        # Replan trigger: rerun plan_from_obs on the current obs and overwrite
        # the tail dense_targets[t:env.horizon] with a fresh plan anchored at
        # the current EE. t=0 plan was already computed above; first replan is
        # at t=replan_freq. The post-grasp hold + descend latch still take
        # precedence on the resulting target since they sit on top of
        # dense_targets[t], so a replan during those overrides is wasted but
        # harmless.
        if replan_freq > 0 and t > 0 and t % replan_freq == 0:
            new_targets, _ = plan_from_obs(
                obs, predict_fn,
                dz_scale=dz_scale, n_steps=env.horizon - t,
                viz_dir=_plan_viz_dir(t),
            )
            dense_targets[t:env.horizon] = new_targets
            logger.info(
                "replan at t=%d: rewrote dense_targets[%d:%d] (%d steps)",
                t, t, env.horizon, env.horizon - t,
            )
        prev_state = oracle.state
        gripper = oracle.step(obs.ee_pose[:3], obs.object_pos, obs.goal_pos,
                              obs.goal_half_extents)
        if prev_state != oracle.state and oracle.state == "CLOSED_ON_OBJECT":
            closed_at_t = t
            ee_at_close = obs.ee_pose[:3].copy()
            obj_at_close = obs.object_pos.copy()
        if prev_state != oracle.state and oracle.state == "OPEN_AT_GOAL":
            opened_at_t = t
            ee_at_open = obs.ee_pose[:3].copy()
            basket_at_open = obs.goal_pos.copy()
        # Post-grasp override: hold the EE at the grasp pose for HOLD_STEPS so
        # the gripper can finish closing before lateral motion shears the
        # marginal grip. After the buffer, resume dense_targets shifted
        # backward by HOLD_STEPS so the trajectory's shape is preserved --
        # only the last HOLD_STEPS planned steps fall off the horizon end.
        if closed_at_t is not None and t > closed_at_t:
            delta = t - closed_at_t
            if delta <= POST_GRASP_HOLD_STEPS:
                target = ee_at_close
            else:
                idx = max(closed_at_t, t - POST_GRASP_HOLD_STEPS)
                target = dense_targets[min(idx, len(dense_targets) - 1)]
        else:
            target = dense_targets[t]
        # Descend-before-release override (latched): once the object first
        # enters the goal region's xy footprint with the gripper still closed,
        # set placing_latched=True and clamp target.z to (goal_z + half_z +
        # GOAL_Z_PLACE_OFFSET) for the rest of the carry phase. The trace
        # still drives target.xy. The latch matters because the trace can
        # graze the region for just 1-2 steps before drifting back out, and
        # the OSC controller (5 cm/step max) cannot descend the EE
        # meaningfully in that window if we only override while currently
        # inside. In 'trace' mode this branch is skipped and target.z follows
        # the trace verbatim.
        # Latch trigger uses a wider radius than the region's actual footprint
        # (PLACE_LATCH_RADIUS, default 15 cm) so the descent begins *before*
        # the mug reaches the region. With a tighter trigger (e.g. inside_xy)
        # the trace can graze the region in 1-2 steps and the OSC controller
        # has no time to lower the EE before release fires.
        if (placement_mode == "descend"
                and oracle.state == "CLOSED_ON_OBJECT"
                and not placing_latched):
            xy_to_goal = float(np.linalg.norm(
                obs.object_pos[:2] - obs.goal_pos[:2]
            ))
            if xy_to_goal < PLACE_LATCH_RADIUS:
                placing_latched = True
                placing_started_at_t = t
                logger.info("placement: descend latched at t=%d (object_xy=%s,"
                            " goal_xy=%s, |xy|=%.3f m)", t,
                            np.round(obs.object_pos[:2], 3).tolist(),
                            np.round(obs.goal_pos[:2], 3).tolist(),
                            xy_to_goal)
        if placement_mode == "descend" and placing_latched \
                and oracle.state == "CLOSED_ON_OBJECT":
            half_z = (float(obs.goal_half_extents[2])
                      if obs.goal_half_extents is not None else 0.0)
            place_z = float(obs.goal_pos[2]) + half_z + GOAL_Z_PLACE_OFFSET
            target = np.array([target[0], target[1], place_z], dtype=np.float32)
        action = build_osc_action(target, gripper, obs.ee_pose[:3])
        obs, _, done, info = env.step(action)
        obj_z_log.append(float(obs.object_pos[2]))
        ee_err_log.append(float(np.linalg.norm(target - obs.ee_pose[:3])))
        ee_obj_log.append(float(np.linalg.norm(obs.ee_pose[:3] - obs.object_pos)))
        ee_obj_xy_log.append(float(np.linalg.norm(obs.ee_pose[:2] - obs.object_pos[:2])))
        ee_obj_z_log.append(float(obs.ee_pose[2] - obs.object_pos[2]))
        ee_basket_xy_log.append(float(np.linalg.norm(obs.ee_pose[:2] - obs.goal_pos[:2])))
        ee_basket_z_log.append(float(obs.ee_pose[2] - obs.goal_pos[2]))
        if rgb_frames is not None:
            rgb_frames.append(np.ascontiguousarray(obs.rgb[::-1]))
        final_obs = obs
        if info.get("success", False):
            success = True
            steps = t
            break
        if done:
            # Env signaled termination without success -- stop stepping a
            # finished env and let classify_failure_mode decide why.
            steps = t
            break

    if viz_dir is not None and rgb_frames:
        from sim.viz import save_rollout_video
        save_rollout_video(rgb_frames, viz_dir / "rollout", fps=30)

    ee_obj_arr = np.asarray(ee_obj_log, dtype=np.float32)
    ee_obj_xy_arr = np.asarray(ee_obj_xy_log, dtype=np.float32)
    ee_obj_z_arr = np.asarray(ee_obj_z_log, dtype=np.float32)
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
        closed_at_t,
        float(np.asarray(ee_basket_xy_log, dtype=np.float32).min())
        if ee_basket_xy_log else float("nan"),
        float(np.asarray(ee_basket_z_log, dtype=np.float32).min())
        if ee_basket_z_log else float("nan"),
        float(np.asarray(ee_basket_z_log, dtype=np.float32).max())
        if ee_basket_z_log else float("nan"),
        opened_at_t,
    )
    if closed_at_t is not None:
        logger.info(
            "grasp event: t=%d ee=%s obj=%s |ee-obj xy|=%.3f m, z_above=%.3f m",
            closed_at_t,
            np.round(ee_at_close, 3).tolist(),
            np.round(obj_at_close, 3).tolist(),
            float(np.linalg.norm(ee_at_close[:2] - obj_at_close[:2])),
            float(ee_at_close[2] - obj_at_close[2]),
        )
    if opened_at_t is not None:
        logger.info(
            "release event: t=%d ee=%s basket=%s |ee-basket xy|=%.3f m, "
            "z_above=%.3f m",
            opened_at_t,
            np.round(ee_at_open, 3).tolist(),
            np.round(basket_at_open, 3).tolist(),
            float(np.linalg.norm(ee_at_open[:2] - basket_at_open[:2])),
            float(ee_at_open[2] - basket_at_open[2]),
        )
    # Release-gate diagnostics: tells you which condition failed when
    # opened_at_t is None. ever_inside_xy=False means the object never entered
    # the region's xy footprint; ever_above=False means it was never high
    # enough above the box top + clearance; both True but
    # released_via=None means the closest-approach trigger never fired (e.g.
    # the trace passed through with monotonically decreasing xy_dist) AND the
    # exit-fallback didn't catch it.
    logger.info(
        "release gate diag: %s; min_xy_inside_above=%s",
        {k: v for k, v in oracle.diag.items() if k != "min_xy_dist_inside_above"},
        f"{oracle.diag['min_xy_dist_inside_above']:.3f} m"
        if oracle.diag["min_xy_dist_inside_above"] != float("inf") else "inf",
    )

    failure_mode = classify_failure_mode(
        success=success,
        object_z_trajectory=np.asarray(obj_z_log, dtype=np.float32),
        object_xy_final=final_obs.object_pos[:2],
        basket_xy=final_obs.goal_pos[:2],
        ee_target_errors=np.asarray(ee_err_log, dtype=np.float32),
    )
    return TrialResult(success=success, steps=steps,
                       failure_mode=failure_mode, diag=sel_diag,
                       initial_rgb=obs0.rgb)
