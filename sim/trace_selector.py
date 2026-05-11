"""Single-trace selection from TraceGen's 400-grid output."""
import logging

import numpy as np

from sim.constants import TRACE_MOTION_THRESHOLD, TRACE_WARN_DISTANCE

logger = logging.getLogger(__name__)


def select_trace(pred_trace_rel: np.ndarray, grid_xy: np.ndarray,
                 ee_pixel_norm: np.ndarray,
                 target_pixel_norm: np.ndarray | None = None,
                 motion_threshold: float = TRACE_MOTION_THRESHOLD,
                 warn_distance: float = TRACE_WARN_DISTANCE
                 ) -> tuple[np.ndarray, int, dict]:
    """Pick one trace: motion-filter then closest-to-target.

    pred_trace_rel:    [N, T, 3] - normalized xy deltas + z deltas.
    grid_xy:           [N, 2]   - normalized grid start positions.
    ee_pixel_norm:     [2]      - projected EE in normalized image coords.
                                  Used for the fallback/diagnostic only.
    target_pixel_norm: [2]      - pixel to prefer; the picked trace's START
                                  will be the mover closest to it. TraceGen
                                  trains on foreground-object motion, so the
                                  useful traces live at the object's pixel,
                                  not the gripper. If None, falls back to
                                  ee_pixel_norm (legacy behavior).
    Returns: (trace_rel [T, 3], chosen_idx int, diag dict).
    """
    target = target_pixel_norm if target_pixel_norm is not None else ee_pixel_norm
    abs_xy = grid_xy[:, None, :] + np.cumsum(pred_trace_rel[..., :2], axis=1)
    total_motion = np.linalg.norm(abs_xy[:, -1] - abs_xy[:, 0], axis=-1)
    movers = total_motion > motion_threshold

    fallback_no_movers = not bool(movers.any())
    if fallback_no_movers:
        logger.warning("no traces exceed motion threshold %.4f; falling back to all traces",
                       motion_threshold)
        movers = np.ones_like(movers, dtype=bool)

    candidate_starts = abs_xy[movers, 0]
    dists = np.linalg.norm(candidate_starts - target[None, :], axis=-1)
    local_idx = int(np.argmin(dists))
    chosen_idx = int(np.where(movers)[0][local_idx])

    min_dist = float(dists[local_idx])
    if min_dist > warn_distance:
        logger.warning("chosen trace starts %.3f from target (> warn=%.3f); target may be occluded",
                       min_dist, warn_distance)

    diag = {
        "n_movers": int(movers.sum()),
        "distance_from_target": min_dist,
        "distance_from_ee": float(
            np.linalg.norm(grid_xy[chosen_idx] - ee_pixel_norm)
        ),
        "motion_magnitude": float(total_motion[chosen_idx]),
        "fallback_no_movers": fallback_no_movers,
    }
    return pred_trace_rel[chosen_idx].copy(), chosen_idx, diag


def select_top_k_traces(traces_3d: np.ndarray,
                        ee_pixel_px: np.ndarray,
                        img_size: int,
                        k: int = 5,
                        motion_threshold: float = 0.01,
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pick K traces whose start pixels are nearest the EE pixel, after motion filter.

    Operates directly on the lifted traces_3d (so the trace's start pixel IS
    `traces_3d[:, 0, :2]` -- no need to recompute from grid_xy + cumsum).

    traces_3d:        [N, 1+T, 3] - (px_x, px_y, z_metric) in top-down pixel
                      coords at `img_size` x `img_size`.
    ee_pixel_px:      [2] EE pixel in the same top-down frame as traces_3d.
    img_size:         pixel-frame side length used to normalize path length so
                      `motion_threshold` is unitless (matches the convention in
                      sim.visualize_pred.save_overlay).
    k:                how many traces to keep.
    motion_threshold: trace.path_length / img_size must exceed this.

    Returns:
        selected:     [K, 1+T, 3]  -- ordered by ascending distance to EE.
        indices:      [K]          -- indices into traces_3d.
        diag dict.
    """
    px_xy = traces_3d[..., :2]
    diffs = np.diff(px_xy, axis=1)
    path_len = np.linalg.norm(diffs, axis=-1).sum(axis=-1) / float(img_size)
    movers = path_len > motion_threshold

    fallback = not bool(movers.any())
    if fallback:
        logger.warning("no traces exceed motion threshold %.4f; selecting from all",
                       motion_threshold)
        movers = np.ones_like(movers, dtype=bool)

    dists = np.full(traces_3d.shape[0], np.inf, dtype=np.float32)
    dists[movers] = np.linalg.norm(
        px_xy[movers, 0] - ee_pixel_px[None, :].astype(np.float32), axis=-1
    )
    k_eff = min(int(k), int(movers.sum()))
    # argpartition gives the k smallest unsorted; sort them so the caller can
    # rely on rank-0 being the closest.
    cand = np.argpartition(dists, k_eff - 1)[:k_eff]
    indices = cand[np.argsort(dists[cand])]

    diag = {
        "k": int(k_eff),
        "n_movers": int(movers.sum()),
        "fallback_no_movers": fallback,
        "min_distance_px": float(dists[indices[0]]),
        "max_distance_px": float(dists[indices[-1]]),
        "ee_pixel_px": [float(ee_pixel_px[0]), float(ee_pixel_px[1])],
    }
    return traces_3d[indices].copy(), indices, diag
