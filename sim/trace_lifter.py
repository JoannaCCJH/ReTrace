"""Pinhole-camera math for TraceGen -> LIBERO.

Contains:
- sample_depth_bilinear: sub-pixel depth sampling.
- lift_to_world: pixel+depth -> world-frame 3D.
- project_ee_to_pixel: world-frame pose -> normalized image pixel.
- reconstruct_absolute_trace: relative-delta trace -> absolute pixel/metric.
- lift_all_traces_to_camera_frame: all N grid traces -> camera-frame 2.5D.
"""
import numpy as np

from sim.constants import GRID_XY


def sample_depth_bilinear(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample a [H, W] depth map at float (u, v) coords.

    u is column (x), v is row (y). Out-of-bounds coords are clamped.
    """
    H, W = depth.shape
    u = np.clip(u, 0.0, W - 1.0)
    v = np.clip(v, 0.0, H - 1.0)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = (u - u0).astype(np.float32)
    dv = (v - v0).astype(np.float32)
    d00 = depth[v0, u0]; d01 = depth[v0, u1]
    d10 = depth[v1, u0]; d11 = depth[v1, u1]
    return (d00 * (1 - du) * (1 - dv) +
            d01 * du       * (1 - dv) +
            d10 * (1 - du) * dv       +
            d11 * du       * dv).astype(np.float32)


def lift_to_world(trace_px: np.ndarray, depth_map: np.ndarray | None = None,
                  K: np.ndarray = None, cam_to_world: np.ndarray = None,
                  z_override: np.ndarray | None = None) -> np.ndarray:
    """Lift [N, 2] pixel trajectory to [N, 3] world frame via depth + pinhole.

    trace_px:     [N, 2] in original image pixel coords (u, v).
    depth_map:    [H, W] metric depth. Required only when z_override is None.
    K:            [3, 3] intrinsics.
    cam_to_world: [4, 4] extrinsics.
    z_override:   [N] metric depth to use instead of sampling depth_map.
                  Pass the predicted trace_z when the trajectory lives above
                  the scene surface (the depth buffer sees the table/object
                  top, not the EE's flight path).
    """
    u = trace_px[:, 0].astype(np.float32)
    v = trace_px[:, 1].astype(np.float32)
    if z_override is None:
        assert depth_map is not None, \
            "lift_to_world: must provide depth_map when z_override is None"
        z = sample_depth_bilinear(depth_map, u, v)
    else:
        z = np.asarray(z_override, dtype=np.float32)
    # robosuite's get_camera_extrinsic_matrix applies a camera_axis_correction
    # = diag(1, -1, -1, 1) before returning cam_to_world, so the matrix is
    # already in OpenCV convention (+x right, +y DOWN, +z forward). Build the
    # camera-frame point straight from top-down (u, v) + positive z; no extra
    # y-flip is needed (an extra flip mirrors the world Y axis).
    x_cam = (u - K[0, 2]) * z / K[0, 0]
    y_cam = (v - K[1, 2]) * z / K[1, 1]
    ones = np.ones_like(z)
    pts_cam_h = np.stack([x_cam, y_cam, z, ones], axis=-1)
    pts_world = (cam_to_world @ pts_cam_h.T).T[:, :3]
    return pts_world.astype(np.float32)


def project_world_to_pixel(world_pts: np.ndarray, K: np.ndarray,
                           cam_to_world: np.ndarray,
                           img_shape: tuple[int, int]
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized world -> ([N, 2] normalized pixel, [N] camera-frame depth).

    Algebraic inverse of lift_to_world. cam_to_world from robosuite is already
    OpenCV convention (+x right, +y DOWN, +z forward) thanks to its built-in
    camera_axis_correction, so the standard pinhole formula applies directly --
    no y-flip.
    """
    world_to_cam = np.linalg.inv(cam_to_world)
    pts_h = np.concatenate(
        [world_pts.astype(np.float32),
         np.ones((world_pts.shape[0], 1), dtype=np.float32)], axis=1
    )
    pts_cam = (world_to_cam @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    H, W = img_shape
    px_norm = np.stack([u / W, v / H], axis=-1).astype(np.float32)
    return px_norm, z.astype(np.float32)


def project_ee_to_pixel(world_pt: np.ndarray, K: np.ndarray,
                        cam_to_world: np.ndarray,
                        img_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """World-frame 3D point -> (normalized [0,1] pixel, depth).

    cam_to_world from robosuite is OpenCV convention (+x right, +y DOWN,
    +z forward) -- get_camera_extrinsic_matrix bakes a camera_axis_correction
    into it. Standard pinhole projection applies directly; v is naturally
    top-down (v=0 at the top of the image), matching how rgb and depth are
    stored after the prepare_model_inputs flip and how GRID_XY is laid out.

    img_shape: (H, W) of the image the caller will compare against.
    Returns px_norm [2] and depth scalar (camera-frame z, positive in front).
    """
    world_to_cam = np.linalg.inv(cam_to_world)
    pt_h = np.append(world_pt.astype(np.float32), 1.0)
    pt_cam = (world_to_cam @ pt_h)[:3]
    z = float(pt_cam[2])
    u = K[0, 0] * pt_cam[0] / z + K[0, 2]
    v = K[1, 1] * pt_cam[1] / z + K[1, 2]
    H, W = img_shape
    return np.array([u / W, v / H], dtype=np.float32), z


def reconstruct_absolute_trace(trace_rel: np.ndarray, grid_xy_start: np.ndarray,
                               depth_map: np.ndarray, img_h: int, img_w: int,
                               z0_metric: float | None = None,
                               dz_scale: float = 1.0,
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Relative-delta trace -> absolute pixel + metric-z.

    trace_rel:     [T, 3] - normalized xy deltas + z deltas (from selected trace).
    grid_xy_start: [2]    - trace's grid origin in normalized [0,1].
    depth_map:     [H, W] - metric depth for sampling starting z when
                            z0_metric is None.
    z0_metric:     metric anchor (camera-frame z) for the trace start. When
                   the model was trained on monocular-scale depth, sampling
                   z0 from the metric depth buffer mixes units inside the
                   cumsum. Pass an externally-known metric z (e.g. the EE
                   camera-frame depth from project_ee_to_pixel) to avoid that.
    dz_scale:      multiplier on dz before cumsum. Converts model-space dz
                   (in training-monocular units) to metric meters. Calibrate
                   offline; default 1.0 leaves the unit mismatch in place.
    Returns trace_px [T, 2] (absolute pixels, clamped to image bounds) and
    trace_z [T] (metric).

    TraceGen is trained on keypoints that were clamped to [0, 1], but during
    inference diffusion sampling can drift the cumulative xy outside that
    range. Targets projected from out-of-bounds pixels land outside the
    camera's field of view (and often outside the robot's reachable
    workspace). Clamp abs_xy_norm to [0, 1] so trace_px stays inside the
    image and the lifted world targets stay inside the scene.
    """
    abs_xy_norm = grid_xy_start[None, :] + np.cumsum(trace_rel[:, :2], axis=0)
    abs_xy_norm = np.clip(abs_xy_norm, 0.0, 1.0)
    trace_px = np.stack([abs_xy_norm[:, 0] * img_w,
                         abs_xy_norm[:, 1] * img_h], axis=-1).astype(np.float32)
    if z0_metric is None:
        u0 = int(round(float(grid_xy_start[0] * img_w)))
        v0 = int(round(float(grid_xy_start[1] * img_h)))
        u0 = max(0, min(img_w - 1, u0))
        v0 = max(0, min(img_h - 1, v0))
        z0 = float(depth_map[v0, u0])
    else:
        z0 = float(z0_metric)
    trace_z = (z0 + np.cumsum(dz_scale * trace_rel[:, 2], axis=0)).astype(np.float32)
    return trace_px, trace_z


def lift_all_traces_to_camera_frame(pred: np.ndarray, depth_map: np.ndarray,
                                    img_h: int, img_w: int,
                                    dz_scale: float = 1.0,
                                    depth_ratio: np.ndarray | None = None,
                                    ) -> np.ndarray:
    """All N grid traces -> [N, 1+T, 3] (px_x, px_y, z_metric).

    Mirrors test_helpers.create_trajectory_visualization's
    `get_reconstructed_trajectory=True` branch:
      1. Prepend each trace's grid_xy as time-step 0, cumsum xy deltas in
         normalized coords, scale to pixel space.
      2. Sample z0 from depth_map at each trace's start pixel, then cumsum
         [z0, dz_scale * dz_1, ..., dz_scale * dz_T] along time.
      3. Optionally rescale each step's z by depth_ratio[iy, ix] (sensored /
         predicted depth ratio). Skip when depth_map already matches the
         sensor's metric scale -- LIBERO obs.depth is one such case.

    pred:        [N, T, 3] - normalized (dx, dy, dz) deltas. xy are [0,1] image
                 fractions; dz is in whatever units the model was trained on.
    depth_map:   [img_h, img_w] - metric depth at the same pixel frame as the
                 trace xy (top-down, model-input size).
    dz_scale:    multiplier on the model's dz output before cumsum. Use to
                 convert training-monocular dz units to metric meters; calibrate
                 from the model's per-channel action_min/max ranges.
    depth_ratio: [img_h, img_w] or None.
    """
    xy_with_start = np.concatenate([GRID_XY[:, None, :], pred[..., :2]], axis=1)
    abs_xy_norm = np.cumsum(xy_with_start, axis=1)
    abs_xy_norm = np.clip(abs_xy_norm, 0.0, 1.0)
    px_xy = np.stack([abs_xy_norm[..., 0] * img_w,
                      abs_xy_norm[..., 1] * img_h], axis=-1).astype(np.float32)

    ix = np.clip(np.rint(px_xy[..., 0]).astype(np.int64), 0, img_w - 1)
    iy = np.clip(np.rint(px_xy[..., 1]).astype(np.int64), 0, img_h - 1)
    z0 = depth_map[iy[:, 0], ix[:, 0]].astype(np.float32)
    dz = dz_scale * pred[..., 2].astype(np.float32)
    z_with_start = np.concatenate([z0[:, None], dz], axis=1)
    z_traj = np.cumsum(z_with_start, axis=1).astype(np.float32)

    if depth_ratio is not None:
        z_traj = z_traj * depth_ratio[iy, ix].astype(np.float32)

    return np.concatenate([px_xy, z_traj[..., None]], axis=-1)
