"""Trace overlay rendering for per-trial debugging."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _flip_for_display(rgb: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip rgb + trace-y vertically for display.

    robosuite's agentview camera is mounted above and behind the robot, so the
    raw buffer naturally puts the robot in the foreground (bottom) with the
    workspace receding to the top of the frame. For human readability we flip
    both the rgb and the trace v so the gripper ends up at the top of the
    figure with the table and objects below -- a bird's-eye style view.
    Projection math stays in top-down image coords; the flip is display-only.
    """
    H = rgb.shape[0]
    rgb_disp = np.ascontiguousarray(rgb[::-1, :, :])
    px_disp = px.copy().astype(np.float32)
    px_disp[..., 1] = (H - 1) - px_disp[..., 1]
    return rgb_disp, px_disp


def save_trace_overlay(rgb: np.ndarray, trace_px: np.ndarray, out_path: Path) -> None:
    """Draw the chosen 2D trace on the RGB image and save as PNG.

    rgb:      [H, W, 3] uint8, top-down storage.
    trace_px: [T, 2] absolute pixel coords in the top-down frame.
    """
    rgb_disp, trace_disp = _flip_for_display(rgb, trace_px)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(rgb_disp)
    ax.plot(trace_disp[:, 0], trace_disp[:, 1], "-", color="yellow", linewidth=2.0)
    ax.scatter(trace_disp[0, 0], trace_disp[0, 1], c="lime", s=40, label="start")
    ax.scatter(trace_disp[-1, 0], trace_disp[-1, 1], c="red", s=40, label="end")
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_all_traces_overlay(rgb: np.ndarray, all_traces_px: np.ndarray,
                            chosen_idx: int, out_path: Path) -> None:
    """Draw all N candidate traces on the RGB image with the chosen one highlighted.

    rgb:           [H, W, 3] uint8, top-down storage.
    all_traces_px: [N, T, 2] absolute pixel coords for each grid-start trace.
    chosen_idx:    index into all_traces_px of the trace that was executed.
    """
    rgb_disp, all_disp = _flip_for_display(rgb, all_traces_px)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(rgb_disp)
    for i, t in enumerate(all_disp):
        if i == chosen_idx:
            continue
        ax.plot(t[:, 0], t[:, 1], "-", color="cyan", linewidth=0.4, alpha=0.35)

    chosen = all_disp[chosen_idx]
    ax.plot(chosen[:, 0], chosen[:, 1], "-", color="yellow", linewidth=2.0)
    ax.scatter(chosen[0, 0], chosen[0, 1], c="lime", s=40, label="chosen start")
    ax.scatter(chosen[-1, 0], chosen[-1, 1], c="red", s=40, label="chosen end")
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_top_k_traces_overlay(rgb: np.ndarray,
                              all_traces_px: np.ndarray,
                              selected_indices: np.ndarray,
                              ee_pixel_px: np.ndarray,
                              out_path: Path,
                              motion_threshold: float = 0.01) -> None:
    """RGB + EE pixel + top-K selected traces (highlighted) + other movers (faint).

    rgb:              [H, W, 3] uint8 in display orientation -- the same
                      top-down tensor the model saw, no flip needed (matches
                      sim.visualize_pred.save_overlay).
    all_traces_px:    [N, T, 2] absolute pixel coords (top-down, matching rgb).
    selected_indices: [K] indices into all_traces_px (assumed already ordered
                      closest-first; turbo color picks rank rather than index).
    ee_pixel_px:      [2] EE pixel in the same top-down frame as all_traces_px.
    motion_threshold: trace.path_length / max(H, W) must exceed this for the
                      faint-overlay layer (matches save_overlay's convention).
    """
    H, W = rgb.shape[:2]
    diffs = np.diff(all_traces_px, axis=1)
    path_len = np.linalg.norm(diffs, axis=-1).sum(axis=-1) / max(H, W)
    movers = path_len > motion_threshold
    selected_set = {int(i) for i in selected_indices}
    K = len(selected_indices)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(rgb)

    # Faint background context: motion-filtered traces that weren't picked.
    for i, t in enumerate(all_traces_px):
        if not movers[i] or i in selected_set:
            continue
        ax.plot(t[:, 0], t[:, 1], "-", color="white", linewidth=0.4, alpha=0.3)

    # Highlighted: the selected K, colored by rank (closest -> turbo low).
    cmap = plt.get_cmap("turbo")
    for rank, idx in enumerate(selected_indices):
        t = all_traces_px[int(idx)]
        color = cmap(rank / max(K - 1, 1))
        ax.plot(t[:, 0], t[:, 1], "-", color=color, linewidth=1.6, alpha=0.95)
        ax.scatter(t[0, 0], t[0, 1], c="lime", s=22, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.scatter(t[-1, 0], t[-1, 1], c="red", s=22, zorder=3,
                   edgecolors="black", linewidths=0.5)

    # EE pixel marker on top of everything.
    ax.scatter(ee_pixel_px[0], ee_pixel_px[1], c="yellow", s=120, marker="*",
               edgecolors="black", linewidths=1.0, zorder=5, label="EE")

    ax.set_axis_off()
    ax.set_title(f"top-{K} traces near EE (motion>{motion_threshold})", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_world_trace_overlay(rgb: np.ndarray, trace_pixel_xy: np.ndarray,
                             ee_pixel_px: np.ndarray, out_path: Path) -> None:
    """RGB + the back-projected world trace + EE pixel marker.

    Diagnostic for the world-frame round-trip: if `trace_pixel_xy` lands on the
    same path the original 2.5D trace took, lift_to_world / project_world_to_
    pixel are consistent and the world coords are in the robot's frame.

    rgb:            [H, W, 3] uint8 in display orientation (top-down).
    trace_pixel_xy: [T, 2] pixel coords from project_world_to_pixel(trace_world)
                    scaled to rgb's pixel frame.
    ee_pixel_px:    [2] EE pixel marker in the same frame.
    """
    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(rgb)
    ax.plot(trace_pixel_xy[:, 0], trace_pixel_xy[:, 1], "-",
            color="cyan", linewidth=2.0, alpha=0.95, label="world trace")
    ax.scatter(trace_pixel_xy[0, 0], trace_pixel_xy[0, 1], c="lime", s=40,
               edgecolors="black", linewidths=0.5, zorder=4, label="start")
    ax.scatter(trace_pixel_xy[-1, 0], trace_pixel_xy[-1, 1], c="red", s=40,
               edgecolors="black", linewidths=0.5, zorder=4, label="end")
    ax.scatter(ee_pixel_px[0], ee_pixel_px[1], c="yellow", s=120, marker="*",
               edgecolors="black", linewidths=1.0, zorder=5, label="EE")
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_axis_off()
    ax.set_title("world trace -> back to pixel", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_rollout_video(frames: list[np.ndarray], out_path: Path,
                       fps: int = 30, scale: int = 4) -> None:
    """Write a sequence of [H, W, 3] uint8 frames to mp4 (falls back to gif).

    frames are assumed to be in display orientation already (caller flips the
    raw bottom-up obs.rgb buffer top-down before appending). Writes nothing
    when the list is empty or imageio isn't installed.

    scale: integer upscale factor. LIBERO's offscreen render is small (128x128
           by default); upsampling pixel-replicate keeps the video readable
           without re-rendering the env. Pass 1 to disable.
    """
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        logger.warning("imageio not installed; skipping rollout video at %s",
                       out_path)
        return
    if scale and scale > 1:
        frames = [np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
                  for f in frames]
    try:
        # macro_block_size=1 disables ffmpeg's resize warning for non-16-multiple
        # dims (LIBERO defaults to 128x128 which is fine, but bigger renders may
        # not be).
        imageio.mimwrite(str(out_path.with_suffix(".mp4")), frames,
                         fps=fps, macro_block_size=1)
    except Exception as e:
        logger.warning("mp4 write failed (%s); writing gif instead", e)
        imageio.mimwrite(str(out_path.with_suffix(".gif")), frames,
                         duration=1.0 / fps)
