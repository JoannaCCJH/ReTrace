"""Single-frame prediction visualizer.

Resets a LIBERO task, runs the diffusion model on the first agentview, and
draws all 400 grid-trace predictions on the RGB. No rollout, no oracle, no
trace-selection -- just "what does the model think every grid point will do."

Mirrors test_example.py's recovery: concat the grid xy as the trace's start
point, cumsum across time, scale to pixels.
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np

# Required so torch.load can unpickle DictToNamespace under this module's name.
from trainer.trainer import DictToNamespace  # noqa: F401

from sim.constants import GRID_XY
from sim.model_adapter import build_predict_fn

logger = logging.getLogger(__name__)


def overlay_checkpoint_arch(cfg_dict: dict, ckpt_path: str) -> None:
    """Overlay model.decoder + d_model from the checkpoint's saved config onto cfg.

    Yaml files alongside checkpoints often record training-time defaults rather
    than the --override values that were actually used to construct the model.
    The authoritative architecture lives in ckpt['config']. Pulling it back out
    here means the script "just works" without the caller needing to remember
    the right --override flags.
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("config")
    if saved is None:
        return

    def to_dict(o):
        if hasattr(o, "__dict__"):
            return {k: to_dict(v) for k, v in o.__dict__.items()}
        if isinstance(o, dict):
            return {k: to_dict(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to_dict(x) for x in o]
        return o

    saved_dict = to_dict(saved)
    decoder = saved_dict.get("model", {}).get("decoder")
    if decoder:
        # DictToNamespace.get inherits as a method when serialized; strip that key.
        decoder = {k: v for k, v in decoder.items() if k != "get"}
        cfg_dict.setdefault("model", {})
        existing = cfg_dict["model"].get("decoder", {})
        existing.update(decoder)
        cfg_dict["model"]["decoder"] = existing
        logger.info("decoder arch from checkpoint: heads=%s head_dim=%s layers=%s",
                    decoder.get("num_attention_heads"),
                    decoder.get("attention_head_dim"),
                    decoder.get("num_layers"))
    if "d_model" in saved_dict:
        cfg_dict["d_model"] = saved_dict["d_model"]


def reconstruct_absolute_traces(pred: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Concat grid xy as start, cumsum, scale to pixels. Returns [N, 1+T, 2].

    Intentionally not clipped to [0, 1] here -- diffusion drift past image
    bounds is itself a diagnostic signal we want to see in the overlay.
    """
    xy_with_start = np.concatenate([GRID_XY[:, None, :], pred[..., :2]], axis=1)
    abs_xy_norm = np.cumsum(xy_with_start, axis=1)
    return np.stack([abs_xy_norm[..., 0] * img_w,
                     abs_xy_norm[..., 1] * img_h], axis=-1).astype(np.float32)


def save_depth_viz(depth: np.ndarray, out_path: Path) -> None:
    """Save the prepped depth tensor as a magma-colored heatmap with colorbar.

    depth: [H, W] float, in whatever units the model received (post-flip and
    post-resize -- exactly what came out of prepare_model_inputs). Per-image
    min/max normalization is automatic via imshow.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    im = ax.imshow(depth, cmap="magma")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_overlay(rgb: np.ndarray, traces_px: np.ndarray, out_path: Path,
                 movement_threshold: float = 0.01) -> int:
    """Overlay all traces on rgb. Returns count rendered.

    movement_threshold is in normalized [0, 1] image coords; traces whose total
    L2 path length falls below it are dropped to reduce clutter from grid cells
    that the model predicts as essentially stationary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # rgb is already in display orientation (the same tensor the model saw).
    H, W = rgb.shape[:2]
    rgb_disp = rgb
    traces_disp = traces_px

    diffs = np.diff(traces_px, axis=1)
    path_len = np.linalg.norm(diffs, axis=-1).sum(axis=-1) / max(H, W)
    keep = path_len > movement_threshold

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(rgb_disp)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    cmap = plt.get_cmap("turbo")
    rendered = 0
    for i, t in enumerate(traces_disp):
        if not keep[i]:
            continue
        color = cmap(i / len(traces_disp))
        ax.plot(t[:, 0], t[:, 1], "-", color=color, linewidth=0.6, alpha=0.7)
        ax.scatter(t[0, 0], t[0, 1], c="lime", s=4, zorder=3)
        ax.scatter(t[-1, 0], t[-1, 1], c="red", s=6, zorder=3)
        rendered += 1
    ax.set_axis_off()
    ax.set_title(f"{rendered}/{len(traces_px)} traces (movement > {movement_threshold})",
                 fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return rendered


KNOWN_BENCHMARKS = ["libero_10", "libero_90", "libero_goal",
                    "libero_object", "libero_spatial"]


def resolve_benchmark(task_name: str, preferred: str | None) -> tuple[str, "object"]:
    """Find the benchmark that actually contains task_name. Returns (name, benchmark).

    Tries the preferred benchmark first, then falls through KNOWN_BENCHMARKS.
    Raises SystemExit with close-match diagnostics if the task is nowhere.
    """
    from libero.libero.benchmark import get_benchmark
    order = []
    if preferred:
        order.append(preferred)
    order += [b for b in KNOWN_BENCHMARKS if b not in order]

    close: dict[str, list[str]] = {}
    tokens = [t for t in task_name.split("_") if len(t) > 3]
    for bn in order:
        try:
            bm = get_benchmark(bn)()
        except Exception as e:
            logger.debug("benchmark %s skipped: %s", bn, e)
            continue
        names = [t.name for t in bm.tasks]
        if task_name in names:
            if bn != preferred:
                logger.info("task '%s' not in '%s'; found in '%s' instead",
                            task_name, preferred, bn)
            return bn, bm
        hits = [n for n in names if any(tok in n for tok in tokens)]
        if hits:
            close[bn] = hits[:5]

    msg = f"task '{task_name}' not in any of {order}."
    if close:
        msg += "\nClose matches per benchmark:"
        for bn, names in close.items():
            msg += f"\n  {bn}: {names}"
    raise SystemExit(msg)


def get_first_view(benchmark_name: str, task_name: str, img_hw: tuple[int, int],
                   seed: int, with_depth: bool = False
                   ) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Render the first agentview frame. Returns (rgb, depth_or_None, language).

    When with_depth=True, depth is converted to metric meters via
    robosuite's get_real_depth_map (same path LiberoEnv uses), so units match
    what the rollout-time predict_fn would see.
    """
    os.environ.setdefault("MUJOCO_GL", "glx")
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_name, benchmark = resolve_benchmark(task_name, benchmark_name)
    task_names = [t.name for t in benchmark.tasks]
    task_idx = task_names.index(task_name)
    bddl = benchmark.get_task_bddl_file_path(task_idx)
    language = benchmark.tasks[task_idx].language
    logger.info("using benchmark=%s task=%s", benchmark_name, task_name)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=img_hw[0],
        camera_widths=img_hw[1],
        camera_names=["agentview"],
        camera_depths=with_depth,
        controller="OSC_POSE",
    )
    try:
        env.seed(seed)
        obs = env.reset()
        depth = None
        if with_depth:
            import robosuite.utils.camera_utils as camera_utils
            depth_raw = np.asarray(obs["agentview_depth"]).astype(np.float32)
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.squeeze(-1)
            depth = camera_utils.get_real_depth_map(
                env.sim, depth_raw[..., None]
            ).squeeze(-1).astype(np.float32)
    finally:
        # Holding the env open for one reset is wasteful; close before model load.
        try:
            env.close()
        except Exception:
            pass

    rgb = np.asarray(obs["agentview_image"]).astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb, depth, language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/n/netscratch/kempner_sham_lab/Lab/jcheng65/"
                                            "checkpoints/TraceGen_Libero/train.yaml")
    parser.add_argument("--resume", default="/n/netscratch/kempner_sham_lab/Lab/jcheng65/"
                                            "checkpoints/TraceGen_Libero/tracegen_libero.pth")
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task",
                        default="STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy")
    parser.add_argument("--output", default="./pred_viz/task1_depth")
    parser.add_argument("--img_size", type=int, default=128,
                        help="LIBERO render size (square). 128 matches obs.depth path; "
                             "256 reads better in the overlay.")
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--use_depth", action="store_true",
                        help="If set, feed obs.depth to the model. Default off (mask token).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--movement_threshold", type=float, default=0.01)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config dot-notation overrides (e.g. model.decoder.num_layers=6)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: render the first view from LIBERO before loading the model -- the
    # robosuite GLX context and torch CUDA init can fight for the same display.
    logger.info("rendering first view: benchmark=%s task=%s seed=%d size=%d use_depth=%s",
                args.benchmark, args.task, args.seed, args.img_size, args.use_depth)
    rgb_buf, depth_buf, language = get_first_view(
        args.benchmark, args.task,
        (args.img_size, args.img_size), args.seed,
        with_depth=args.use_depth,
    )
    logger.info("language: %s", language)

    # Step 2: load checkpoint.
    from test_example import TrajectoryDiffusionTest, load_config
    cfg_dict = load_config(args.config, overrides=args.override)
    overlay_checkpoint_arch(cfg_dict, args.resume)
    trainer = TrajectoryDiffusionTest(cfg_dict, output_dir=str(out_dir))
    trainer.load_checkpoint(args.resume)
    trainer.model.diffusion_decoder.set_data_act_statistics(
        trainer.action_max, trainer.action_min
    )
    trainer.model.eval()

    image_size = cfg_dict["model"]["vision_encoder"]["image_size"]

    # Step 3: run inference. predict_fn returns the prepped (top-down, resized)
    # rgb + depth alongside the prediction so the saved frames are a ground-truth
    # answer to "what does the model actually see?" without re-prepping here.
    predict_fn = build_predict_fn(trainer, image_size=image_size,
                                  guidance_scale=args.guidance_scale,
                                  use_depth=args.use_depth)
    logger.info("running diffusion (guidance_scale=%g)", args.guidance_scale)
    rgb_t, depth_t, pred = predict_fn(rgb_buf, depth=depth_buf, language=language)
    logger.info("pred shape=%s, xy range=[%.4f, %.4f], dz range=[%.4f, %.4f]",
                pred.shape, pred[..., :2].min(), pred[..., :2].max(),
                pred[..., 2].min(), pred[..., 2].max())

    rgb_input = (rgb_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(rgb_input).save(out_dir / "first_view.png")
    logger.info("saved model-input view to %s", out_dir / "first_view.png")

    if depth_t is not None:
        depth_input = depth_t[0].cpu().numpy()
        logger.info("depth fed to model: shape=%s range=[%.4f, %.4f]",
                    depth_input.shape, float(depth_input.min()), float(depth_input.max()))
        save_depth_viz(depth_input, out_dir / "first_depth.png")
        logger.info("saved model-input depth to %s", out_dir / "first_depth.png")

    # Step 4: reconstruct + draw on the same image the model saw.
    H, W = rgb_input.shape[:2]
    traces_px = reconstruct_absolute_traces(pred, img_h=H, img_w=W)
    out_png = out_dir / "pred_traces.png"
    rendered = save_overlay(rgb_input, traces_px, out_png,
                            movement_threshold=args.movement_threshold)
    logger.info("rendered %d/%d traces -> %s", rendered, traces_px.shape[0], out_png)


if __name__ == "__main__":
    main()
