"""Adapter that turns TrajectoryDiffusionTest into a predict_fn for run_trial."""
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF

from sim.constants import TRAJECTORY_HORIZON


def prepare_model_inputs(rgb: np.ndarray, depth: np.ndarray | None,
                         image_size: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn raw LIBERO obs into model-ready tensors.

    robosuite's agentview buffer is OpenGL bottom-up (row 0 = bottom of scene).
    Training PNGs were saved in natural orientation (row 0 = top of scene), so
    the model expects top-down input. Flip both rgb and depth here so callers
    can pass raw obs.rgb / obs.depth without remembering the convention.

    Returns (rgb_t [1,3,S,S], depth_t [1,1,S,S], valid [1]) on `device`.
    """
    rgb = np.ascontiguousarray(rgb[::-1])
    if depth is not None:
        depth = np.ascontiguousarray(depth[::-1])

    # Training feeds [0, 1] tensors via NormalizeTransform(normalize_image=False);
    # the backbone processors (SigLIP, DINOv2) do their own normalization.
    rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    rgb_t = TF.resize(rgb_t, [image_size, image_size], antialias=True)
    rgb_t = rgb_t.unsqueeze(0).to(device)

    if depth is None:
        # valid=0 → model substitutes learnable_depth_mask_token; tensor content is unused.
        depth_t = torch.zeros(1, 1, image_size, image_size, device=device)
        valid = torch.zeros(1, dtype=torch.float32, device=device)
    else:
        depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        depth_t = TF.resize(depth_t, [image_size, image_size], antialias=True)
        depth_t = depth_t.to(device)
        valid = torch.ones(1, dtype=torch.float32, device=device)
    return rgb_t, depth_t, valid


def build_predict_fn(trainer, image_size: int, guidance_scale: float = 2.0,
                     seed: int | None = 0,
                     use_depth: bool = False) -> Callable:
    """Return fn(rgb_uint8, depth, language) -> (rgb_t [3,S,S], depth_t [1,S,S] | None, pred [400,T,3]).

    rgb_t / depth_t are the exact top-down, resized tensors the model saw
    (unbatched, on trainer.device). depth_t is None when no real depth was
    conditioned on (use_depth=False, or depth=None) -- in that case the model
    used learnable_depth_mask_token and the placeholder zeros it received are
    not meaningful to surface.

    Diffusion sampling draws fresh noise on each call; pass seed to make
    predictions reproducible (the seed is re-applied before every inference
    so back-to-back calls with the same inputs give the same trace).

    use_depth: when False, the depth argument to the returned predict_fn is
    ignored and the model's learnable_depth_mask_token branch is used (matches
    the depth-missing samples seen during training). LIBERO's obs.depth is
    metric meters, while the training-time depth distribution was monocular
    relative scale, so feeding it directly is OOD; the cleanest way to avoid
    that mismatch when no calibrated monocular depth source is available is
    to skip depth conditioning entirely.
    """
    from test_helpers import predict_trajectory_simple

    def predict_fn(rgb: np.ndarray, depth: np.ndarray | None, language: str):
        used_depth = depth if use_depth else None
        rgb_t, depth_t, valid = prepare_model_inputs(
            rgb, used_depth, image_size, trainer.device
        )
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        pred = predict_trajectory_simple(
            trainer=trainer,
            images=rgb_t,
            texts=[language],
            depth_maps=depth_t,
            is_depth_valid=valid,
            guidance_scale=guidance_scale,
        )
        pred = pred[0].detach().cpu().numpy()
        if pred.shape[-1] == 2:
            pred = np.concatenate([pred, np.zeros_like(pred[..., :1])], axis=-1)
        pred = pred[:, :TRAJECTORY_HORIZON, :]
        out_depth = depth_t[0] if used_depth is not None else None
        return rgb_t[0], out_depth, pred.astype(np.float32)

    return predict_fn
