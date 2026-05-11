"""Runtime adapter that plugs the trained trigger MLP into `run_segment`.

Used as the `replan_decider` callable: at each step `t`, returns True iff
we should replan. Queries the MLP only at multiples of `trigger_freq`; at
other steps returns False without running the model.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim.rollout import RolloutState
from trigger.features import build_features
from trigger.model import TriggerMLP


class TriggerDecider:
    def __init__(self, *, model: TriggerMLP, threshold: float,
                 trigger_freq: int, horizon: int,
                 device: str | torch.device = "cpu"):
        if trigger_freq < 1:
            raise ValueError(f"trigger_freq must be >= 1, got {trigger_freq}")
        self.model = model.to(device).eval()
        self.threshold = float(threshold)
        self.trigger_freq = int(trigger_freq)
        self.horizon = int(horizon)
        self.device = torch.device(device)

    def __call__(self, state: RolloutState, t: int) -> bool:
        if t <= 0 or (t % self.trigger_freq) != 0:
            return False
        feats = build_features(
            ee_pos=state.obs.ee_pose[:3],
            dense_target=state.dense_targets[t],
            object_pos=state.obs.object_pos,
            goal_pos=state.obs.goal_pos,
            t=t, horizon=self.horizon,
        )
        x = torch.from_numpy(feats).to(self.device).unsqueeze(0)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x)).item()
        return p > self.threshold


def load_trigger_decider(ckpt_path: str | Path, *, threshold: float,
                         trigger_freq: int, horizon: int,
                         device: str | torch.device = "cpu") -> TriggerDecider:
    """Construct a TriggerDecider from a checkpoint saved by
    `trigger/train.py`. The checkpoint is a dict with `model_state` and
    `input_dim` keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TriggerMLP(input_dim=int(ckpt.get("input_dim", 13)))
    model.load_state_dict(ckpt["model_state"])
    return TriggerDecider(model=model, threshold=threshold,
                          trigger_freq=trigger_freq, horizon=horizon,
                          device=device)
