"""Replan-trigger MLP (Q3 in docs/method.txt).

Input layout (13 dims, in order):
    0:3   tracking_error_vec  (dense_targets[t] - ee_pose[:3], world frame)
    3     tracking_error_norm
    4     phase               (t / horizon)
    5:8   ee_to_object_vec    (object_pos - ee_pose[:3])
    8     ee_to_object_dist
    9:12  ee_to_goal_vec      (goal_pos - ee_pose[:3])
    12    ee_to_goal_dist

Normalization stats are stored as buffers in the module so the deployed model
takes raw features and the calling code never has to remember to normalize.
"""
import torch
import torch.nn as nn

INPUT_DIM = 13
HIDDEN_DIM = 16


class TriggerMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("mu", torch.zeros(input_dim))
        self.register_buffer("sd", torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].bias)

    def set_norm_stats(self, mu, sd) -> None:
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mu) / self.sd
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
