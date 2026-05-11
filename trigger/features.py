"""Feature builder for the replan-trigger MLP.

Channel layout MUST match the documented order in `trigger/model.py`. The
MLP's normalization buffers and weights are trained on this exact column
order, so any reorder silently breaks the deployed model.
"""
import numpy as np

from trigger.model import INPUT_DIM


def build_features(*, ee_pos: np.ndarray, dense_target: np.ndarray,
                   object_pos: np.ndarray, goal_pos: np.ndarray,
                   t: int, horizon: int) -> np.ndarray:
    """Return the 13-dim feature vector at checkpoint t.

    All inputs are world-frame; vectors are length-3.
    """
    ee = np.asarray(ee_pos, dtype=np.float32)
    tgt = np.asarray(dense_target, dtype=np.float32)
    obj = np.asarray(object_pos, dtype=np.float32)
    goal = np.asarray(goal_pos, dtype=np.float32)

    track_err = tgt - ee
    ee_to_obj = obj - ee
    ee_to_goal = goal - ee

    feats = np.empty(INPUT_DIM, dtype=np.float32)
    feats[0:3] = track_err
    feats[3] = np.linalg.norm(track_err)
    feats[4] = float(t) / float(horizon)
    feats[5:8] = ee_to_obj
    feats[8] = np.linalg.norm(ee_to_obj)
    feats[9:12] = ee_to_goal
    feats[12] = np.linalg.norm(ee_to_goal)
    return feats
