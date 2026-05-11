"""Snapshot/restore wrapping MuJoCo state + the Python rollout state.

A `Snapshot` is everything you need to rewind a rollout to an earlier t:
  - The MuJoCo sim state (qpos, qvel, time, ...) via `env.save_sim_state`.
  - A deep copy of the `RolloutState` (oracle, dense_targets, latches, logs).
  - The numpy + torch RNG state at capture time, so plan_from_obs calls
    during replan branches are reproducible.

The caller is responsible for splicing the restored state back into its
rollout loop. `restore` also refreshes `state.obs` from the env so the
next iteration of `run_segment` reads the post-restore observation.
"""
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sim.rollout import RolloutState


@dataclass
class Snapshot:
    sim_state: Any                 # opaque MjSimState
    rollout_state: RolloutState    # deep-copied
    numpy_rng_state: tuple
    torch_rng_state: torch.Tensor


def capture(env, state: RolloutState) -> Snapshot:
    """Bundle env sim state + rollout state + RNG into a `Snapshot`.

    Deep-copies the rollout state so post-capture mutations of `state`
    don't affect the snapshot.
    """
    return Snapshot(
        sim_state=env.save_sim_state(),
        rollout_state=copy.deepcopy(state),
        numpy_rng_state=np.random.get_state(),
        torch_rng_state=torch.random.get_rng_state(),
    )


def restore(env, snap: Snapshot) -> RolloutState:
    """Rewind the env + RNG, return a fresh RolloutState ready for
    `run_segment`. The returned state is a deep copy of the snapshot's,
    with `obs` refreshed from the restored env so the next step reads
    the post-restore observation.
    """
    refreshed_obs = env.restore_sim_state(snap.sim_state)
    np.random.set_state(snap.numpy_rng_state)
    torch.random.set_rng_state(snap.torch_rng_state)
    state = copy.deepcopy(snap.rollout_state)
    state.obs = refreshed_obs
    state.final_obs = refreshed_obs
    return state
