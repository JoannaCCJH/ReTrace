"""Thin LIBERO wrapper: agentview only, OSC_POSE controller, metric depth.

This cluster's EGL driver does not support PLATFORM_DEVICE, so robosuite's
GPU EGL path fails. We force MUJOCO_GL=glx at import time, which requires
an X display (e.g. run under xvfb-run).
"""
import os
os.environ.setdefault("MUJOCO_GL", "glx")

from dataclasses import dataclass

import numpy as np

from sim.constants import EPISODE_HORIZON

BENCHMARK_NAME = None
DEFAULT_TASK_NAME = None
OBJECT_OBS_KEY = None
GOAL_BODY_NAME = None


@dataclass
class LiberoObs:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    cam_to_world: np.ndarray
    ee_pose: np.ndarray
    object_pos: np.ndarray
    goal_pos: np.ndarray
    # Box half-extents (x, y, z) for site-based goals. Used to gate "is the
    # object inside the region's footprint and above its top face?" When the
    # goal is a body (not a region), this is None and the oracle falls back to
    # the legacy GOAL_XY_RADIUS / GOAL_Z_CLEARANCE radius gate.
    goal_half_extents: np.ndarray | None
    language: str


class LiberoEnv:
    horizon = EPISODE_HORIZON

    def __init__(self, task_name: str = DEFAULT_TASK_NAME,
                 benchmark_name: str = BENCHMARK_NAME,
                 object_obs_key: str = OBJECT_OBS_KEY,
                 goal_body_name: str | None = GOAL_BODY_NAME,
                 goal_site_name: str | None = None,
                 perturb_body_name: str | None = None,
                 img_hw: tuple[int, int] = (128, 128)):
        if (goal_body_name is None) == (goal_site_name is None):
            raise ValueError(
                "LiberoEnv: provide exactly one of goal_body_name or "
                "goal_site_name (use site for region-based goals like "
                "'<table>_<region>_region')."
            )
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        benchmark = get_benchmark(benchmark_name)()
        task_idx = next(i for i, t in enumerate(benchmark.tasks) if t.name == task_name)
        task_bddl = benchmark.get_task_bddl_file_path(task_idx)
        task = benchmark.tasks[task_idx]

        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl,
            camera_heights=img_hw[0],
            camera_widths=img_hw[1],
            camera_names=["agentview"],
            camera_depths=True,
            controller="OSC_POSE",
        )
        self._language = task.language
        self._img_hw = img_hw
        self._object_obs_key = object_obs_key
        self._goal_body_name = goal_body_name
        self._goal_site_name = goal_site_name
        self._perturb_body_name = perturb_body_name
        self._last_obs: dict | None = None

    def reset(self, seed: int = 0) -> LiberoObs:
        self._env.seed(seed)
        obs = self._env.reset()
        self._last_obs = obs
        return self._build_obs(obs)

    def step(self, action: np.ndarray) -> tuple[LiberoObs, float, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        info = dict(info) if info else {}
        info["success"] = bool(self._env.env._check_success())
        return self._build_obs(obs), float(reward), bool(done), info

    def close(self) -> None:
        self._env.close()

    def save_sim_state(self):
        """Snapshot the MuJoCo state. Returned object is opaque -- pass it
        back to `restore_sim_state` to rewind.
        """
        return self._env.sim.get_state()

    def restore_sim_state(self, state) -> "LiberoObs":
        """Rewind the MuJoCo state and re-derive obs. The caller still needs
        to restore any *Python-level* rollout state (oracle, dense_targets,
        latches) separately; this method handles the env side only.

        Returns the obs at the restored state so the caller can refresh
        `state.obs` without an explicit observe() step.
        """
        self._env.sim.set_state(state)
        self._env.sim.forward()  # re-derive xpos / contact / sensors
        return self._build_obs(self._env.env._get_observations(force_update=True))

    def perturb_object_xy(self, dx: float, dy: float) -> "LiberoObs":
        """Translate the configured perturb body in the xy plane by
        (dx, dy) meters, then re-derive obs.

        Implemented by mutating the freejoint qpos slice for the body.
        Requires `perturb_body_name` to have been set at construction.
        Returns the refreshed obs so the rollout loop's in-memory obs can
        be updated without re-stepping.
        """
        if self._perturb_body_name is None:
            raise RuntimeError(
                "LiberoEnv: perturb_object_xy requires `perturb_body_name` "
                "to be set at construction."
            )
        sim = self._env.sim
        body_id = sim.model.body_name2id(self._perturb_body_name)
        jnt_adr = sim.model.body_jntadr[body_id]
        if jnt_adr < 0:
            raise RuntimeError(
                f"LiberoEnv: body {self._perturb_body_name!r} has no joint "
                "(can't perturb its qpos)."
            )
        qpos_adr = sim.model.jnt_qposadr[jnt_adr]
        sim.data.qpos[qpos_adr + 0] += float(dx)
        sim.data.qpos[qpos_adr + 1] += float(dy)
        sim.forward()
        return self._build_obs(self._env.env._get_observations(force_update=True))

    def _build_obs(self, obs: dict) -> LiberoObs:
        import robosuite.utils.camera_utils as camera_utils
        H, W = self._img_hw
        K = camera_utils.get_camera_intrinsic_matrix(self._env.sim, "agentview", H, W)
        cam_to_world = camera_utils.get_camera_extrinsic_matrix(self._env.sim, "agentview")

        rgb = np.asarray(obs["agentview_image"]).astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))

        depth_raw = np.asarray(obs["agentview_depth"]).astype(np.float32)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw.squeeze(-1)
        metric_depth = camera_utils.get_real_depth_map(
            self._env.sim, depth_raw[..., None]
        ).squeeze(-1).astype(np.float32)

        ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        ee_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        ee_pose = np.concatenate([ee_pos, ee_quat]).astype(np.float32)

        sim = self._env.sim
        if self._goal_site_name is not None:
            sid = sim.model.site_name2id(self._goal_site_name)
            goal_pos = np.asarray(sim.data.site_xpos[sid]).astype(np.float32)
            # site_size is half-extents (x, y, z) for box sites; for non-box
            # sites the components have other meanings. LIBERO's region sites
            # are boxes, so this is fine. (If you point at a non-box site the
            # gate will treat its first three numbers as half-extents.)
            goal_half_extents = np.asarray(
                sim.model.site_size[sid][:3]
            ).astype(np.float32)
        else:
            goal_bid = sim.model.body_name2id(self._goal_body_name)
            goal_pos = sim.data.body_xpos[goal_bid].astype(np.float32)
            goal_half_extents = None

        return LiberoObs(
            rgb=rgb,
            depth=metric_depth,
            K=K.astype(np.float32),
            cam_to_world=cam_to_world.astype(np.float32),
            ee_pose=ee_pose,
            object_pos=np.asarray(obs[self._object_obs_key], dtype=np.float32),
            goal_pos=goal_pos,
            goal_half_extents=goal_half_extents,
            language=self._language,
        )
