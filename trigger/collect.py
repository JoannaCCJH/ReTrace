"""Data-collection driver for the replan-trigger MLP.

Per task: 110 episodes (55 nominal + 55 perturbed). Per episode:
  1. Scan pass (nominal rollout, no replan) to discover t_approach.
  2. Sample K checkpoints uniformly from [0, t_approach).
     For perturbed episodes, also sample t_perturb in [0, t_approach) and a
     unit-circle direction with Uniform(1, 3) cm magnitude.
  3. Base pass (acts as the no-replan branch): rollout with `on_step`
     hook that (a) perturbs the mug at t_perturb, (b) snapshots + builds
     features at each checkpoint.
  4. Replan branches: restore each snapshot, replan once at t_cp,
     continue to horizon, record outcome.
  5. Label by counterfactual table; emit per-episode npz.

Outputs (under --output dir):
  raw/<task>/episode_NNN.npz
  raw/<task>/summary.json   # per-episode metadata + skip reasons

Usage:
  python -m trigger.collect \\
      --config cfg/train.yaml --resume <ckpt> \\
      --benchmark libero_90 --task <task_name> \\
      --object_obs_key porcelain_mug_1_pos \\
      --perturb_body_name porcelain_mug_1_main \\
      --goal_site_name study_table_desk_caddy_right_region \\
      --output trigger/data/raw/STUDY_SCENE3_white_mug \\
      --num_episodes 110 --num_nominal 55 --K 5 --R_close 0.05
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

# DictToNamespace must be importable under this module so torch.load can
# unpickle checkpoints saved when __main__ was the eval entry point. See
# sim/run_libero_eval.py for the same workaround.
from trainer.trainer import DictToNamespace  # noqa: F401

from sim.libero_env import LiberoEnv
from sim.model_adapter import build_predict_fn
from sim.rollout import build_initial_state, finalize_trial, run_segment
from trigger.features import build_features
from trigger.perturb import sample_perturbation
from trigger.snapshot import capture, restore

logger = logging.getLogger(__name__)

R_CLOSE_DEFAULT = 0.05


def detect_t_approach(logs: dict, r_close: float) -> int:
    """First step t where ||ee_xy - object_xy|| < r_close. -1 if never."""
    arr = np.asarray(logs["ee_obj_xy"], dtype=np.float32)
    mask = arr < r_close
    if not mask.any():
        return -1
    return int(np.argmax(mask))


def label_outcome(*, outcome_base: bool, outcome_replan: bool) -> int:
    """Counterfactual label: 1 iff replan succeeded AND base failed."""
    return 1 if (outcome_replan and not outcome_base) else 0


def run_one_episode(env, predict_fn, *, episode_idx: int, num_nominal: int,
                    seed_base: int, K: int, R_close: float, dz_scale: float,
                    placement_mode: str) -> dict:
    seed = seed_base + episode_idx
    cond = "nominal" if episode_idx < num_nominal else "perturbed"
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Scan pass: discover t_approach.
    scan_state = build_initial_state(env, predict_fn, seed,
                                     dz_scale=dz_scale,
                                     placement_mode=placement_mode,
                                     viz_replans="none", viz_dir=None)
    scan_state = run_segment(env, scan_state, predict_fn,
                             t_end=env.horizon, replan_at=[],
                             dz_scale=dz_scale,
                             placement_mode=placement_mode,
                             viz_replans="none", viz_dir=None)
    t_approach = detect_t_approach(scan_state.logs, R_close)
    if t_approach <= 0:
        return {"skipped": True, "reason": "no_approach", "seed": int(seed),
                "condition": cond, "t_approach": int(t_approach),
                "elapsed_s": time.time() - t0}

    # Sample checkpoints + perturbation.
    k_eff = int(min(K, t_approach))
    cps = sorted(rng.choice(t_approach, size=k_eff, replace=False).tolist())
    if cond == "perturbed":
        t_perturb = int(rng.integers(0, t_approach))
        dxdy = sample_perturbation(rng)
    else:
        t_perturb, dxdy = -1, (0.0, 0.0)

    # Base pass.
    state = build_initial_state(env, predict_fn, seed,
                                dz_scale=dz_scale,
                                placement_mode=placement_mode,
                                viz_replans="none", viz_dir=None)
    snapshots: dict[int, object] = {}
    feats: dict[int, np.ndarray] = {}
    cps_set = set(cps)

    def _on_step(s, t: int) -> None:
        if t == t_perturb:
            s.obs = env.perturb_object_xy(*dxdy)
        if t in cps_set:
            snapshots[t] = capture(env, s)
            feats[t] = build_features(
                ee_pos=s.obs.ee_pose[:3],
                dense_target=s.dense_targets[t],
                object_pos=s.obs.object_pos,
                goal_pos=s.obs.goal_pos,
                t=t, horizon=env.horizon,
            )

    state = run_segment(env, state, predict_fn, t_end=env.horizon,
                        replan_at=[], dz_scale=dz_scale,
                        placement_mode=placement_mode,
                        viz_replans="none", viz_dir=None,
                        on_step=_on_step)
    outcome_base = bool(state.success)

    # Replan branches. Skip any checkpoint whose snapshot didn't fire
    # (e.g., early termination of the base pass shortened the loop).
    outcomes_replan: list[bool] = []
    used_cps: list[int] = []
    used_feats: list[np.ndarray] = []
    for t_cp in cps:
        if t_cp not in snapshots:
            continue
        snap = snapshots[t_cp]
        rstate = restore(env, snap)
        rstate = run_segment(env, rstate, predict_fn,
                             t_end=env.horizon,
                             replan_at=[t_cp], dz_scale=dz_scale,
                             placement_mode=placement_mode,
                             viz_replans="none", viz_dir=None)
        outcomes_replan.append(bool(rstate.success))
        used_cps.append(int(t_cp))
        used_feats.append(feats[t_cp])

    if not used_cps:
        return {"skipped": True, "reason": "no_snapshots_fired",
                "seed": int(seed), "condition": cond,
                "t_approach": int(t_approach),
                "elapsed_s": time.time() - t0}

    labels = [label_outcome(outcome_base=outcome_base, outcome_replan=o_r)
              for o_r in outcomes_replan]
    return {
        "skipped": False,
        "X": np.stack(used_feats).astype(np.float32),
        "y": np.array(labels, dtype=np.int8),
        "checkpoints": np.array(used_cps, dtype=np.int32),
        "outcome_base": outcome_base,
        "outcome_replan": np.array(outcomes_replan, dtype=bool),
        "condition": cond,
        "t_approach": int(t_approach),
        "t_perturb": int(t_perturb),
        "dxdy": (float(dxdy[0]), float(dxdy[1])),
        "seed": int(seed),
        "elapsed_s": time.time() - t0,
    }


def _save_episode_npz(out_path: Path, ep: dict) -> None:
    np.savez(
        out_path,
        X=ep["X"], y=ep["y"],
        checkpoints=ep["checkpoints"],
        outcome_base=np.bool_(ep["outcome_base"]),
        outcome_replan=ep["outcome_replan"],
        condition=np.bytes_(ep["condition"]),
        t_approach=np.int32(ep["t_approach"]),
        t_perturb=np.int32(ep["t_perturb"]),
        dxdy=np.array(ep["dxdy"], dtype=np.float32),
        seed=np.int32(ep["seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True, help="checkpoint path")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--object_obs_key", required=True)
    parser.add_argument("--perturb_body_name", required=True,
                        help="MuJoCo body name of the target object whose "
                             "freejoint qpos gets perturbed.")
    parser.add_argument("--goal_body_name", default=None)
    parser.add_argument("--goal_site_name", default=None)
    parser.add_argument("--output", required=True,
                        help="Per-task directory: writes episode_NNN.npz + "
                             "summary.json under this path.")
    parser.add_argument("--num_episodes", type=int, default=110)
    parser.add_argument("--num_nominal", type=int, default=55)
    parser.add_argument("--K", type=int, default=5,
                        help="Checkpoints per episode.")
    parser.add_argument("--R_close", type=float, default=R_CLOSE_DEFAULT,
                        help="ee_xy <-> obj_xy distance defining t_approach.")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Per-episode seed = seed_base + episode_idx.")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--dz_scale", type=float, default=1.0)
    parser.add_argument("--placement_mode", choices=["trace", "descend"],
                        default="descend")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config dot-notation overrides."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    from test_example import TrajectoryDiffusionTest, load_config
    cfg_dict = load_config(args.config, overrides=args.override)
    trainer = TrajectoryDiffusionTest(cfg_dict, output_dir=str(out_dir))
    trainer.load_checkpoint(args.resume)
    trainer.model.diffusion_decoder.set_data_act_statistics(
        trainer.action_max, trainer.action_min
    )
    trainer.model.eval()
    image_size = cfg_dict["model"]["vision_encoder"]["image_size"]
    predict_fn = build_predict_fn(trainer, image_size=image_size,
                                  guidance_scale=args.guidance_scale,
                                  use_depth=args.use_depth)

    goal_body = None if args.goal_site_name else args.goal_body_name
    env = LiberoEnv(
        task_name=args.task,
        benchmark_name=args.benchmark,
        object_obs_key=args.object_obs_key,
        goal_body_name=goal_body,
        goal_site_name=args.goal_site_name,
        perturb_body_name=args.perturb_body_name,
    )
    logger.info("env: benchmark=%s task=%s perturb_body=%s",
                args.benchmark, args.task, args.perturb_body_name)

    summary: list[dict] = []
    try:
        for i in range(args.num_episodes):
            ep = run_one_episode(env, predict_fn,
                                 episode_idx=i,
                                 num_nominal=args.num_nominal,
                                 seed_base=args.seed_base,
                                 K=args.K, R_close=args.R_close,
                                 dz_scale=args.dz_scale,
                                 placement_mode=args.placement_mode)
            ep_path = out_dir / f"episode_{i:03d}.npz"
            if not ep.get("skipped"):
                _save_episode_npz(ep_path, ep)
                logger.info(
                    "episode %d (%s seed=%d) saved: outcome_base=%s "
                    "labels=%s checkpoints=%s elapsed=%.1fs",
                    i, ep["condition"], ep["seed"],
                    ep["outcome_base"], ep["y"].tolist(),
                    ep["checkpoints"].tolist(), ep["elapsed_s"],
                )
            else:
                logger.warning("episode %d skipped: %s", i, ep["reason"])
            # Append a JSON-serializable summary entry.
            summary.append({
                "episode_idx": i,
                "skipped": bool(ep.get("skipped", False)),
                "reason": ep.get("reason"),
                "seed": ep.get("seed"),
                "condition": ep.get("condition"),
                "t_approach": ep.get("t_approach"),
                "t_perturb": ep.get("t_perturb"),
                "dxdy": list(ep.get("dxdy", (0.0, 0.0))),
                "outcome_base": bool(ep["outcome_base"])
                if not ep.get("skipped") else None,
                "outcome_replan": ep["outcome_replan"].tolist()
                if not ep.get("skipped") else None,
                "labels": ep["y"].tolist() if not ep.get("skipped") else None,
                "checkpoints": ep["checkpoints"].tolist()
                if not ep.get("skipped") else None,
                "elapsed_s": ep.get("elapsed_s"),
            })
            # Persist progressively so a crash leaves a partial summary.
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
