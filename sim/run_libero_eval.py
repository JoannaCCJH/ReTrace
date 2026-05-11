"""CLI entrypoint: run N trials on one LIBERO task, report success rate."""
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

# DictToNamespace must be importable under this module's namespace so that
# torch.load can unpickle the checkpoint (the training run saved it under
# __main__, which is `sim.run_libero_eval` when invoked with `python -m`).
from trainer.trainer import DictToNamespace  # noqa: F401

from sim.libero_env import LiberoEnv
from sim.model_adapter import build_predict_fn
from sim.rollout import TrialResult, run_trial

logger = logging.getLogger(__name__)


def aggregate_trials(results: Iterable[TrialResult], out_dir: Path) -> dict:
    results = list(results)
    success_mask = [r.success for r in results]
    success_steps = [r.steps for r in results if r.success]
    failure_modes: dict[str, int] = {}
    for r in results:
        failure_modes[r.failure_mode] = failure_modes.get(r.failure_mode, 0) + 1
    summary = {
        "num_trials": len(results),
        "success_rate": float(np.mean(success_mask)) if results else 0.0,
        "mean_steps_to_success": float(np.mean(success_steps)) if success_steps else None,
        "failure_modes": failure_modes,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _save_trial_artifacts(trial_dir: Path, result: TrialResult) -> None:
    """Save the per-trial summary + the reset-frame rgb.

    Pixel-space + world-frame overlays already get written into the same
    trial_dir by run_trial (selected_traces.png, world_trace_overlay.png,
    pred_traces.png, rollout.mp4); no need to re-render them here.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(json.dumps({
        "success": result.success,
        "steps": result.steps,
        "failure_mode": result.failure_mode,
        "diag": result.diag,
    }, indent=2))
    if result.initial_rgb is not None:
        # obs.rgb is the raw OpenGL bottom-up buffer; flip top-down so the PNG
        # matches the orientation of the other artifacts in this dir
        # (pred_traces.png, selected_traces.png, rollout.mp4 are all top-down).
        from PIL import Image
        import numpy as np
        rgb_topdown = np.ascontiguousarray(result.initial_rgb[::-1])
        Image.fromarray(rgb_topdown).save(trial_dir / "initial_rgb.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True, help="checkpoint path")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--output", default="./libero_results/run1")
    parser.add_argument("--benchmark", default="libero_goal",
                        help="LIBERO benchmark name (e.g. libero_goal, libero_10).")
    parser.add_argument("--task", default="put_the_cream_cheese_in_the_bowl",
                        help="Task name within --benchmark.")
    parser.add_argument("--object_obs_key", default="cream_cheese_1_pos",
                        help="Key in the obs dict for the manipulation target's "
                             "world position. Must be set per task -- the default "
                             "is for put_the_cream_cheese_in_the_bowl.")
    parser.add_argument("--goal_body_name", default="akita_black_bowl_1_main",
                        help="MuJoCo body name for the placement goal. Default "
                             "matches the cream-cheese task's bowl. Use this "
                             "when the goal IS a body (e.g. drop into a bowl).")
    parser.add_argument("--goal_site_name", default=None,
                        help="MuJoCo site name for region-based goals (e.g. "
                             "'study_table_desk_caddy_right_region' for 'place "
                             "the mug to the right of the caddy'). When set, "
                             "overrides --goal_body_name. LIBERO auto-creates a "
                             "site at each table region whose name is "
                             "<table_name>_<region_name>.")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--dz_scale", type=float, default=1.0,
                        help="Scale factor on model dz output (training-monocular -> metric meters)")
    parser.add_argument("--placement_mode", choices=["trace", "descend"],
                        default="trace",
                        help="How carry-phase target.z is computed once the "
                             "object enters the goal region's xy footprint. "
                             "'trace': follow dense_targets verbatim (the "
                             "model + linear-interp anchor decide altitude). "
                             "'descend': override target.z to (goal_z + half_z "
                             "+ GOAL_Z_PLACE_OFFSET) so the EE actively drops "
                             "to a low placement altitude before the gripper "
                             "opens. Use 'descend' when the trace's altitude "
                             "drift is too slow to bring the EE down before "
                             "release.")
    parser.add_argument("--replan_freq", type=int, default=0,
                        help="Replan every N env steps. 0 (default) = one-shot "
                             "at t=0. With N>0, the predict->lift->top-K->world "
                             "->anchor->interp pipeline reruns at t in {N, 2N, "
                             "...} on the current obs, and the tail of "
                             "dense_targets is overwritten with a fresh plan. "
                             "Mutually exclusive with --trigger_ckpt.")
    parser.add_argument("--trigger_ckpt", default=None,
                        help="Path to a trigger MLP checkpoint (from "
                             "trigger.train). When set, replan decisions are "
                             "made at runtime by the MLP -- queried every "
                             "--trigger_freq steps; replans fire when the "
                             "predicted probability exceeds --trigger_threshold.")
    parser.add_argument("--trigger_threshold", type=float, default=0.5,
                        help="Sigmoid threshold above which the trigger MLP "
                             "fires a replan. Only used when --trigger_ckpt "
                             "is set.")
    parser.add_argument("--trigger_freq", type=int, default=20,
                        help="Query the trigger MLP every N env steps. Only "
                             "used when --trigger_ckpt is set. The model is "
                             "not called on intermediate steps.")
    parser.add_argument("--viz_replans", choices=["first", "all", "none"],
                        default="all",
                        help="Per-plan viz artifact policy. 'all' (default) "
                             "writes pred_traces.png / selected_traces.png / "
                             "world_trace_overlay.png / traces_3d.npz / "
                             "combined_trace_world.npz under "
                             "trial_NNN/replan_TTT/ for every plan. 'first' "
                             "only writes them for the t=0 plan. 'none' "
                             "suppresses per-plan artifacts (rollout.mp4 is "
                             "unaffected).")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--use_depth", action="store_true",
                        help="Feed obs.depth into the model. Off by default: the "
                             "training-time depth distribution was monocular "
                             "relative, while obs.depth is metric meters; feeding "
                             "it is OOD. With the flag off, the model's "
                             "learnable_depth_mask_token is used instead.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config dot-notation overrides (e.g. model.decoder.attention_head_dim=96)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    from test_example import TrajectoryDiffusionTest, load_config
    cfg_dict = load_config(args.config, overrides=args.override)
    trainer = TrajectoryDiffusionTest(cfg_dict, output_dir=str(out_dir))
    trainer.load_checkpoint(args.resume)
    trainer.model.diffusion_decoder.set_data_act_statistics(
        trainer.action_max, trainer.action_min
    )
    trainer.model.eval()

    # Action stats let you eyeball the model's output scale per channel.
    # Channel 2 (dz) magnitude tells you what dz_scale should compensate for:
    # if action_max[2]-action_min[2] is O(0.05) it's normalized monocular,
    # O(0.5) is roughly inverse-depth, O(2.0) is closer to metric meters.
    am = trainer.action_max.cpu().numpy() if hasattr(trainer.action_max, 'cpu') else np.asarray(trainer.action_max)
    an = trainer.action_min.cpu().numpy() if hasattr(trainer.action_min, 'cpu') else np.asarray(trainer.action_min)
    logger.info("action_min per channel: %s", an.ravel().tolist())
    logger.info("action_max per channel: %s", am.ravel().tolist())
    logger.info("dz range (channel 2): [%.6f, %.6f]; using dz_scale=%g",
                float(an.ravel()[2]), float(am.ravel()[2]), args.dz_scale)

    image_size = cfg_dict["model"]["vision_encoder"]["image_size"]
    predict_fn = build_predict_fn(trainer, image_size=image_size,
                                  guidance_scale=args.guidance_scale,
                                  use_depth=args.use_depth)
    logger.info("model depth conditioning: %s",
                "obs.depth (metric)" if args.use_depth else "mask token (depth=None)")

    if args.replan_freq > 0 and args.trigger_ckpt is not None:
        parser.error("--replan_freq and --trigger_ckpt are mutually "
                     "exclusive: pick a fixed schedule OR the learned trigger.")

    replan_decider = None
    if args.trigger_ckpt is not None:
        from trigger.decider import load_trigger_decider
        from sim.constants import EPISODE_HORIZON
        replan_decider = load_trigger_decider(
            args.trigger_ckpt,
            threshold=args.trigger_threshold,
            trigger_freq=args.trigger_freq,
            horizon=EPISODE_HORIZON,
        )
        logger.info("trigger: ckpt=%s threshold=%.3f freq=%d",
                    args.trigger_ckpt, args.trigger_threshold,
                    args.trigger_freq)

    # site overrides body when both are present (the body default is the
    # cream-cheese task's bowl; for region-goal tasks the user passes a site).
    goal_body = None if args.goal_site_name else args.goal_body_name
    env = LiberoEnv(
        task_name=args.task,
        benchmark_name=args.benchmark,
        object_obs_key=args.object_obs_key,
        goal_body_name=goal_body,
        goal_site_name=args.goal_site_name,
    )
    logger.info("LIBERO env: benchmark=%s task=%s object=%s goal=%s",
                args.benchmark, args.task, args.object_obs_key,
                f"site={args.goal_site_name}" if args.goal_site_name
                else f"body={goal_body}")
    results = []
    for i in range(args.num_trials):
        trial_dir = out_dir / f"trial_{i:03d}"
        r = run_trial(env, predict_fn=predict_fn, seed=args.seed_offset + i,
                      dz_scale=args.dz_scale,
                      placement_mode=args.placement_mode,
                      replan_freq=args.replan_freq,
                      replan_decider=replan_decider,
                      viz_replans=args.viz_replans,
                      viz_dir=trial_dir)
        _save_trial_artifacts(trial_dir, r)
        results.append(r)
        logger.info("trial %d: success=%s mode=%s steps=%d",
                    i, r.success, r.failure_mode, r.steps)
    env.close()

    summary = aggregate_trials(results, out_dir)
    logger.info("run_summary: %s", summary)


if __name__ == "__main__":
    main()
