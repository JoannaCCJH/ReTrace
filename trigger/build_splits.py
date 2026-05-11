"""Aggregate per-task `episode_NNN.npz` files into train.npz / val.npz.

Per task, the first `train_episodes_per_task` episodes contribute rows to
train, the next `val_episodes_per_task` contribute to val. Rows within
each split are shuffled deterministically.

Usage:
  python -m trigger.build_splits \\
      --raw_dirs trigger/data/raw/task_a trigger/data/raw/task_b \\
      --out_dir trigger/data \\
      --train_episodes_per_task 100 --val_episodes_per_task 10
"""
import argparse
import json
from pathlib import Path

import numpy as np


def collect_split_rows(task_dir: Path,
                       episode_idx_range) -> tuple[np.ndarray, np.ndarray,
                                                    list[dict]]:
    """Walk `task_dir/episode_NNN.npz` for the given episode indices and
    return stacked X, y plus per-row metadata. Missing episodes are
    skipped (logged via metadata list).
    """
    xs, ys, meta = [], [], []
    for idx in episode_idx_range:
        path = task_dir / f"episode_{idx:03d}.npz"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=False)
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.int8)
        cps = d["checkpoints"].astype(np.int32)
        for j in range(X.shape[0]):
            xs.append(X[j])
            ys.append(int(y[j]))
            meta.append({
                "task_dir": str(task_dir),
                "episode_idx": int(idx),
                "checkpoint": int(cps[j]),
                "condition": str(d["condition"].item().decode()
                                 if hasattr(d["condition"].item(), "decode")
                                 else d["condition"]),
                "seed": int(d["seed"]),
            })
    if not xs:
        return (np.zeros((0, 13), dtype=np.float32),
                np.zeros((0,), dtype=np.int8), meta)
    return (np.stack(xs).astype(np.float32),
            np.asarray(ys, dtype=np.int8),
            meta)


def write_splits(raw_dirs: list[Path], out_dir: Path, *,
                 train_episodes_per_task: int,
                 val_episodes_per_task: int,
                 seed: int = 0) -> None:
    train_X, train_y, train_meta = [], [], []
    val_X, val_y, val_meta = [], [], []
    for task_dir in raw_dirs:
        tr_X, tr_y, tr_m = collect_split_rows(
            task_dir,
            range(0, train_episodes_per_task),
        )
        va_X, va_y, va_m = collect_split_rows(
            task_dir,
            range(train_episodes_per_task,
                  train_episodes_per_task + val_episodes_per_task),
        )
        train_X.append(tr_X); train_y.append(tr_y); train_meta.extend(tr_m)
        val_X.append(va_X);   val_y.append(va_y);   val_meta.extend(va_m)

    train_X = np.concatenate(train_X, axis=0) if train_X else np.zeros((0, 13), dtype=np.float32)
    train_y = np.concatenate(train_y, axis=0) if train_y else np.zeros((0,), dtype=np.int8)
    val_X = np.concatenate(val_X, axis=0) if val_X else np.zeros((0, 13), dtype=np.float32)
    val_y = np.concatenate(val_y, axis=0) if val_y else np.zeros((0,), dtype=np.int8)

    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(train_X.shape[0])
    val_perm = rng.permutation(val_X.shape[0])
    train_X = train_X[train_perm]; train_y = train_y[train_perm]
    train_meta = [train_meta[i] for i in train_perm]
    val_X = val_X[val_perm]; val_y = val_y[val_perm]
    val_meta = [val_meta[i] for i in val_perm]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train.npz", X=train_X, y=train_y)
    np.savez(out_dir / "val.npz", X=val_X, y=val_y)
    (out_dir / "build_meta.json").write_text(json.dumps({
        "seed": seed,
        "train_episodes_per_task": train_episodes_per_task,
        "val_episodes_per_task": val_episodes_per_task,
        "raw_dirs": [str(d) for d in raw_dirs],
        "train_rows": train_meta,
        "val_rows": val_meta,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--train_episodes_per_task", type=int, default=100)
    ap.add_argument("--val_episodes_per_task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    write_splits(args.raw_dirs, args.out_dir,
                 train_episodes_per_task=args.train_episodes_per_task,
                 val_episodes_per_task=args.val_episodes_per_task,
                 seed=args.seed)
    print(f"wrote train.npz / val.npz / build_meta.json under {args.out_dir}")


if __name__ == "__main__":
    main()
