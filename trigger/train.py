"""Train and validate the replan-trigger MLP.

Expected data files:
    <data-dir>/train.npz   with arrays X [N, 13] float32, y [N] in {0, 1}
    <data-dir>/val.npz     same schema

Usage:
    python -m trigger.train \
        --data-dir trigger/data \
        --out trigger/checkpoints/trigger.pt \
        --epochs 200
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trigger.model import INPUT_DIM, TriggerMLP


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{path}: expected X of shape [N, {INPUT_DIM}], got {X.shape}"
        )
    if y.shape != (X.shape[0],):
        raise ValueError(
            f"{path}: expected y of shape [{X.shape[0]}], got {y.shape}"
        )
    return X, y


def compute_norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             threshold: float = 0.5) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y).item()
        p = torch.sigmoid(logits)
    pred = (p > threshold).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"bce": bce, "acc": acc, "prec": prec, "rec": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("trigger/data"))
    ap.add_argument("--out", type=Path,
                    default=Path("trigger/checkpoints/trigger.pt"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_tr, y_tr = load_split(args.data_dir / "train.npz")
    X_va, y_va = load_split(args.data_dir / "val.npz")
    print(f"train: N={X_tr.shape[0]:5d}  pos={y_tr.mean()*100:5.1f}%")
    print(f"val:   N={X_va.shape[0]:5d}  pos={y_va.mean()*100:5.1f}%")

    mu, sd = compute_norm_stats(X_tr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True,
    )
    X_va_t = torch.from_numpy(X_va).to(args.device)
    y_va_t = torch.from_numpy(y_va).to(args.device)

    model = TriggerMLP().to(args.device)
    model.set_norm_stats(mu, sd)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = model(xb)
            loss = bce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_bce = running / n

        m = evaluate(model, X_va_t, y_va_t)
        print(f"ep {epoch:4d}  train_bce {train_bce:.4f}  "
              f"val_bce {m['bce']:.4f}  acc {m['acc']:.3f}  "
              f"prec {m['prec']:.3f}  rec {m['rec']:.3f}")

    # Save the final-epoch model. With ~10 val examples, val_bce is too noisy
    # to reliably pick a best epoch; rely on weight_decay to prevent overfit.
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": INPUT_DIM,
        "epoch": args.epochs - 1,
        "val_metrics": m,
        "args": vars(args) | {"data_dir": str(args.data_dir),
                              "out": str(args.out)},
    }, args.out)
    print(f"\nfinal val_bce = {m['bce']:.4f}  acc {m['acc']:.3f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
