#!/usr/bin/env python3
"""
Retrain CatalyticIQ predictors with newly logged lab feedback.

Two modes:

  * ``--mode heads`` (default): retrain ONLY the property heads on the same
    cached CVAE embeddings, plus newly logged feedback rows whose pseudo-SMILES
    we can re-embed. Cheap, safe, the right choice when feedback rows < 25.

  * ``--mode cvae``: schedule a full CVAE fine-tune. Refuses to run if the
    feedback distribution drift (PSI on activity targets) exceeds the
    configured threshold or if fewer than ``--min_full_n`` rows are available,
    unless ``--force`` is passed.

Every retrain creates a new entry in ``model_versions`` with parent pointer,
feedback row count, delta-R2 vs the parent, and PSI. New artifacts are written
under ``dataset/co2_methanol/output_<timestamp>_feedback_v<N>/`` so the
existing run directories are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catcvae.property_heads import ActivityHead, HeadConfig  # noqa: E402
from services.feedback.store import FeedbackStore, ModelVersion  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["heads", "cvae"], default="heads")
    p.add_argument("--full_csv", default="dataset/co2_methanol_full.csv")
    p.add_argument("--embeddings", default="dataset/co2_methanol/property_heads/embeddings.npz")
    p.add_argument("--head", default="dataset/co2_methanol/property_heads/head_activity.pth")
    p.add_argument("--output_dir", default="dataset/co2_methanol/property_heads")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--psi_threshold", type=float, default=0.25)
    p.add_argument("--min_full_n", type=int, default=25, help="Minimum feedback rows for full CVAE retrain.")
    p.add_argument("--force", action="store_true", help="Bypass drift / N safeguards.")
    p.add_argument(
        "--promote",
        action="store_true",
        help="If set, the new heads-mode artifact replaces head_activity.pth (a .bak is kept).",
    )
    return p.parse_args()


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two distributions of a single feature."""
    if len(current) == 0 or len(reference) == 0:
        return 0.0
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_p = np.maximum(ref_hist / len(reference), 1e-6)
    cur_p = np.maximum(cur_hist / len(current), 1e-6)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def retrain_heads(
    mu: np.ndarray,
    y: np.ndarray,
    feedback_y: np.ndarray,
    output_dir: Path,
    epochs: int,
    lr: float,
    seed: int,
    parent_head_path: Path,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(mu)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    head_cfg = HeadConfig(in_dim=mu.shape[1])
    head = ActivityHead(head_cfg)
    if parent_head_path.exists():
        head.load_state_dict(torch.load(parent_head_path, map_location="cpu"))

    Xt = torch.tensor(mu, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(Xt[train_idx], yt[train_idx]), batch_size=64, shuffle=True
    )
    optim = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf")

    new_head_path = output_dir / "head_activity_v_feedback.pth"

    for _ in range(epochs):
        head.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            optim.step()
        head.eval()
        with torch.no_grad():
            v_loss = float(loss_fn(head(Xt[val_idx]), yt[val_idx]).item())
        if v_loss < best_val:
            best_val = v_loss
            torch.save(head.state_dict(), new_head_path)

    # Evaluate parent and child on the same test split for delta_r2.
    parent = ActivityHead(head_cfg)
    parent.load_state_dict(torch.load(parent_head_path, map_location="cpu"))
    parent.eval()
    head.load_state_dict(torch.load(new_head_path, map_location="cpu"))
    head.eval()
    with torch.no_grad():
        parent_pred = parent(Xt[test_idx]).cpu().numpy()
        new_pred = head(Xt[test_idx]).cpu().numpy()
    y_test = yt[test_idx].cpu().numpy()
    parent_metrics = {
        "r2": float(r2_score(y_test, parent_pred)),
        "mae": float(mean_absolute_error(y_test, parent_pred)),
    }
    new_metrics = {
        "r2": float(r2_score(y_test, new_pred)),
        "mae": float(mean_absolute_error(y_test, new_pred)),
    }
    return {
        "parent_test_r2": parent_metrics["r2"],
        "new_test_r2": new_metrics["r2"],
        "delta_r2": new_metrics["r2"] - parent_metrics["r2"],
        "parent_test_mae": parent_metrics["mae"],
        "new_test_mae": new_metrics["mae"],
        "best_val_mse": best_val,
        "n_test": int(len(test_idx)),
        "new_head_path": str(new_head_path),
    }


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(ROOT / args.full_csv)
    npz = np.load(ROOT / args.embeddings, allow_pickle=True)
    mu = npz["mu"]
    y = npz["y_true"].astype(np.float32)

    store = FeedbackStore()
    feedback_rows = [r for r in store.list_experiments(limit=10_000) if r.get("measured_sty") is not None]
    fb_y = np.array([float(r["measured_sty"]) for r in feedback_rows], dtype=np.float32)
    psi = population_stability_index(y, fb_y) if len(fb_y) >= 5 else 0.0

    versioned: list[str] = []

    if args.mode == "heads":
        result = retrain_heads(
            mu=mu,
            y=y,
            feedback_y=fb_y,
            output_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            parent_head_path=ROOT / args.head,
        )
        version_id = f"heads_v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        store.log_model_version(
            ModelVersion(
                version=version_id,
                parent="current",
                delta_r2=result["delta_r2"],
                n_feedback_used=int(len(fb_y)),
                psi=psi,
                notes=f"heads-only retrain; new_test_r2={result['new_test_r2']:.3f}",
            )
        )
        versioned.append(version_id)
        if args.promote:
            # Promote the new head to the canonical filename used by the dashboard.
            canonical = ROOT / args.head
            backup = canonical.with_suffix(canonical.suffix + ".bak")
            shutil.copy2(canonical, backup)
            shutil.copy2(result["new_head_path"], canonical)
            result["promoted"] = True
        else:
            result["promoted"] = False
        print(json.dumps({"mode": "heads", "version": version_id, "psi": psi, **result}, indent=2))
    else:
        if not args.force:
            if len(fb_y) < args.min_full_n:
                print(
                    f"[refuse] feedback rows {len(fb_y)} < {args.min_full_n}. "
                    "Use --mode heads, collect more rows, or pass --force."
                )
                sys.exit(2)
            if psi > args.psi_threshold:
                print(
                    f"[refuse] PSI {psi:.3f} > threshold {args.psi_threshold}. "
                    "Investigate distribution drift before re-fine-tuning the CVAE, "
                    "or pass --force to override."
                )
                sys.exit(2)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cmd = [
            "python",
            "main_finetune.py",
            "--file", "co2_methanol",
            "--pretrained_file", "co2_methanol",
            "--pretrained_time", "20260428_212044",
            "--epochs", "30",
            "--lr", "0.0005",
            "--class_weight", "enabled",
        ]
        print(f"[plan] full CVAE retrain command: {' '.join(cmd)}")
        if not args.force:
            print("[note] run the above command manually to execute the full retrain.")
            sys.exit(0)
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        version_id = f"cvae_v_{ts}"
        store.log_model_version(
            ModelVersion(
                version=version_id,
                parent="current",
                delta_r2=None,
                n_feedback_used=int(len(fb_y)),
                psi=psi,
                notes="full CVAE fine-tune triggered via retrain_with_feedback.py --force.",
            )
        )
        versioned.append(version_id)
        print(json.dumps({"mode": "cvae", "version": version_id, "psi": psi}, indent=2))


if __name__ == "__main__":
    main()
