#!/usr/bin/env python3
"""
Retrain CatalyticIQ predictors with newly logged lab feedback.

Two modes:

  * ``--mode heads`` (default): retrain ONLY the property heads. Training batches
    use the literature 80/10 split on cached μ, **plus** all measured feedback
    rows that can be re-embedded through the frozen CVAE (same graph schema as
    the fine-tuned reaction). Val/test remain literature-only for comparable R².

  * ``--mode cvae``: schedule a full CVAE fine-tune. Refuses to run if the
    feedback distribution drift (PSI on activity targets) exceeds the
    configured threshold or if fewer than ``--min_full_n`` rows are available,
    unless ``--force`` is passed.

Every retrain creates a new entry in ``model_versions`` with parent pointer,
feedback row count, delta-R2 vs the parent, and PSI. New artifacts are written
under ``dataset/<file>/property_heads/`` (heads) or a new ``output_*`` dir for CVAE.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catcvae.ae import CVAE  # noqa: E402
from catcvae.feedback_embed import embed_feedback_mu  # noqa: E402
from catcvae.prediction import NN, NN_TASK  # noqa: E402
from catcvae.property_heads import ActivityHead, HeadConfig  # noqa: E402
from catcvae.setup import ModelArgumentParser  # noqa: E402
from services.feedback.store import FeedbackStore, ModelVersion  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["heads", "cvae"], default="heads")
    p.add_argument("--file", default="co2_methanol", help="Dataset key in dataset/_dataset.py.")
    p.add_argument(
        "--pretrained_time",
        default="20260507_173839",
        help="Timestamp suffix of the CVAE run under dataset/<file>/output_0_<ts>/.",
    )
    p.add_argument(
        "--full_csv",
        default=None,
        help="Merged CSV with targets (default: dataset/<file>_full.csv).",
    )
    p.add_argument(
        "--embeddings",
        default=None,
        help="Cached μ (default: dataset/<file>/property_heads/embeddings.npz).",
    )
    p.add_argument(
        "--head",
        default=None,
        help="Activity head checkpoint (default: dataset/<file>/property_heads/head_activity.pth).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Head output dir (default: dataset/<file>/property_heads).",
    )
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--feedback_weight",
        type=int,
        default=3,
        help="Repeat each measured feedback row this many times in heads-mode training.",
    )
    p.add_argument("--psi_threshold", type=float, default=0.25)
    p.add_argument("--min_full_n", type=int, default=25, help="Minimum feedback rows for full CVAE retrain.")
    p.add_argument("--force", action="store_true", help="Bypass drift / N safeguards.")
    p.add_argument(
        "--promote",
        action="store_true",
        help="If set, replace head_activity.pth only when held-out R2 does not regress unless --force is also set.",
    )
    return p.parse_args()


def _default_paths(file: str) -> dict[str, Path]:
    return {
        "full_csv": ROOT / f"dataset/{file}_full.csv",
        "embeddings": ROOT / f"dataset/{file}/property_heads/embeddings.npz",
        "head": ROOT / f"dataset/{file}/property_heads/head_activity.pth",
        "output_dir": ROOT / f"dataset/{file}/property_heads",
    }


def setup_cvae_args(file: str, pretrained_time: str, seed: int):
    parser = ModelArgumentParser()
    return parser.setArgument(
        arguments=[
            "--file",
            file,
            "--pretrained_file",
            file,
            "--pretrained_time",
            pretrained_time,
            "--seed",
            str(seed),
            "--epochs",
            "0",
            "--class_weight",
            "disabled",
        ]
    )


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


def _load_cvae(args, output_model_dir: Path) -> CVAE:
    AE = CVAE(
        embedding_setting=args.embedding_setting,
        encoding_setting=args.encoding_setting,
        decoding_setting=args.decoding_setting,
        emb_dim=args.emb_dim,
        emb_cond_dim=args.emb_cond_dim,
        cond_dim=args.cond_dim,
        device=args.device,
    ).to(args.device)
    AE.load_state_dict(torch.load(output_model_dir / "model_ae.pth", map_location=args.device))
    AE.eval()
    return AE


def retrain_heads(
    mu: np.ndarray,
    y: np.ndarray,
    mu_fb: np.ndarray,
    y_fb: np.ndarray,
    output_dir: Path,
    epochs: int,
    lr: float,
    seed: int,
    parent_head_path: Path,
    feedback_weight: int,
) -> dict:
    """Literature 80/10/10 split; feedback (μ,y) appended only to the training set."""
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

    mu_train_lit = mu[train_idx]
    y_train_lit = y[train_idx]
    feedback_weight = max(1, int(feedback_weight))
    if len(mu_fb) > 0:
        mu_fb_train = np.repeat(mu_fb, feedback_weight, axis=0)
        y_fb_train = np.repeat(y_fb, feedback_weight, axis=0)
        mu_train = np.vstack([mu_train_lit, mu_fb_train])
        y_train = np.concatenate([y_train_lit, y_fb_train])
    else:
        mu_train = mu_train_lit
        y_train = y_train_lit

    Xt_train = torch.tensor(mu_train, dtype=torch.float32)
    yt_train = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(Xt_train, yt_train), batch_size=64, shuffle=True)

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

    if not new_head_path.exists():
        torch.save(head.state_dict(), new_head_path)

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
        "n_feedback_train_rows": int(len(mu_fb)),
        "feedback_weight": int(feedback_weight),
        "n_feedback_weighted_rows": int(len(mu_fb) * feedback_weight),
    }


def main() -> None:
    args = parse_args()
    paths = _default_paths(args.file)
    full_csv = Path(args.full_csv) if args.full_csv else paths["full_csv"]
    emb_path = Path(args.embeddings) if args.embeddings else paths["embeddings"]
    head_path = Path(args.head) if args.head else paths["head"]
    out_dir = Path(args.output_dir) if args.output_dir else paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(emb_path, allow_pickle=True)
    mu = npz["mu"]
    y = npz["y_true"].astype(np.float32)

    store = FeedbackStore()
    feedback_rows = [r for r in store.list_experiments(limit=10_000) if r.get("measured_sty") is not None]
    fb_y = np.array([float(r["measured_sty"]) for r in feedback_rows], dtype=np.float32)
    psi = population_stability_index(y, fb_y) if len(fb_y) >= 5 else 0.0

    mu_fb = np.zeros((0, mu.shape[1]), dtype=np.float32)
    y_fb = np.zeros((0,), dtype=np.float32)
    if args.file == "co2_methanol" and feedback_rows:
        cvae_args = setup_cvae_args(args.file, args.pretrained_time, args.seed)
        out_cvae = ROOT / "dataset" / args.file / f"output_{cvae_args.seed}_{args.pretrained_time}"
        if out_cvae.is_dir():
            AE = _load_cvae(cvae_args, out_cvae)
            mu_fb, y_fb = embed_feedback_mu(
                cvae_args,
                feedback_rows,
                AE,
                cvae_args.device,
                reaction="co2_methanol",
                emb_dim=int(cvae_args.emb_dim),
            )
        elif out_cvae.is_dir():
            print(
                json.dumps(
                    {
                        "warn": (
                            f"missing CVAE weights at {ckpt_ae}; feedback rows are NOT embedded for this "
                            "heads retrain (literature-only μ). Copy model_ae.pth into that folder, or pass "
                            "--cvae-run-dir pointing to a run directory that contains it."
                        ),
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "warn": f"CVAE dir missing {out_cvae}; feedback not embedded for heads retrain.",
                    },
                    indent=2,
                )
            )

    versioned: list[str] = []

    if args.mode == "heads":
        result = retrain_heads(
            mu=mu,
            y=y,
            mu_fb=mu_fb,
            y_fb=y_fb,
            output_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            parent_head_path=head_path,
            feedback_weight=args.feedback_weight,
        )
        version_id = f"heads_v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        store.log_model_version(
            ModelVersion(
                version=version_id,
                parent="current",
                delta_r2=result["delta_r2"],
                n_feedback_used=int(len(fb_y)),
                psi=psi,
                notes=(
                    f"heads-only retrain; new_test_r2={result['new_test_r2']:.3f}; "
                    f"embedded_feedback_train={result['n_feedback_train_rows']}"
                ),
            )
        )
        versioned.append(version_id)
        should_promote = bool(args.promote) and (result["delta_r2"] >= 0.0 or bool(args.force))
        if should_promote:
            canonical = head_path
            backup = canonical.with_suffix(canonical.suffix + ".bak")
            shutil.copy2(canonical, backup)
            shutil.copy2(result["new_head_path"], canonical)
            result["promoted"] = True
        else:
            result["promoted"] = False
            if args.promote and result["delta_r2"] < 0.0 and not args.force:
                result["promotion_skipped_reason"] = (
                    f"held-out R2 regressed by {result['delta_r2']:.6f}; "
                    "pass --force to promote anyway"
                )
        print(
            json.dumps(
                {
                    "mode": "heads",
                    "version": version_id,
                    "psi": psi,
                    "n_feedback_logged": int(len(fb_y)),
                    **result,
                },
                indent=2,
            )
        )
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
            "--file",
            args.file,
            "--pretrained_file",
            args.file,
            "--pretrained_time",
            args.pretrained_time,
            "--epochs",
            "30",
            "--lr",
            "0.0005",
            "--class_weight",
            "enabled",
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
