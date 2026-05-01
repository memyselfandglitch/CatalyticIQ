#!/usr/bin/env python3
"""
Train CatalyticIQ property heads (activity, selectivity) on cached CVAE
embeddings, and evaluate the descriptor-based stability proxy.

The script:

  1. Loads the existing CVAE checkpoint (default: dataset/co2_methanol/output_0_<TS>/).
  2. Embeds the merged dataset once and caches mu / latent / id arrays to
     dataset/co2_methanol/property_heads/embeddings.npz.
  3. Trains an ActivityHead (predicts methanol_sty) and a SelectivityHead
     (predicts MeOH selectivity %; CO selectivity column is left as a
     placeholder until a CO-selectivity-bearing dataset is integrated).
  4. Evaluates the descriptor-based StabilityHead on every row.
  5. Writes per-head metrics (R^2, MAE, calibration plots) under
     dataset/co2_methanol/property_heads/.

Run:
    python scripts/train_property_heads.py --pretrained_time 20260428_212044
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader as TorchLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from catcvae.ae import CVAE  # noqa: E402
from catcvae.dataset import (  # noqa: E402
    getDataLoader,
    getDatasetSplittingFinetune,
)
from catcvae.latent import embed  # noqa: E402
from catcvae.prediction import NN, NN_TASK  # noqa: E402
from catcvae.property_heads import (  # noqa: E402
    ActivityHead,
    HeadConfig,
    SelectivityHead,
    StabilityHead,
)
from catcvae.setup import ModelArgumentParser  # noqa: E402
from catcvae.stability_descriptors import composition_stability_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="co2_methanol", help="Dataset key in dataset/_dataset.py.")
    p.add_argument(
        "--pretrained_time",
        required=True,
        help="Timestamp suffix of the run directory (e.g. 20260428_212044).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full_csv", default="dataset/co2_methanol_full.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--output_dir",
        default="dataset/co2_methanol/property_heads",
        help="Directory for trained heads, embeddings cache and plots.",
    )
    p.add_argument(
        "--rebuild_embeddings",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists.",
    )
    return p.parse_args()


def setup_cvae_args(file: str, pretrained_time: str, seed: int) -> "argparse.Namespace":
    parser = ModelArgumentParser()
    return parser.setArgument(
        arguments=[
            "--file", file,
            "--pretrained_file", file,
            "--pretrained_time", pretrained_time,
            "--seed", str(seed),
            "--epochs", "0",
            "--class_weight", "disabled",
        ]
    )


def parse_components(catalyst_smiles: str) -> list[str]:
    return re.findall(r"\[([A-Za-z]+)\d*\]", str(catalyst_smiles))


def compute_or_load_embeddings(
    args,
    full_df: pd.DataFrame,
    output_model_dir: Path,
    cache_path: Path,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not rebuild:
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    print("[embed] reconstructing CVAE and NN head from", output_model_dir)

    AE = CVAE(
        embedding_setting=args.embedding_setting,
        encoding_setting=args.encoding_setting,
        decoding_setting=args.decoding_setting,
        emb_dim=args.emb_dim,
        emb_cond_dim=args.emb_cond_dim,
        cond_dim=args.cond_dim,
        device=args.device,
    ).to(args.device)
    if args.predictiontask == "yield":
        NN_PREDICTION = NN(in_dim=args.emb_dim + (3 * args.emb_cond_dim) + args.cond_dim, out_dim_class=1).to(args.device)
    else:
        NN_PREDICTION = NN_TASK(in_dim=args.emb_dim + (3 * args.emb_cond_dim) + args.cond_dim, out_dim_class=1).to(args.device)

    AE.load_state_dict(torch.load(output_model_dir / "model_ae.pth", map_location=args.device))
    NN_PREDICTION.load_state_dict(torch.load(output_model_dir / "model_nn.pth", map_location=args.device))
    AE.eval(); NN_PREDICTION.eval()

    train_split, val_split, test_split = getDatasetSplittingFinetune(args, datasets_df=None, datasets_dobj=None, augmentation=0)
    all_dobj = list(train_split) + list(val_split) + list(test_split)
    print(f"[embed] dataset objects: train={len(train_split)} val={len(val_split)} test={len(test_split)} total={len(all_dobj)}")

    from torch_geometric.loader import DataLoader as PyGLoader
    loader_all = PyGLoader(
        all_dobj,
        batch_size=args.batch_size,
        shuffle=False,
        follow_batch=["x_reactant", "x_reagent", "x_product", "x_catalyst"],
    )

    latent, mu, y_true, _y_pred, ids, condition = embed(loader_all, AE, NN_PREDICTION, device=args.device)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        latent=latent,
        mu=mu,
        y_true=y_true,
        ids=np.array([str(i) for i in ids]),
        condition=condition,
    )
    print(f"[embed] cached embeddings -> {cache_path}")
    return {
        "latent": latent,
        "mu": mu,
        "y_true": y_true,
        "ids": np.array([str(i) for i in ids]),
        "condition": condition,
    }


def join_embeddings_with_full(
    embeddings: dict[str, np.ndarray], full_df: pd.DataFrame
) -> pd.DataFrame:
    df_emb = pd.DataFrame({"id": embeddings["ids"]})
    df_emb["mu_index"] = np.arange(len(df_emb))
    full = full_df.copy()
    full["id"] = full["index"].astype(str)
    merged = df_emb.merge(full, on="id", how="left")
    if merged["index"].isna().any():
        missing = merged["id"][merged["index"].isna()].head().tolist()
        print(f"[warn] {merged['index'].isna().sum()} embedded ids not found in full csv; first few: {missing}")
    return merged


def train_regression_head(
    name: str,
    head: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    multi_target: bool = False,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    if multi_target and yt.ndim == 1:
        yt = yt.unsqueeze(-1)

    train_loader = TorchLoader(
        TensorDataset(Xt[train_idx], yt[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )

    optim = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = []

    for epoch in range(epochs):
        head.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optim.zero_grad()
            pred = head(xb)
            if multi_target and pred.ndim == 1:
                pred = pred.unsqueeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_idx)

        head.eval()
        with torch.no_grad():
            pv = head(Xt[val_idx])
            if multi_target and pv.ndim == 1:
                pv = pv.unsqueeze(-1)
            val_loss = loss_fn(pv, yt[val_idx]).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(head.state_dict(), output_dir / f"head_{name}.pth")

    head.load_state_dict(torch.load(output_dir / f"head_{name}.pth"))
    head.eval()
    with torch.no_grad():
        pred_test = head(Xt[test_idx])
        if multi_target and pred_test.ndim == 1:
            pred_test = pred_test.unsqueeze(-1)
        pred_test = pred_test.cpu().numpy()
    y_test = yt[test_idx].cpu().numpy()

    if multi_target:
        r2 = r2_score(y_test, pred_test, multioutput="uniform_average")
        mae = mean_absolute_error(y_test, pred_test, multioutput="uniform_average")
    else:
        if pred_test.ndim > 1:
            pred_test = pred_test.squeeze(-1)
        if y_test.ndim > 1:
            y_test = y_test.squeeze(-1)
        r2 = float(r2_score(y_test, pred_test))
        mae = float(mean_absolute_error(y_test, pred_test))

    fig, ax = plt.subplots(figsize=(5, 5))
    if multi_target:
        for k in range(y_test.shape[1]):
            ax.scatter(y_test[:, k], pred_test[:, k], alpha=0.5, label=f"target {k}")
    else:
        ax.scatter(y_test, pred_test, alpha=0.5)
    lims = [
        min(np.min(y_test), float(np.min(pred_test))),
        max(np.max(y_test), float(np.max(pred_test))),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name} head (R^2={r2:.3f}, MAE={mae:.3f})")
    if multi_target:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"calibration_{name}.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(history).to_csv(output_dir / f"history_{name}.csv", index=False)
    return {"r2": float(r2), "mae": float(mae), "n_train": int(n_train), "n_val": int(n_val), "n_test": int(len(test_idx))}


def evaluate_stability_head(merged: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    components_lists = [parse_components(s) for s in merged["catalyst"].fillna("")]
    temps = merged["temperature_c"].fillna(merged["temperature_c"].median()).to_numpy()
    scores = np.array(
        [composition_stability_score(c, t) for c, t in zip(components_lists, temps)]
    )

    out_df = merged[["id", "catalyst", "temperature_c"]].copy()
    out_df["stability_proxy"] = scores
    out_df.to_csv(output_dir / "stability_per_row.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=30, edgecolor="black")
    ax.set_xlabel("Stability proxy [0, 1]")
    ax.set_ylabel("Number of training rows")
    ax.set_title("Descriptor-based stability proxy distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "stability_histogram.png", dpi=150)
    plt.close(fig)

    return {
        "min": float(np.nanmin(scores)),
        "median": float(np.nanmedian(scores)),
        "max": float(np.nanmax(scores)),
        "share_high": float((scores > 0.6).mean()),
        "share_low": float((scores < 0.3).mean()),
    }


def main() -> None:
    cli = parse_args()
    output_dir = ROOT / cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_csv = ROOT / cli.full_csv
    if not full_csv.exists():
        print(f"[fatal] full csv not found: {full_csv}. Run scripts/prepare_co2_methanol_dataset.py first.")
        sys.exit(1)
    full_df = pd.read_csv(full_csv)

    args = setup_cvae_args(cli.file, cli.pretrained_time, cli.seed)
    output_model_dir = ROOT / "dataset" / args.file / f"output_{cli.seed}_{cli.pretrained_time}"
    if not output_model_dir.exists():
        print(f"[fatal] pretrained run dir missing: {output_model_dir}")
        sys.exit(1)

    cache_path = output_dir / "embeddings.npz"
    embeddings = compute_or_load_embeddings(args, full_df, output_model_dir, cache_path, cli.rebuild_embeddings)
    merged = join_embeddings_with_full(embeddings, full_df)

    mu = embeddings["mu"]
    in_dim = int(mu.shape[1])
    head_cfg = HeadConfig(in_dim=in_dim, hidden_dim=cli.hidden, dropout=cli.dropout)

    metrics: dict[str, dict[str, float]] = {}

    activity_mask = merged["methanol_sty"].notna().to_numpy()
    if activity_mask.sum() > 50:
        activity_head = ActivityHead(head_cfg)
        metrics["activity"] = train_regression_head(
            name="activity",
            head=activity_head,
            X=mu[activity_mask],
            y=merged.loc[activity_mask, "methanol_sty"].to_numpy(dtype=np.float32),
            output_dir=output_dir,
            epochs=cli.epochs,
            lr=cli.lr,
            batch_size=cli.batch_size,
            seed=cli.seed,
        )
    else:
        print("[skip] activity head: insufficient data.")

    sel_mask = merged["selectivity_meoh_pct"].notna().to_numpy()
    if sel_mask.sum() > 50:
        sel_head = SelectivityHead(head_cfg)
        # The dataset only ships methanol selectivity (TheMeCat). We still
        # supervise both outputs by setting CO selectivity = 100 - MeOH%
        # as a coarse complement; this gives the head a useful signal when
        # CO-selectivity data is later joined.
        meoh = merged.loc[sel_mask, "selectivity_meoh_pct"].to_numpy(dtype=np.float32)
        co = np.clip(100.0 - meoh, 0.0, 100.0)
        y_sel = np.stack([meoh, co], axis=1)
        metrics["selectivity"] = train_regression_head(
            name="selectivity",
            head=sel_head,
            X=mu[sel_mask],
            y=y_sel,
            output_dir=output_dir,
            epochs=cli.epochs,
            lr=cli.lr,
            batch_size=cli.batch_size,
            seed=cli.seed,
            multi_target=True,
        )
    else:
        print("[skip] selectivity head: insufficient labelled rows.")

    metrics["stability_proxy"] = evaluate_stability_head(merged, output_dir)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[done] metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
