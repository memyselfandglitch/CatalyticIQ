#!/usr/bin/env python3
"""Train a lightweight surrogate from simulation sweep outputs."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_PREFIXES = ("feed_",)
DEFAULT_FEATURE_COLUMNS = [
    "temperature_c",
    "pressure_bar",
    "ghsv_h",
    "time_on_stream_h",
    "predicted_sty_g_h_gcat",
    "validation_score",
    "known_family_similarity",
    "thermodynamic_delta_g_kj_mol",
    "equilibrium_constant_kp",
    "equilibrium_conversion_pct",
    "cantera_equilibrium_conversion_pct",
    "reaction_forward_margin",
    "catalyst_rate_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simulation surrogate from a sweep CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Sweep CSV from generate_cantera_sweep.py")
    parser.add_argument("--target", default="simulated_sty_g_h_gcat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    features = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns and c != target]
    features.extend(c for c in df.columns if c.startswith(DEFAULT_FEATURE_PREFIXES) and c != target)
    return sorted(set(features), key=features.index)


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        return None
    return float(r2_score(y_true, y_pred))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    features = _feature_columns(df, args.target)
    if not features:
        raise ValueError("No numeric feature columns found for surrogate training.")

    model_df = df.copy()
    y = pd.to_numeric(model_df[args.target], errors="coerce")
    valid = y.notna()
    model_df = model_df.loc[valid].copy()
    y = y.loc[valid].astype(float)
    if len(model_df) < 4:
        raise ValueError(f"Need at least 4 valid rows to train a surrogate, got {len(model_df)}.")

    x = _numeric_frame(model_df, features)
    test_size = min(max(args.test_size, 0.1), 0.5)
    if len(model_df) < 12:
        test_size = max(1 / len(model_df), 0.25)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=args.seed,
    )
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    pred_test = pipe.predict(x_test)
    pred_train = pipe.predict(x_train)

    metrics = {
        "input_csv": str(args.input),
        "target": args.target,
        "n_rows": int(len(model_df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "features": features,
        "train_r2": _safe_r2(y_train.to_numpy(), pred_train),
        "test_r2": _safe_r2(y_test.to_numpy(), pred_test),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "simulation_surrogate.pkl"
    metrics_path = args.output_dir / "metrics.json"
    with model_path.open("wb") as fh:
        pickle.dump({"model": pipe, "features": features, "target": args.target}, fh)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote surrogate -> {model_path}")
    print(f"Wrote metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
