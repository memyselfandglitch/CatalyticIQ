#!/usr/bin/env python3
"""Import lab-feedback rows into the CatalyticIQ feedback store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.feedback.store import ExperimentRecord, FeedbackStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import feedback CSV rows into cache/feedback.duckdb.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("cache/feedback.duckdb"))
    parser.add_argument("--model-version", default="current")
    parser.add_argument("--user", default=None, help="Override user column.")
    return parser.parse_args()


def _maybe_float(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name not in row:
            continue
        value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return None


def _conditions(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src, dst in (
        ("temperature_c", "T_C"),
        ("pressure_bar", "P_bar"),
        ("h2_co2_ratio", "h2_co2"),
        ("ghsv_h", "ghsv_h"),
        ("time_on_stream_h", "time_on_stream_h"),
    ):
        if src in row and pd.notna(row[src]):
            out[dst] = row[src].item() if hasattr(row[src], "item") else row[src]
    if "conditions_json" in row and isinstance(row["conditions_json"], str) and row["conditions_json"].strip():
        try:
            out.update(json.loads(row["conditions_json"]))
        except json.JSONDecodeError:
            pass
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    store = FeedbackStore(args.db)
    count = 0
    for i, row in df.iterrows():
        pseudo = str(row.get("pseudo_smiles", row.get("candidate_id", f"row_{i}")))
        record = ExperimentRecord(
            candidate_id=str(row.get("candidate_id", pseudo)),
            pseudo_smiles=pseudo,
            composition_view=str(row.get("composition_view", pseudo)),
            measured_sty=_maybe_float(row, "measured_sty_g_h_gcat", "measured_sty"),
            predicted_sty=_maybe_float(row, "predicted_sty_g_h_gcat", "predicted_sty", "activity_head_sty"),
            measured_selectivity=_maybe_float(row, "measured_selectivity_pct", "measured_selectivity"),
            predicted_selectivity=_maybe_float(row, "predicted_selectivity_pct", "predicted_selectivity", "selectivity_proxy_pct"),
            measured_yield=_maybe_float(row, "measured_yield_pct", "measured_yield"),
            predicted_yield=_maybe_float(row, "predicted_yield_pct", "predicted_yield"),
            measured_stability_tos_h=_maybe_float(row, "measured_stability_tos_h", "stability_h"),
            predicted_stability_tos_h=_maybe_float(row, "predicted_stability_tos_h", "predicted_stability_h"),
            measured_enzyme_activity=_maybe_float(row, "measured_enzyme_activity", "measured_enzyme_activity_umol_min_mg"),
            predicted_enzyme_activity=_maybe_float(row, "predicted_enzyme_activity", "predicted_enzyme_activity_umol_min_mg"),
            conditions=_conditions(row),
            user=str(args.user or row.get("user", "demo_researcher")),
            notes=str(row.get("notes", "")),
            model_version=args.model_version,
        )
        store.log_experiment(record)
        count += 1
    print(f"Imported {count} feedback rows -> {args.db}")


if __name__ == "__main__":
    main()
