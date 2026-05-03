#!/usr/bin/env python3
"""
Map an external syngas→ethanol corpus into CatalyticIQ CSVs:
``dataset/syngas_ethanol.csv`` (CVAE training) and ``dataset/syngas_ethanol_full.csv``.

**Example sources:** Zenodo ``11639494`` (Suvarna et al., FeCoCuZr HAS / modelling
xlsx files), PNNL higher-alcohol work (often PDF — digitise or use paper SI).

Expected input columns (rename your source headers to match, or pass ``--rename-json``):

  * ``catalyst`` — composition text or pseudo-SMILES token string
  * ``ethanol_sty`` — space-time yield in g EtOH / h / g_cat (adjust with --sty_scale if mg-scale)
  * ``temperature_c``, ``pressure_bar`` — process conditions
  * ``time_h`` — dummy time column for the loader (default 1.0 if missing)
  * ``h2_co_ratio`` — optional; defaults to 2.0
  * ``ghsv_h-1`` — optional; defaults to median or 1000

Gas-phase SMILES defaults (modifiable with CLI flags):
  CO + H2 → ethanol, matching the graph pipeline used for CO2→methanol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_OUT = [
    "index",
    "reactant",
    "reagent",
    "product",
    "catalyst",
    "ethanol_sty",
    "time_h",
    "temperature_c",
    "pressure_bar",
    "h2_co_ratio",
    "ghsv_h-1",
    "catalyst_components_count",
    "catalyst_primary_loading_wt",
    "source_dataset",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build syngas_ethanol dataset CSVs for CatalyticIQ.")
    p.add_argument("--input", type=Path, required=True, help="Source CSV or XLSX path.")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset" / "syngas_ethanol.csv",
        help="Legacy training CSV path.",
    )
    p.add_argument(
        "--sty_scale",
        type=float,
        default=1.0,
        help="Multiply yield column by this factor (e.g. 0.001 if input is mg/g/h).",
    )
    p.add_argument(
        "--reactant_smiles",
        default="[C-]#[O+]",
        help="Default gas-phase CO SMILES when the input table has no reactant column.",
    )
    p.add_argument("--reagent_smiles", default="[H][H]")
    p.add_argument("--product_smiles", default="CCO")
    p.add_argument(
        "--rename-json",
        type=Path,
        default=None,
        help="JSON map of {source_column: target_column} applied after reading --input "
        "(e.g. map Excel headers to catalyst, ethanol_sty, temperature_c).",
    )
    return p.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[fatal] input missing columns {missing}. Have: {list(df.columns)}")


def build_frame(df: pd.DataFrame, sty_scale: float, r_sm: str, rg_sm: str, p_sm: str) -> pd.DataFrame:
    _require_columns(df, ["catalyst", "ethanol_sty"])

    out = pd.DataFrame()
    out["index"] = df.get("index", pd.Series(np.arange(len(df)))).astype(str)
    out["reactant"] = df["reactant"] if "reactant" in df.columns else r_sm
    out["reagent"] = df["reagent"] if "reagent" in df.columns else rg_sm
    out["product"] = df["product"] if "product" in df.columns else p_sm
    out["catalyst"] = df["catalyst"].astype(str)

    sty = pd.to_numeric(df["ethanol_sty"], errors="coerce").astype(float) * sty_scale
    out["ethanol_sty"] = sty

    if "time_h" in df.columns:
        out["time_h"] = pd.to_numeric(df["time_h"], errors="coerce")
    else:
        out["time_h"] = 1.0
    out["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    out["pressure_bar"] = pd.to_numeric(df["pressure_bar"], errors="coerce")

    if "h2_co_ratio" in df.columns:
        out["h2_co_ratio"] = pd.to_numeric(df["h2_co_ratio"], errors="coerce")
    else:
        out["h2_co_ratio"] = 2.0

    if "ghsv_h-1" in df.columns:
        out["ghsv_h-1"] = pd.to_numeric(df["ghsv_h-1"], errors="coerce")
    else:
        out["ghsv_h-1"] = 1000.0

    if "catalyst_components_count" in df.columns:
        out["catalyst_components_count"] = pd.to_numeric(df["catalyst_components_count"], errors="coerce")
    else:
        out["catalyst_components_count"] = out["catalyst"].str.count(r"\[") + out["catalyst"].str.count(r"\.")

    if "catalyst_primary_loading_wt" in df.columns:
        out["catalyst_primary_loading_wt"] = pd.to_numeric(df["catalyst_primary_loading_wt"], errors="coerce")
    else:
        out["catalyst_primary_loading_wt"] = 40.0

    out["source_dataset"] = df.get("source_dataset", "external").astype(str)

    for c in ["temperature_c", "pressure_bar", "time_h", "h2_co_ratio", "ghsv_h-1"]:
        med = float(out[c].median(skipna=True))
        out[c] = out[c].fillna(med)

    out = out[np.isfinite(out["ethanol_sty"])]
    out = out[out["ethanol_sty"] >= 0]
    return out.reset_index(drop=True)


def main() -> None:
    cli = parse_args()
    raw = _read_table(cli.input)
    if cli.rename_json is not None:
        mapping = json.loads(cli.rename_json.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise SystemExit("[fatal] --rename-json must be a JSON object of strings to strings")
        skip = {k for k in mapping if str(k).startswith("_")}
        raw = raw.rename(
            columns={str(k): str(v) for k, v in mapping.items() if k not in skip}
        )
    built = build_frame(
        raw,
        sty_scale=cli.sty_scale,
        r_sm=cli.reactant_smiles,
        rg_sm=cli.reagent_smiles,
        p_sm=cli.product_smiles,
    )

    for c in REQUIRED_OUT:
        if c not in built.columns:
            raise SystemExit(f"[fatal] internal column {c} missing")

    out_path = cli.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    built[REQUIRED_OUT].to_csv(out_path, index=False)

    full_path = out_path.parent / (out_path.stem + "_full.csv")
    full = built.copy()
    if "selectivity_etoh_pct" in raw.columns:
        r = raw.copy()
        r["index"] = r.get("index", pd.Series(np.arange(len(r)))).astype(str)
        sel = r[["index", "selectivity_etoh_pct"]].drop_duplicates("index")
        full = full.merge(sel, on="index", how="left")
    full.to_csv(full_path, index=False)

    print(f"[done] wrote {out_path} ({len(built)} rows) and {full_path}")
    print("Next: fine-tune from ORD weights, e.g.:")
    print(
        "  python main_finetune.py --file syngas_ethanol --pretrained_file ord "
        "--pretrained_time ord_pretrained_aug5 --epochs 30 --lr 0.0005 --class_weight enabled"
    )


if __name__ == "__main__":
    main()
