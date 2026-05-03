#!/usr/bin/env python3
"""
Map an external syngas→ethanol corpus into CatalyticIQ CSVs:
``dataset/syngas_ethanol.csv`` (CVAE training) and ``dataset/syngas_ethanol_full.csv``.

**Example sources:** Zenodo ``11639494`` (Suvarna et al., FeCoCuZr HAS / modelling
xlsx files), PNNL higher-alcohol work (often PDF — digitise or use paper SI).

Expected simple input columns (rename your source headers to match, or pass ``--rename-json``):

  * ``catalyst`` — composition text or pseudo-SMILES token string
  * ``ethanol_sty`` — space-time yield in g EtOH / h / g_cat (adjust with --sty_scale if mg-scale)
  * ``temperature_c``, ``pressure_bar`` — process conditions
  * ``time_h`` — dummy time column for the loader (default 1.0 if missing)
  * ``h2_co_ratio`` — optional; defaults to 2.0
  * ``ghsv_h-1`` — optional; defaults to median or 1000

The script also understands the Zenodo 11639494 workbook
``Full_catalytic_performance_data.xlsx`` from Suvarna et al. It derives
``ethanol_sty`` from measured higher-alcohol productivity:

    ethanol_sty = STYHA[mg h-1 gcat-1] * HA_C2_selectivity / 1000

where HA_C2 is the ethanol fraction inside the higher-alcohol selectivity block.

Gas-phase SMILES defaults (modifiable with CLI flags):
  CO + H2 → ethanol, matching the graph pipeline used for CO2→methanol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

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


def _read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None, header=None)


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[fatal] input missing columns {missing}. Have: {list(df.columns)}")


def _find_block(row0: pd.Series, row1: pd.Series, group: str, labels: list[str]) -> dict[str, int]:
    """Find a grouped block in the two-row Zenodo workbook header."""
    matches = [i for i, v in row0.items() if str(v).strip() == group]
    for start in reversed(matches):
        out: dict[str, int] = {}
        for offset in range(0, len(labels) + 8):
            idx = start + offset
            if idx >= len(row1):
                break
            label = str(row1.iloc[idx]).strip()
            if label in labels and label not in out:
                out[label] = idx
        if all(label in out for label in labels):
            return out
    raise ValueError(f"could not find block {group!r} with labels {labels}")


def _find_group_label(row0: pd.Series, row1: pd.Series, group: str, label: str) -> int:
    """Find a column whose first header row is group and second row is label."""
    for i in range(len(row0)):
        if str(row0.iloc[i]).strip() == group and str(row1.iloc[i]).strip() == label:
            return i
    raise ValueError(f"could not find grouped column {group!r}/{label!r}")


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def composition_to_pseudo_smiles(comp: dict[str, float], slots: int = 20) -> str:
    """Encode approximate composition by repeated element tokens.

    The original CVAE only sees pseudo-SMILES tokens, not separate numeric
    composition columns. Repeating tokens keeps Fe65Co19Cu5Zr11 distinct from
    Fe20Co20Cu50Zr10 while staying compatible with the existing graph pipeline.
    """
    finite = {k: v for k, v in comp.items() if np.isfinite(v) and v > 0}
    if not finite:
        return ""
    total = sum(finite.values())
    if total <= 0:
        return ""

    toks: list[str] = []
    for element in ("Fe", "Co", "Cu", "Zr"):
        frac = finite.get(element, 0.0) / total
        if frac <= 0:
            continue
        n = max(1, int(round(frac * slots)))
        toks.extend([f"[{element}]"] * n)
    return ".".join(toks)


def build_zenodo_has_frame(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a syngas->ethanol frame from Zenodo 11639494 HAS workbook."""
    rows: list[dict[str, object]] = []
    for sheet_name, sheet in sheets.items():
        if sheet.shape[0] < 3:
            continue
        row0 = sheet.iloc[0]
        row1 = sheet.iloc[1]
        try:
            comp_cols = _find_block(row0, row1, "Actual composition (XRF)", ["Zr", "Cu", "Co", "Fe"])
            cond_cols = _find_block(
                row0,
                row1,
                "Actual reaction conditions",
                ["T [°C]", "H2/CO", "GHSV [cm3/(h*gcat)]"],
            )
            activity_cols = _find_block(
                row0,
                row1,
                "Measured activity",
                ["STYHA [mg/(h*gcat)]", "XCO"],
            )
            selectivity_cols = _find_block(
                row0,
                row1,
                "Measured selectivity",
                ["CO2", "CH4", "MeOH", "HA"],
            )
            styha_col = activity_cols["STYHA [mg/(h*gcat)]"]
            xco_col = activity_cols["XCO"]
            co2_sel_col = selectivity_cols["CO2"]
            ch4_sel_col = selectivity_cols["CH4"]
            meoh_sel_col = selectivity_cols["MeOH"]
            ha_sel_col = selectivity_cols["HA"]
            ethanol_ha_col = _find_group_label(row0, row1, "HA", "C2")
        except ValueError:
            continue

        for local_i in range(2, len(sheet)):
            rec = sheet.iloc[local_i]
            styha_mg = _to_float(rec.iloc[styha_col])
            ethanol_frac_in_ha = _to_float(rec.iloc[ethanol_ha_col])
            t_c = _to_float(rec.iloc[cond_cols["T [°C]"]])
            pressure_bar = 50.0
            h2_co = _to_float(rec.iloc[cond_cols["H2/CO"]])
            ghsv = _to_float(rec.iloc[cond_cols["GHSV [cm3/(h*gcat)]"]])
            comp = {
                "Zr": _to_float(rec.iloc[comp_cols["Zr"]]),
                "Cu": _to_float(rec.iloc[comp_cols["Cu"]]),
                "Co": _to_float(rec.iloc[comp_cols["Co"]]),
                "Fe": _to_float(rec.iloc[comp_cols["Fe"]]),
            }
            catalyst = composition_to_pseudo_smiles(comp)
            if not catalyst or not np.isfinite(styha_mg) or not np.isfinite(ethanol_frac_in_ha):
                continue
            if not np.isfinite(t_c):
                continue

            nonzero_components = sum(1 for v in comp.values() if np.isfinite(v) and v > 0)
            primary_loading = max([v for v in comp.values() if np.isfinite(v)] or [0.0]) * 100.0
            rows.append(
                {
                    "index": f"{sheet_name}-{int(_to_float(rec.iloc[1])) if np.isfinite(_to_float(rec.iloc[1])) else local_i}",
                    "reactant": "[C-]#[O+]",
                    "reagent": "[H][H]",
                    "product": "CCO",
                    "catalyst": catalyst,
                    "ethanol_sty": styha_mg * ethanol_frac_in_ha / 1000.0,
                    "time_h": 1.0,
                    "temperature_c": t_c,
                    "pressure_bar": pressure_bar,
                    "h2_co_ratio": h2_co if np.isfinite(h2_co) else 2.0,
                    "ghsv_h-1": ghsv if np.isfinite(ghsv) else 1000.0,
                    "catalyst_components_count": nonzero_components,
                    "catalyst_primary_loading_wt": primary_loading,
                    "source_dataset": "zenodo_11639494_" + sheet_name.replace(" ", "_").lower(),
                    "styha_g_h_gcat": styha_mg / 1000.0,
                    "ha_c2_fraction": ethanol_frac_in_ha,
                    "co_conversion": _to_float(rec.iloc[xco_col]),
                    "selectivity_co2": _to_float(rec.iloc[co2_sel_col]),
                    "selectivity_ch4": _to_float(rec.iloc[ch4_sel_col]),
                    "selectivity_meoh": _to_float(rec.iloc[meoh_sel_col]),
                    "selectivity_ha": _to_float(rec.iloc[ha_sel_col]),
                    "xrf_zr": comp["Zr"],
                    "xrf_cu": comp["Cu"],
                    "xrf_co": comp["Co"],
                    "xrf_fe": comp["Fe"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("[fatal] could not parse Zenodo HAS workbook into training rows")
    out = out[np.isfinite(out["ethanol_sty"])]
    out = out[out["ethanol_sty"] >= 0]
    return out.reset_index(drop=True)


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
    raw: pd.DataFrame | None = None
    if cli.input.suffix.lower() in {".xlsx", ".xls"}:
        workbook = _read_workbook(cli.input)
        first_header = pd.read_excel(cli.input, nrows=1)
        if {"catalyst", "ethanol_sty"}.issubset(first_header.columns):
            raw = _read_table(cli.input)
        else:
            built = build_zenodo_has_frame(workbook)
    else:
        raw = _read_table(cli.input)

    if raw is not None and cli.rename_json is not None:
        mapping = json.loads(cli.rename_json.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise SystemExit("[fatal] --rename-json must be a JSON object of strings to strings")
        skip = {k for k in mapping if str(k).startswith("_")}
        raw = raw.rename(
            columns={str(k): str(v) for k, v in mapping.items() if k not in skip}
        )

    if raw is not None:
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
    if raw is not None and "selectivity_etoh_pct" in raw.columns:
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
