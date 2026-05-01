#!/usr/bin/env python3
"""
Prepare a CatalyticIQ-compatible CO2->methanol dataset from TheMeCat + Suvarna files.

This script standardizes heterogeneous-catalysis tabular data into the schema
expected by `dataset/_dataset.py` for the `co2_methanol` entry.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "index",
    "reactant",
    "reagent",
    "product",
    "catalyst",
    "methanol_sty",
    "time_h",
    "temperature_c",
    "pressure_bar",
    "h2_co2_ratio",
    "ghsv_h-1",
    "catalyst_components_count",
    "catalyst_primary_loading_wt",
]

# Optional property columns surfaced into the *_full.csv but not into the
# legacy schema consumed by the existing CVAE training pipeline.
OPTIONAL_PROPERTY_COLUMNS = [
    "selectivity_meoh_pct",
    "co2_conversion_pct",
    "yield_meoh_pct",
]


THEMECAT_COLUMNS = {
    "active_1": "active_comp_1",
    "active_2": "active_comp_2",
    "support_1": "support_comp_1",
    "support_2": "support_comp_2",
    "active_1_percent": "active_1_percent",
    "active_2_percent": "active_2_percent",
    "support_1_percent": "support_1_percent",
    "support_2_percent": "support_2_percent",
    "temperature_k": "temperature_k",
    "pressure_bar": "pressure_bar",
    "h2_co2_ratio": "pH2_pCO2_ratio",
    "ghsv": "GHSV_nlph_gcat",
    "sty": "STY_g_per_gcath",
    "selectivity_meoh": "selectivity_CH3OH",
    "co2_conversion": "CO2_conversion",
    "yield_meoh": "yield_CH3OH",
}

SUVARNA_COLUMNS = {
    "family": "Family ",
    "metal_loading": " Metal Loading [wt.%]",
    "support_1": " Support 1",
    "support_2": "Name of Support2",
    "support_3": "Name of Support 3",
    "promoter_1": " Promoter 1",
    "promoter_2": "Promoter 2",
    "temperature_k": "Temperature [K]",
    "pressure_mpa": "Pressure [Mpa]",
    "h2_co2_ratio": "H2/CO2 [-]",
    "ghsv": "GHSV [cm3 h-1 gcat-1]",
    "sty_mg": "STY [mgMeOH h-1 gcat-1]",
}

ELEMENTS_ORD_PATH = Path("catcvae/utils/elements_ord.csv")


def get_allowed_elements() -> set[str]:
    if ELEMENTS_ORD_PATH.exists():
        table = pd.read_csv(ELEMENTS_ORD_PATH)
        return set(table["Symbol"].dropna().astype(str).tolist())
    # Fallback subset if file not found for some reason.
    return {
        "H", "Li", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
        "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
        "As", "Se", "Br", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
        "Sn", "Sb", "Te", "I", "Cs", "Ba", "La", "Ce", "Nd", "Sm", "Eu", "Dy", "Yb",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    }


def safe_numeric(series: pd.Series, fallback: float) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    return parsed.fillna(parsed.median(skipna=True)).fillna(fallback)


def parse_components_and_loading(
    catalyst_text: str, allowed_elements: set[str]
) -> Tuple[List[str], Optional[float]]:
    if not catalyst_text:
        return [], None

    token_pattern = re.compile(r"([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)")
    component_tokens = token_pattern.findall(catalyst_text)

    components: List[str] = []
    for tok in component_tokens:
        element = re.match(r"[A-Z][a-z]?", tok)
        if not element:
            continue
        symbol = element.group(0)
        if symbol in allowed_elements:
            components.append(symbol)

    if not components:
        return [], None

    seen = set()
    unique_components: List[str] = []
    for comp in components:
        if comp not in seen:
            seen.add(comp)
            unique_components.append(comp)

    percent_matches = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", catalyst_text)]
    primary_loading = max(percent_matches) if percent_matches else None
    return unique_components, primary_loading


def catalyst_to_pseudo_smiles(catalyst_text: object, allowed_elements: set[str]) -> Tuple[str, int, float]:
    text = "" if catalyst_text is None else str(catalyst_text)
    components, primary_loading = parse_components_and_loading(text, allowed_elements)
    if not components:
        components = ["Cu"]

    pseudo_smiles = ".".join(f"[{el}]" for el in components)
    count = len(components)
    loading = 100.0 if primary_loading is None else float(primary_loading)
    return pseudo_smiles, count, loading


def build_themecat_frame(df: pd.DataFrame, allowed_elements: set[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["index"] = [f"themecat-{i}" for i in range(len(df))]
    out["reactant"] = "O=C=O"
    out["reagent"] = "[H][H]"
    out["product"] = "CO"
    cat_text = (
        df[THEMECAT_COLUMNS["active_1"]].fillna("").astype(str)
        + "."
        + df[THEMECAT_COLUMNS["active_2"]].fillna("").astype(str)
        + "."
        + df[THEMECAT_COLUMNS["support_1"]].fillna("").astype(str)
        + "."
        + df[THEMECAT_COLUMNS["support_2"]].fillna("").astype(str)
    )
    pseudo = cat_text.apply(lambda x: catalyst_to_pseudo_smiles(x, allowed_elements))
    out["catalyst"] = pseudo.apply(lambda x: x[0])
    out["catalyst_components_count"] = pseudo.apply(lambda x: x[1]).astype(float)
    loadings = pd.concat(
        [
            pd.to_numeric(df[THEMECAT_COLUMNS["active_1_percent"]], errors="coerce"),
            pd.to_numeric(df[THEMECAT_COLUMNS["active_2_percent"]], errors="coerce"),
            pd.to_numeric(df[THEMECAT_COLUMNS["support_1_percent"]], errors="coerce"),
            pd.to_numeric(df[THEMECAT_COLUMNS["support_2_percent"]], errors="coerce"),
        ],
        axis=1,
    )
    out["catalyst_primary_loading_wt"] = loadings.max(axis=1).fillna(50.0)
    out["methanol_sty"] = pd.to_numeric(df[THEMECAT_COLUMNS["sty"]], errors="coerce")
    out["temperature_c"] = pd.to_numeric(df[THEMECAT_COLUMNS["temperature_k"]], errors="coerce") - 273.15
    out["pressure_bar"] = pd.to_numeric(df[THEMECAT_COLUMNS["pressure_bar"]], errors="coerce")
    out["time_h"] = 1.0
    out["ghsv_h-1"] = pd.to_numeric(df[THEMECAT_COLUMNS["ghsv"]], errors="coerce")
    out["h2_co2_ratio"] = pd.to_numeric(df[THEMECAT_COLUMNS["h2_co2_ratio"]], errors="coerce")
    # Optional performance properties.
    out["selectivity_meoh_pct"] = pd.to_numeric(
        df[THEMECAT_COLUMNS["selectivity_meoh"]], errors="coerce"
    )
    out["co2_conversion_pct"] = pd.to_numeric(
        df[THEMECAT_COLUMNS["co2_conversion"]], errors="coerce"
    )
    out["yield_meoh_pct"] = pd.to_numeric(
        df[THEMECAT_COLUMNS["yield_meoh"]], errors="coerce"
    )
    out["source_dataset"] = "themecat"
    return out


def build_suvarna_frame(df: pd.DataFrame, allowed_elements: set[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["index"] = [f"suvarna-{i}" for i in range(len(df))]
    out["reactant"] = "O=C=O"
    out["reagent"] = "[H][H]"
    out["product"] = "CO"
    cat_text = (
        df[SUVARNA_COLUMNS["family"]].fillna("").astype(str)
        + "."
        + df[SUVARNA_COLUMNS["support_1"]].fillna("").astype(str)
        + "."
        + df[SUVARNA_COLUMNS["support_2"]].fillna("").astype(str)
        + "."
        + df[SUVARNA_COLUMNS["support_3"]].fillna("").astype(str)
        + "."
        + df[SUVARNA_COLUMNS["promoter_1"]].fillna("").astype(str)
        + "."
        + df[SUVARNA_COLUMNS["promoter_2"]].fillna("").astype(str)
    )
    pseudo = cat_text.apply(lambda x: catalyst_to_pseudo_smiles(x, allowed_elements))
    out["catalyst"] = pseudo.apply(lambda x: x[0])
    out["catalyst_components_count"] = pseudo.apply(lambda x: x[1]).astype(float)
    out["catalyst_primary_loading_wt"] = pd.to_numeric(
        df[SUVARNA_COLUMNS["metal_loading"]], errors="coerce"
    ).fillna(50.0)
    # Suvarna STY is mgMeOH h-1 gcat-1; convert to gMeOH h-1 gcat-1.
    out["methanol_sty"] = pd.to_numeric(df[SUVARNA_COLUMNS["sty_mg"]], errors="coerce") / 1000.0
    out["temperature_c"] = pd.to_numeric(df[SUVARNA_COLUMNS["temperature_k"]], errors="coerce") - 273.15
    out["pressure_bar"] = pd.to_numeric(df[SUVARNA_COLUMNS["pressure_mpa"]], errors="coerce") * 10.0
    out["time_h"] = 1.0
    out["ghsv_h-1"] = pd.to_numeric(df[SUVARNA_COLUMNS["ghsv"]], errors="coerce")
    out["h2_co2_ratio"] = pd.to_numeric(df[SUVARNA_COLUMNS["h2_co2_ratio"]], errors="coerce")
    # Suvarna ships only STY; selectivity / conversion / yield are not present.
    out["selectivity_meoh_pct"] = np.nan
    out["co2_conversion_pct"] = np.nan
    out["yield_meoh_pct"] = np.nan
    out["source_dataset"] = "suvarna"
    return out


def _finalize_common(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filtering + numeric coercion shared by both schemas."""
    df = df.copy()
    df = df[np.isfinite(df["methanol_sty"])]
    df = df[df["methanol_sty"] >= 0]

    df["temperature_c"] = safe_numeric(df["temperature_c"], fallback=250.0)
    df["pressure_bar"] = safe_numeric(df["pressure_bar"], fallback=30.0)
    df["time_h"] = safe_numeric(df["time_h"], fallback=1.0)
    df["ghsv_h-1"] = safe_numeric(df["ghsv_h-1"], fallback=2000.0)
    df["h2_co2_ratio"] = safe_numeric(df["h2_co2_ratio"], fallback=3.0)
    df["catalyst_components_count"] = safe_numeric(df["catalyst_components_count"], fallback=2.0)
    df["catalyst_primary_loading_wt"] = safe_numeric(df["catalyst_primary_loading_wt"], fallback=50.0)

    for col in OPTIONAL_PROPERTY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return the legacy schema used by the existing CVAE training pipeline."""
    common = _finalize_common(df)
    return common[REQUIRED_COLUMNS + ["source_dataset"]].reset_index(drop=True)


def finalize_full(df: pd.DataFrame) -> pd.DataFrame:
    """Return the extended schema including selectivity / conversion / yield.

    NaN is preserved on the optional columns so downstream property heads can
    mask out rows that lack the relevant target. The legacy CVAE pipeline must
    keep using `finalize()` — this richer file is for the property heads only.
    """
    common = _finalize_common(df)
    return common[REQUIRED_COLUMNS + OPTIONAL_PROPERTY_COLUMNS + ["source_dataset"]].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare co2_methanol dataset for CatalyticIQ.")
    parser.add_argument(
        "--themecat",
        type=Path,
        required=True,
        help="Path to TheMeCat CSV file.",
    )
    parser.add_argument(
        "--suvarna",
        type=Path,
        required=True,
        help="Path to Suvarna CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/co2_methanol.csv"),
        help="Output CatalyticIQ-ready CSV path (legacy schema).",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=None,
        help="Output path for the extended schema with selectivity/conversion/yield. Defaults to <output>_full.csv.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format for {path}. Use CSV or XLSX.")


def main() -> None:
    args = parse_args()
    allowed_elements = get_allowed_elements()
    themecat_df = load_table(args.themecat)
    suvarna_df = load_table(args.suvarna)

    merged = pd.concat(
        [
            build_themecat_frame(themecat_df, allowed_elements),
            build_suvarna_frame(suvarna_df, allowed_elements),
        ],
        ignore_index=True,
    )
    final_df = finalize(merged)
    full_df = finalize_full(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    full_path = args.output.with_name(args.output.stem + "_full.csv") if args.full_output is None else args.full_output
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(full_path, index=False)

    print(f"Wrote {len(final_df)} rows -> {args.output}")
    print(f"Wrote {len(full_df)} rows -> {full_path} (with selectivity/conversion/yield)")
    print("Columns (full):", list(full_df.columns))
    print("Source counts:")
    print(final_df["source_dataset"].value_counts().to_string())
    coverage = (
        full_df[OPTIONAL_PROPERTY_COLUMNS].notna().mean() * 100.0
    ).round(2).to_dict()
    print("Optional column coverage (%):", coverage)


if __name__ == "__main__":
    main()
