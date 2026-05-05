#!/usr/bin/env python3
"""Generate simulation-labelled sweep data from a reaction YAML and shortlist."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.simulation.cantera_validator import validate_candidate  # noqa: E402
from services.simulation.reaction_config import load_reaction_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a YAML-defined condition sweep and write simulation training data."
    )
    parser.add_argument("--reaction-config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=None, help="Override sweep.n_samples.")
    parser.add_argument("--seed", type=int, default=None, help="Override sweep.seed.")
    parser.add_argument(
        "--limit-candidates",
        type=int,
        default=None,
        help="Use only the first N candidates for a fast smoke sweep.",
    )
    return parser.parse_args()


def _normalise_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if "candidate_id" not in out.columns:
        out["candidate_id"] = [f"cand_{i:04d}" for i in range(len(out))]
    if "pseudo_smiles" not in out.columns:
        if "composition" in out.columns:
            out["pseudo_smiles"] = out["composition"].astype(str)
        elif "catalyst_name" in out.columns:
            out["pseudo_smiles"] = out["catalyst_name"].astype(str)
        else:
            out["pseudo_smiles"] = out["candidate_id"].astype(str)
    if "composition_view" not in out.columns:
        if "composition" in out.columns:
            out["composition_view"] = out["composition"].astype(str)
        elif "catalyst_name" in out.columns:
            out["composition_view"] = out["catalyst_name"].astype(str)
        else:
            out["composition_view"] = out["pseudo_smiles"].astype(str)
    if "predicted_sty_g_h_gcat" not in out.columns:
        for col in ("predicted_activity", "score", "activity_score"):
            if col in out.columns:
                out["predicted_sty_g_h_gcat"] = pd.to_numeric(out[col], errors="coerce").fillna(1.0)
                break
        else:
            out["predicted_sty_g_h_gcat"] = 1.0
    return out


def _sample_unit_cube(n_samples: int, dimensions: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
        return sampler.random(n=n_samples)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, dimensions))


def _sweep_dimensions(sweep: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    dims: list[tuple[str, str, float, float]] = []
    for section_name in ("variables", "feed_ratios"):
        section = sweep.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for name, bounds in section.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"sweep.{section_name}.{name} must be [min, max]")
            lo, hi = float(bounds[0]), float(bounds[1])
            if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
                raise ValueError(f"Invalid bounds for sweep.{section_name}.{name}: {bounds}")
            dims.append((section_name, str(name), lo, hi))
    if not dims:
        raise ValueError("Reaction config has no sweep.variables or sweep.feed_ratios entries.")
    return dims


def _build_cases(config_path: Path, n_samples: int | None, seed: int | None) -> list[dict[str, Any]]:
    cfg = load_reaction_config(config_path)
    sweep = cfg.sweep
    if not sweep:
        raise ValueError(f"Reaction config has no sweep section: {config_path}")

    count = int(n_samples if n_samples is not None else sweep.get("n_samples", 200))
    random_seed = int(seed if seed is not None else sweep.get("seed", 42))
    dims = _sweep_dimensions(sweep)
    unit = _sample_unit_cube(count, len(dims), random_seed)

    cases: list[dict[str, Any]] = []
    for row_idx in range(count):
        case: dict[str, Any] = {
            "sweep_case_id": f"{cfg.id}_sweep_{row_idx:04d}",
            "temperature_c": float(cfg.default_conditions["temperature_c"]),
            "pressure_bar": float(cfg.default_conditions["pressure_bar"]),
            "feed": dict(cfg.feed),
        }
        reference_species = cfg.reference_species
        reference_basis = max(float(cfg.feed.get(reference_species, 1.0)), 1e-30)
        case["feed"][reference_species] = reference_basis

        for dim_idx, (section, name, lo, hi) in enumerate(dims):
            value = lo + (hi - lo) * float(unit[row_idx, dim_idx])
            if section == "variables":
                case[name] = value
            else:
                case["feed"][name] = reference_basis * value

        case.setdefault("ghsv_h", None)
        case.setdefault("time_on_stream_h", None)
        cases.append(case)
    return cases


def _row_for_output(candidate: pd.Series, case: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    out = {
        "sweep_case_id": case["sweep_case_id"],
        "candidate_id": candidate.get("candidate_id"),
        "pseudo_smiles": candidate.get("pseudo_smiles"),
        "composition_view": candidate.get("composition_view"),
        "temperature_c": round(float(case["temperature_c"]), 6),
        "pressure_bar": round(float(case["pressure_bar"]), 6),
        "ghsv_h": case.get("ghsv_h"),
        "time_on_stream_h": case.get("time_on_stream_h"),
    }
    for species, amount in sorted(case["feed"].items()):
        out[f"feed_{species}"] = round(float(amount), 8)
    for col in (
        "predicted_sty_g_h_gcat",
        "validation_score",
        "validation_tier",
        "known_family_similarity",
        "is_novel_composition",
    ):
        if col in candidate:
            out[col] = candidate.get(col)
    out.update(sim)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_reaction_config(args.reaction_config)
    candidates = _normalise_candidate_columns(pd.read_csv(args.candidates))
    if args.limit_candidates is not None:
        candidates = candidates.head(args.limit_candidates).copy()
    cases = _build_cases(args.reaction_config, args.n_samples, args.seed)

    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        for case in cases:
            case_cfg = replace(
                cfg,
                default_conditions={
                    **cfg.default_conditions,
                    "temperature_c": float(case["temperature_c"]),
                    "pressure_bar": float(case["pressure_bar"]),
                },
                feed={str(k): float(v) for k, v in case["feed"].items()},
            )
            sim = asdict(
                validate_candidate(
                    candidate,
                    config=case_cfg,
                    temperature_c=float(case["temperature_c"]),
                    pressure_bar=float(case["pressure_bar"]),
                )
            )
            rows.append(_row_for_output(candidate, case, sim))

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} sweep rows -> {output}")
    if not df.empty:
        show = [
            "candidate_id",
            "composition_view",
            "temperature_c",
            "pressure_bar",
            "equilibrium_conversion_pct",
            "catalyst_rate_score",
            "simulated_sty_g_h_gcat",
        ]
        print(df.head(8)[show].to_string(index=False))


if __name__ == "__main__":
    main()
