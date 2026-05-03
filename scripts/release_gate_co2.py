#!/usr/bin/env python3
"""Fail CI / release if CO2→methanol metrics fall below config thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--criteria",
        type=Path,
        default=ROOT / "config" / "release_criteria_co2.json",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "dataset" / "co2_methanol" / "property_heads" / "metrics.json",
    )
    p.add_argument(
        "--encoder-report",
        type=Path,
        default=ROOT / "dataset" / "co2_methanol" / "validation" / "encoder_report.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    criteria = json.loads(args.criteria.read_text(encoding="utf-8"))
    failures: list[str] = []

    if args.metrics.exists():
        m = json.loads(args.metrics.read_text(encoding="utf-8"))
        act = m.get("activity", {})
        r2 = act.get("r2")
        mae = act.get("mae")
        ph = criteria.get("property_heads", {})
        if r2 is not None and r2 < float(ph.get("activity_r2_min", 0.0)):
            failures.append(f"property_heads activity R2 {r2:.3f} < {ph['activity_r2_min']}")
        if mae is not None and mae > float(ph.get("activity_mae_max", 1e9)):
            failures.append(f"property_heads activity MAE {mae:.3f} > {ph['activity_mae_max']}")
    else:
        failures.append(f"missing metrics file: {args.metrics}")

    if args.encoder_report.exists():
        rep = json.loads(args.encoder_report.read_text(encoding="utf-8"))
        er = criteria.get("encoder_report", {})
        ho = rep.get("held_out", {})
        r2 = ho.get("r2")
        cov = ho.get("coverage_90pct")
        if r2 is not None and r2 < float(er.get("held_out_r2_min", 0.0)):
            failures.append(f"encoder held_out R2 {r2:.3f} < {er['held_out_r2_min']}")
        if cov is not None and cov < float(er.get("held_out_coverage_90pct_min", 0.0)):
            failures.append(f"encoder 90% coverage {cov:.3f} < {er['held_out_coverage_90pct_min']}")
        lj = rep.get("latent_neighbours", {}).get("mean_jaccard")
        if lj is not None and lj < float(er.get("latent_neighbours_mean_jaccard_min", 0.0)):
            failures.append(f"latent neighbour Jaccard {lj:.3f} < {er['latent_neighbours_mean_jaccard_min']}")
    else:
        failures.append(f"missing encoder report: {args.encoder_report}")

    if failures:
        print("[release_gate] FAILED")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print("[release_gate] OK")


if __name__ == "__main__":
    main()
