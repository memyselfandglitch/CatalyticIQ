#!/usr/bin/env python3
"""Run thermodynamic/reactor validation for a CatalyticIQ shortlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.simulation.cantera_validator import write_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate shortlist with thermodynamic/reactor equations.")
    parser.add_argument("--candidates", type=Path, required=True, help="generated_candidates_clean.csv")
    parser.add_argument(
        "--reaction-config",
        type=Path,
        default=Path("config/reactions/co2_methanol.yaml"),
        help="YAML file defining reaction thermodynamics, Cantera mechanism, feed, and catalyst descriptors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: simulation_validation.csv next to --candidates.",
    )
    parser.add_argument("--temperature-c", type=float, default=240.0)
    parser.add_argument("--pressure-bar", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (args.candidates.parent / "simulation_validation.csv")
    df = write_validation(
        args.candidates,
        output,
        reaction_config=args.reaction_config,
        temperature_c=args.temperature_c,
        pressure_bar=args.pressure_bar,
    )
    print(f"Wrote {len(df)} simulation validation rows -> {output}")
    if not df.empty:
        show = [
            "composition_view",
            "simulation_backend",
            "equilibrium_conversion_pct",
            "catalyst_rate_score",
            "simulated_sty_g_h_gcat",
            "simulation_confidence",
        ]
        print(df.head(8)[show].to_string(index=False))


if __name__ == "__main__":
    main()
