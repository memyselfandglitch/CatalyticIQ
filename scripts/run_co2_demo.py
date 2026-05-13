#!/usr/bin/env python3
"""Run the CO2-to-methanol demo artifact refresh in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "dataset" / "co2_methanol" / "output_0_20260507_173839"
DEFAULT_CONFIG = ROOT / "config" / "reactions" / "co2_methanol.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the CO2-to-methanol demo loop: postprocess -> simulation validation "
            "-> YAML sweep -> simulation surrogate."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--raw-candidates", type=Path, default=None)
    parser.add_argument("--reaction-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sweep-output", type=Path, default=ROOT / "dataset/simulation/co2_methanol_sweep.csv")
    parser.add_argument(
        "--surrogate-output-dir",
        type=Path,
        default=ROOT / "dataset/simulation/surrogates/co2_methanol",
    )
    parser.add_argument("--sweep-samples", type=int, default=200)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--temperature-c", type=float, default=240.0)
    parser.add_argument("--pressure-bar", type=float, default=50.0)
    parser.add_argument(
        "--skip-surrogate",
        action="store_true",
        help="Stop after writing the simulation sweep CSV.",
    )
    return parser.parse_args()


def _repo_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _latest_generated_csv(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("generated_mol_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No generated_mol_*.csv files found in {run_dir}")
    return files[0]


def _run(cmd: list[str]) -> None:
    printable = " ".join(_repo_path(Path(x)) if x.startswith(str(ROOT)) else x for x in cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    raw_candidates = args.raw_candidates.resolve() if args.raw_candidates else _latest_generated_csv(run_dir)
    clean_csv = run_dir / "generated_candidates_clean.csv"
    simulation_csv = run_dir / "simulation_validation.csv"

    if not raw_candidates.exists():
        raise FileNotFoundError(raw_candidates)
    if not args.reaction_config.exists():
        raise FileNotFoundError(args.reaction_config)

    _run(
        [
            sys.executable,
            "scripts/postprocess_candidates.py",
            "--candidates",
            str(raw_candidates),
            "--training",
            "dataset/co2_methanol.csv",
            "--dataset-file",
            "co2_methanol",
            "--use-activity-head",
            "--cvae-run-dir",
            str(run_dir),
            "--output",
            str(clean_csv),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/validate_shortlist_simulation.py",
            "--candidates",
            str(clean_csv),
            "--reaction-config",
            str(args.reaction_config),
            "--output",
            str(simulation_csv),
            "--temperature-c",
            str(args.temperature_c),
            "--pressure-bar",
            str(args.pressure_bar),
        ]
    )

    sweep_cmd = [
        sys.executable,
        "scripts/generate_cantera_sweep.py",
        "--reaction-config",
        str(args.reaction_config),
        "--candidates",
        str(clean_csv),
        "--output",
        str(args.sweep_output),
        "--n-samples",
        str(args.sweep_samples),
    ]
    if args.limit_candidates is not None:
        sweep_cmd.extend(["--limit-candidates", str(args.limit_candidates)])
    _run(sweep_cmd)

    if not args.skip_surrogate:
        _run(
            [
                sys.executable,
                "scripts/train_simulation_surrogate.py",
                "--input",
                str(args.sweep_output),
                "--target",
                "simulated_sty_g_h_gcat",
                "--output-dir",
                str(args.surrogate_output_dir),
            ]
        )

    print(
        "\nDemo artifacts refreshed.\n"
        f"- Shortlist: {_repo_path(clean_csv)}\n"
        f"- Simulation validation: {_repo_path(simulation_csv)}\n"
        f"- Sweep CSV: {_repo_path(args.sweep_output)}\n"
        f"- Surrogate dir: {_repo_path(args.surrogate_output_dir)}\n\n"
        "Launch dashboard:\n"
        "conda run --no-capture-output -n catdrx streamlit run app.py "
        "--server.port 8501 --server.address 127.0.0.1\n"
    )


if __name__ == "__main__":
    main()
