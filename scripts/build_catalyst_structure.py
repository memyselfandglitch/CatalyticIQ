#!/usr/bin/env python3
"""Build one approximate catalyst heterostructure from Materials Project bulk inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.structure_builder import CatalystStructureSpec, build_catalyst_structure, parse_miller_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-label", default="Pt/TiO2/ZnO")
    p.add_argument("--active-metal", default="Pt")
    p.add_argument("--surface-species", default="", help="Comma-separated surface species. Defaults to --active-metal.")
    p.add_argument("--active-atoms", type=int, default=1)
    p.add_argument("--substrate-material-id", default="mp-2133", help="Default: ZnO wurtzite.")
    p.add_argument("--substrate-label", default="ZnO")
    p.add_argument("--substrate-miller", default="0 0 1")
    p.add_argument("--substrate-thickness", type=float, default=3.0)
    p.add_argument("--film-material-id", default="mp-390", help="Default: TiO2 reference bulk.")
    p.add_argument("--film-label", default="TiO2")
    p.add_argument("--film-miller", default="1 0 1")
    p.add_argument("--film-thickness", type=float, default=4.0)
    p.add_argument("--interface-gap-a", type=float, default=2.5)
    p.add_argument("--vacuum-a", type=float, default=15.0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "dataset" / "generated_structures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = CatalystStructureSpec(
        candidate_label=args.candidate_label,
        active_metal=args.active_metal,
        surface_species=tuple(s.strip() for s in args.surface_species.split(",") if s.strip()) or (args.active_metal,),
        active_atoms=args.active_atoms,
        substrate_material_id=args.substrate_material_id,
        substrate_label=args.substrate_label,
        substrate_miller=parse_miller_index(args.substrate_miller, (0, 0, 1)),
        substrate_thickness=args.substrate_thickness,
        film_material_id=args.film_material_id or None,
        film_label=args.film_label or None,
        film_miller=parse_miller_index(args.film_miller, (1, 0, 1)),
        film_thickness=args.film_thickness,
        interface_gap_a=args.interface_gap_a,
        vacuum_a=args.vacuum_a,
    )
    result = build_catalyst_structure(spec, output_dir=args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "structure"}, indent=2))


if __name__ == "__main__":
    main()
