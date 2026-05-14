"""Build approximate catalyst heterostructures from composition-level candidates.

This module intentionally separates Materials Project *bulk inputs* from the
generated catalyst structure. MP provides periodic bulk phases; the full
catalyst geometry is built here from explicit modeling assumptions.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _patch_typing_for_mp_api() -> None:
    """Let newer emmet-core/mp-api releases import on Python 3.10."""
    import typing

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
    if hasattr(typing, "NotRequired"):
        return
    try:
        from typing_extensions import NotRequired
    except Exception:
        return
    typing.NotRequired = NotRequired  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CatalystStructureSpec:
    candidate_label: str
    active_metal: str = "Pt"
    surface_species: tuple[str, ...] = ()
    active_atoms: int = 1
    metal_height_a: float = 2.1
    substrate_material_id: str = "mp-2133"
    substrate_label: str = "ZnO"
    substrate_miller: tuple[int, int, int] = (0, 0, 1)
    substrate_thickness: float = 3.0
    film_material_id: str | None = "mp-390"
    film_label: str | None = "TiO2"
    film_miller: tuple[int, int, int] = (1, 0, 1)
    film_thickness: float = 4.0
    interface_gap_a: float = 2.5
    vacuum_a: float = 15.0
    max_interface_area: float = 200.0


def parse_miller_index(raw: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    nums = re.findall(r"-?\d+", str(raw))
    if len(nums) != 3:
        return default
    return tuple(int(n) for n in nums)  # type: ignore[return-value]


def fetch_mp_structure(material_id: str):
    key = (os.environ.get("MP_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("MP_API_KEY is required to fetch live bulk structures from Materials Project.")
    try:
        _patch_typing_for_mp_api()
        from mp_api.client import MPRester  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Could not import mp-api client: {exc}") from exc

    with MPRester(key) as mpr:
        return mpr.get_structure_by_material_id(str(material_id), conventional_unit_cell=True)


def _strain_score(interface: Any) -> float:
    props = getattr(interface, "interface_properties", {}) or {}
    for key in ("strain", "von_mises_strain", "mean_abs_strain", "max_strain"):
        raw = props.get(key)
        try:
            return abs(float(raw))
        except (TypeError, ValueError):
            pass
    return 0.0


def _single_slab(structure: Any, miller: tuple[int, int, int], thickness: float, vacuum: float):
    from pymatgen.core.surface import SlabGenerator  # type: ignore

    gen = SlabGenerator(
        structure,
        miller,
        min_slab_size=float(thickness),
        min_vacuum_size=float(vacuum),
        center_slab=True,
        in_unit_planes=True,
    )
    slabs = gen.get_slabs(symmetrize=False)
    if not slabs:
        raise RuntimeError(f"No slab could be generated for Miller index {miller}.")
    slabs.sort(key=lambda s: len(s.sites))
    return slabs[0]


def _cartesian_bounds(structure: Any) -> tuple[float, float]:
    zs = [float(site.coords[2]) for site in structure.sites]
    return min(zs), max(zs)


def _stack_slabs_fallback(substrate: Any, film: Any, spec: CatalystStructureSpec):
    """Approximate interface when ZSL coherent matching finds no solution."""
    from pymatgen.core import Lattice, Structure  # type: ignore

    substrate_slab = _single_slab(substrate, spec.substrate_miller, spec.substrate_thickness, 0.0)
    film_slab = _single_slab(film, spec.film_miller, spec.film_thickness, 0.0)

    sub_matrix = substrate_slab.lattice.matrix.copy()
    film_matrix = film_slab.lattice.matrix.copy()
    sub_min_z, sub_max_z = _cartesian_bounds(substrate_slab)
    film_min_z, film_max_z = _cartesian_bounds(film_slab)
    film_shift = sub_max_z - film_min_z + float(spec.interface_gap_a)

    c_len = (
        (sub_max_z - sub_min_z)
        + (film_max_z - film_min_z)
        + float(spec.interface_gap_a)
        + float(spec.vacuum_a)
    )
    lattice = Lattice.from_parameters(
        a=float(substrate_slab.lattice.a),
        b=float(substrate_slab.lattice.b),
        c=max(float(c_len), float(substrate_slab.lattice.c), 10.0),
        alpha=90,
        beta=90,
        gamma=float(substrate_slab.lattice.gamma),
    )
    stacked = Structure(lattice, [], [])
    for site in substrate_slab.sites:
        stacked.append(site.specie, site.coords, coords_are_cartesian=True)
    for site in film_slab.sites:
        frac = site.frac_coords
        cart_xy = frac[0] * sub_matrix[0] + frac[1] * sub_matrix[1]
        cart = [
            float(cart_xy[0]),
            float(cart_xy[1]),
            float(site.coords[2] + film_shift),
        ]
        stacked.append(site.specie, cart, coords_are_cartesian=True)

    meta = {
        "mode": "stacked_slab_fallback",
        "substrate_material_id": spec.substrate_material_id,
        "film_material_id": spec.film_material_id,
        "fallback_reason": "No coherent ZSL interface was generated; stacked unstrained slabs in substrate cell.",
        "substrate_a_b": [float(substrate_slab.lattice.a), float(substrate_slab.lattice.b)],
        "film_original_a_b": [float(film_matrix[0][0]), float(film_matrix[1][1])],
    }
    return stacked, meta


def _interface_structure(spec: CatalystStructureSpec):
    from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder  # type: ignore
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator  # type: ignore

    substrate = fetch_mp_structure(spec.substrate_material_id)
    if not spec.film_material_id:
        slab = _single_slab(substrate, spec.substrate_miller, spec.substrate_thickness, spec.vacuum_a)
        return slab, {"mode": "single_slab", "substrate_material_id": spec.substrate_material_id}

    film = fetch_mp_structure(spec.film_material_id)
    zsl = ZSLGenerator(max_area=float(spec.max_interface_area), max_length_tol=0.05, max_angle_tol=0.01)
    builder = CoherentInterfaceBuilder(
        substrate_structure=substrate,
        film_structure=film,
        film_miller=spec.film_miller,
        substrate_miller=spec.substrate_miller,
        zslgen=zsl,
    )
    if not builder.terminations:
        return _stack_slabs_fallback(substrate, film, spec)

    interfaces = list(
        builder.get_interfaces(
            termination=builder.terminations[0],
            gap=float(spec.interface_gap_a),
            vacuum_over_film=float(spec.vacuum_a),
            film_thickness=float(spec.film_thickness),
            substrate_thickness=float(spec.substrate_thickness),
        )
    )
    if not interfaces:
        return _stack_slabs_fallback(substrate, film, spec)

    best = min(interfaces, key=_strain_score)
    meta = {
        "mode": "coherent_interface",
        "substrate_material_id": spec.substrate_material_id,
        "film_material_id": spec.film_material_id,
        "termination": str(builder.terminations[0]),
        "interface_count": len(interfaces),
        "strain_score": _strain_score(best),
    }
    return best, meta


def _add_surface_species(structure: Any, symbols: tuple[str, ...], count_each: int, height_a: float):
    struct = structure.copy()
    species = tuple(s for s in symbols if s)
    if not species:
        return struct, {}
    count_each = max(1, int(count_each))
    added_counts = {symbol: 0 for symbol in species}
    a_vec, b_vec, _c_vec = struct.lattice.matrix
    top_z = max(float(site.coords[2]) for site in struct.sites)
    base = 0.5 * a_vec + 0.5 * b_vec
    spacing = 2.75
    offsets = [
        (0.0, 0.0),
        (spacing, 0.0),
        (0.5 * spacing, 0.866 * spacing),
        (-0.5 * spacing, 0.866 * spacing),
        (0.0, -spacing),
        (-spacing, 0.0),
        (spacing, spacing),
        (-spacing, -spacing),
    ]
    placements = [(symbol, n) for symbol in species for n in range(count_each)]
    for i, (symbol, _n) in enumerate(placements):
        dx, dy = offsets[i % len(offsets)]
        layer = i // len(offsets)
        coords = [float(base[0] + dx), float(base[1] + dy), float(top_z + height_a + 0.3 * layer)]
        struct.append(symbol, coords, coords_are_cartesian=True)
        added_counts[symbol] = added_counts.get(symbol, 0) + 1
    return struct, added_counts


def validate_structure(
    structure: Any,
    surface_species: tuple[str, ...] = ("Pt",),
    added_surface_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    import numpy as np

    dm = structure.distance_matrix
    positive = dm[dm > 1e-8]
    min_dist = float(np.min(positive)) if positive.size else None
    surface_counts = added_surface_counts or {
        symbol: int(sum(1 for site in structure.sites if str(site.specie.symbol) == symbol))
        for symbol in surface_species
    }
    return {
        "n_atoms": int(len(structure.sites)),
        "formula": structure.composition.reduced_formula,
        "min_interatomic_distance_a": min_dist,
        "slab_density_g_cm3": float(structure.density),
        "surface_species": list(surface_species),
        "surface_atom_counts": surface_counts,
        "warnings": _validation_warnings(min_dist, float(structure.density), surface_counts),
    }


def _validation_warnings(min_dist: float | None, density: float, surface_counts: dict[str, int]) -> list[str]:
    warnings: list[str] = []
    if min_dist is not None and min_dist < 1.5:
        warnings.append(f"Atoms may overlap: minimum distance is {min_dist:.2f} A.")
    if density < 0.05 or density > 12.0:
        warnings.append(f"Slab cell density is unusual at {density:.2f} g/cm3; vacuum lowers slab density.")
    missing = [symbol for symbol, count in surface_counts.items() if count == 0]
    if missing:
        warnings.append(f"No surface atom(s) were added for: {', '.join(missing)}.")
    return warnings


def build_catalyst_structure(spec: CatalystStructureSpec, output_dir: Path | None = None) -> dict[str, Any]:
    base, meta = _interface_structure(spec)
    surface_species = spec.surface_species or (spec.active_metal,)
    full, added_counts = _add_surface_species(base, surface_species, spec.active_atoms, spec.metal_height_a)
    validation = validate_structure(full, surface_species, added_counts)
    out: dict[str, Any] = {
        "spec": asdict(spec),
        "build": meta,
        "validation": validation,
        "structure": full.as_dict(),
        "files": {},
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.candidate_label).strip("_") or "candidate"
        poscar = output_dir / f"{slug}_heterostructure.vasp"
        cif = output_dir / f"{slug}_heterostructure.cif"
        full.to(fmt="poscar", filename=str(poscar))
        full.to(fmt="cif", filename=str(cif))
        out["files"] = {"poscar": str(poscar), "cif": str(cif)}
    return out
