"""Materials Project adapter.

Live mode uses the official `mp-api` client when an `MP_API_KEY` is exported
in the environment and the package is importable. Offline mode reads from a
curated cache committed to the repo so the demo always renders something.

Public API:
  * `fetch_known_catalysts(reaction)` - returns list[KnownEntry] for the named
    reaction (e.g. "co2_to_methanol"). Uses live API if available, otherwise
    cache.
  * `fetch_by_composition(symbols)` - returns entries whose composition is a
    superset of `symbols`.
  * `seed_offline_cache(cache)` - writes a small curated set of well-known
    MP entries into the cache for offline use.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

from .cache import RetrievalCache
from .types import KnownEntry

logger = logging.getLogger(__name__)

# Curated offline cache of known catalyst phases for CO2->methanol synthesis.
# Properties are taken from public Materials Project entries; values are
# rounded for readability and labelled clearly as a static snapshot.
OFFLINE_MP_SEED: list[KnownEntry] = [
    KnownEntry(
        source="MP",
        identifier="mp-30",
        name="Cu (fcc)",
        composition=["Cu"],
        reaction="co2_to_methanol",
        properties={
            "structure": "Fm-3m",
            "formation_energy_per_atom_ev": 0.0,
            "band_gap_ev": 0.0,
            "density_g_cc": 8.96,
            "role": "active metal",
        },
        citation="Materials Project mp-30 (https://next-gen.materialsproject.org/materials/mp-30)",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-2133",
        name="ZnO (wurtzite)",
        composition=["Zn", "O"],
        reaction="co2_to_methanol",
        properties={
            "structure": "P63mc",
            "formation_energy_per_atom_ev": -1.74,
            "band_gap_ev": 3.37,
            "density_g_cc": 5.67,
            "role": "structural support / hydrogen spillover",
        },
        citation="Materials Project mp-2133",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-1143",
        name="Al2O3 (corundum)",
        composition=["Al", "O"],
        reaction="co2_to_methanol",
        properties={
            "structure": "R-3c",
            "formation_energy_per_atom_ev": -3.42,
            "band_gap_ev": 6.0,
            "density_g_cc": 3.98,
            "role": "structural promoter (high SBET)",
        },
        citation="Materials Project mp-1143",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-2858",
        name="ZrO2 (monoclinic)",
        composition=["Zr", "O"],
        reaction="co2_to_methanol",
        properties={
            "structure": "P21/c",
            "formation_energy_per_atom_ev": -3.70,
            "band_gap_ev": 4.1,
            "density_g_cc": 5.85,
            "role": "redox-active support / Cu anchor",
        },
        citation="Materials Project mp-2858",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-22598",
        name="In2O3 (bixbyite)",
        composition=["In", "O"],
        reaction="co2_to_methanol",
        properties={
            "structure": "Ia-3",
            "formation_energy_per_atom_ev": -3.30,
            "band_gap_ev": 2.7,
            "density_g_cc": 7.18,
            "role": "active phase (oxygen vacancies)",
        },
        citation="Materials Project mp-22598",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-2624",
        name="CeO2 (fluorite)",
        composition=["Ce", "O"],
        reaction="co2_to_methanol",
        properties={
            "structure": "Fm-3m",
            "formation_energy_per_atom_ev": -3.50,
            "band_gap_ev": 3.0,
            "density_g_cc": 7.22,
            "role": "redox / oxygen storage support",
        },
        citation="Materials Project mp-2624",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-2",
        name="Pd (fcc)",
        composition=["Pd"],
        reaction="co2_to_methanol",
        properties={
            "structure": "Fm-3m",
            "formation_energy_per_atom_ev": 0.0,
            "band_gap_ev": 0.0,
            "density_g_cc": 12.02,
            "role": "active metal (Pd/ZnO bifunctional)",
        },
        citation="Materials Project mp-2",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-74",
        name="Ni (fcc)",
        composition=["Ni"],
        reaction="co2_to_methanol",
        properties={
            "structure": "Fm-3m",
            "formation_energy_per_atom_ev": 0.0,
            "band_gap_ev": 0.0,
            "density_g_cc": 8.91,
            "role": "active metal; CH4 selectivity risk",
        },
        citation="Materials Project mp-74",
    ),
    KnownEntry(
        source="MP",
        identifier="mp-126",
        name="Pt (fcc)",
        composition=["Pt"],
        reaction="co2_to_methanol",
        properties={
            "structure": "Fm-3m",
            "formation_energy_per_atom_ev": 0.0,
            "band_gap_ev": 0.0,
            "density_g_cc": 21.45,
            "role": "active metal; expensive",
        },
        citation="Materials Project mp-126",
    ),
]


def _live_fetch(reaction: str) -> list[KnownEntry] | None:
    """Live `mp-api` query. Returns None if the live API is unavailable."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        return None
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception as exc:
        logger.info("mp-api not importable (%s); using offline cache.", exc)
        return None

    # We map a small set of reaction keys to composition queries.
    reaction_to_compositions = {
        "co2_to_methanol": [
            ["Cu"], ["Zn", "O"], ["Al", "O"], ["Zr", "O"], ["In", "O"],
            ["Ce", "O"], ["Pd"], ["Pt"], ["Ni"],
        ],
    }
    targets = reaction_to_compositions.get(reaction, [])
    if not targets:
        return []

    try:
        with MPRester(api_key) as mpr:
            collected: list[KnownEntry] = []
            for chemsys in targets:
                docs = mpr.summary.search(
                    chemsys="-".join(chemsys),
                    fields=["material_id", "formula_pretty", "symmetry", "band_gap", "formation_energy_per_atom", "density"],
                )
                for d in docs[:3]:
                    collected.append(
                        KnownEntry(
                            source="MP",
                            identifier=str(d.material_id),
                            name=str(d.formula_pretty),
                            composition=list(chemsys),
                            reaction=reaction,
                            properties={
                                "structure": str(getattr(d.symmetry, "symbol", "")),
                                "formation_energy_per_atom_ev": float(getattr(d, "formation_energy_per_atom", 0.0) or 0.0),
                                "band_gap_ev": float(getattr(d, "band_gap", 0.0) or 0.0),
                                "density_g_cc": float(getattr(d, "density", 0.0) or 0.0),
                            },
                            citation=f"Materials Project {d.material_id} (live API)",
                        )
                    )
        return collected
    except Exception as exc:
        logger.warning("Live MP query failed: %s. Falling back to cache.", exc)
        return None


def seed_offline_cache(cache: RetrievalCache | None = None) -> int:
    cache = cache or RetrievalCache()
    return cache.upsert_mp_entries(OFFLINE_MP_SEED)


def fetch_known_catalysts(
    reaction: str,
    cache: RetrievalCache | None = None,
    prefer_live: bool = True,
) -> list[KnownEntry]:
    cache = cache or RetrievalCache()
    if prefer_live:
        live = _live_fetch(reaction)
        if live is not None:
            cache.upsert_mp_entries(live)
            return live
    cached = cache.fetch_mp_by_reaction(reaction)
    if not cached:
        seed_offline_cache(cache)
        cached = cache.fetch_mp_by_reaction(reaction)
    return cached


def fetch_by_composition(
    symbols: Sequence[str],
    cache: RetrievalCache | None = None,
) -> list[KnownEntry]:
    cache = cache or RetrievalCache()
    cached = cache.fetch_mp_by_composition(symbols)
    if not cached:
        seed_offline_cache(cache)
        cached = cache.fetch_mp_by_composition(symbols)
    return cached
