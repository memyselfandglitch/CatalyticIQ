"""
Literature-derived stability descriptors for heterogeneous catalysts.

The CatalyticIQ training corpus does not include time-on-stream measurements,
so we cannot data-fit a stability model the way we fit activity / selectivity.
Instead we expose a transparent, descriptor-based stability score in [0, 1]
built from established physical-chemistry quantities:

  * Tammann temperature  (T_T = 0.5 * T_melt) - sintering onset for the active metal.
  * Hüttig temperature   (T_H = 0.3 * T_melt) - surface-mobility onset.
  * A redox-stability class (1 = poor, 2 = moderate, 3 = good) for the dominant
    oxide-forming partner in the composition. Values are taken from XPS / TPR
    literature on CO2-hydrogenation and methanol-synthesis catalysts.

The aggregator is composition-weighted across the catalyst's element list and
clipped to [0, 1]. The score is intentionally labelled as a *proxy* in every
UI surface that consumes it - it is not a regression on measured TOS.

References used to seed the table (representative, not exhaustive):
- Bartholomew, "Mechanisms of catalyst deactivation", Appl. Catal. A 212 (2001).
- Hansen et al., "Sintering of Catalytic Nanoparticles", Acc. Chem. Res. 46 (2013).
- Behrens et al., "The active site of methanol synthesis over Cu/ZnO/Al2O3",
  Science 336 (2012).
- Fichtl et al., "Counting of oxygen defects vs metal surface sites in
  methanol synthesis catalysts", Angew. Chem. 53 (2014).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ElementStability:
    symbol: str
    melting_k: float          # Pure-element melting point (K).
    redox_class: int          # 1 (poor) - 3 (good) under H2 / H2O at 200-300 C.
    notes: str = ""

    @property
    def tammann_k(self) -> float:
        return 0.5 * self.melting_k

    @property
    def huttig_k(self) -> float:
        return 0.3 * self.melting_k


# Active metals + supports + promoters seen in CO2->methanol literature.
# Melting points are well-tabulated; redox_class is a literature-informed
# qualitative score under typical methanol-synthesis conditions
# (200-300 C, 30-80 bar H2-rich gas).
ELEMENT_STABILITY: dict[str, ElementStability] = {
    # Active metals.
    "Cu": ElementStability("Cu", 1357.77, 2, "Sinters readily above ~250 C; ZnO support stabilises."),
    "Pd": ElementStability("Pd", 1828.05, 3, "Stable under H2; resists redox cycling."),
    "Pt": ElementStability("Pt", 2041.4, 3, "Highly stable; expensive."),
    "Rh": ElementStability("Rh", 2237.0, 3, "Stable; ZrO2 support enhances anchoring."),
    "Ru": ElementStability("Ru", 2607.0, 3, "Good under H2; volatile RuO4 if O2 ingress."),
    "Ni": ElementStability("Ni", 1728.0, 2, "Carbon deposition risk above 300 C."),
    "Co": ElementStability("Co", 1768.0, 2, "Cobalt-carbide formation possible."),
    "Fe": ElementStability("Fe", 1811.0, 1, "Iron oxide reduction / re-oxidation cycling."),
    "Ag": ElementStability("Ag", 1234.93, 2, "Low T_melt -> easy sintering."),
    "Au": ElementStability("Au", 1337.33, 2, "Sinters above ~150 C if unsupported."),
    "Ir": ElementStability("Ir", 2719.0, 3, "Highly stable; expensive."),
    "Re": ElementStability("Re", 3459.0, 3, "Stable under H2."),
    "In": ElementStability("In", 429.75, 2, "In2O3 itself is the active phase; loss to In0 above 300 C."),
    "Mn": ElementStability("Mn", 1519.0, 2, "Multiple oxidation states."),
    "Mo": ElementStability("Mo", 2896.0, 3, "Sulfur-tolerant."),
    # Supports / structural promoters.
    "Zn": ElementStability("Zn", 692.68, 2, "ZnO; reducible at high H2 pressure."),
    "Al": ElementStability("Al", 933.47, 3, "Al2O3; thermally robust."),
    "Zr": ElementStability("Zr", 2128.0, 3, "ZrO2; thermally stable; good Cu anchor."),
    "Ti": ElementStability("Ti", 1941.0, 3, "TiO2; SMSI but stable."),
    "Ce": ElementStability("Ce", 1068.0, 3, "CeO2; redox-active oxygen storage."),
    "Si": ElementStability("Si", 1687.0, 3, "SiO2; weakly interacting support."),
    "Mg": ElementStability("Mg", 923.0, 3, "MgO; basic support."),
    "Ga": ElementStability("Ga", 302.91, 1, "Ga2O3; volatile reduction risk."),
    "La": ElementStability("La", 1193.0, 3, "La2O3; structural promoter."),
    "Y":  ElementStability("Y",  1799.0, 3, "Y2O3; structural promoter."),
    "Hf": ElementStability("Hf", 2506.0, 3, "HfO2; structural promoter."),
    # Alkali / alkaline-earth promoters.
    "K":  ElementStability("K",  336.7,  1, "Volatilises above ~400 C; stable below."),
    "Cs": ElementStability("Cs", 301.59, 1, "Highly volatile."),
    "Na": ElementStability("Na", 370.95, 1, "Volatile."),
    "Ca": ElementStability("Ca", 1115.0, 3, "CaO; structural promoter."),
    "Ba": ElementStability("Ba", 1000.0, 2, "BaO; moderate stability."),
    "Li": ElementStability("Li", 453.65, 1, "Volatile."),
    "Rb": ElementStability("Rb", 312.46, 1, "Volatile."),
}


def element_stability_score(symbol: str, temperature_c: float) -> float:
    """Return a 0-1 stability score for a single element at reaction T.

    Uses the Tammann-temperature heuristic: stability decays sharply once the
    operating temperature approaches T_T (~0.5 T_melt). Combined linearly
    with redox class normalised to [0, 1].
    """
    desc = ELEMENT_STABILITY.get(symbol)
    if desc is None:
        return 0.5  # Unknown element: neutral prior.

    operating_k = float(temperature_c) + 273.15
    # Sintering term: 1.0 well below T_Hüttig, 0.0 above T_Tammann.
    if operating_k <= desc.huttig_k:
        sinter_score = 1.0
    elif operating_k >= desc.tammann_k:
        sinter_score = max(0.0, 1.0 - (operating_k - desc.tammann_k) / desc.tammann_k)
    else:
        span = max(desc.tammann_k - desc.huttig_k, 1.0)
        sinter_score = 1.0 - (operating_k - desc.huttig_k) / span

    redox_score = (desc.redox_class - 1) / 2.0  # 1->0, 2->0.5, 3->1.

    return float(max(0.0, min(1.0, 0.6 * sinter_score + 0.4 * redox_score)))


def composition_stability_score(
    components: Iterable[str],
    temperature_c: float,
    pressure_bar: float | None = None,
) -> float:
    """Aggregate per-element stability into a composition-level proxy.

    A simple geometric mean across components is used so a single weak link
    (e.g. an alkali promoter at very high T) drags the overall score down,
    matching the chemistry intuition that the *worst* component governs
    deactivation.

    `pressure_bar` is currently unused but reserved for a future reducing-
    atmosphere correction.
    """
    scores: list[float] = []
    for c in components:
        scores.append(element_stability_score(c, temperature_c))
    if not scores:
        return 0.0
    # Geometric mean.
    product = 1.0
    for s in scores:
        product *= max(s, 1e-3)
    return float(product ** (1.0 / len(scores)))


__all__ = [
    "ElementStability",
    "ELEMENT_STABILITY",
    "element_stability_score",
    "composition_stability_score",
]
