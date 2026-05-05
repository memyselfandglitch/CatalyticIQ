"""YAML-driven thermodynamic/reactor validation for CatalyticIQ shortlists."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .reaction_config import ReactionConfig, load_reaction_config

R_J_MOL_K = 8.31446261815324


@dataclass(frozen=True)
class SimulationResult:
    reaction_id: str
    pseudo_smiles: str
    composition_view: str
    simulation_backend: str
    thermodynamic_delta_g_kj_mol: float
    equilibrium_constant_kp: float
    equilibrium_conversion_pct: float
    cantera_equilibrium_conversion_pct: float | None
    reaction_forward_margin: float
    catalyst_rate_score: float
    simulated_sty_g_h_gcat: float
    simulation_confidence: str
    simulation_notes: str


def _components_from_row(row: pd.Series) -> list[str]:
    components = row.get("components")
    if isinstance(components, str) and components.strip():
        return [c for c in components.split("|") if c]
    composition_view = str(row.get("composition_view", ""))
    if "/" in composition_view:
        return [c for c in composition_view.split("/") if c]
    pseudo = str(row.get("pseudo_smiles", ""))
    return [tok.strip("[]") for tok in pseudo.split(".") if tok.startswith("[") and tok.endswith("]")]


def _maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def delta_g_kj_mol(config: ReactionConfig, temperature_c: float) -> float:
    temperature_k = temperature_c + 273.15
    dh = config.thermo["delta_h_298_kj_mol"]
    ds = config.thermo["delta_s_298_j_mol_k"]
    return dh - temperature_k * (ds / 1000.0)


def equilibrium_constant_kp(config: ReactionConfig, temperature_c: float) -> float:
    temperature_k = temperature_c + 273.15
    dg_j_mol = delta_g_kj_mol(config, temperature_c) * 1000.0
    return math.exp(-dg_j_mol / (R_J_MOL_K * temperature_k))


def _max_extent(config: ReactionConfig) -> float:
    limits = []
    for species, nu in config.stoichiometry.items():
        if nu < 0:
            feed = config.feed.get(species, 0.0)
            limits.append(feed / abs(nu))
    return min(limits) if limits else 1.0


def _species_moles_at_extent(config: ReactionConfig, extent: float) -> dict[str, float]:
    species = set(config.feed) | set(config.stoichiometry)
    moles: dict[str, float] = {}
    for sp in species:
        n = config.feed.get(sp, 0.0) + config.stoichiometry.get(sp, 0.0) * extent
        moles[sp] = max(1e-30, n)
    return moles


def reaction_quotient(config: ReactionConfig, extent: float, pressure_bar: float) -> float:
    moles = _species_moles_at_extent(config, extent)
    total = sum(moles.values())
    q = 1.0
    for species, nu in config.stoichiometry.items():
        p_i = max(1e-30, moles.get(species, 1e-30) / total * pressure_bar)
        q *= p_i**nu
    return q


def equilibrium_conversion(config: ReactionConfig, temperature_c: float, pressure_bar: float) -> float:
    kp = equilibrium_constant_kp(config, temperature_c)
    extent_max = _max_extent(config)
    lo, hi = 1e-12, max(1e-12, extent_max * 0.999999)
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if reaction_quotient(config, mid, pressure_bar) < kp:
            lo = mid
        else:
            hi = mid
    extent = 0.5 * (lo + hi)
    reference_feed = config.feed.get(config.reference_species, 1.0)
    reference_nu = abs(config.stoichiometry.get(config.reference_species, -1.0))
    return max(0.0, min(1.0, reference_nu * extent / max(reference_feed, 1e-30)))


def cantera_equilibrium_conversion(
    config: ReactionConfig,
    temperature_c: float,
    pressure_bar: float,
) -> float | None:
    try:
        import cantera as ct
    except Exception:
        return None

    try:
        gas = ct.Solution(config.cantera_mechanism)
        required = set(config.required_species)
        if not required.issubset(set(gas.species_names)):
            return None
        gas.TPX = temperature_c + 273.15, pressure_bar * 1e5, config.feed
        gas.equilibrate("TP")
        ref = config.reference_species
        ref_in = config.feed.get(ref, 0.0) / max(sum(config.feed.values()), 1e-30)
        ref_out = gas[ref].X[0]
        return max(0.0, min(1.0, 1.0 - ref_out / max(ref_in, 1e-30)))
    except Exception:
        return None


def _component_aliases(component: str) -> set[str]:
    aliases = {component}
    for suffix in ("O2", "O3", "2O3"):
        if component.endswith(suffix):
            aliases.add(component.removesuffix(suffix))
    return aliases


def catalyst_rate_score(config: ReactionConfig, components: Sequence[str], temperature_c: float) -> float:
    comp_aliases: set[str] = set()
    for component in components:
        comp_aliases.update(_component_aliases(component))

    desc = config.catalyst_descriptor
    weights = {str(k): float(v) for k, v in desc.get("active_site_weights", {}).items()}
    base = sum(weights.get(c, 0.0) for c in comp_aliases) / max(len(comp_aliases), 1)
    synergy = 0.0
    for rule in desc.get("synergy_rules", []):
        required = {str(x) for x in rule.get("required", [])}
        if required.issubset(comp_aliases):
            synergy += float(rule.get("bonus", 0.0))

    poison_elements = {str(x) for x in desc.get("poison_elements", [])}
    poison = float(desc.get("poison_penalty", 0.0)) if comp_aliases & poison_elements else 0.0
    optimum = float(desc.get("temperature_optimum_c", temperature_c))
    spread = max(1.0, float(desc.get("temperature_spread_c", 80.0)))
    temp_factor = math.exp(-((temperature_c - optimum) / spread) ** 2)
    return max(0.0, min(1.0, (base + synergy - poison) * (0.55 + 0.45 * temp_factor)))


def validate_candidate(
    row: pd.Series,
    *,
    config: ReactionConfig | None = None,
    temperature_c: float | None = None,
    pressure_bar: float | None = None,
) -> SimulationResult:
    cfg = config or load_reaction_config()
    t_c = float(temperature_c if temperature_c is not None else cfg.default_conditions["temperature_c"])
    p_bar = float(pressure_bar if pressure_bar is not None else cfg.default_conditions["pressure_bar"])

    pseudo = str(row.get("pseudo_smiles", ""))
    composition = str(row.get("composition_view", ""))
    components = _components_from_row(row)
    predicted_sty = _maybe_float(row.get("predicted_sty_g_h_gcat"), default=0.0)

    kp = equilibrium_constant_kp(cfg, t_c)
    conversion = equilibrium_conversion(cfg, t_c, p_bar)
    cantera_conversion = cantera_equilibrium_conversion(cfg, t_c, p_bar)
    rate_score = catalyst_rate_score(cfg, components, t_c)
    dg = delta_g_kj_mol(cfg, t_c)

    low_extent = _max_extent(cfg) * 0.05
    q_low = reaction_quotient(cfg, low_extent, p_bar)
    forward_margin = math.log10(max(kp, 1e-300) / max(q_low, 1e-300))
    simulated_sty = predicted_sty * rate_score * max(conversion, 0.0)

    confidence = "high"
    notes = f"YAML-driven analytic thermodynamic equilibrium + catalyst descriptor screen ({cfg.id})."
    backend = "yaml_analytic_thermo_microkinetic"
    if cantera_conversion is not None:
        backend = "yaml_cantera_plus_analytic_microkinetic"
        notes = f"YAML-driven Cantera equilibrium cross-check + catalyst descriptor screen ({cfg.id})."
    elif conversion < 0.02:
        confidence = "medium"
        notes += " Equilibrium conversion is low at selected T/P."
    if rate_score < 0.2:
        confidence = "low"
        notes += " Catalyst descriptor score is weak."

    return SimulationResult(
        reaction_id=cfg.id,
        pseudo_smiles=pseudo,
        composition_view=composition,
        simulation_backend=backend,
        thermodynamic_delta_g_kj_mol=round(dg, 4),
        equilibrium_constant_kp=kp,
        equilibrium_conversion_pct=round(100.0 * conversion, 4),
        cantera_equilibrium_conversion_pct=(
            round(100.0 * cantera_conversion, 4) if cantera_conversion is not None else None
        ),
        reaction_forward_margin=round(forward_margin, 4),
        catalyst_rate_score=round(rate_score, 4),
        simulated_sty_g_h_gcat=round(simulated_sty, 6),
        simulation_confidence=confidence,
        simulation_notes=notes,
    )


def validate_shortlist(
    candidates: pd.DataFrame,
    *,
    config: ReactionConfig | None = None,
    temperature_c: float | None = None,
    pressure_bar: float | None = None,
) -> pd.DataFrame:
    cfg = config or load_reaction_config()
    results = [
        asdict(validate_candidate(row, config=cfg, temperature_c=temperature_c, pressure_bar=pressure_bar))
        for _, row in candidates.iterrows()
    ]
    return pd.DataFrame(results)


def write_validation(
    candidates_csv: Path,
    output_csv: Path,
    *,
    reaction_config: Path | str | None = None,
    temperature_c: float | None = None,
    pressure_bar: float | None = None,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_csv)
    cfg = load_reaction_config(reaction_config)
    out = validate_shortlist(candidates, config=cfg, temperature_c=temperature_c, pressure_bar=pressure_bar)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
