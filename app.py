from __future__ import annotations

import io
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _mirror_mp_api_key_from_streamlit_secrets() -> None:
    """Materials Project code uses os.environ; Streamlit Cloud stores keys in st.secrets."""
    if os.environ.get("MP_API_KEY"):
        return
    try:
        sec = getattr(st, "secrets", None)
        if sec is None:
            return
        val = sec["MP_API_KEY"] if "MP_API_KEY" in sec else None
    except Exception:
        return
    if val:
        os.environ["MP_API_KEY"] = str(val).strip()


_mirror_mp_api_key_from_streamlit_secrets()

ROOT = Path(__file__).resolve().parent
CONDA_ENV = os.environ.get("CATALYTICIQ_CONDA_ENV", "catalyticiq")
# Dashboard "Generate + rank": fewer samples than CLI default (1000) for faster loops; same code path / model weights.
# Default 250 is temporary; override with CATALYTICIQ_GENERATION_N_SAMPLES.
_GENERATION_N_SAMPLES = int(os.environ.get("CATALYTICIQ_GENERATION_N_SAMPLES", "250"))


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _clean_csv_uses_activity_head(clean: Path | None) -> bool:
    if clean is None or not clean.is_file():
        return False
    try:
        cols = pd.read_csv(clean, nrows=0).columns.tolist()
    except Exception:
        return False
    return "activity_head_sty" in cols


def _rerank_delta_frame(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate rank / score before and after an ActivityHead rerank."""
    needed = {"composition_view", "predicted_sty_g_h_gcat"}
    if before.empty or after.empty or not needed.issubset(before.columns) or not needed.issubset(after.columns):
        return pd.DataFrame()

    before_ranked = before.copy()
    after_ranked = after.copy()
    before_ranked["rank_before"] = before_ranked["predicted_sty_g_h_gcat"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    after_ranked["rank_after"] = after_ranked["predicted_sty_g_h_gcat"].rank(
        method="first",
        ascending=False,
    ).astype(int)

    cols = ["composition_view", "predicted_sty_g_h_gcat"]
    merged = before_ranked[cols + ["rank_before"]].merge(
        after_ranked[cols + ["rank_after"]],
        on="composition_view",
        how="outer",
        suffixes=("_before", "_after"),
    )
    merged["status"] = "reranked"
    merged.loc[merged["rank_before"].isna(), "status"] = "new"
    merged.loc[merged["rank_after"].isna(), "status"] = "removed"
    merged["sty_delta"] = (
        merged["predicted_sty_g_h_gcat_after"] - merged["predicted_sty_g_h_gcat_before"]
    )
    merged["rank_movement"] = merged["rank_before"] - merged["rank_after"]
    merged["rank_before"] = merged["rank_before"].astype("Int64")
    merged["rank_after"] = merged["rank_after"].astype("Int64")
    return (
        merged.sort_values(
            ["status", "rank_after", "rank_before"],
            ascending=[True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


from services.reaction_registry import (  # noqa: E402
    dataset_dir,
    list_profiles_for_ui,
)


# =========================================================================
# Discovery + parsing helpers
# =========================================================================

def discover_output_runs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("output_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def discover_generated_csv(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("generated_mol_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def discover_generated_stats(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("generated_stats_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)


def discover_clean_csv(run_dir: Path) -> Path | None:
    target = run_dir / "generated_candidates_clean.csv"
    return target if target.exists() else None


def discover_simulation_csv(run_dir: Path) -> Path | None:
    target = run_dir / "simulation_validation.csv"
    return target if target.exists() else None


def merge_co2_demo_output_runs(runs: list[Path], dataset_dir: Path) -> list[Path]:
    """Keep normal discovery order, but always surface key CO2 demo folders if they have raw generation CSVs."""
    if dataset_dir.name != "co2_methanol":
        return runs
    extras = [dataset_dir / "output_0_20260512_184642"]
    seen = {p.resolve() for p in runs}
    out = list(runs)
    for ex in extras:
        if ex.is_dir() and ex.resolve() not in seen and discover_generated_csv(ex):
            out.append(ex)
            seen.add(ex.resolve())
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def simulation_surrogate_metrics_path(reaction_id: str) -> Path:
    return ROOT / "dataset" / "simulation" / "surrogates" / reaction_id / "metrics.json"


def simulation_sweep_path(reaction_id: str) -> Path:
    return ROOT / "dataset" / "simulation" / f"{reaction_id}_sweep.csv"


@st.cache_data(show_spinner=False)
def load_candidate_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["candidate", "score"])
    df["candidate"] = df["candidate"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_clean_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["predicted_sty_g_h_gcat"] = pd.to_numeric(df["predicted_sty_g_h_gcat"], errors="coerce")
    df["raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce")
    if "validation_score" in df.columns:
        df["validation_score"] = pd.to_numeric(df["validation_score"], errors="coerce")
    for col in (
        "passes_validation_gate",
        "is_novel_composition",
        "has_known_support",
        "has_known_promoter",
    ):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    if "n_components" in df.columns:
        df["n_components"] = pd.to_numeric(df["n_components"], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_simulation_validation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (
        "thermodynamic_delta_g_kj_mol",
        "equilibrium_constant_kp",
        "equilibrium_conversion_pct",
        "cantera_equilibrium_conversion_pct",
        "reaction_forward_margin",
        "catalyst_rate_score",
        "simulated_sty_g_h_gcat",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _candidate_active_metal(row: Any) -> str:
    text = f"{row.get('pseudo_smiles', '')} {row.get('components', '')} {row.get('composition_view', '')}"
    for symbol in ("Pt", "Pd", "Cu", "Ni", "Rh", "Ru", "Ag", "Au", "Co", "Fe"):
        if re.search(rf"(^|[^A-Za-z]){symbol}([^a-z]|$)", text):
            return symbol
    return "Pt"


def _candidate_elements(row: Any) -> list[str]:
    text = f"{row.get('components', '')}|{row.get('composition_view', '')}|{row.get('pseudo_smiles', '')}"
    out: list[str] = []
    for symbol in re.findall(r"[A-Z][a-z]?", text):
        if symbol not in out:
            out.append(symbol)
    return out


def _candidate_component_tokens(row: Any) -> list[str]:
    text = str(row.get("composition_view", "") or row.get("components", ""))
    tokens = [t.strip() for t in re.split(r"[/|,+;]", text) if t.strip()]
    return tokens


def _candidate_has_elemental_component(row: Any, symbol: str) -> bool:
    pattern = re.compile(rf"^{re.escape(symbol)}(?:\d|\.|$)")
    return any(pattern.match(token) for token in _candidate_component_tokens(row))


def _surface_species_for_candidate(row: Any, support_refs: list[dict[str, Any]]) -> tuple[str, ...]:
    support_elements: set[str] = {"O"}
    for ref in support_refs:
        support_elements.update(re.findall(r"[A-Z][a-z]?", str(ref.get("formula_pretty", ""))))
        support_elements.update(str(ref.get("chemsys", "")).split("-"))
    species = [el for el in _candidate_elements(row) if el and el not in support_elements]
    if species:
        return tuple(species)
    return (_candidate_active_metal(row),)


def _bulk_ref_elements(ref: dict[str, Any]) -> set[str]:
    elements = set(re.findall(r"[A-Z][a-z]?", str(ref.get("formula_pretty", ""))))
    elements.update(part for part in str(ref.get("chemsys", "")).split("-") if part)
    return elements


def _is_oxide_ref(ref: dict[str, Any]) -> bool:
    return "O" in _bulk_ref_elements(ref)


def _surface_elements_from_ref(ref: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(el for el in _bulk_ref_elements(ref) if el != "O"))


def _structure_element_counts(struct_dict: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in struct_dict.get("sites", []):
        if not isinstance(site, dict):
            continue
        species = site.get("species") or []
        if not species or not isinstance(species[0], dict):
            continue
        symbol = str(species[0].get("element") or species[0].get("label") or "").strip()
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def _element_model_role(
    symbol: str,
    substrate_ref: dict[str, Any],
    film_ref: dict[str, Any] | None,
    surface_species: tuple[str, ...],
) -> str:
    roles: list[str] = []
    if symbol in _bulk_ref_elements(substrate_ref):
        roles.append("substrate bulk")
    if isinstance(film_ref, dict) and symbol in _bulk_ref_elements(film_ref):
        roles.append("film/interface bulk")
    if symbol in surface_species:
        roles.append("adsorbed surface additive")
    return " + ".join(roles) if roles else "generated atom"


@st.cache_data(ttl=3600, show_spinner=False)
def _dash_build_catalyst_structure(
    candidate_label: str,
    active_metal: str,
    surface_species: tuple[str, ...],
    active_atoms: int,
    substrate_material_id: str,
    substrate_label: str,
    substrate_miller: tuple[int, int, int],
    substrate_thickness: float,
    film_material_id: str | None,
    film_label: str | None,
    film_miller: tuple[int, int, int],
    film_thickness: float,
    interface_gap_a: float,
    vacuum_a: float,
    output_dir: str,
) -> dict[str, Any]:
    from services.structure_builder import CatalystStructureSpec, build_catalyst_structure

    spec = CatalystStructureSpec(
        candidate_label=candidate_label,
        active_metal=active_metal,
        surface_species=surface_species,
        active_atoms=active_atoms,
        substrate_material_id=substrate_material_id,
        substrate_label=substrate_label,
        substrate_miller=substrate_miller,
        substrate_thickness=substrate_thickness,
        film_material_id=film_material_id,
        film_label=film_label,
        film_miller=film_miller,
        film_thickness=film_thickness,
        interface_gap_a=interface_gap_a,
        vacuum_a=vacuum_a,
    )
    return build_catalyst_structure(spec, output_dir=Path(output_dir))


@st.cache_data(show_spinner=False)
def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_generated_stats(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 2:
            continue
        key = parts[0]
        numbers = []
        for p in parts[1:]:
            try:
                numbers.append(float(p))
            except ValueError:
                continue
        if numbers:
            out[key] = numbers[0] if len(numbers) == 1 else numbers
    return out


def generation_stats_summary(stats: dict[str, Any]) -> pd.DataFrame:
    """Convert raw CatDRX generation stats into judge-readable rows."""
    if not stats:
        return pd.DataFrame()

    def _pair(key: str) -> tuple[float | None, float | None]:
        value = stats.get(key)
        if isinstance(value, list):
            first = value[0] if len(value) > 0 else None
            second = value[1] if len(value) > 1 else None
            return first, second
        if isinstance(value, (int, float)):
            return value, None
        return None, None

    validity_n, validity_pct = _pair("Validity")
    unique_n, unique_pct = _pair("Uniqueness")
    novel_n, novel_pct_unique = _pair("Novelty")
    intdiv_mean, intdiv_std = _pair("IntDiv")
    snn_mean, snn_std = _pair("SNN")
    fcd, _ = _pair("FCD")

    rows = [
        {
            "metric": "Validity",
            "value": f"{int(validity_n or 0):,} valid / {validity_pct:.1f}%" if validity_pct is not None else "N/A",
            "what it means": "Generated strings that pass the syntax/chemistry parser.",
        },
        {
            "metric": "Uniqueness",
            "value": f"{int(unique_n or 0):,} unique / {unique_pct:.1f}% of generated" if unique_pct is not None else "N/A",
            "what it means": "Deduplicated candidates after generation.",
        },
        {
            "metric": "Novelty",
            "value": (
                f"{int(novel_n or 0):,} novel / {novel_pct_unique:.1f}% of unique"
                if novel_pct_unique is not None
                else "N/A"
            ),
            "what it means": "Unique candidates not found verbatim in the training corpus.",
        },
        {
            "metric": "Internal diversity",
            "value": f"{intdiv_mean:.3f} ± {intdiv_std:.3f}" if intdiv_mean is not None else "N/A",
            "what it means": "How spread out the generated candidates are; higher is more diverse.",
        },
        {
            "metric": "SNN",
            "value": f"{snn_mean:.3f} ± {snn_std:.3f}" if snn_mean is not None else "N/A",
            "what it means": "Similarity to nearest training examples; lower means less copy-like.",
        },
        {
            "metric": "FCD",
            "value": "not available for this pseudo-SMILES run" if pd.isna(fcd) else f"{fcd:.3f}",
            "what it means": "Distribution-distance metric; skipped here because catalyst pseudo-SMILES are sparse.",
        },
    ]
    return pd.DataFrame(rows)


def parse_training_metrics(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    pat = re.compile(
        r"epoch:\s*(?P<epoch>\d+)\s+"
        r"t_loss:\s*(?P<t_loss>[-+]?\d*\.?\d+)\s+"
        r"v_loss:\s*(?P<v_loss>[-+]?\d*\.?\d+)\s+"
        r"opt_loss:\s*(?P<opt_loss>[-+]?\d*\.?\d+)\s+"
        r"valid:\s*(?P<valid>[-+]?\d*\.?\d+)\s+"
        r"diver:\s*(?P<diver>[-+]?\d*\.?\d+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if not m:
            continue
        rows.append({k: float(v) for k, v in m.groupdict().items()})
    return pd.DataFrame(rows)


def parse_loss_metrics(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    pat = re.compile(
        r"epoch:\s*(?P<epoch>\d+)\s+"
        r"recon_t:\s*(?P<recon_t>[-+]?\d*\.?\d+)\s+"
        r"kl_t:\s*(?P<kl_t>[-+]?\d*\.?\d+)\s+"
        r"nn_t:\s*(?P<nn_t>[-+]?\d*\.?\d+)\s+"
        r"recon_v:\s*(?P<recon_v>[-+]?\d*\.?\d+)\s+"
        r"kl_v:\s*(?P<kl_v>[-+]?\d*\.?\d+)\s+"
        r"nn_v:\s*(?P<nn_v>[-+]?\d*\.?\d+)\s+"
        r"an_step:\s*(?P<an_step>[-+]?\d*\.?\d+)\s+"
        r"slop:\s*(?P<slop>[-+]?\d*\.?\d+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if not m:
            continue
        rows.append({k: float(v) for k, v in m.groupdict().items()})
    return pd.DataFrame(rows)


def parse_hyper_result(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        rec: dict[str, Any] = {"pretrained": parts[0], "run": parts[1]}
        for token in parts[2:]:
            if ":" not in token:
                continue
            key, val = token.split(":", 1)
            try:
                rec[key.strip()] = float(val.strip())
            except ValueError:
                rec[key.strip()] = val.strip()
        rows.append(rec)
    return pd.DataFrame(rows)


def element_frequency(candidates: pd.Series) -> pd.DataFrame:
    pat = re.compile(r"\[([A-Za-z0-9]+)\]")
    c: Counter[str] = Counter()
    for s in candidates:
        for element in pat.findall(s):
            c[element] += 1
    if not c:
        return pd.DataFrame(columns=["element", "count"])
    return (
        pd.DataFrame(c.items(), columns=["element", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def _components_from_smiles(smiles: str) -> list[str]:
    return re.findall(r"\[([A-Z][a-z]?)\d*\]", str(smiles))


# =========================================================================
# Chemistry helpers (energy, retrieval, stability)
# =========================================================================


@st.cache_data(show_spinner=False)
def compute_energy_profile(smiles: str, mechanism: str, backend: str) -> dict[str, Any]:
    from catcvae.reaction_energy import estimate_pathway

    components = _components_from_smiles(smiles) or ["Cu"]
    profile = estimate_pathway(components, mechanism=mechanism, backend=backend)
    return {
        "components": components,
        "mechanism": profile.mechanism,
        "intermediates": profile.intermediates,
        "delta_g_ev": list(profile.delta_g_ev),
        "backend": profile.backend,
        "citation": profile.citation,
        "notes": profile.notes,
        "extras": dict(profile.extras),
    }


def render_energy_diagram(profile: dict[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intermediates = profile["intermediates"]
    dg = profile["delta_g_ev"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xs = list(range(len(intermediates)))
    for i, energy in enumerate(dg):
        ax.hlines(energy, i - 0.35, i + 0.35, linewidth=3)
        ax.text(i, energy + 0.08, f"{energy:+.2f}", ha="center", va="bottom", fontsize=9)
    for i in range(len(dg) - 1):
        ax.plot([i + 0.35, i + 1 - 0.35], [dg[i], dg[i + 1]], linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace(" ", "\n") for s in intermediates], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Relative free energy (eV)")
    ax.set_title(f"{profile['mechanism']} mechanism — {profile['backend']} tier")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=140)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def load_known_catalysts(reaction: str) -> list[dict[str, Any]]:
    from services.retrieval.cache import RetrievalCache
    from services.retrieval.materials_project import fetch_known_catalysts

    cache = RetrievalCache()
    entries = fetch_known_catalysts(reaction, cache=cache, prefer_live=True)
    return [
        {
            "source": e.source,
            "identifier": e.identifier,
            "name": e.name,
            "composition": list(e.composition),
            "role": e.properties.get("role", ""),
            "structure": e.properties.get("structure", ""),
            "formation_energy_per_atom_ev": e.properties.get("formation_energy_per_atom_ev"),
            "band_gap_ev": e.properties.get("band_gap_ev"),
            "density_g_cc": e.properties.get("density_g_cc"),
        }
        for e in entries
    ]


@st.cache_data(show_spinner=False)
def load_ocp_for_composition(composition: tuple[str, ...]) -> list[dict[str, Any]]:
    from services.retrieval.cache import RetrievalCache
    from services.retrieval.open_catalyst import fetch_binding_energies

    cache = RetrievalCache()
    return fetch_binding_energies(list(composition), cache=cache, prefer_live=True)


def stability_score_for(smiles: str, temperature_c: float = 240.0) -> float:
    try:
        from catcvae.stability_descriptors import composition_stability_score
    except Exception:
        return float("nan")
    components = _components_from_smiles(smiles)
    if not components:
        return float("nan")
    return composition_stability_score(components, temperature_c)


# Composition-weighted methanol-selectivity prior used in the Compare tab when
# we don't have direct lab data on the candidate. Values are qualitative,
# loosely calibrated against TheMeCat MeOH selectivity column.
SELECTIVITY_PRIOR: dict[str, float] = {
    "Cu": 70.0, "Pd": 50.0, "Pt": 35.0, "Rh": 30.0, "Ru": 25.0,
    "Ni": 20.0, "Co": 25.0, "Fe": 25.0, "Ag": 60.0, "Au": 55.0,
    "In": 70.0, "Mn": 35.0, "Mo": 30.0, "Re": 40.0, "Ir": 30.0,
    "Zn": 65.0, "Zr": 60.0, "Ti": 55.0, "Ce": 65.0, "Al": 60.0,
    "Si": 55.0, "Mg": 55.0, "Ga": 60.0, "La": 55.0, "Y": 55.0,
    "Hf": 55.0, "K": 60.0, "Cs": 60.0, "Na": 55.0, "Ca": 55.0,
    "Ba": 55.0, "Li": 55.0, "Rb": 55.0,
}


def selectivity_proxy(components: list[str]) -> float:
    if not components:
        return float("nan")
    vals = [SELECTIVITY_PRIOR.get(c, 50.0) for c in components]
    return float(sum(vals) / len(vals))


def _pct_error(actual: Any, predicted: Any) -> float | None:
    try:
        actual_f = float(actual)
        predicted_f = float(predicted)
    except (TypeError, ValueError):
        return None
    if pd.isna(actual_f) or pd.isna(predicted_f):
        return None
    return 100.0 * (actual_f - predicted_f) / max(abs(predicted_f), 1e-9)


def _optional_float(raw: Any) -> float | None:
    """Parse optional numeric UI text fields without treating blank as zero."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if pd.isna(value):
        return None
    return value


def _run_demo_command(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run_timestamp(run_name: str) -> str:
    match = re.match(r"output_\d+_(.+)", run_name)
    return match.group(1) if match else run_name


def _feedback_discrepancy_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in records:
        sty_err = _pct_error(r.get("measured_sty"), r.get("predicted_sty"))
        sel_delta = None
        if r.get("measured_selectivity") is not None and r.get("predicted_selectivity") is not None:
            sel_delta = float(r["measured_selectivity"]) - float(r["predicted_selectivity"])
        yield_delta = None
        if r.get("measured_yield") is not None and r.get("predicted_yield") is not None:
            yield_delta = float(r["measured_yield"]) - float(r["predicted_yield"])

        flags: list[str] = []
        if sty_err is not None and abs(sty_err) >= 25:
            flags.append("activity_gap")
        if sel_delta is not None and abs(sel_delta) >= 10:
            flags.append("selectivity_gap")
        if yield_delta is not None and abs(yield_delta) >= 5:
            flags.append("yield_gap")

        if "activity_gap" in flags and sty_err is not None and sty_err < 0:
            hypothesis = "Activity model overestimated productivity; inspect nearest-family similarity, descriptor confidence, and reaction-condition coverage."
        elif "selectivity_gap" in flags:
            hypothesis = "Selectivity prior may be underweighted for this composition; prioritize measured selectivity rows in the next head retrain."
        elif "yield_gap" in flags:
            hypothesis = "Yield gap suggests condition sensitivity; add more sweep/lab rows near this T/P/feed window."
        elif flags:
            hypothesis = "Prediction differs from measured outcome; include this row in the next feedback retrain."
        else:
            hypothesis = "Prediction is within current demo tolerance."

        rows.append(
            {
                "logged_at": r.get("logged_at"),
                "composition": r.get("composition_view"),
                "predicted_sty": r.get("predicted_sty"),
                "measured_sty": r.get("measured_sty"),
                "sty_error_pct": sty_err,
                "predicted_selectivity": r.get("predicted_selectivity"),
                "measured_selectivity": r.get("measured_selectivity"),
                "selectivity_delta": sel_delta,
                "predicted_yield": r.get("predicted_yield"),
                "measured_yield": r.get("measured_yield"),
                "yield_delta": yield_delta,
                "flags": ", ".join(flags) if flags else "ok",
                "hypothesis": hypothesis,
            }
        )
    return pd.DataFrame(rows)


# =========================================================================
# Sidebar + state
# =========================================================================

st.set_page_config(page_title="CatalyticIQ Dashboard", layout="wide")

profiles = list_profiles_for_ui()
if not profiles:
    st.error("No reaction profiles with a dataset folder found. Add dataset/<name>/ and CVAE output_* runs.")
    st.stop()
profile_by_label = {p.label: p for p in profiles}
reaction_label = st.sidebar.selectbox("Reaction", list(profile_by_label.keys()))
profile = profile_by_label[reaction_label]
DATASET_DIR = dataset_dir(profile)
HYPER_RESULT_PATH = DATASET_DIR / "hyper_result.txt"
PROPERTY_DIR = DATASET_DIR / "property_heads"
VALIDATION_DIR = DATASET_DIR / "validation"

st.title(f"CatalyticIQ — {profile.label}")
simple_ui = st.sidebar.checkbox(
    "Simple view",
    value=True,
    help="Fewer metrics, auto-pick latest stats file, and tuck charts / JSON / CLI into closed sections.",
)
if not simple_ui:
    st.caption(profile.caption)

runs = discover_output_runs(DATASET_DIR)
if profile.id == "co2_methanol":
    runs = merge_co2_demo_output_runs(runs, DATASET_DIR)
if not runs:
    st.warning(f"{profile.label} is registered as a pilot reaction, but no generated demo run exists yet.")
    st.caption(profile.caption)
    st.markdown("**What exists for this reaction profile**")
    status_rows = [
        {
            "asset": "Reaction YAML",
            "path": profile.reaction_config_relative,
            "status": "present" if (ROOT / profile.reaction_config_relative).exists() else "missing",
        },
        {
            "asset": "Cleaned dataset",
            "path": f"dataset/{profile.dataset_subdir}.csv",
            "status": "present" if (ROOT / "dataset" / f"{profile.dataset_subdir}.csv").exists() else "missing",
        },
        {
            "asset": "Full training CSV",
            "path": profile.full_csv_relative,
            "status": "present" if (ROOT / profile.full_csv_relative).exists() else "missing",
        },
        {
            "asset": "Generated output_* run",
            "path": f"dataset/{profile.dataset_subdir}/output_*",
            "status": "missing",
        },
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    st.info(
        "The shipped hackathon demo is CO2-to-methanol. This pilot profile shows the intended "
        "multi-reaction architecture, but it needs a reaction-specific CVAE run, property heads, "
        "and simulation validation artifacts before the full dashboard can render."
    )
    st.code(
        f"conda run -n {CONDA_ENV} python scripts/run_co2_demo.py --sweep-samples 200\n"
        "# For this pilot, first add/promote an equivalent reaction-specific generation pipeline.",
        language="bash",
    )
    st.stop()

candidate_ready_runs = [p for p in runs if discover_generated_csv(p)]
if not candidate_ready_runs:
    st.error(
        f"No generated CSV found in any output folder under "
        f"{ROOT / 'dataset' / profile.dataset_subdir}."
    )
    st.stop()

if len(candidate_ready_runs) < len(runs):
    skipped = len(runs) - len(candidate_ready_runs)
    st.sidebar.caption(f"Hiding {skipped} run folder(s) without generated candidates.")

run_map = {p.name: p for p in candidate_ready_runs}
_run_labels = list(run_map.keys())
_preferred_run_idx = 0
for _pref in ("20260512", "173839"):
    _hit = next((i for i, n in enumerate(_run_labels) if _pref in n), None)
    if _hit is not None:
        _preferred_run_idx = _hit
        break
selected_run_name = st.sidebar.selectbox(
    "Candidate-ready run folder",
    _run_labels,
    index=_preferred_run_idx,
    help="Defaults to `output_0_*20260512*` when present (newer CVAE + generation), else `*173839*`.",
)
selected_run = run_map[selected_run_name]
selected_run_time = _run_timestamp(selected_run_name)

gen_csv_files = discover_generated_csv(selected_run)
gen_stats_files = discover_generated_stats(selected_run)
if not gen_csv_files:
    st.error(f"No generated CSV found in {selected_run}.")
    st.stop()

selected_gen_csv = st.sidebar.selectbox(
    "Generated candidates file", [p.name for p in gen_csv_files]
)
selected_gen_csv_path = selected_run / selected_gen_csv

selected_stats_path = None
if gen_stats_files:
    if simple_ui:
        selected_stats_path = gen_stats_files[0]
        st.sidebar.caption(f"Stats file: `{gen_stats_files[0].name}`")
    else:
        selected_stats_name = st.sidebar.selectbox(
            "Generation stats file", [p.name for p in gen_stats_files]
        )
        selected_stats_path = selected_run / selected_stats_name

top_n = st.sidebar.slider("Top-N shortlist size", min_value=5, max_value=100, value=20, step=5)
must_have_metal = st.sidebar.checkbox("Require metal-containing candidates", value=True)
search_query = st.sidebar.text_input("Search candidate text")

st.sidebar.markdown("---")
if simple_ui:
    st.sidebar.caption(f"Data: `dataset/{profile.dataset_subdir}/` · CLI under **Refresh shortlist**.")
else:
    st.sidebar.caption(
        f"Artifacts: `dataset/{profile.dataset_subdir}/`. Refresh with postprocess_candidates, "
        "train_property_heads, validate_encoder, and retrain_with_feedback.py "
        f"(pass --file {profile.id} for feedback retrain)."
    )

_train_csv_rel = f"dataset/{profile.dataset_subdir}.csv"
_gen_rel = _relative_to_repo(selected_gen_csv_path)
_run_rel = _relative_to_repo(selected_run)
_out_rel = f"{_run_rel}/generated_candidates_clean.csv"
_pp_base = (
    f"conda run -n {CONDA_ENV} python scripts/postprocess_candidates.py \\\n"
    f"  --candidates {_gen_rel} \\\n"
    f"  --training {_train_csv_rel} \\\n"
    f"  --output {_out_rel}"
)
_pp_activity = (
    f"conda run -n {CONDA_ENV} python scripts/postprocess_candidates.py \\\n"
    f"  --candidates {_gen_rel} \\\n"
    f"  --training {_train_csv_rel} \\\n"
    f"  --dataset-file {profile.id} \\\n"
    f"  --use-activity-head \\\n"
    f"  --cvae-run-dir {_run_rel} \\\n"
    f"  --output {_out_rel}"
)
_sim_rel = f"{_run_rel}/simulation_validation.csv"
_reaction_cfg_rel = profile.reaction_config_relative
_sim_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/validate_shortlist_simulation.py \\\n"
    f"  --candidates {_out_rel} \\\n"
    f"  --reaction-config {_reaction_cfg_rel} \\\n"
    f"  --output {_sim_rel} \\\n"
    f"  --temperature-c 240 \\\n"
    f"  --pressure-bar 50"
)
_sweep_rel = _relative_to_repo(simulation_sweep_path(profile.id))
_surrogate_rel = f"dataset/simulation/surrogates/{profile.id}"
_sweep_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/generate_cantera_sweep.py \\\n"
    f"  --reaction-config {_reaction_cfg_rel} \\\n"
    f"  --candidates {_out_rel} \\\n"
    f"  --output {_sweep_rel}"
)
_surrogate_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/train_simulation_surrogate.py \\\n"
    f"  --input {_sweep_rel} \\\n"
    f"  --target simulated_sty_g_h_gcat \\\n"
    f"  --output-dir {_surrogate_rel}"
)
_co2_demo_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/run_co2_demo.py "
    "--sweep-samples 200"
)
_feedback_import_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/import_feedback_csv.py \\\n"
    "  --input dataset/feedback/co2_methanol_lab_results_example.csv"
)
_feedback_heads_prep_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/train_property_heads.py \\\n"
    f"  --file {profile.id} \\\n"
    f"  --pretrained_time {selected_run_time} \\\n"
    "  --epochs 100"
)
_feedback_retrain_cmd = (
    f"conda run -n {CONDA_ENV} python scripts/retrain_with_feedback.py \\\n"
    f"  --file {profile.id} \\\n"
    f"  --pretrained_time {selected_run_time} \\\n"
    f"  --mode heads \\\n"
    f"  --promote"
)
with st.sidebar.expander("Refresh shortlist (`generated_candidates_clean.csv`)"):
    st.caption("Run from repo root. First = NN rank calibration; second = ActivityHead (recommended).")
    st.code(_pp_base, language="bash")
    st.code(_pp_activity, language="bash")
with st.sidebar.expander("Run simulation validation"):
    st.caption("Thermodynamic equilibrium + reactor descriptor validation. Uses Cantera when installed.")
    st.code(_sim_cmd, language="bash")
with st.sidebar.expander("Run full CO2 demo loop"):
    st.caption("Postprocess -> validation -> YAML sweep -> surrogate. Recommended for the video demo.")
    st.code(_co2_demo_cmd, language="bash")
with st.sidebar.expander("Train simulation surrogate"):
    st.caption("Use after simulation validation. The YAML defines the sweep; the sweep writes CSV labels.")
    st.code(_sweep_cmd, language="bash")
    st.code(_surrogate_cmd, language="bash")
with st.sidebar.expander("Feedback/retraining demo"):
    st.caption(
        "Imports example lab outcomes, refreshes property-head artifacts, then retrains the ranking head. "
        "Full CVAE retrain waits for more rows."
    )
    st.code(_feedback_import_cmd, language="bash")
    st.code(_feedback_heads_prep_cmd, language="bash")
    st.code(_feedback_retrain_cmd, language="bash")


# =========================================================================
# Shared dataframes
# =========================================================================

raw_candidates = load_candidate_csv(selected_gen_csv_path)
clean_path = discover_clean_csv(selected_run)
clean_ranked_by_activity_head = _clean_csv_uses_activity_head(clean_path)
clean_df = load_clean_candidates(clean_path) if clean_path is not None else pd.DataFrame()
simulation_path = discover_simulation_csv(selected_run)
simulation_df = load_simulation_validation(simulation_path) if simulation_path is not None else pd.DataFrame()
surrogate_metrics_path = simulation_surrogate_metrics_path(profile.id)
surrogate_metrics = load_json_file(surrogate_metrics_path)
sweep_csv_path = simulation_sweep_path(profile.id)
if not clean_df.empty:
    if not simulation_df.empty and "pseudo_smiles" in simulation_df.columns:
        sim_cols = [
            c for c in simulation_df.columns
            if c not in {"composition_view"} and c in {
                "pseudo_smiles",
                "simulation_backend",
                "thermodynamic_delta_g_kj_mol",
                "equilibrium_conversion_pct",
                "cantera_equilibrium_conversion_pct",
                "reaction_forward_margin",
                "catalyst_rate_score",
                "simulated_sty_g_h_gcat",
                "simulation_confidence",
                "simulation_notes",
            }
        ]
        clean_df = clean_df.merge(simulation_df[sim_cols], on="pseudo_smiles", how="left")
    if must_have_metal and "has_active_metal" in clean_df.columns:
        clean_df = clean_df[clean_df["has_active_metal"]].copy()
    if search_query.strip():
        mask = clean_df["composition_view"].str.contains(search_query.strip(), case=False, na=False) | (
            clean_df["pseudo_smiles"].str.contains(search_query.strip(), case=False, na=False)
        )
        clean_df = clean_df[mask].copy()
    clean_df = (
        clean_df.sort_values("predicted_sty_g_h_gcat", ascending=False).head(top_n).reset_index(drop=True)
    )
    clean_df["stability_proxy"] = clean_df["pseudo_smiles"].apply(stability_score_for)
    clean_df["selectivity_proxy_pct"] = clean_df["pseudo_smiles"].apply(
        lambda s: selectivity_proxy(_components_from_smiles(s))
    )

stats = parse_generated_stats(selected_stats_path) if selected_stats_path else {}
train_df = (
    parse_training_metrics(selected_run / "report.txt")
    if (selected_run / "report.txt").exists()
    else pd.DataFrame()
)
loss_df = (
    parse_loss_metrics(selected_run / "loss.txt")
    if (selected_run / "loss.txt").exists()
    else pd.DataFrame()
)
hyper_df = parse_hyper_result(HYPER_RESULT_PATH)
hyper_row = (
    hyper_df[hyper_df["run"] == selected_run_name].tail(1) if not hyper_df.empty else pd.DataFrame()
)


# =========================================================================
# Tab layout
# =========================================================================

tab_discover, tab_pathway, tab_compare, tab_kb, tab_validation, tab_feedback = st.tabs(
    ["Discover", "Pathway", "Compare", "Knowledge Base", "Validation", "Feedback"]
)


# ------------------------------------------------------------- DISCOVER
with tab_discover:
    if simple_ui:
        c1, c2, c3 = st.columns(3)
        c1.metric("Generated", f"{len(raw_candidates):,}")
        c2.metric("Unique SMILES", f"{raw_candidates['candidate'].nunique():,}")
        if (PROPERTY_DIR / "metrics.json").exists():
            m = json.loads((PROPERTY_DIR / "metrics.json").read_text(encoding="utf-8"))
            r2 = m.get("activity", {}).get("r2")
            c3.metric("Activity head R² (lit.)", f"{r2:.3f}" if r2 is not None else "N/A")
        else:
            c3.metric("Activity head R²", "N/A")
        _valid = stats.get("Validity")
        _novel = stats.get("Novelty")
        vtxt = (
            f"{int(_valid[0]):,} valid / {_valid[1]:.1f}%"
            if isinstance(_valid, list) and len(_valid) >= 2
            else "N/A"
        )
        ntxt = (
            f"{int(_novel[0]):,} novel / {_novel[1]:.1f}% of unique"
            if isinstance(_novel, list) and len(_novel) >= 2
            else "N/A"
        )
        st.caption(f"Generation validity **{vtxt}** · novelty **{ntxt}** (see *Technical details* for charts).")
        with st.expander("Closed-loop demo status", expanded=True):
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Known catalyst baseline", f"{len(load_known_catalysts(profile.retrieval_reaction))}")
            d2.metric("Validated shortlist rows", f"{len(simulation_df):,}" if not simulation_df.empty else "0")
            _n_ct_discover = (
                int(simulation_df["simulation_backend"].astype(str).str.contains("cantera", case=False).sum())
                if not simulation_df.empty and "simulation_backend" in simulation_df.columns
                else 0
            )
            d3.metric("Cantera-backed rows", f"{_n_ct_discover:,}")
            d4.metric(
                "Condition sweep rows",
                f"{surrogate_metrics.get('n_rows', 0):,}" if surrogate_metrics else ("ready" if sweep_csv_path.exists() else "0"),
            )
            test_r2 = surrogate_metrics.get("test_r2") if surrogate_metrics else None
            d5.metric("Simulation surrogate R²", f"{test_r2:.3f}" if test_r2 is not None else "N/A")
            st.caption(
                "Baseline = retrieved/offline known catalyst entries. Validated shortlist = row count in "
                "**this** output run's `simulation_validation.csv` (thermodynamic + descriptor layer). "
                "**Cantera-backed** counts rows whose `simulation_backend` includes a Cantera equilibrium cross-check. "
                "Condition sweep + surrogate R² come from the shared reaction profile CSV under `dataset/simulation/` "
                "(independent of which generation run is selected). Surrogate R² is fit to simulation labels, not "
                "wet-lab accuracy."
            )
            if _n_ct_discover == 0 and not simulation_df.empty:
                st.caption(
                    "Cantera-backed is 0 for this CSV: install Cantera and regenerate `simulation_validation.csv` "
                    "(sidebar **Run simulation validation**). The code now resolves `gri30.yaml` from Cantera's data "
                    "directory or `config/reactions/`."
                )
            _need_sim = clean_path is not None and (simulation_path is None or simulation_df.empty)
            if _need_sim:
                st.info(
                    "Validated shortlist is **0** because this run has no simulation output yet (or the CSV is empty). "
                    "Post-processing alone does not run the simulator. Use sidebar **Run simulation validation** "
                    "so `simulation_validation.csv` is written next to `generated_candidates_clean.csv`, then "
                    "**Reload data** if counts look stale."
                )
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Generated", f"{len(raw_candidates):,}")
        col2.metric("Unique", f"{raw_candidates['candidate'].nunique():,}")
        col3.metric("Validity", f"{stats.get('Validity', 'N/A')}")
        col4.metric("Novelty", f"{stats.get('Novelty', 'N/A')}")
        if (PROPERTY_DIR / "metrics.json").exists():
            m = json.loads((PROPERTY_DIR / "metrics.json").read_text(encoding="utf-8"))
            r2 = m.get("activity", {}).get("r2")
            col5.metric("Activity R^2", f"{r2:.3f}" if r2 is not None else "N/A")
        else:
            col5.metric("Activity R^2", "N/A")

    st.subheader("Top Ranked Candidates")
    if clean_df.empty:
        st.info(
            "No post-processed candidates found. Run `python scripts/postprocess_candidates.py "
            f"--candidates {selected_gen_csv_path}` to generate generated_candidates_clean.csv."
        )
        st.dataframe(raw_candidates.head(top_n), use_container_width=True)
    else:
        if clean_ranked_by_activity_head:
            st.caption(
                "Ranking flow: CVAE proposes catalyst compositions -> ActivityHead predicts methanol productivity "
                f"from the frozen CVAE latent vector -> chemistry gate checks catalyst relevance -> "
                "simulation validation estimates thermodynamic feasibility before export."
            )
        else:
            st.caption(
                "**STY** = NN score mapped to training quantiles (not the ActivityHead). "
                "Use sidebar **Refresh shortlist** → second command for head-aligned STY. "
                "Selectivity / stability = priors."
            )
        display_cols = [
            "composition_view",
            "predicted_sty_g_h_gcat",
            "validation_tier",
            "validation_score",
            "selectivity_proxy_pct",
            "stability_proxy",
            "pseudo_smiles",
        ]
        if "n_components" in clean_df.columns:
            display_cols.append("n_components")
        if "matched_methanol_family" in clean_df.columns:
            display_cols.append("matched_methanol_family")
        for col in (
            "equilibrium_conversion_pct",
            "catalyst_rate_score",
            "simulated_sty_g_h_gcat",
            "simulation_confidence",
        ):
            if col in clean_df.columns:
                display_cols.append(col)
        if "mp_structure_summary" in clean_df.columns:
            display_cols.append("mp_structure_summary")
        display_cols = [c for c in display_cols if c in clean_df.columns]
        column_labels = {
            "composition_view": "candidate",
            "predicted_sty_g_h_gcat": "predicted STY",
            "validation_tier": "gate tier",
            "validation_score": "gate score",
            "selectivity_proxy_pct": "selectivity prior",
            "stability_proxy": "stability descriptor",
            "pseudo_smiles": "pseudo-SMILES",
            "n_components": "components",
            "matched_methanol_family": "matched family",
            "equilibrium_conversion_pct": "eq. conversion %",
            "catalyst_rate_score": "catalyst rate score",
            "simulated_sty_g_h_gcat": "simulated STY",
            "simulation_confidence": "simulation confidence",
            "mp_structure_summary": "MP bulk refs (summary)",
        }
        st.dataframe(
            clean_df[display_cols].rename(columns=column_labels),
            use_container_width=True,
        )
        st.caption(
            "For the demo, focus on predicted STY, gate tier/score, simulated STY, and simulation confidence. "
            "Selectivity and stability are descriptor priors until real lab labels are logged."
        )
        _show_structure_builder = not clean_df.empty and (
            "components" in clean_df.columns or "mp_structure_refs" in clean_df.columns
        )
        if _show_structure_builder:
            from services.retrieval.mp_bulk_structures import (
                mp_crystal_structure_page_url,
                mp_refs_from_clean_csv_row,
            )
            from services.structure_builder import parse_miller_index

            with st.expander("Generated catalyst 3D structure", expanded=False):
                st.caption(
                    "Build an approximate full catalyst heterostructure from explicit modeling assumptions. "
                    "Materials Project supplies bulk inputs; the visualized object is the generated slab/interface "
                    "plus adsorbed metal, not a direct Materials Project crystal page."
                )
                mp_row_count = min(len(clean_df), 50)
                candidate_options = list(range(mp_row_count))
                if not candidate_options:
                    st.info("No shortlist rows available for structure generation.")
                else:
                    candidate_idx = st.selectbox(
                        "Catalyst candidate",
                        candidate_options,
                        index=None,
                        placeholder="Choose a shortlist candidate...",
                        format_func=lambda j: str(clean_df.iloc[j].get("composition_view", f"row {j + 1}")),
                        key="catalyst_structure_candidate_pick",
                    )
                    if candidate_idx is None:
                        st.caption("Select a candidate to configure and lazily build its full 3D catalyst model.")
                    else:
                        row = clean_df.iloc[candidate_idx]
                        with st.spinner("Resolving Materials Project bulk inputs for the selected candidate..."):
                            refs = mp_refs_from_clean_csv_row(row)

                        if not refs:
                            st.info(
                                "No Materials Project bulk inputs resolved for this candidate. Set **`MP_API_KEY`** "
                                "for live lookup and rebuild the structure."
                            )
                        else:
                            def _ref_label(ref: dict[str, Any]) -> str:
                                return (
                                    f"{ref.get('formula_pretty', ref.get('material_id', 'MP phase'))} "
                                    f"({ref.get('material_id', 'unknown')})"
                                )

                            oxide_refs = [r for r in refs if _is_oxide_ref(r)]
                            if not oxide_refs:
                                st.info("No oxide/support bulk phases were found for this candidate.")
                            else:
                                candidate_label = str(row.get("composition_view", f"candidate_{candidate_idx + 1}"))
                                active_metal = _candidate_active_metal(row)
                                role_options = ["substrate", "film", "surface additive", "dopant/vacancy", "ignore"]
                                ref_roles: dict[str, str] = {}

                                st.markdown("**Component roles**")
                                st.caption(
                                    "For candidates with three or more ingredients, choose one bulk support substrate, "
                                    "optionally one film/interface bulk, and place the remaining active species above "
                                    "the surface. Dopants/vacancies are tracked as assumptions but are not inserted "
                                    "into the lattice yet."
                                )

                                zn_substrate_id = next(
                                    (
                                        str(r.get("material_id", ""))
                                        for r in oxide_refs
                                        if "ZnO" in str(r.get("formula_pretty", ""))
                                    ),
                                    "",
                                )
                                first_oxide_id = str(oxide_refs[0].get("material_id", ""))
                                tio2_film_id = next(
                                    (
                                        str(r.get("material_id", ""))
                                        for r in oxide_refs
                                        if "TiO2" in str(r.get("formula_pretty", ""))
                                    ),
                                    "",
                                )

                                def _default_ref_role(ref: dict[str, Any]) -> str:
                                    mid = str(ref.get("material_id", ""))
                                    if mid == (zn_substrate_id or first_oxide_id):
                                        return "substrate"
                                    if mid == tio2_film_id and mid != (zn_substrate_id or first_oxide_id):
                                        return "film"
                                    if not _is_oxide_ref(ref):
                                        if any(_candidate_has_elemental_component(row, el) for el in _surface_elements_from_ref(ref)):
                                            return "surface additive"
                                        return "ignore"
                                    return "ignore"

                                for ref in refs:
                                    mid = str(ref.get("material_id", "unknown"))
                                    default_role = _default_ref_role(ref)
                                    ref_roles[mid] = st.selectbox(
                                        _ref_label(ref),
                                        role_options,
                                        index=role_options.index(default_role),
                                        key=f"catalyst_role_ref_{candidate_idx}_{mid}",
                                    )

                                covered_elements: set[str] = set()
                                for ref in refs:
                                    if ref_roles.get(str(ref.get("material_id", ""))) not in {"ignore"}:
                                        covered_elements.update(_bulk_ref_elements(ref))
                                extra_element_roles: dict[str, str] = {}
                                extra_elements = [
                                    el for el in _candidate_elements(row)
                                    if el != "O" and el not in covered_elements
                                ]
                                if extra_elements:
                                    st.caption("Candidate-only components")
                                    extra_cols = st.columns(min(3, len(extra_elements)))
                                    for i, symbol in enumerate(extra_elements):
                                        with extra_cols[i % len(extra_cols)]:
                                            extra_element_roles[symbol] = st.selectbox(
                                                symbol,
                                                role_options[2:],
                                                index=0,
                                                key=f"catalyst_role_extra_{candidate_idx}_{symbol}",
                                            )

                                substrate_refs = [
                                    r for r in refs
                                    if ref_roles.get(str(r.get("material_id", ""))) == "substrate" and _is_oxide_ref(r)
                                ]
                                film_refs = [
                                    r for r in refs
                                    if ref_roles.get(str(r.get("material_id", ""))) == "film" and _is_oxide_ref(r)
                                ]
                                can_build_structure = bool(substrate_refs)
                                if not can_build_structure:
                                    st.warning("Select one oxide bulk as the substrate before building.")
                                    substrate_ref = oxide_refs[0]
                                elif len(substrate_refs) > 1:
                                    st.warning("Multiple substrate bulks selected; using the first one.")
                                    substrate_ref = substrate_refs[0]
                                else:
                                    substrate_ref = substrate_refs[0]
                                film_ref = next(
                                    (
                                        r for r in film_refs
                                        if str(r.get("material_id", "")) != str(substrate_ref.get("material_id", ""))
                                    ),
                                    None,
                                )
                                support_refs = [substrate_ref] + ([film_ref] if isinstance(film_ref, dict) else [])

                                surface_species_list: list[str] = []
                                for ref in refs:
                                    if ref_roles.get(str(ref.get("material_id", ""))) == "surface additive":
                                        surface_species_list.extend(_surface_elements_from_ref(ref))
                                for symbol, role in extra_element_roles.items():
                                    if role == "surface additive":
                                        surface_species_list.append(symbol)
                                surface_species = tuple(dict.fromkeys(el for el in surface_species_list if el and el != "O"))
                                if not surface_species:
                                    surface_species = _surface_species_for_candidate(
                                        row,
                                        [substrate_ref] + ([film_ref] if isinstance(film_ref, dict) else []),
                                    )
                                surface_label = ", ".join(surface_species)
                                support_elements = set().union(*(_bulk_ref_elements(r) for r in support_refs))
                                duplicate_surface_elements = [
                                    el for el in surface_species
                                    if el in support_elements and el != "O"
                                ]
                                dopant_components = [
                                    _ref_label(r)
                                    for r in refs
                                    if ref_roles.get(str(r.get("material_id", ""))) == "dopant/vacancy"
                                ] + [
                                    symbol for symbol, role in extra_element_roles.items()
                                    if role == "dopant/vacancy"
                                ]
                                ignored_components = [
                                    _ref_label(r)
                                    for r in refs
                                    if ref_roles.get(str(r.get("material_id", ""))) == "ignore"
                                ] + [
                                    symbol for symbol, role in extra_element_roles.items()
                                    if role == "ignore"
                                ]

                                st.info(
                                    "Approximation indicator: generated slab/interface model. "
                                    f"Substrate = {_ref_label(substrate_ref)}; "
                                    f"film = {_ref_label(film_ref) if isinstance(film_ref, dict) else 'none'}; "
                                    f"surface additives = {surface_label or 'none'}."
                                )
                                if dopant_components:
                                    st.warning(
                                        "Dopant/vacancy roles are recorded as assumptions but are not atomistically "
                                        "inserted yet: " + ", ".join(dopant_components)
                                    )
                                if ignored_components:
                                    st.caption("Ignored for this generated structure: " + ", ".join(ignored_components))
                                if duplicate_surface_elements:
                                    st.warning(
                                        "These surface additives are already present in the selected substrate/film: "
                                        + ", ".join(duplicate_surface_elements)
                                        + ". Keeping them as surface additives means adding extra adsorbed atoms, "
                                        "not just using the support composition."
                                    )

                                role_rows = []
                                for ref in refs:
                                    mid = str(ref.get("material_id", ""))
                                    role = ref_roles.get(mid, "ignore")
                                    modeled_as = {
                                        "substrate": "bulk slab support",
                                        "film": "bulk film/interface slab",
                                        "surface additive": "adsorbed atom(s) above surface",
                                        "dopant/vacancy": "tracked only; not inserted yet",
                                        "ignore": "not included in generated geometry",
                                    }.get(role, role)
                                    role_rows.append(
                                        {
                                            "component": _ref_label(ref),
                                            "role": role,
                                            "modeled as": modeled_as,
                                        }
                                    )
                                for symbol, role in extra_element_roles.items():
                                    role_rows.append(
                                        {
                                            "component": symbol,
                                            "role": role,
                                            "modeled as": {
                                                "surface additive": "adsorbed atom(s) above surface",
                                                "dopant/vacancy": "tracked only; not inserted yet",
                                                "ignore": "not included in generated geometry",
                                            }.get(role, role),
                                        }
                                    )
                                with st.expander("Component role summary", expanded=False):
                                    st.dataframe(pd.DataFrame(role_rows), use_container_width=True, hide_index=True)

                                with st.expander("Modeling assumptions", expanded=True):
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        substrate_miller_raw = st.text_input(
                                            "Substrate Miller index",
                                            value="0 0 1",
                                            key=f"substrate_miller_{candidate_idx}",
                                        )
                                        substrate_layers = st.number_input(
                                            "Substrate thickness (layers)",
                                            min_value=1.0,
                                            max_value=12.0,
                                            value=3.0,
                                            step=1.0,
                                            key=f"substrate_layers_{candidate_idx}",
                                        )
                                    with c2:
                                        film_miller_raw = st.text_input(
                                            "Film Miller index",
                                            value="1 0 1",
                                            disabled=film_ref is None,
                                            key=f"film_miller_{candidate_idx}",
                                        )
                                        film_layers = st.number_input(
                                            "Film thickness (layers)",
                                            min_value=1.0,
                                            max_value=12.0,
                                            value=4.0,
                                            step=1.0,
                                            disabled=film_ref is None,
                                            key=f"film_layers_{candidate_idx}",
                                        )
                                    with c3:
                                        active_atoms = st.number_input(
                                            f"Atoms per surface species ({surface_label})",
                                            min_value=1,
                                            max_value=12,
                                            value=1,
                                            step=1,
                                            key=f"active_atoms_{candidate_idx}",
                                        )
                                        metal_height = st.number_input(
                                            "Metal height above surface (A)",
                                            min_value=1.0,
                                            max_value=5.0,
                                            value=2.1,
                                            step=0.1,
                                            key=f"metal_height_{candidate_idx}",
                                        )
                                    g1, g2 = st.columns(2)
                                    with g1:
                                        interface_gap = st.number_input(
                                            "Interface gap (A)",
                                            min_value=1.0,
                                            max_value=6.0,
                                            value=2.5,
                                            step=0.1,
                                            disabled=film_ref is None,
                                            key=f"interface_gap_{candidate_idx}",
                                        )
                                    with g2:
                                        vacuum = st.number_input(
                                            "Vacuum above surface (A)",
                                            min_value=8.0,
                                            max_value=30.0,
                                            value=15.0,
                                            step=1.0,
                                            key=f"vacuum_{candidate_idx}",
                                        )

                                st.caption(
                                    f"Surface additives selected for this generated model: **{surface_label}**. "
                                    "These are placed above the support/interface model."
                                )

                                for ref in support_refs:
                                    mid = str(ref.get("material_id", "")).strip()
                                    fp = str(ref.get("formula_pretty", mid))
                                    st.link_button(
                                        f"MP bulk input: {fp} ({mid})",
                                        str(ref.get("mp_crystal_url") or mp_crystal_structure_page_url(mid, fp)),
                                        use_container_width=False,
                                    )

                                build_clicked = st.button(
                                    "Build full catalyst structure",
                                    type="primary",
                                    disabled=not can_build_structure,
                                    key=f"build_catalyst_structure_{candidate_idx}",
                                )
                                if build_clicked:
                                    try:
                                        with st.spinner("Fetching MP bulks and building slab/interface catalyst structure..."):
                                            result = _dash_build_catalyst_structure(
                                                candidate_label=candidate_label,
                                                active_metal=active_metal,
                                                surface_species=surface_species,
                                                active_atoms=int(active_atoms),
                                                substrate_material_id=str(substrate_ref.get("material_id", "")),
                                                substrate_label=str(substrate_ref.get("formula_pretty", "")),
                                                substrate_miller=parse_miller_index(substrate_miller_raw, (0, 0, 1)),
                                                substrate_thickness=float(substrate_layers),
                                                film_material_id=(
                                                    str(film_ref.get("material_id", "")) if isinstance(film_ref, dict) else None
                                                ),
                                                film_label=(
                                                    str(film_ref.get("formula_pretty", "")) if isinstance(film_ref, dict) else None
                                                ),
                                                film_miller=parse_miller_index(film_miller_raw, (1, 0, 1)),
                                                film_thickness=float(film_layers),
                                                interface_gap_a=float(interface_gap),
                                                vacuum_a=float(vacuum),
                                                output_dir=str(selected_run / "generated_structures"),
                                            )
                                        st.session_state[f"catalyst_structure_result_{candidate_idx}"] = result
                                    except Exception as exc:  # noqa: BLE001
                                        st.error(f"Could not build catalyst structure: {exc}")

                                result = st.session_state.get(f"catalyst_structure_result_{candidate_idx}")
                                if isinstance(result, dict):
                                    result_spec = result.get("spec", {})
                                    result_species = tuple(result_spec.get("surface_species") or (result_spec.get("active_metal"),))
                                    result_substrate_id = str(result_spec.get("substrate_material_id", ""))
                                    result_film_id = str(result_spec.get("film_material_id") or "")
                                    selected_film_id = str(film_ref.get("material_id", "")) if isinstance(film_ref, dict) else ""
                                    if (
                                        result_species != surface_species
                                        or result_substrate_id != str(substrate_ref.get("material_id", ""))
                                        or result_film_id != selected_film_id
                                    ):
                                        st.info(
                                            "The displayed structure was built with older component-role assumptions. "
                                            "Click **Build full catalyst structure** again to regenerate it."
                                        )
                                        result = None
                                if isinstance(result, dict):
                                    validation = result.get("validation", {})
                                    st.markdown("**Generated structure validation**")
                                    v1, v2, v3, v4 = st.columns(4)
                                    v1.metric("Atoms", validation.get("n_atoms", "N/A"))
                                    v2.metric("Formula", validation.get("formula", "N/A"))
                                    min_dist = validation.get("min_interatomic_distance_a")
                                    v3.metric("Min distance", f"{min_dist:.2f} A" if isinstance(min_dist, (int, float)) else "N/A")
                                    surface_counts = validation.get("surface_atom_counts", {})
                                    if isinstance(surface_counts, dict):
                                        surface_count_label = ", ".join(f"{k}:{v}" for k, v in surface_counts.items())
                                    else:
                                        surface_count_label = "N/A"
                                    v4.metric("Surface atoms", surface_count_label)
                                    for warning in validation.get("warnings") or []:
                                        st.warning(str(warning))
                                    files = result.get("files", {})
                                    if files:
                                        st.caption(
                                            "Saved generated catalyst files: "
                                            + ", ".join(f"`{_relative_to_repo(Path(p))}`" for p in files.values())
                                        )
                                    build_meta = result.get("build", {})
                                    if build_meta.get("mode") == "stacked_slab_fallback":
                                        st.warning(
                                            "No coherent low-strain interface match was found, so this view uses an "
                                            "approximate stacked-slab fallback. Treat it as a visual starting geometry, "
                                            "not a DFT-ready relaxed interface."
                                        )
                                    elif build_meta.get("mode") == "coherent_interface":
                                        strain_score = build_meta.get("strain_score")
                                        st.caption(
                                            "Coherent interface generated"
                                            + (f" (strain score {strain_score:.4f})." if isinstance(strain_score, (int, float)) else ".")
                                        )
                                    try:
                                        from services.viz.mp_streamlit import (
                                            element_color,
                                            py3dmol_html_with_legend,
                                            py3dmol_view_from_cif,
                                            structure_dict_to_cif,
                                        )

                                        element_counts = _structure_element_counts(result["structure"])
                                        legend_items: list[dict[str, Any]] = []
                                        if element_counts:
                                            legend_items = [
                                                {
                                                    "element": symbol,
                                                    "atoms": count,
                                                    "role": _element_model_role(
                                                        symbol,
                                                        substrate_ref,
                                                        film_ref,
                                                        surface_species,
                                                    ),
                                                    "color": element_color(symbol),
                                                }
                                                for symbol, count in element_counts.items()
                                            ]
                                            st.caption(
                                                "The bottom-right legend gives species, base-cell counts, and model roles. "
                                                "The viewer repeats the saved base cell as 2x2x1 only for inspection."
                                            )

                                        cif = structure_dict_to_cif(result["structure"], supercell=(2, 2, 1))
                                        view = py3dmol_view_from_cif(
                                            cif,
                                            width=900,
                                            height=560,
                                            elements=list(element_counts.keys()),
                                        )
                                        st.caption(
                                            "Viewer shows a 2x2x1 visual supercell with unit-cell edges for clarity; "
                                            "saved files remain the generated base cell."
                                        )
                                        st.components.v1.html(py3dmol_html_with_legend(view, legend_items), height=610)
                                    except Exception as exc:  # noqa: BLE001
                                        st.info(f"Could not render the generated structure inline: {exc}")
        rerank_delta_state = st.session_state.get("latest_rerank_delta")
        if isinstance(rerank_delta_state, pd.DataFrame) and not rerank_delta_state.empty:
            with st.expander("Latest reranking impact", expanded=True):
                st.caption(
                    "This compares the shortlist immediately before and after the latest **Generate + rank** run. "
                    "Positive rank movement means the candidate moved up."
                )
                show = [
                    "composition_view",
                    "status",
                    "rank_before",
                    "rank_after",
                    "rank_movement",
                    "predicted_sty_g_h_gcat_before",
                    "predicted_sty_g_h_gcat_after",
                    "sty_delta",
                ]
                st.dataframe(
                    rerank_delta_state[show],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "composition_view": "candidate",
                        "rank_movement": st.column_config.NumberColumn("rank movement", format="%d"),
                        "predicted_sty_g_h_gcat_before": st.column_config.NumberColumn("STY before", format="%.4f"),
                        "predicted_sty_g_h_gcat_after": st.column_config.NumberColumn("STY after", format="%.4f"),
                        "sty_delta": st.column_config.NumberColumn("STY delta", format="%+.4f"),
                    },
                )

        csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download shortlist CSV",
            data=csv_bytes,
            file_name=f"shortlist_{selected_run_name}_{top_n}.csv",
            mime="text/csv",
        )

    _tech_expanded = not simple_ui
    with st.expander("Technical details (elements, raw stats, training curves)", expanded=_tech_expanded):
        st.subheader("Element frequency")
        elements_df = element_frequency(raw_candidates["candidate"])
        if elements_df.empty:
            st.info("No element tokens found.")
        else:
            st.bar_chart(elements_df.set_index("element")["count"].head(15))

        st.subheader("Generation metrics")
        summary_df = generation_stats_summary(stats)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            with st.expander("Raw generation stats file", expanded=False):
                st.caption(
                    "`*_2` keys are confirmation ratios written by the original CatDRX generator "
                    "(for example 0.22 = 22%). The table above is the judge-facing version."
                )
                st.json(stats)
        else:
            st.info("No generation_stats_*.txt found for this run.")

        st.subheader("Training curves")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("Train / val loss")
            if train_df.empty:
                st.info("report.txt not found.")
            else:
                st.line_chart(train_df.set_index("epoch")[["t_loss", "v_loss", "opt_loss"]])
        with tcol2:
            st.markdown("Reconstruction + NN loss")
            if loss_df.empty:
                st.info("loss.txt not found.")
            else:
                st.line_chart(loss_df.set_index("epoch")[["recon_t", "recon_v", "nn_t", "nn_v"]])

        if not hyper_row.empty:
            st.markdown("**Hyperparameter snapshot**")
            st.dataframe(hyper_row, use_container_width=True)


# ------------------------------------------------------------- PATHWAY
with tab_pathway:
    st.subheader("Reaction Pathway (free energy)")
    if profile.id != "co2_methanol":
        st.info(
            "Free-energy pathway diagrams in this build are parameterized for CO₂→methanol "
            "(HCOO / RWGS). Add a reaction-specific energy module before using this tab for "
            f"{profile.label}."
        )
    elif clean_df.empty:
        st.info("Run scripts/postprocess_candidates.py to populate the candidate list first.")
    else:
        choices = clean_df.apply(
            lambda r: f"{r['composition_view']} | STY {r['predicted_sty_g_h_gcat']:.2f}",
            axis=1,
        ).tolist()
        ccol, mcol = st.columns([2, 1])
        with ccol:
            choice_idx = st.selectbox(
                "Candidate",
                range(len(choices)),
                format_func=lambda i: choices[i],
                key="pathway_candidate",
            )
        with mcol:
            mechanism = st.radio(
                "Mechanism",
                ["HCOO", "RWGS"],
                horizontal=True,
                key="pathway_mechanism",
                help=(
                    "HCOO is dominant on Cu/ZnO catalysts; RWGS is the alternative *CO route. "
                    "The chosen mechanism does not change the activity ranking, only the diagram."
                ),
            )
        chosen_smiles = clean_df.iloc[choice_idx]["pseudo_smiles"]
        backend = "heuristic_scaling"
        energy_profile = compute_energy_profile(chosen_smiles, mechanism, backend)
        st.caption(f"Pathway backend: **{energy_profile['backend']}**. {energy_profile['citation']}")
        st.caption(
            "This diagram is a mechanism-level descriptor view for CO2-to-methanol, not a full reactor simulation. "
            "For each candidate, the graph is computed from composition-weighted adsorption/binding descriptors."
        )
        st.image(render_energy_diagram(energy_profile), use_container_width=True)
        if energy_profile["notes"]:
            st.info(energy_profile["notes"])
        if energy_profile["extras"]:
            with st.expander("Underlying binding-energy descriptors (eV)"):
                st.json(energy_profile["extras"])


# ------------------------------------------------------------- COMPARE
with tab_compare:
    st.subheader("Activity vs Selectivity")
    if clean_df.empty:
        st.info("Run scripts/postprocess_candidates.py first.")
    else:
        novel = pd.DataFrame(
            {
                "label": clean_df["composition_view"],
                "predicted_sty_g_h_gcat": clean_df["predicted_sty_g_h_gcat"],
                "selectivity_proxy_pct": clean_df["selectivity_proxy_pct"],
                "stability_proxy": clean_df["stability_proxy"],
                "source": "CatalyticIQ-novel",
            }
        )
        known_rows = load_known_catalysts(profile.retrieval_reaction)
        if known_rows:
            known_df = pd.DataFrame(
                [
                    {
                        "label": r["name"],
                        "predicted_sty_g_h_gcat": clean_df["predicted_sty_g_h_gcat"].median()
                        if not clean_df["predicted_sty_g_h_gcat"].empty
                        else 0.5,
                        "selectivity_proxy_pct": selectivity_proxy(r["composition"]),
                        "stability_proxy": stability_score_for(
                            ".".join(f"[{c}]" for c in r["composition"])
                        ),
                        "source": f"Known ({r['source']})",
                    }
                    for r in known_rows
                ]
            )
            combined = pd.concat([novel, known_df], ignore_index=True)
        else:
            combined = novel

        st.scatter_chart(
            combined,
            x="selectivity_proxy_pct",
            y="predicted_sty_g_h_gcat",
            color="source",
            size="stability_proxy",
            use_container_width=True,
        )
        _n_novel = len(novel)
        _n_known_plot = len(known_df) if known_rows else 0
        st.caption(
            "Use this as a prioritisation map, not final truth: y-axis is ActivityHead STY; x-axis is a composition "
            "selectivity prior; bubble size is a descriptor-based stability prior. Logged lab feedback replaces "
            "these priors over time. "
            f"**Points shown:** {_n_novel:,} from this run’s shortlist (`generated_candidates_clean.csv`) plus "
            f"{_n_known_plot:,} known baseline entries (Discover’s “Known catalyst baseline” count). "
            "Known catalysts are given the **median shortlist STY** on y only so they appear on the same scale as "
            "novel candidates; that value is not a measured literature STY per composition."
        )
        st.dataframe(combined.sort_values("predicted_sty_g_h_gcat", ascending=False).head(20), use_container_width=True)


# ------------------------------------------------------------- KNOWLEDGE BASE
with tab_kb:
    st.subheader("Known catalysts")
    known = load_known_catalysts(profile.retrieval_reaction)
    if not known:
        st.info("No known catalyst entries loaded.")
    else:
        st.caption(
            f"{len(known)} entries from the offline Materials Project/OCP-style seed cache. "
            "These are baseline priors for demo retrieval; live API adapters are the pilot path."
        )
        known_df = pd.DataFrame(known)
        if "composition" in known_df.columns:
            known_df["composition"] = known_df["composition"].apply(lambda xs: "/".join(xs))
        st.dataframe(known_df, use_container_width=True)

        _ocp_title = "Optional: adsorption lookup (qualitative)"
        if simple_ui:
            with st.expander(_ocp_title, expanded=False):
                st.caption("Offline representative adsorbate binding energies for a composition probe.")
                comp_pick = st.text_input("Composition (e.g. Cu/Zn)", value="Cu/Zn", key="ocp_probe_tab")
                if comp_pick.strip():
                    symbols = tuple(s.strip() for s in comp_pick.split("/") if s.strip())
                    ocp_rows = load_ocp_for_composition(symbols)
                    if ocp_rows:
                        ocp_df = pd.DataFrame(ocp_rows)
                        ocp_df = ocp_df.drop(columns=["citation"], errors="ignore")
                        st.dataframe(ocp_df, use_container_width=True)
                    else:
                        st.info(f"No OCP entries for {'/'.join(symbols)}.")
        else:
            st.markdown(f"**{_ocp_title}**")
            st.caption("Offline representative adsorbate binding energies for a composition probe.")
            comp_pick = st.text_input(
                "Probe binding energies for composition (slash-separated, e.g. Cu/Zn)",
                value="Cu/Zn",
                key="ocp_probe_tab",
            )
            if comp_pick.strip():
                symbols = tuple(s.strip() for s in comp_pick.split("/") if s.strip())
                ocp_rows = load_ocp_for_composition(symbols)
                if ocp_rows:
                    ocp_df = pd.DataFrame(ocp_rows)
                    ocp_df = ocp_df.drop(columns=["citation"], errors="ignore")
                    st.dataframe(ocp_df, use_container_width=True)
                else:
                    st.info(f"No OCP entries cached for composition {'/'.join(symbols)}.")


# ------------------------------------------------------------- VALIDATION
with tab_validation:
    st.subheader("Simulation validation")
    if simulation_df.empty:
        st.info("No simulation_validation.csv found for this run. Use the sidebar simulation command.")
    else:
        n_cantera = (
            int(simulation_df["simulation_backend"].astype(str).str.contains("cantera", case=False).sum())
            if "simulation_backend" in simulation_df.columns
            else 0
        )
        sim1, sim2, sim3, sim4 = st.columns(4)
        sim1.metric("Validated candidates", f"{len(simulation_df):,}")
        sim2.metric(
            "Cantera-backed rows",
            f"{n_cantera:,}",
        )
        sim3.metric(
            "Mean eq. conversion",
            f"{simulation_df['equilibrium_conversion_pct'].mean():.1f}%"
            if "equilibrium_conversion_pct" in simulation_df.columns
            else "N/A",
        )
        sim4.metric(
            "Mean simulated STY",
            f"{simulation_df['simulated_sty_g_h_gcat'].mean():.3f}"
            if "simulated_sty_g_h_gcat" in simulation_df.columns
            else "N/A",
        )
        st.dataframe(simulation_df.head(30), use_container_width=True)
        if n_cantera == 0:
            st.caption(
                "If this stays at 0: install **Cantera** (`conda install -c conda-forge cantera`), ensure `gri30.yaml` "
                "is discoverable (bundled with Cantera or next to `config/reactions/co2_methanol.yaml`), then re-run "
                "the sidebar **validate_shortlist_simulation** command to regenerate `simulation_validation.csv`."
            )
        st.caption(
            "Equilibrium conversion is the thermodynamic CO2 conversion predicted at the selected T/P/feed before "
            "lab calibration. Cantera-backed rows mean Cantera successfully cross-checked gas equilibrium; the "
            "catalyst-specific part still comes from the YAML descriptor screen."
        )

    st.subheader("Sweep surrogate")
    if not surrogate_metrics:
        st.info("No simulation surrogate metrics found. Generate a sweep and train the surrogate from the sidebar.")
    else:
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Sweep rows", f"{surrogate_metrics.get('n_rows', 0):,}")
        sm2.metric("Train rows", f"{surrogate_metrics.get('n_train', 0):,}")
        sm3.metric("Test R²", f"{surrogate_metrics.get('test_r2', float('nan')):.3f}")
        sm4.metric("Test MAE", f"{surrogate_metrics.get('test_mae', float('nan')):.4f}")
        st.caption(
            "The surrogate is a fast regressor trained on generated simulation-sweep labels. High R² here means it "
            "reproduces the simulation layer well; it is not a claim of wet-lab accuracy."
        )
        with st.expander("Surrogate metrics JSON", expanded=not simple_ui):
            st.json(surrogate_metrics)

    st.subheader("Encoder validation")
    report_json = VALIDATION_DIR / "encoder_report.json"
    report_pdf = VALIDATION_DIR / "encoder_report.pdf"
    if not report_json.exists():
        st.info(
            "Run `python scripts/validate_encoder.py` to generate encoder_report.pdf and "
            "encoder_report.json."
        )
    else:
        report = json.loads(report_json.read_text(encoding="utf-8"))
        held = report.get("held_out", {})
        pareto = report.get("pareto", {})
        al = report.get("active_learning", {})
        nbr = report.get("latent_neighbours", {})
        coh = report.get("top_decile_coherence", {})

        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        vc1.metric("Held-out R^2", f"{held.get('r2', float('nan')):.3f}")
        vc2.metric("90% coverage", f"{held.get('coverage_90pct', float('nan')):.0%}")
        vc3.metric("Neighbour Jaccard", f"{nbr.get('mean_jaccard', float('nan')):.2f}")
        vc4.metric("Top-decile coherence", f"{coh.get('mean_top_share', float('nan')):.0%}")
        vc5.metric(
            "AL recovery (top20 in top50)",
            f"{al.get('recovered_in_top50', 0)}/{al.get('n_target', 0)}",
        )

        if report_pdf.exists():
            with open(report_pdf, "rb") as f:
                st.download_button(
                    "Download encoder validation PDF",
                    data=f.read(),
                    file_name="encoder_report.pdf",
                    mime="application/pdf",
                )

        _val_extra = not simple_ui
        with st.expander("More validation (Pareto toy comparison + raw JSON)", expanded=_val_extra):
            pareto_df = pd.DataFrame(
                [
                    {"source": "random", "mean": pareto.get("random_mean", 0.0), "p95": pareto.get("random_p95", 0.0)},
                    {"source": "GA", "mean": pareto.get("ga_mean", 0.0), "p95": pareto.get("ga_p95", 0.0)},
                    {"source": "CVAE", "mean": pareto.get("cvae_mean", 0.0), "p95": pareto.get("cvae_p95", 0.0)},
                ]
            )
            st.markdown("Pareto-style comparison (toy baselines)")
            st.caption(
                "Optional technical slide only: this compares the encoder/ranking distribution against random and "
                "small-GA baselines. Skip it in the main video unless judges ask how the latent ranking was checked."
            )
            st.bar_chart(pareto_df.set_index("source"))
            st.json(report)


# ------------------------------------------------------------- FEEDBACK
with tab_feedback:
    st.subheader("Lab feedback loop")
    try:
        from services.feedback.store import ExperimentRecord, FeedbackStore

        feedback_store: FeedbackStore | None = FeedbackStore()
    except Exception as exc:  # noqa: BLE001
        feedback_store = None
        st.error(f"Feedback store unavailable: {exc}")

    if feedback_store is not None:
        st.info(
            "Demo loop: import example outcomes -> refresh property-head artifacts -> "
            "retrain the ActivityHead ranking model. "
            "Full CVAE fine-tuning is triggered only after enough validated lab rows pass drift checks."
        )
        feedback_rows = feedback_store.list_experiments(limit=10_000)
        measured_feedback_rows = [r for r in feedback_rows if r.get("measured_sty") is not None]
        versions_preview = feedback_store.list_model_versions()

        st.markdown("**Closed-loop demo controls**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Measured feedback rows", len(measured_feedback_rows))
        c2.metric("Current CVAE run", selected_run_time)
        c3.metric("Last feedback model", versions_preview[0]["version"] if versions_preview else "not retrained")

        action_left, action_mid, action_right, action_clear = st.columns([1, 1, 1, 1])
        with action_left:
            retrain_clicked = st.button(
                "Retrain activity head",
                type="primary",
                use_container_width=True,
                disabled=not measured_feedback_rows,
                help=(
                    "Runs scripts/retrain_with_feedback.py --mode heads --promote for the CVAE run selected in the "
                    "sidebar. Retrains ActivityHead using cached literature embeddings (embeddings.npz) plus measured "
                    "STY rows from the feedback store. For the demo path, --promote always activates the newly "
                    "retrained head (even if held-out literature R² regresses). Does not delete logged experiments."
                ),
            )
        with action_mid:
            generate_clicked = st.button(
                "Generate + rank candidates",
                use_container_width=True,
                help=(
                    "Runs generation.py for the selected CVAE run, then postprocess_candidates.py with "
                    "--use-activity-head. Overwrites generated_candidates_clean.csv in that run folder and refreshes "
                    "cached CSV reads. The rerank delta table compares predicted STY/rank before vs after this run "
                    "(same compositions only); it is not lab measured vs predicted. "
                ),
            )
        with action_right:
            refresh_clicked = st.button(
                "Refresh dashboard",
                use_container_width=True,
                help=(
                    "Clears Streamlit cached data and reloads files from disk (shortlists, simulation CSVs, etc.). "
                    "Use after manual edits or subprocess steps outside the app. Does not train models or change DuckDB."
                ),
            )
        with action_clear:
            clear_logs_clicked = st.button(
                "Clear logged experiments",
                use_container_width=True,
                help=(
                    "Deletes every row in DuckDB table experiments (cache/feedback.duckdb). "
                    "Does not remove model_versions or dataset checkpoints. "
                    "Re-import examples with scripts/import_feedback_csv.py if needed."
                ),
            )

        if "_feedback_cleared_count" in st.session_state:
            _cnt = st.session_state.pop("_feedback_cleared_count")
            st.success(f"Cleared {_cnt} logged experiment row(s) from cache/feedback.duckdb.")

        st.caption(
            "Logged lab rows live in `cache/feedback.duckdb` (`experiments`). "
            "Clear logged experiments wipes that table only; CVAE runs and `model_versions` are unchanged."
        )

        if clear_logs_clicked:
            _cleared = feedback_store.clear_experiments()
            st.cache_data.clear()
            st.session_state["_feedback_cleared_count"] = _cleared
            st.rerun()

        if refresh_clicked:
            st.cache_data.clear()
            st.rerun()

        if retrain_clicked:
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                CONDA_ENV,
                "python",
                "scripts/retrain_with_feedback.py",
                "--file",
                profile.id,
                "--pretrained_time",
                selected_run_time,
                "--mode",
                "heads",
                "--promote",
            ]
            with st.spinner("Retraining ActivityHead with logged feedback..."):
                result = _run_demo_command(cmd, timeout=900)
            st.session_state["feedback_last_action"] = {
                "label": "Retrain activity head",
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-2000:],
            }
            if result.returncode == 0:
                st.cache_data.clear()
                st.success("Retraining complete. The promoted ActivityHead is now active for ranking.")
            else:
                st.error("Retraining failed. Expand the run log below.")

        if generate_clicked:
            before_rerank_df = pd.DataFrame()
            before_clean_path = discover_clean_csv(selected_run)
            if before_clean_path is not None:
                try:
                    before_rerank_df = pd.read_csv(before_clean_path)
                except Exception:
                    before_rerank_df = pd.DataFrame()
            gen_cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                CONDA_ENV,
                "python",
                "generation.py",
                "--file",
                profile.id,
                "--pretrained_file",
                profile.id,
                "--pretrained_time",
                selected_run_time,
                "--correction",
                "enabled",
                "--from_around_mol",
                "enabled",
                "--n_samples",
                str(_GENERATION_N_SAMPLES),
            ]
            with st.spinner("Generating a fresh CVAE candidate batch..."):
                gen_result = _run_demo_command(gen_cmd, timeout=1200)
            newest_generated = discover_generated_csv(selected_run)[0] if discover_generated_csv(selected_run) else selected_gen_csv_path
            if gen_result.returncode == 0:
                post_cmd = [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    CONDA_ENV,
                    "python",
                    "scripts/postprocess_candidates.py",
                    "--candidates",
                    _relative_to_repo(newest_generated),
                    "--training",
                    _train_csv_rel,
                    "--dataset-file",
                    profile.id,
                    "--use-activity-head",
                    "--cvae-run-dir",
                    _run_rel,
                    "--output",
                    _out_rel,
                ]
                with st.spinner("Ranking and cleaning candidates with the current ActivityHead..."):
                    post_result = _run_demo_command(post_cmd, timeout=900)
            else:
                post_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
            rerank_delta = pd.DataFrame()
            if gen_result.returncode == 0 and post_result.returncode == 0:
                after_clean_path = selected_run / "generated_candidates_clean.csv"
                try:
                    after_rerank_df = pd.read_csv(after_clean_path)
                    rerank_delta = _rerank_delta_frame(before_rerank_df, after_rerank_df)
                except Exception:
                    rerank_delta = pd.DataFrame()
                st.session_state["latest_rerank_delta"] = rerank_delta
                st.session_state["latest_rerank_meta"] = {
                    "raw_file": newest_generated.name,
                    "rows_before": int(len(before_rerank_df)),
                    "rows_after": int(len(after_rerank_df)) if "after_rerank_df" in locals() else 0,
                }
            st.session_state["feedback_last_action"] = {
                "label": "Generate + rank candidates",
                "returncode": gen_result.returncode or post_result.returncode,
                "stdout": (gen_result.stdout + "\n" + post_result.stdout)[-5000:],
                "stderr": (gen_result.stderr + "\n" + post_result.stderr)[-3000:],
            }
            if gen_result.returncode == 0 and post_result.returncode == 0:
                st.cache_data.clear()
                st.success(f"Generated and ranked candidates. Latest raw file: {newest_generated.name}")
                if not rerank_delta.empty:
                    moved = rerank_delta[
                        (rerank_delta["status"] == "reranked")
                        & (
                            (rerank_delta["sty_delta"].abs() > 1e-9)
                            | (rerank_delta["rank_movement"].fillna(0) != 0)
                        )
                    ]
                    st.info(
                        f"Reranking impact captured: {len(moved)} existing candidate(s) changed score/rank. "
                        "See **Reranking impact** below and on the Discover tab."
                    )
            else:
                st.error("Generation or ranking failed. Expand the run log below.")

        rerank_delta_state = st.session_state.get("latest_rerank_delta")
        if isinstance(rerank_delta_state, pd.DataFrame) and not rerank_delta_state.empty:
            meta = st.session_state.get("latest_rerank_meta", {})
            with st.expander("Reranking impact from latest Generate + Rank", expanded=True):
                st.caption(
                    f"Raw batch: `{meta.get('raw_file', 'latest')}` · "
                    f"shortlist rows {meta.get('rows_before', '?')} -> {meta.get('rows_after', '?')}. "
                    "Positive rank movement means the candidate moved up."
                )
                show = [
                    "composition_view",
                    "status",
                    "rank_before",
                    "rank_after",
                    "rank_movement",
                    "predicted_sty_g_h_gcat_before",
                    "predicted_sty_g_h_gcat_after",
                    "sty_delta",
                ]
                st.dataframe(
                    rerank_delta_state[show],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "composition_view": "candidate",
                        "rank_movement": st.column_config.NumberColumn("rank movement", format="%d"),
                        "predicted_sty_g_h_gcat_before": st.column_config.NumberColumn("STY before", format="%.4f"),
                        "predicted_sty_g_h_gcat_after": st.column_config.NumberColumn("STY after", format="%.4f"),
                        "sty_delta": st.column_config.NumberColumn("STY delta", format="%+.4f"),
                    },
                )

        last_action = st.session_state.get("feedback_last_action")
        if last_action:
            status = "success" if last_action["returncode"] == 0 else "error"
            with st.expander(f"Last action log: {last_action['label']} ({status})", expanded=last_action["returncode"] != 0):
                if last_action.get("stdout"):
                    st.markdown("**stdout**")
                    st.code(last_action["stdout"], language="text")
                if last_action.get("stderr"):
                    st.markdown("**stderr**")
                    st.code(last_action["stderr"], language="text")

        with st.expander("Feedback import and retraining commands", expanded=False):
            st.code(_feedback_import_cmd, language="bash")
            st.code(_feedback_heads_prep_cmd, language="bash")
            st.code(_feedback_retrain_cmd, language="bash")
            example_path = ROOT / "dataset" / "feedback" / "co2_methanol_lab_results_example.csv"
            if example_path.exists():
                st.download_button(
                    "Download example lab feedback CSV",
                    data=example_path.read_bytes(),
                    file_name="co2_methanol_lab_results_example.csv",
                    mime="text/csv",
                )

        fl, fr = st.columns([1, 1])

        with fl:
            st.markdown("**Log a new experiment**")
            candidate_choices: list[str] = []
            if not clean_df.empty:
                candidate_choices = clean_df.apply(
                    lambda r: f"{r['composition_view']} ({r['pseudo_smiles']})",
                    axis=1,
                ).tolist()
            candidate_source = st.radio(
                "Candidate source",
                ["Latest shortlist", "New candidate"],
                horizontal=True,
                key="feedback_candidate_source",
            )
            chosen = "(no shortlist available)"
            if candidate_source == "Latest shortlist":
                chosen = st.selectbox(
                    "Candidate from latest shortlist",
                    options=candidate_choices or ["(no shortlist available)"],
                    key="feedback_candidate",
                )
            with st.form("feedback_form", clear_on_submit=True):
                if candidate_source == "New candidate":
                    custom_comp_view = st.text_input(
                        "Candidate name / composition",
                        placeholder="Example: Cu/ZnO/ZrO2",
                    )
                    custom_pseudo = st.text_input(
                        "Pseudo-SMILES / component tokens",
                        placeholder="Example: [Cu].[Zn].[Zr]",
                    )
                    st.caption("Optional predictions are stored for discrepancy analysis only; measured STY drives heads retraining.")
                    manual_pred_sty = st.text_input("Predicted STY (optional)", placeholder="Example: 0.84")
                    manual_pred_sel = st.text_input("Predicted selectivity % (optional)", placeholder="Example: 68")
                    manual_pred_yield = st.text_input("Predicted yield % (optional)", placeholder="Example: 16")
                else:
                    custom_comp_view = ""
                    custom_pseudo = ""
                    manual_pred_sty = ""
                    manual_pred_sel = ""
                    manual_pred_yield = ""
                measured_sty = st.number_input("Measured STY (g MeOH / h / g cat)", min_value=0.0, step=0.05, value=0.0)
                measured_sel = st.number_input("Measured MeOH selectivity (%)", min_value=0.0, max_value=100.0, step=1.0, value=0.0)
                measured_yield = st.number_input("Measured MeOH yield (%)", min_value=0.0, max_value=100.0, step=1.0, value=0.0)
                measured_tos = st.number_input("Measured stability (h on stream)", min_value=0.0, step=10.0, value=0.0)
                t_c = st.number_input("Temperature (C)", min_value=100.0, max_value=400.0, value=240.0)
                p_bar = st.number_input("Pressure (bar)", min_value=1.0, max_value=200.0, value=50.0)
                h2_co2 = st.number_input("H2/CO2 ratio", min_value=1.0, max_value=10.0, value=3.0)
                user = st.text_input("Logged by", value="researcher")
                note = st.text_area("Notes", value="")
                submit = st.form_submit_button("Save experiment")
                if submit:
                    predicted_row = pd.DataFrame()
                    if candidate_source == "Latest shortlist":
                        if "(" not in chosen:
                            st.error("Select a shortlist candidate before saving.")
                            st.stop()
                        comp_view = chosen.split(" (")[0]
                        pseudo = chosen.split("(", 1)[1].rstrip(")")
                        predicted_row = (
                            clean_df[clean_df["pseudo_smiles"] == pseudo].head(1)
                            if not clean_df.empty and "pseudo_smiles" in clean_df.columns
                            else pd.DataFrame()
                        )
                        predicted_sty = (
                            float(predicted_row.iloc[0]["predicted_sty_g_h_gcat"])
                            if not predicted_row.empty and pd.notna(predicted_row.iloc[0].get("predicted_sty_g_h_gcat"))
                            else None
                        )
                        predicted_selectivity = (
                            float(predicted_row.iloc[0]["selectivity_proxy_pct"])
                            if not predicted_row.empty and pd.notna(predicted_row.iloc[0].get("selectivity_proxy_pct"))
                            else None
                        )
                        predicted_yield = None
                        if predicted_selectivity is not None and "equilibrium_conversion_pct" in predicted_row.columns:
                            conv = predicted_row.iloc[0].get("equilibrium_conversion_pct")
                            if pd.notna(conv):
                                predicted_yield = float(conv) * predicted_selectivity / 100.0
                    else:
                        comp_view = custom_comp_view.strip()
                        pseudo = custom_pseudo.strip() or comp_view
                        if not comp_view or not pseudo:
                            st.error("Enter both a candidate name/composition and pseudo-SMILES/component tokens.")
                            st.stop()
                        predicted_sty = _optional_float(manual_pred_sty)
                        predicted_selectivity = _optional_float(manual_pred_sel)
                        predicted_yield = _optional_float(manual_pred_yield)

                    rec = ExperimentRecord(
                        candidate_id=pseudo,
                        pseudo_smiles=pseudo,
                        composition_view=comp_view,
                        measured_sty=float(measured_sty) if measured_sty > 0 else None,
                        predicted_sty=predicted_sty,
                        measured_selectivity=float(measured_sel) if measured_sel > 0 else None,
                        predicted_selectivity=predicted_selectivity,
                        measured_yield=float(measured_yield) if measured_yield > 0 else None,
                        predicted_yield=predicted_yield,
                        measured_stability_tos_h=float(measured_tos) if measured_tos > 0 else None,
                        conditions={"T_C": float(t_c), "P_bar": float(p_bar), "h2_co2": float(h2_co2)},
                        user=user.strip() or "anonymous",
                        notes=note.strip(),
                        model_version="current",
                    )
                    feedback_store.log_experiment(rec)
                    st.success(f"Logged experiment for {comp_view}.")

        with fr:
            st.markdown("**Logged experiments**")
            recent = feedback_store.list_experiments(limit=20)
            if not recent:
                st.info("No experiments logged yet.")
            else:
                recent_df = pd.DataFrame(
                    [
                        {
                            "logged_at": r["logged_at"],
                            "composition": r["composition_view"],
                            "pseudo_smiles": r["pseudo_smiles"],
                            "predicted_sty": r.get("predicted_sty"),
                            "measured_sty": r["measured_sty"],
                            "predicted_selectivity": r.get("predicted_selectivity"),
                            "measured_selectivity": r["measured_selectivity"],
                            "predicted_yield": r.get("predicted_yield"),
                            "measured_yield": r.get("measured_yield"),
                            "stability_h": r["measured_stability_tos_h"],
                            "conditions": r.get("conditions"),
                            "user": r["user"],
                            "model_version": r["model_version"],
                        }
                        for r in recent
                    ]
                )
                st.caption("Raw rows from cache/feedback.duckdb. Gap columns below are derived, not separately logged.")
                st.dataframe(
                    recent_df,
                    use_container_width=True,
                    column_config={
                        "predicted_sty": st.column_config.NumberColumn("Pred STY", format="%.4f"),
                        "measured_sty": st.column_config.NumberColumn("Measured STY", format="%.4f"),
                        "predicted_selectivity": st.column_config.NumberColumn("Pred sel %", format="%.1f"),
                        "measured_selectivity": st.column_config.NumberColumn("Measured sel %", format="%.1f"),
                        "predicted_yield": st.column_config.NumberColumn("Pred yield %", format="%.1f"),
                        "measured_yield": st.column_config.NumberColumn("Measured yield %", format="%.1f"),
                    },
                )

                discrepancy_df = _feedback_discrepancy_frame(recent)
                if not discrepancy_df.empty:
                    with st.expander("Prediction vs actual analysis", expanded=True):
                        flagged = discrepancy_df[discrepancy_df["flags"] != "ok"].copy()
                        if flagged.empty:
                            st.success("No major discrepancies under current demo thresholds.")
                        else:
                            st.warning(f"{len(flagged)} feedback row(s) exceed discrepancy thresholds.")
                        st.dataframe(
                            discrepancy_df,
                            use_container_width=True,
                            column_config={
                                "sty_error_pct": st.column_config.NumberColumn("STY error %", format="%.1f"),
                                "selectivity_delta": st.column_config.NumberColumn("Sel delta", format="%.1f"),
                                "yield_delta": st.column_config.NumberColumn("Yield delta", format="%.1f"),
                            },
                        )

                n_pending = feedback_store.count_since_last_train("current")
                st.caption(
                    f"**{n_pending}** row(s) since last train · retrain: "
                    f"`python scripts/retrain_with_feedback.py --file {profile.id} "
                    f"--pretrained_time <cvae_ts> --mode heads --promote`"
                )

        versions = feedback_store.list_model_versions()
        if versions:
            with st.expander("Model version history", expanded=not simple_ui):
                st.dataframe(pd.DataFrame(versions), use_container_width=True)
