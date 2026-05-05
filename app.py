from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


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
# Chemistry helpers (RDKit, energy, retrieval, stability)
# =========================================================================

@st.cache_data(show_spinner=False)
def render_smiles_png(smiles: str, size: int = 220) -> bytes | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        img = Draw.MolToImage(mol, size=(size, size))
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
            "citation": e.citation,
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
if not runs:
    st.error(f"No output runs found under {DATASET_DIR}. Fine-tune the CVAE for this reaction first.")
    st.stop()

run_map = {p.name: p for p in runs}
selected_run_name = st.sidebar.selectbox("Run folder", list(run_map.keys()))
selected_run = run_map[selected_run_name]

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
    f"conda run -n catdrx python scripts/postprocess_candidates.py \\\n"
    f"  --candidates {_gen_rel} \\\n"
    f"  --training {_train_csv_rel} \\\n"
    f"  --output {_out_rel}"
)
_pp_activity = (
    f"conda run -n catdrx python scripts/postprocess_candidates.py \\\n"
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
    f"conda run -n catdrx python scripts/validate_shortlist_simulation.py \\\n"
    f"  --candidates {_out_rel} \\\n"
    f"  --reaction-config {_reaction_cfg_rel} \\\n"
    f"  --output {_sim_rel} \\\n"
    f"  --temperature-c 240 \\\n"
    f"  --pressure-bar 50"
)
_sweep_rel = _relative_to_repo(simulation_sweep_path(profile.id))
_surrogate_rel = f"dataset/simulation/surrogates/{profile.id}"
_sweep_cmd = (
    f"conda run -n catdrx python scripts/generate_cantera_sweep.py \\\n"
    f"  --reaction-config {_reaction_cfg_rel} \\\n"
    f"  --candidates {_out_rel} \\\n"
    f"  --output {_sweep_rel}"
)
_surrogate_cmd = (
    f"conda run -n catdrx python scripts/train_simulation_surrogate.py \\\n"
    f"  --input {_sweep_rel} \\\n"
    f"  --target simulated_sty_g_h_gcat \\\n"
    f"  --output-dir {_surrogate_rel}"
)
_co2_demo_cmd = (
    "conda run -n catdrx python scripts/run_co2_demo.py "
    "--sweep-samples 200"
)
_feedback_import_cmd = (
    "conda run -n catdrx python scripts/import_feedback_csv.py \\\n"
    "  --input dataset/feedback/co2_methanol_lab_results_example.csv"
)
_feedback_retrain_cmd = (
    f"conda run -n catdrx python scripts/retrain_with_feedback.py \\\n"
    f"  --file {profile.id} \\\n"
    f"  --pretrained_time 20260503_190505 \\\n"
    f"  --mode heads"
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
    st.caption("Imports example lab outcomes, then retrains the ranking head. Full CVAE retrain waits for more rows.")
    st.code(_feedback_import_cmd, language="bash")
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
        vtxt = stats.get("Validity", "—")
        ntxt = stats.get("Novelty", "—")
        st.caption(f"Generation validity **{vtxt}** · novelty **{ntxt}** (see *Technical details* for charts).")
        with st.expander("Closed-loop demo status", expanded=True):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Known baseline", f"{len(load_known_catalysts(profile.retrieval_reaction))}")
            d2.metric("Simulation rows", f"{len(simulation_df):,}" if not simulation_df.empty else "0")
            d3.metric(
                "Sweep rows",
                f"{surrogate_metrics.get('n_rows', 0):,}" if surrogate_metrics else ("ready" if sweep_csv_path.exists() else "0"),
            )
            test_r2 = surrogate_metrics.get("test_r2") if surrogate_metrics else None
            d4.metric("Surrogate test R²", f"{test_r2:.3f}" if test_r2 is not None else "N/A")
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
                f"Ranking: CVAE generates candidates -> ActivityHead predicts methanol STY "
                f"(μ from CVAE; median T/P from `{_train_csv_rel}`) -> validation gate -> "
                "simulation/surrogate validation before export. `raw_score` = generation NN."
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
        st.dataframe(clean_df[display_cols], use_container_width=True)

        st.markdown("**Top candidate 2D depictions**")
        n_show = min(8, len(clean_df))
        cols = st.columns(min(4, max(1, n_show)))
        for i in range(n_show):
            row = clean_df.iloc[i]
            png = render_smiles_png(row["pseudo_smiles"])
            with cols[i % len(cols)]:
                if png is not None:
                    st.image(
                        png,
                        caption=f"{row['composition_view']} — STY {row['predicted_sty_g_h_gcat']:.2f}",
                    )
                else:
                    st.write(f"{row['composition_view']} (no 2D)")

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

        st.subheader("Generation metrics (raw)")
        if stats:
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
        ccol, mcol, bcol = st.columns([2, 1, 1])
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
        with bcol:
            backend = st.radio(
                "Backend tier",
                ["heuristic_scaling", "xtb_topn", "dft_topk"],
                index=0,
                key="pathway_backend",
                help=(
                    "Tier A always runs. Tier B / C activate when xtb-python / fairchem are "
                    "installed; otherwise they degrade gracefully and report the actual backend used."
                ),
            )

        chosen_smiles = clean_df.iloc[choice_idx]["pseudo_smiles"]
        energy_profile = compute_energy_profile(chosen_smiles, mechanism, backend)
        st.caption(f"Backend used: **{energy_profile['backend']}**. {energy_profile['citation']}")
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
        st.caption(
            "Bubble size ≈ stability proxy (descriptor). Selectivity ≈ composition prior "
            + ("(methanol-oriented placeholder for this reaction)." if profile.id != "co2_methanol" else "(TheMeCat-style prior).")
        )
        st.dataframe(combined.sort_values("predicted_sty_g_h_gcat", ascending=False).head(20), use_container_width=True)


# ------------------------------------------------------------- KNOWLEDGE BASE
with tab_kb:
    st.subheader("Known catalysts (Materials Project + OCP)")
    known = load_known_catalysts(profile.retrieval_reaction)
    if not known:
        st.info("Knowledge base is empty.")
    else:
        st.caption(f"{len(known)} entries (MP + OCP cache when offline).")
        known_df = pd.DataFrame(known)
        if "composition" in known_df.columns:
            known_df["composition"] = known_df["composition"].apply(lambda xs: "/".join(xs))
        st.dataframe(known_df, use_container_width=True)

        _ocp_title = "Optional: OCP binding lookup"
        if simple_ui:
            with st.expander(_ocp_title, expanded=False):
                comp_pick = st.text_input("Composition (e.g. Cu/Zn)", value="Cu/Zn", key="ocp_probe_tab")
                if comp_pick.strip():
                    symbols = tuple(s.strip() for s in comp_pick.split("/") if s.strip())
                    ocp_rows = load_ocp_for_composition(symbols)
                    if ocp_rows:
                        st.dataframe(pd.DataFrame(ocp_rows), use_container_width=True)
                    else:
                        st.info(f"No OCP entries for {'/'.join(symbols)}.")
        else:
            st.markdown(f"**{_ocp_title}**")
            comp_pick = st.text_input(
                "Probe OCP binding energies for composition (slash-separated, e.g. Cu/Zn)",
                value="Cu/Zn",
                key="ocp_probe_tab",
            )
            if comp_pick.strip():
                symbols = tuple(s.strip() for s in comp_pick.split("/") if s.strip())
                ocp_rows = load_ocp_for_composition(symbols)
                if ocp_rows:
                    st.dataframe(pd.DataFrame(ocp_rows), use_container_width=True)
                else:
                    st.info(f"No OCP entries cached for composition {'/'.join(symbols)}.")


# ------------------------------------------------------------- VALIDATION
with tab_validation:
    st.subheader("Simulation validation")
    if simulation_df.empty:
        st.info("No simulation_validation.csv found for this run. Use the sidebar simulation command.")
    else:
        sim1, sim2, sim3, sim4 = st.columns(4)
        sim1.metric("Validated candidates", f"{len(simulation_df):,}")
        sim2.metric(
            "Cantera-backed rows",
            f"{int(simulation_df['simulation_backend'].astype(str).str.contains('cantera', case=False).sum()):,}"
            if "simulation_backend" in simulation_df.columns
            else "N/A",
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

    st.subheader("Sweep surrogate")
    if not surrogate_metrics:
        st.info("No simulation surrogate metrics found. Generate a sweep and train the surrogate from the sidebar.")
    else:
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Sweep rows", f"{surrogate_metrics.get('n_rows', 0):,}")
        sm2.metric("Train rows", f"{surrogate_metrics.get('n_train', 0):,}")
        sm3.metric("Test R²", f"{surrogate_metrics.get('test_r2', float('nan')):.3f}")
        sm4.metric("Test MAE", f"{surrogate_metrics.get('test_mae', float('nan')):.4f}")
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
            "Demo loop: import example outcomes -> retrain the ActivityHead ranking model. "
            "Full CVAE fine-tuning is triggered only after enough validated lab rows pass drift checks."
        )
        with st.expander("Feedback import and retraining commands", expanded=True):
            st.code(_feedback_import_cmd, language="bash")
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
            chosen = st.selectbox(
                "Candidate from latest shortlist",
                options=candidate_choices or ["(no shortlist available)"],
                key="feedback_candidate",
            )
            with st.form("feedback_form", clear_on_submit=True):
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
                if submit and "(" in chosen:
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
            st.markdown("**Recent experiments**")
            recent = feedback_store.list_experiments(limit=20)
            if not recent:
                st.info("No experiments logged yet.")
            else:
                recent_df = pd.DataFrame(
                    [
                        {
                            "logged_at": r["logged_at"],
                            "composition": r["composition_view"],
                            "predicted_sty": r.get("predicted_sty"),
                            "measured_sty": r["measured_sty"],
                            "predicted_selectivity": r.get("predicted_selectivity"),
                            "measured_selectivity": r["measured_selectivity"],
                            "predicted_yield": r.get("predicted_yield"),
                            "measured_yield": r.get("measured_yield"),
                            "stability_h": r["measured_stability_tos_h"],
                            "user": r["user"],
                            "model_version": r["model_version"],
                        }
                        for r in recent
                    ]
                )
                st.dataframe(recent_df, use_container_width=True)

                discrepancy_df = _feedback_discrepancy_frame(recent)
                if not discrepancy_df.empty:
                    st.markdown("**Prediction vs actual discrepancy analysis**")
                    flagged = discrepancy_df[discrepancy_df["flags"] != "ok"].copy()
                    if flagged.empty:
                        st.success("No major discrepancies under current demo thresholds.")
                    else:
                        st.warning(f"{len(flagged)} feedback row(s) exceed discrepancy thresholds.")
                    st.dataframe(discrepancy_df, use_container_width=True)

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
