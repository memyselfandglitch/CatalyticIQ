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
DATASET_DIR = ROOT / "dataset" / "co2_methanol"
HYPER_RESULT_PATH = DATASET_DIR / "hyper_result.txt"
PROPERTY_DIR = DATASET_DIR / "property_heads"
VALIDATION_DIR = DATASET_DIR / "validation"


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
    if "n_components" in df.columns:
        df["n_components"] = pd.to_numeric(df["n_components"], errors="coerce").fillna(0).astype(int)
    return df


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


# =========================================================================
# Sidebar + state
# =========================================================================

st.set_page_config(page_title="CatalyticIQ Dashboard", layout="wide")
st.title("CatalyticIQ — CO2-to-Methanol Catalyst Discovery Loop")
st.caption("Generative AI + multi-property prediction + reaction-energy estimation + lab feedback.")

runs = discover_output_runs(DATASET_DIR)
if not runs:
    st.error("No output runs found under dataset/co2_methanol.")
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
    selected_stats_name = st.sidebar.selectbox(
        "Generation stats file", [p.name for p in gen_stats_files]
    )
    selected_stats_path = selected_run / selected_stats_name

top_n = st.sidebar.slider("Top-N shortlist size", min_value=5, max_value=100, value=20, step=5)
must_have_metal = st.sidebar.checkbox("Require metal-containing candidates", value=True)
search_query = st.sidebar.text_input("Search candidate text")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Use scripts/postprocess_candidates.py, train_property_heads.py, "
    "validate_encoder.py and retrain_with_feedback.py to refresh the artifacts read here."
)


# =========================================================================
# Shared dataframes
# =========================================================================

raw_candidates = load_candidate_csv(selected_gen_csv_path)
clean_path = discover_clean_csv(selected_run)
clean_df = load_clean_candidates(clean_path) if clean_path is not None else pd.DataFrame()
if not clean_df.empty:
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
        st.caption(
            f"Cleaned shortlist from {clean_path.name}. STY calibrated to training-set rank "
            "percentiles; selectivity and stability shown as priors / proxies until lab data lands."
        )
        display_cols = [
            "composition_view",
            "predicted_sty_g_h_gcat",
            "selectivity_proxy_pct",
            "stability_proxy",
            "pseudo_smiles",
        ]
        if "n_components" in clean_df.columns:
            display_cols.append("n_components")
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

    st.subheader("Element Frequency")
    elements_df = element_frequency(raw_candidates["candidate"])
    if elements_df.empty:
        st.info("No element tokens found.")
    else:
        st.bar_chart(elements_df.set_index("element")["count"].head(15))

    st.subheader("Generation Metrics")
    if stats:
        st.json(stats)
    else:
        st.info("No generation_stats_*.txt found for this run.")

    st.subheader("Training Trends")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("**Report: train/val loss**")
        if train_df.empty:
            st.info("report.txt not found.")
        else:
            st.line_chart(train_df.set_index("epoch")[["t_loss", "v_loss", "opt_loss"]])
    with tcol2:
        st.markdown("**Loss breakdown**")
        if loss_df.empty:
            st.info("loss.txt not found.")
        else:
            st.line_chart(loss_df.set_index("epoch")[["recon_t", "recon_v", "nn_t", "nn_v"]])

    if not hyper_row.empty:
        with st.expander("Hyperparameter snapshot"):
            st.dataframe(hyper_row, use_container_width=True)


# ------------------------------------------------------------- PATHWAY
with tab_pathway:
    st.subheader("Reaction Pathway (free energy)")
    if clean_df.empty:
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
        profile = compute_energy_profile(chosen_smiles, mechanism, backend)
        st.caption(f"Backend used: **{profile['backend']}**. {profile['citation']}")
        st.image(render_energy_diagram(profile), use_container_width=True)
        if profile["notes"]:
            st.info(profile["notes"])
        if profile["extras"]:
            with st.expander("Underlying binding-energy descriptors (eV)"):
                st.json(profile["extras"])


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
        known_rows = load_known_catalysts("co2_to_methanol")
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
            "Bubble size encodes the descriptor-based stability proxy. Selectivity is a "
            "composition-weighted prior calibrated against TheMeCat — replace with measured "
            "values via the Feedback tab as lab data lands."
        )
        st.dataframe(combined.sort_values("predicted_sty_g_h_gcat", ascending=False).head(20), use_container_width=True)


# ------------------------------------------------------------- KNOWLEDGE BASE
with tab_kb:
    st.subheader("Known catalysts (Materials Project + OCP)")
    known = load_known_catalysts("co2_to_methanol")
    if not known:
        st.info("Knowledge base is empty.")
    else:
        st.caption(
            f"{len(known)} curated entries. Live mp-api / fairchem are used when keys / packages "
            "are available; otherwise the offline cache (cache/retrieval.duckdb) is consulted."
        )
        known_df = pd.DataFrame(known)
        if "composition" in known_df.columns:
            known_df["composition"] = known_df["composition"].apply(lambda xs: "/".join(xs))
        st.dataframe(known_df, use_container_width=True)

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

        pareto_df = pd.DataFrame(
            [
                {"source": "random", "mean": pareto.get("random_mean", 0.0), "p95": pareto.get("random_p95", 0.0)},
                {"source": "GA", "mean": pareto.get("ga_mean", 0.0), "p95": pareto.get("ga_p95", 0.0)},
                {"source": "CVAE", "mean": pareto.get("cvae_mean", 0.0), "p95": pareto.get("cvae_p95", 0.0)},
            ]
        )
        st.markdown("**Pareto comparison: predicted STY**")
        st.bar_chart(pareto_df.set_index("source"))

        if report_pdf.exists():
            with open(report_pdf, "rb") as f:
                st.download_button(
                    "Download full encoder validation PDF",
                    data=f.read(),
                    file_name="encoder_report.pdf",
                    mime="application/pdf",
                )
        with st.expander("Full validation JSON"):
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
                    rec = ExperimentRecord(
                        candidate_id=pseudo,
                        pseudo_smiles=pseudo,
                        composition_view=comp_view,
                        measured_sty=float(measured_sty) if measured_sty > 0 else None,
                        measured_selectivity=float(measured_sel) if measured_sel > 0 else None,
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
                            "measured_sty": r["measured_sty"],
                            "measured_selectivity": r["measured_selectivity"],
                            "stability_h": r["measured_stability_tos_h"],
                            "user": r["user"],
                            "model_version": r["model_version"],
                        }
                        for r in recent
                    ]
                )
                st.dataframe(recent_df, use_container_width=True)

                n_pending = feedback_store.count_since_last_train("current")
                st.caption(
                    f"Feedback rows queued for retrain: **{n_pending}**. "
                    "Run `python scripts/retrain_with_feedback.py --mode heads --promote` to update."
                )

        versions = feedback_store.list_model_versions()
        if versions:
            with st.expander("Model version history"):
                st.dataframe(pd.DataFrame(versions), use_container_width=True)
