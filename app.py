from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "co2_methanol"
HYPER_RESULT_PATH = DATASET_DIR / "hyper_result.txt"


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


@st.cache_data(show_spinner=False)
def load_candidate_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["candidate", "score"])
    df["candidate"] = df["candidate"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
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
        # examples:
        # Validity,1000,,100.0000
        # IntDiv,,,0.44,0.29
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
            key = key.strip()
            val = val.strip()
            try:
                rec[key] = float(val)
            except ValueError:
                rec[key] = val
        rows.append(rec)
    return pd.DataFrame(rows)


def element_frequency(df: pd.DataFrame) -> pd.DataFrame:
    pat = re.compile(r"\[([A-Za-z0-9]+)\]")
    c: Counter[str] = Counter()
    for s in df["candidate"]:
        for element in pat.findall(s):
            c[element] += 1
    if not c:
        return pd.DataFrame(columns=["element", "count"])
    out = pd.DataFrame(c.items(), columns=["element", "count"]).sort_values("count", ascending=False)
    return out.reset_index(drop=True)


def build_shortlist(df: pd.DataFrame, must_have_metal: bool, top_n: int) -> pd.DataFrame:
    ranked = df.sort_values("score", ascending=False).drop_duplicates(subset=["candidate"]).copy()
    ranked["components"] = ranked["candidate"].str.count(r"\.") + 1
    if must_have_metal:
        # Common catalyst elements for quick filtering.
        metal_pat = re.compile(r"\[(Cu|Zn|Pd|Pt|Ni|Rh|Fe|Co|Mn|Zr|Ag|Au|Ti|Cr|Mo|W)\]")
        ranked = ranked[ranked["candidate"].str.contains(metal_pat, regex=True)]
    return ranked.head(top_n).reset_index(drop=True)


st.set_page_config(page_title="CatalyticIQ Dashboard", layout="wide")
st.title("CatalyticIQ - Candidate Generation Dashboard")
st.caption("Live view from existing training and generation artifacts.")

runs = discover_output_runs(DATASET_DIR)
if not runs:
    st.error("No output runs found under dataset/co2_methanol.")
    st.stop()

run_map = {p.name: p for p in runs}
selected_run_name = st.sidebar.selectbox("Select run folder", list(run_map.keys()))
selected_run = run_map[selected_run_name]

gen_csv_files = discover_generated_csv(selected_run)
gen_stats_files = discover_generated_stats(selected_run)

if not gen_csv_files:
    st.error(f"No generated CSV found in {selected_run}.")
    st.stop()

selected_gen_csv = st.sidebar.selectbox("Generated candidates file", [p.name for p in gen_csv_files])
selected_gen_csv_path = selected_run / selected_gen_csv

selected_stats = None
if gen_stats_files:
    selected_stats_name = st.sidebar.selectbox("Generation stats file", [p.name for p in gen_stats_files])
    selected_stats = selected_run / selected_stats_name

top_n = st.sidebar.slider("Top-N shortlist size", min_value=5, max_value=100, value=20, step=5)
must_have_metal = st.sidebar.checkbox("Require metal-containing candidates", value=True)
search_query = st.sidebar.text_input("Search candidate text")

candidates = load_candidate_csv(selected_gen_csv_path)
if search_query.strip():
    candidates = candidates[candidates["candidate"].str.contains(search_query.strip(), case=False, regex=False)]

shortlist = build_shortlist(candidates, must_have_metal=must_have_metal, top_n=top_n)
elements_df = element_frequency(candidates)

stats = parse_generated_stats(selected_stats) if selected_stats else {}
train_df = parse_training_metrics(selected_run / "report.txt") if (selected_run / "report.txt").exists() else pd.DataFrame()
loss_df = parse_loss_metrics(selected_run / "loss.txt") if (selected_run / "loss.txt").exists() else pd.DataFrame()
hyper_df = parse_hyper_result(HYPER_RESULT_PATH)
hyper_row = hyper_df[hyper_df["run"] == selected_run_name].tail(1) if not hyper_df.empty else pd.DataFrame()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Generated", f"{len(candidates):,}")
col2.metric("Unique", f"{candidates['candidate'].nunique():,}")
col3.metric("Validity", f"{stats.get('Validity', 'N/A')}")
col4.metric("Novelty", f"{stats.get('Novelty', 'N/A')}")
if not hyper_row.empty and "R2_rfr" in hyper_row.columns:
    col5.metric("R2 (RFR)", f"{float(hyper_row.iloc[0]['R2_rfr']):.3f}")
else:
    col5.metric("R2 (RFR)", "N/A")

st.subheader("Top Ranked Candidates")
st.dataframe(shortlist, use_container_width=True)

csv_bytes = shortlist.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download shortlist CSV",
    data=csv_bytes,
    file_name=f"shortlist_{selected_run_name}_{top_n}.csv",
    mime="text/csv",
)

left, right = st.columns(2)
with left:
    st.subheader("Element Frequency")
    if elements_df.empty:
        st.info("No element tokens found.")
    else:
        st.bar_chart(elements_df.set_index("element")["count"].head(15))

with right:
    st.subheader("Generation Metrics")
    if not stats:
        st.info("No generation stats file found.")
    else:
        st.json(stats)

st.subheader("Training Trends")
tcol1, tcol2 = st.columns(2)

with tcol1:
    st.markdown("**Report: train/val loss**")
    if train_df.empty:
        st.info("report.txt not found or not parseable.")
    else:
        st.line_chart(train_df.set_index("epoch")[["t_loss", "v_loss", "opt_loss"]])

with tcol2:
    st.markdown("**Loss breakdown**")
    if loss_df.empty:
        st.info("loss.txt not found or not parseable.")
    else:
        st.line_chart(loss_df.set_index("epoch")[["recon_t", "recon_v", "nn_t", "nn_v"]])

st.subheader("Hyperparameter Summary")
if hyper_row.empty:
    st.info("No matching row found in dataset/co2_methanol/hyper_result.txt for this run.")
else:
    st.dataframe(hyper_row, use_container_width=True)
