#!/usr/bin/env python3
"""
Post-process raw CatalyticIQ generation output into a chemically credible shortlist.

Transforms:
- Deduplicate atom tokens within a SMILES (e.g. "[Zr].[Zr]" -> "[Zr]").
- Drop implausible / non-catalyst tokens (e.g. lone N, lone C, [KH], [CaH2])
  unless explicitly flagged as a known promoter.
- Map common support metals to canonical oxide labels for human-facing display
  (the original pseudo-SMILES is preserved for downstream model compatibility).
- Translate the raw NN-head score into an empirical STY estimate using rank
  percentiles against the training distribution. This avoids displaying
  un-interpretable negative numbers while staying honest about the calibration.

Optional ``--use-activity-head``: embed each candidate through the frozen CVAE,
run ``ActivityHead`` on μ (same ranking signal as ``validate_encoder``), sort
by that value, and write ``predicted_sty_g_h_gcat`` as the head output in
physical units (NN score remains in ``raw_score``).

The cleaned candidates are written to `generated_candidates_clean.csv` next to
the source CSV. The input file is never modified.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Active metals seen in CO2->methanol catalysis literature. Anything outside this
# set + the support / promoter set will be dropped from the human-visible
# composition view.
ACTIVE_METALS: set[str] = {
    "Cu", "Pd", "Pt", "Rh", "Ru", "Ni", "Co", "Fe", "Ag", "Au", "Ir", "Re",
    "In", "Mn", "Mo",
}

# Supports that should be displayed as their oxide form.
SUPPORT_TO_OXIDE: dict[str, str] = {
    "Zn": "ZnO",
    "Al": "Al2O3",
    "Zr": "ZrO2",
    "Ti": "TiO2",
    "Ce": "CeO2",
    "Si": "SiO2",
    "Mg": "MgO",
    "Ga": "Ga2O3",
    "La": "La2O3",
    "Y":  "Y2O3",
    "Hf": "HfO2",
}

# Known promoters in CO2 hydrogenation literature. Kept verbatim in the display.
KNOWN_PROMOTERS: set[str] = {"K", "Cs", "Na", "Ba", "Ca", "Li", "Rb"}

# Tokens we never want to surface to a chemist — these are tokenisation
# artefacts (atomic hydrogen as standalone "ligand", non-metal carbons and
# nitrogens, hydride placeholders) rather than real catalyst components.
BLOCKED_TOKENS: set[str] = {"H", "N", "C", "O", "KH", "CaH2", "NH", "OH"}


TOKEN_PATTERN = re.compile(r"\[([A-Z][a-z]?\d*)\]|([A-Z][a-z]?)")


def parse_tokens(smiles: str) -> list[str]:
    """Extract bracketed atom symbols from a SMILES-style string.

    We deliberately ignore bare-letter atoms (`C`, `O`, `N`, `I`, etc.) because
    the VAE occasionally emits organic-molecule fragments alongside the
    catalyst components. Bracketed tokens are the explicit elemental
    components a heterogeneous-catalysis chemist would recognise.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return []
    return re.findall(r"\[([A-Za-z0-9]+)\]", smiles)


def clean_token(tok: str) -> str | None:
    """Return canonical element symbol if usable, else None."""
    # Many tokens come in like "Zn", "ZnO", "KH", "CaH2". We only keep the
    # leading element symbol.
    m = re.match(r"([A-Z][a-z]?)", tok)
    if not m:
        return None
    sym = m.group(1)
    if sym in BLOCKED_TOKENS:
        return None
    return sym


def deduplicate_components(smiles: str) -> tuple[str, list[str]]:
    """Return (clean_pseudo_smiles, ordered_unique_components)."""
    seen: set[str] = set()
    unique: list[str] = []
    for raw in parse_tokens(smiles):
        sym = clean_token(raw)
        if sym is None:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        unique.append(sym)
    pseudo = ".".join(f"[{c}]" for c in unique)
    return pseudo, unique


def composition_view(components: Iterable[str]) -> str:
    """Human-readable composition string for the dashboard."""
    parts: list[str] = []
    for c in components:
        if c in SUPPORT_TO_OXIDE:
            parts.append(SUPPORT_TO_OXIDE[c])
        elif c in ACTIVE_METALS or c in KNOWN_PROMOTERS:
            parts.append(c)
        else:
            parts.append(c)
    return "/".join(parts) if parts else ""


def has_active_metal(components: Iterable[str]) -> bool:
    return any(c in ACTIVE_METALS for c in components)


def calibrate_scores(scores: np.ndarray, training_sty: np.ndarray) -> np.ndarray:
    """Map raw NN-head scores to an empirical STY estimate via rank percentiles.

    The raw scores produced by the latent NN head are not in physical units.
    We translate by:
      1. Ranking the candidate scores into [0, 1] percentiles.
      2. Mapping each percentile to the corresponding quantile of the training
         STY distribution (gMeOH h-1 gcat-1).

    This keeps the rank order intact while making the displayed value
    physically interpretable.
    """
    if len(scores) == 0:
        return scores
    if len(training_sty) == 0:
        return np.full_like(scores, np.nan, dtype=float)

    valid_train = training_sty[np.isfinite(training_sty)]
    if valid_train.size == 0:
        return np.full_like(scores, np.nan, dtype=float)

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pct = ranks / max(len(scores) - 1, 1)
    quantiles = np.quantile(valid_train, pct)
    return quantiles


def load_training_sty(training_csv: Path | None) -> np.ndarray:
    if training_csv is None or not training_csv.exists():
        return np.array([])
    df = pd.read_csv(training_csv)
    col = None
    for cand in (
        "methanol_sty",
        "methanol_sty_scaled",
        "ethanol_sty",
        "STY",
        "sty",
    ):
        if cand in df.columns:
            col = cand
            break
    if col is None:
        return np.array([])
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()


def _reaction_smiles(dataset_file: str) -> tuple[str, str, str]:
    if dataset_file == "co2_methanol":
        from catcvae.feedback_embed import (
            CO2_METHANOL_PRODUCT,
            CO2_METHANOL_REACTANT,
            CO2_METHANOL_REAGENT,
        )

        return CO2_METHANOL_REACTANT, CO2_METHANOL_REAGENT, CO2_METHANOL_PRODUCT
    if dataset_file == "syngas_ethanol":
        return "[C-]#[O+]", "[H][H]", "CCO"
    raise ValueError(f"unknown --dataset-file {dataset_file!r}")


def _median_temperature_pressure(training_csv: Path | None) -> tuple[float, float]:
    t_c, p_bar = 240.0, 50.0
    if training_csv is None or not training_csv.exists():
        return t_c, p_bar
    df = pd.read_csv(training_csv)
    if "temperature_c" in df.columns:
        t_c = float(pd.to_numeric(df["temperature_c"], errors="coerce").median())
    if "pressure_bar" in df.columns:
        p_bar = float(pd.to_numeric(df["pressure_bar"], errors="coerce").median())
    return t_c, p_bar


def _parse_cvae_run_dir(run_dir: Path) -> tuple[int, str]:
    m = re.match(r"output_(\d+)_(.+)$", run_dir.name)
    if not m:
        raise ValueError(f"expected folder name output_<seed>_<time>, got {run_dir.name!r}")
    return int(m.group(1)), m.group(2)


def _activity_head_scores(
    pseudo_smiles: list[str],
    training_csv: Path | None,
    cvae_run_dir: Path,
    dataset_file: str,
    activity_head_path: Path,
) -> np.ndarray:
    """Embed each pseudo-SMILES through the frozen CVAE and run ``ActivityHead`` (μ only)."""
    import torch

    try:
        from catcvae.ae import CVAE
        from catcvae.dataset import getDataObject
        from catcvae.latent import embed
        from catcvae.property_heads import ActivityHead, HeadConfig
        from catcvae.setup import ModelArgumentParser
        from torch_geometric.loader import DataLoader as PyGLoader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Activity-head scoring requires the same stack as training (torch-geometric, rdkit, …). "
            "Use your project conda env, e.g.\n"
            "  conda run -n catdrx python scripts/postprocess_candidates.py ... --use-activity-head ...\n"
            f"Original import error: {exc}"
        ) from exc

    seed, pretrained_time = _parse_cvae_run_dir(cvae_run_dir.resolve())
    parser = ModelArgumentParser()
    margs = parser.setArgument(
        arguments=[
            "--file",
            dataset_file,
            "--pretrained_file",
            dataset_file,
            "--pretrained_time",
            pretrained_time,
            "--seed",
            str(seed),
            "--epochs",
            "0",
            "--class_weight",
            "disabled",
        ]
    )
    xr, xg, xp = _reaction_smiles(dataset_file)
    t_c, p_bar = _median_temperature_pressure(training_csv)
    cond_key = list(margs.condition_dict.keys())[0]

    out = np.full(len(pseudo_smiles), np.nan, dtype=np.float64)
    graphs: list = []
    idx_map: list[int] = []
    for i, smi in enumerate(pseudo_smiles):
        if not isinstance(smi, str) or not smi.strip():
            continue
        d = {
            "X_reactant": xr,
            "X_reagent": xg,
            "X_product": xp,
            "X_catalyst": smi,
            "X_time": t_c,
            "y": 0.0,
            "ids": f"postprocess-{i}",
            "C_" + cond_key: p_bar,
        }
        dobj = getDataObject(margs, d)
        if dobj is None:
            continue
        graphs.append(dobj)
        idx_map.append(i)
    if not graphs:
        return out

    dev = margs.device
    AE = CVAE(
        embedding_setting=margs.embedding_setting,
        encoding_setting=margs.encoding_setting,
        decoding_setting=margs.decoding_setting,
        emb_dim=margs.emb_dim,
        emb_cond_dim=margs.emb_cond_dim,
        cond_dim=margs.cond_dim,
        device=dev,
    ).to(dev)
    ckpt = cvae_run_dir.resolve() / "model_ae.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing CVAE weights: {ckpt}")
    AE.load_state_dict(torch.load(ckpt, map_location=dev))
    AE.eval()
    loader = PyGLoader(
        graphs,
        batch_size=min(64, len(graphs)),
        shuffle=False,
        follow_batch=["x_reactant", "x_reagent", "x_product", "x_catalyst"],
    )
    _z, mu, *_rest = embed(loader, AE, None, device=dev)
    mu_np = np.asarray(mu, dtype=np.float32)

    if not activity_head_path.is_file():
        raise FileNotFoundError(f"missing ActivityHead: {activity_head_path}")
    cfg = HeadConfig(in_dim=mu_np.shape[1])
    head = ActivityHead(cfg)
    head.load_state_dict(torch.load(activity_head_path, map_location="cpu"))
    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(mu_np, dtype=torch.float32)).numpy().reshape(-1)

    for row_i, p in zip(idx_map, pred):
        out[row_i] = float(p)
    return out


def postprocess(
    candidates_csv: Path,
    training_csv: Path | None,
    output_csv: Path,
    require_active_metal: bool = True,
    *,
    use_activity_head: bool = False,
    cvae_run_dir: Path | None = None,
    dataset_file: str = "co2_methanol",
    activity_head_path: Path | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(candidates_csv, header=None, names=["candidate", "score"])
    raw["candidate"] = raw["candidate"].astype(str)
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
    raw = raw.dropna(subset=["score"]).reset_index(drop=True)

    pseudo: list[str] = []
    comps: list[list[str]] = []
    for s in raw["candidate"]:
        clean, components = deduplicate_components(s)
        pseudo.append(clean)
        comps.append(components)

    df = pd.DataFrame({
        "pseudo_smiles": pseudo,
        "components": comps,
        "n_components": [len(c) for c in comps],
        "raw_score": raw["score"].to_numpy(),
        "composition_view": [composition_view(c) for c in comps],
        "has_active_metal": [has_active_metal(c) for c in comps],
    })

    df = df[df["pseudo_smiles"].str.len() > 0].copy()
    if require_active_metal:
        df = df[df["has_active_metal"]].copy()

    if use_activity_head:
        if cvae_run_dir is None or activity_head_path is None:
            raise ValueError("use_activity_head requires cvae_run_dir and activity_head_path")
        df["activity_head_sty"] = _activity_head_scores(
            df["pseudo_smiles"].tolist(),
            training_csv,
            cvae_run_dir,
            dataset_file,
            activity_head_path,
        )
        df = (
            df.sort_values("activity_head_sty", ascending=False, na_position="last")
            .drop_duplicates(subset=["pseudo_smiles"])
            .reset_index(drop=True)
        )
        df["predicted_sty_g_h_gcat"] = df["activity_head_sty"].to_numpy(dtype=float)
    else:
        df = (
            df.sort_values("raw_score", ascending=False)
            .drop_duplicates(subset=["pseudo_smiles"])
            .reset_index(drop=True)
        )
        training_sty = load_training_sty(training_csv)
        df["predicted_sty_g_h_gcat"] = calibrate_scores(
            df["raw_score"].to_numpy(),
            training_sty,
        )

    cols = [
        "pseudo_smiles",
        "composition_view",
        "predicted_sty_g_h_gcat",
        "raw_score",
        "n_components",
        "has_active_metal",
    ]
    if use_activity_head and "activity_head_sty" in df.columns:
        cols.insert(4, "activity_head_sty")
    df = df[cols + ["components"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["components"] = df_out["components"].apply(lambda xs: "|".join(xs))
    df_out.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CatalyticIQ generation output.")
    parser.add_argument("--candidates", type=Path, required=True, help="Raw generated_mol_*.csv path.")
    parser.add_argument(
        "--training",
        type=Path,
        default=Path("dataset/co2_methanol.csv"),
        help="Training CSV used to calibrate scores into STY units.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV. Defaults to generated_candidates_clean.csv next to --candidates.",
    )
    parser.add_argument(
        "--allow-non-metal",
        action="store_true",
        help="Keep candidates without an active-metal component (off by default).",
    )
    parser.add_argument(
        "--use-activity-head",
        action="store_true",
        help="Re-rank with frozen-CVAE μ + ActivityHead (same signal as validate_encoder); "
        "STY column is the head output in physical units.",
    )
    parser.add_argument(
        "--cvae-run-dir",
        type=Path,
        default=None,
        help="e.g. dataset/co2_methanol/output_0_20260428_212044 (must contain model_ae.pth).",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="co2_methanol",
        help="Dataset key for ModelArgumentParser (co2_methanol or syngas_ethanol).",
    )
    parser.add_argument(
        "--activity-head-path",
        type=Path,
        default=None,
        help="Default: dataset/<dataset-file>/property_heads/head_activity.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_activity_head and args.cvae_run_dir is None:
        raise SystemExit("--cvae-run-dir is required when using --use-activity-head")
    output = args.output or (args.candidates.parent / "generated_candidates_clean.csv")
    head_path = args.activity_head_path
    if args.use_activity_head and head_path is None:
        head_path = ROOT / "dataset" / args.dataset_file / "property_heads" / "head_activity.pth"
    df = postprocess(
        candidates_csv=args.candidates,
        training_csv=args.training,
        output_csv=output,
        require_active_metal=not args.allow_non_metal,
        use_activity_head=args.use_activity_head,
        cvae_run_dir=args.cvae_run_dir,
        dataset_file=args.dataset_file,
        activity_head_path=head_path if args.use_activity_head else None,
    )
    print(f"Wrote {len(df)} cleaned candidates -> {output}")
    if not df.empty:
        show = ["composition_view", "predicted_sty_g_h_gcat", "raw_score"]
        if args.use_activity_head and "activity_head_sty" in df.columns:
            show.insert(2, "activity_head_sty")
        head = df.head(5)[show]
        print(head.to_string(index=False))


if __name__ == "__main__":
    main()
