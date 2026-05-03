"""Embed lab feedback rows through a frozen CVAE to obtain μ for head retraining."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGLoader

from catcvae.dataset import getDataObject

logger = logging.getLogger(__name__)

# Fixed gas-phase skeleton for CO2 hydrogenation to methanol in this codebase
# (matches dataset/co2_methanol.csv).
CO2_METHANOL_REACTANT = "O=C=O"
CO2_METHANOL_REAGENT = "[H][H]"
CO2_METHANOL_PRODUCT = "CO"


def _conditions_from_feedback_row(row: dict[str, Any]) -> tuple[float, float]:
    cond = row.get("conditions") or {}
    if isinstance(cond, str):
        try:
            cond = json.loads(cond)
        except json.JSONDecodeError:
            cond = {}
    t_c = float(cond.get("T_C", 240.0))
    p_bar = float(cond.get("P_bar", 50.0))
    return t_c, p_bar


def feedback_rows_to_graphs(
    args: Any,
    feedback_rows: list[dict[str, Any]],
    reaction: str = "co2_methanol",
) -> list[Any]:
    """Build PyG data objects for measured feedback rows. Skips invalid rows."""
    if reaction != "co2_methanol":
        raise ValueError(f"feedback embedding not implemented for reaction={reaction!r}")

    objects: list[Any] = []
    for i, row in enumerate(feedback_rows):
        sty = row.get("measured_sty")
        if sty is None or (isinstance(sty, float) and np.isnan(sty)):
            continue
        try:
            y_val = float(sty)
        except (TypeError, ValueError):
            continue
        if y_val <= 0.0:
            continue

        pseudo = (row.get("pseudo_smiles") or row.get("candidate_id") or "").strip()
        if not pseudo:
            continue

        t_c, p_bar = _conditions_from_feedback_row(row)
        eid = f"feedback-{row.get('logged_at', i)}-{i}"

        d = {
            "X_reactant": CO2_METHANOL_REACTANT,
            "X_reagent": CO2_METHANOL_REAGENT,
            "X_product": CO2_METHANOL_PRODUCT,
            "X_catalyst": pseudo,
            "X_time": t_c,
            "y": y_val,
            "ids": eid,
            "C_pressure_bar": p_bar,
        }
        dobj = getDataObject(args, d)
        if dobj is None:
            logger.warning("skip feedback row %s: getDataObject failed for %s", eid, pseudo)
            continue
        objects.append(dobj)

    return objects


def embed_feedback_mu(
    args: Any,
    feedback_rows: list[dict[str, Any]],
    AE: torch.nn.Module,
    device: torch.device,
    reaction: str = "co2_methanol",
    emb_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, y) for feedback rows embeddable through ``AE`` (NN not required for μ)."""
    from catcvae.latent import embed

    dim = int(emb_dim if emb_dim is not None else getattr(args, "emb_dim", 256))
    graphs = feedback_rows_to_graphs(args, feedback_rows, reaction=reaction)
    if not graphs:
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    loader = PyGLoader(
        graphs,
        batch_size=min(32, len(graphs)),
        shuffle=False,
        follow_batch=["x_reactant", "x_reagent", "x_product", "x_catalyst"],
    )
    AE.eval()
    _latent, mu, y_true, _yp, _ids, _c = embed(loader, AE, None, device=device)
    return np.asarray(mu, dtype=np.float32), np.asarray(y_true, dtype=np.float32).reshape(-1)
