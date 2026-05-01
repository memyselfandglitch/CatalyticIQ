"""
Property prediction heads operating on the frozen CVAE latent embedding.

Three small MLPs sit on top of the encoder produced by `catcvae.latent.embed`:

  * ActivityHead     -> calibrated methanol space-time yield (gMeOH h-1 gcat-1).
  * SelectivityHead  -> joint MeOH and CO selectivity (% of CO2 carbon converted).
  * StabilityHead    -> stability proxy in [0, 1].

The heads can be trained independently of the CVAE so that:

  1. We can iterate on property prediction without re-running fine-tuning.
  2. The feedback retrain loop can refresh only the heads when the new lab
     data is too small to safely shift the latent space.

`StabilityHead` is initialised as a frozen wrapper around the descriptor table
in `catcvae.stability_descriptors`. We expose it as an `nn.Module` purely for
API parity with the other heads; if a future version wants to fit residuals
on TOS data, this is the integration point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .stability_descriptors import composition_stability_score


@dataclass
class HeadConfig:
    in_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ActivityHead(nn.Module):
    """Regression head for methanol STY (g MeOH / h / g cat)."""

    target_name: str = "methanol_sty"

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.net = _mlp(config.in_dim, 1, config.hidden_dim, config.dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class SelectivityHead(nn.Module):
    """Joint head for (MeOH selectivity %, CO selectivity %).

    Uses a 2-logit softplus so outputs are non-negative; downstream code can
    optionally normalise them to sum to 100. We do not enforce that here
    because real catalyst data routinely shows non-trivial fractions of
    CH4 / DME / higher alcohols, so a softmax would force a false closure.
    """

    target_columns: tuple[str, str] = ("selectivity_meoh_pct", "selectivity_co_pct")

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.net = _mlp(config.in_dim, 2, config.hidden_dim, config.dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        return torch.nn.functional.softplus(raw)


class StabilityHead(nn.Module):
    """Descriptor-based stability proxy.

    This module does not have learnable parameters by default. It evaluates
    `composition_stability_score` on the candidate's element list at the
    requested operating temperature. We keep the same `forward` signature as
    the data-driven heads so that the orchestration code can treat all three
    uniformly.

    `forward` accepts a tuple `(components, temperature_c)` per row; we keep
    the latent input around so that a future learnable variant (e.g. a
    correction MLP fit on TOS data) can be slotted in without touching the
    callers.
    """

    target_name: str = "stability_proxy"

    def __init__(self, config: HeadConfig | None = None):
        super().__init__()
        # Optional learnable residual on top of the descriptor score; off by default.
        self.residual: nn.Module | None = None
        if config is not None:
            self.residual = _mlp(config.in_dim, 1, config.hidden_dim, config.dropout)

    def descriptor_score(
        self,
        components_batch: Sequence[Iterable[str]],
        temperatures_c: Sequence[float],
    ) -> torch.Tensor:
        scores = [
            composition_stability_score(comps, t)
            for comps, t in zip(components_batch, temperatures_c)
        ]
        return torch.tensor(scores, dtype=torch.float32)

    def forward(
        self,
        z: torch.Tensor,
        components_batch: Sequence[Iterable[str]],
        temperatures_c: Sequence[float],
    ) -> torch.Tensor:
        base = self.descriptor_score(components_batch, temperatures_c).to(z.device)
        if self.residual is None:
            return base
        delta = torch.tanh(self.residual(z).squeeze(-1)) * 0.25
        return torch.clamp(base + delta, 0.0, 1.0)


__all__ = [
    "HeadConfig",
    "ActivityHead",
    "SelectivityHead",
    "StabilityHead",
]
