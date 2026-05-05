"""Reaction-family configuration loader for simulation validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REACTION_CONFIG = ROOT / "config" / "reactions" / "co2_methanol.yaml"


@dataclass(frozen=True)
class ReactionConfig:
    id: str
    name: str
    family: str
    cantera_mechanism: str
    default_conditions: dict[str, float]
    feed: dict[str, float]
    reference_species: str
    target_species: list[str]
    byproduct_species: list[str]
    required_species: list[str]
    stoichiometry: dict[str, float]
    thermo: dict[str, float]
    catalyst_descriptor: dict[str, Any]
    sweep: dict[str, Any]
    raw: dict[str, Any]
    path: Path


def load_reaction_config(path: Path | str | None = None) -> ReactionConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_REACTION_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Reaction config must be a YAML mapping: {cfg_path}")

    required = [
        "id",
        "name",
        "family",
        "cantera_mechanism",
        "default_conditions",
        "feed",
        "reference_species",
        "target_species",
        "byproduct_species",
        "required_species",
        "stoichiometry",
        "thermo",
        "catalyst_descriptor",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Reaction config {cfg_path} is missing keys: {missing}")

    return ReactionConfig(
        id=str(data["id"]),
        name=str(data["name"]),
        family=str(data["family"]),
        cantera_mechanism=str(data["cantera_mechanism"]),
        default_conditions={k: float(v) for k, v in data["default_conditions"].items()},
        feed={k: float(v) for k, v in data["feed"].items()},
        reference_species=str(data["reference_species"]),
        target_species=[str(x) for x in data["target_species"]],
        byproduct_species=[str(x) for x in data["byproduct_species"]],
        required_species=[str(x) for x in data["required_species"]],
        stoichiometry={k: float(v) for k, v in data["stoichiometry"].items()},
        thermo={k: float(v) for k, v in data["thermo"].items()},
        catalyst_descriptor=dict(data["catalyst_descriptor"]),
        sweep=dict(data.get("sweep", {})),
        raw=dict(data),
        path=cfg_path,
    )


def config_for_reaction_id(reaction_id: str) -> Path:
    return ROOT / "config" / "reactions" / f"{reaction_id}.yaml"
