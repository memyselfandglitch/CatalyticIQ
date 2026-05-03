"""Reaction profiles for dashboard routing, retrieval keys, and dataset paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ReactionProfile:
    """One deployable reaction loop (dataset + retrieval + copy)."""

    id: str
    label: str
    dataset_subdir: str
    retrieval_reaction: str
    full_csv_relative: str
    caption: str


PROFILES: tuple[ReactionProfile, ...] = (
    ReactionProfile(
        id="co2_methanol",
        label="CO₂ + H₂ → methanol",
        dataset_subdir="co2_methanol",
        retrieval_reaction="co2_to_methanol",
        full_csv_relative="dataset/co2_methanol_full.csv",
        caption="Generative AI + multi-property prediction + reaction-energy estimation + lab feedback.",
    ),
    ReactionProfile(
        id="syngas_ethanol",
        label="Syngas → ethanol (pilot)",
        dataset_subdir="syngas_ethanol",
        retrieval_reaction="syngas_to_ethanol",
        full_csv_relative="dataset/syngas_ethanol_full.csv",
        caption="Pilot track: same discovery stack once `dataset/syngas_ethanol.csv` is built from PNNL / Zenodo corpora.",
    ),
)

_BY_ID = {p.id: p for p in PROFILES}


def get_profile(reaction_id: str) -> ReactionProfile:
    if reaction_id not in _BY_ID:
        raise KeyError(f"unknown reaction id {reaction_id!r}; known: {list(_BY_ID)}")
    return _BY_ID[reaction_id]


def dataset_dir(profile: ReactionProfile) -> Path:
    return ROOT / "dataset" / profile.dataset_subdir


def list_profiles_for_ui() -> list[ReactionProfile]:
    """All registered reactions (each tab shows an error until that reaction has output_* runs)."""
    return list(PROFILES)
