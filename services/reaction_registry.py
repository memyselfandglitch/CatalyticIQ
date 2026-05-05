"""Reaction profiles for dashboard routing, retrieval keys, and dataset paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    reaction_config_relative: str
    caption: str
    public_demo: bool = False


PROFILES: tuple[ReactionProfile, ...] = (
    ReactionProfile(
        id="co2_methanol",
        label="CO₂ + H₂ → methanol",
        dataset_subdir="co2_methanol",
        retrieval_reaction="co2_to_methanol",
        full_csv_relative="dataset/co2_methanol_full.csv",
        reaction_config_relative="config/reactions/co2_methanol.yaml",
        caption="Generative AI + multi-property prediction + reaction-energy estimation + lab feedback.",
        public_demo=True,
    ),
    ReactionProfile(
        id="syngas_ethanol",
        label="Syngas → ethanol (pilot)",
        dataset_subdir="syngas_ethanol",
        retrieval_reaction="syngas_to_ethanol",
        full_csv_relative="dataset/syngas_ethanol_full.csv",
        reaction_config_relative="config/reactions/syngas_ethanol.yaml",
        caption=(
            "Pilot track: same discovery stack once reaction-specific outputs, heads, "
            "and validation artifacts are promoted for demo use."
        ),
        public_demo=False,
    ),
)

_BY_ID = {p.id: p for p in PROFILES}


def get_profile(reaction_id: str) -> ReactionProfile:
    if reaction_id not in _BY_ID:
        raise KeyError(f"unknown reaction id {reaction_id!r}; known: {list(_BY_ID)}")
    return _BY_ID[reaction_id]


def dataset_dir(profile: ReactionProfile) -> Path:
    return ROOT / "dataset" / profile.dataset_subdir


def has_output_runs(profile: ReactionProfile) -> bool:
    base = dataset_dir(profile)
    return base.exists() and any(p.is_dir() and p.name.startswith("output_") for p in base.iterdir())


def list_profiles_for_ui() -> list[ReactionProfile]:
    """Public dashboard profiles.

    The hackathon dashboard is intentionally CO2-focused. Pilot reactions remain
    registered for code reuse, but are hidden unless explicitly enabled with
    ``CATALYTICIQ_SHOW_PILOTS=1``.
    """
    show_pilots = os.environ.get("CATALYTICIQ_SHOW_PILOTS", "").lower() in {"1", "true", "yes"}
    return [
        profile
        for profile in PROFILES
        if profile.public_demo or (show_pilots and has_output_runs(profile))
    ]
