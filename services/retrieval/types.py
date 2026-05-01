"""Shared dataclasses for the retrieval adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnownEntry:
    """A single record returned by a retrieval adapter."""

    source: str                                # "MP" or "OCP".
    identifier: str                            # MP id or OCP id.
    name: str                                  # Human-readable label.
    composition: list[str]                     # Element symbols.
    reaction: str                              # Canonical reaction key (e.g. "co2_to_methanol").
    properties: dict[str, Any] = field(default_factory=dict)
    citation: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
