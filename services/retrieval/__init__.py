"""Adapters for external scientific databases.

Each adapter exposes:
  * `fetch(reaction: str) -> list[KnownEntry]`  - lookup by reaction key.
  * `fetch_by_composition(symbols: Sequence[str]) -> list[KnownEntry]` - lookup
    by elemental composition match.

Both methods are guaranteed to return *something* even when the live API is
unavailable, by reading from a curated offline cache committed to the repo.
"""

from .types import KnownEntry  # noqa: F401
from .cache import RetrievalCache  # noqa: F401
