#!/usr/bin/env python3
"""Remove all rows from the local feedback experiments table (DuckDB)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.feedback.store import FeedbackStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to feedback.duckdb (default: cache/feedback.duckdb under repo root).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    store = FeedbackStore(args.db)
    if not args.yes:
        try:
            ans = input(f"Delete ALL experiment rows from {store.db_path}? [y/N] ")
        except EOFError:
            ans = "n"
        if ans.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return
    removed = store.clear_experiments()
    print(f"Removed {removed} row(s) from experiments.")


if __name__ == "__main__":
    main()
