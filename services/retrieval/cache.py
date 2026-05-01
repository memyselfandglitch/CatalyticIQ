"""DuckDB-backed cache for retrieval adapters.

Schema:

  mp_entries(identifier TEXT PRIMARY KEY,
             name TEXT,
             composition TEXT,
             reaction TEXT,
             properties_json TEXT,
             citation TEXT,
             retrieved_at TIMESTAMP)

  ocp_entries(identifier TEXT PRIMARY KEY,
              composition TEXT,
              adsorbate TEXT,
              binding_energy_ev DOUBLE,
              surface_termination TEXT,
              citation TEXT,
              retrieved_at TIMESTAMP)

  provenance(source TEXT,
             event TEXT,
             ts TIMESTAMP,
             details TEXT)

The cache is purely additive; live API calls upsert by `identifier`. Read
queries automatically fall back to this cache when the live API is offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import duckdb

from .types import KnownEntry

_DEFAULT_CACHE_DIR = Path("cache")
_DEFAULT_CACHE_PATH = _DEFAULT_CACHE_DIR / "retrieval.duckdb"


class RetrievalCache:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_CACHE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS mp_entries(
                    identifier TEXT PRIMARY KEY,
                    name TEXT,
                    composition TEXT,
                    reaction TEXT,
                    properties_json TEXT,
                    citation TEXT,
                    retrieved_at TIMESTAMP
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ocp_entries(
                    identifier TEXT PRIMARY KEY,
                    composition TEXT,
                    adsorbate TEXT,
                    binding_energy_ev DOUBLE,
                    surface_termination TEXT,
                    citation TEXT,
                    retrieved_at TIMESTAMP
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance(
                    source TEXT,
                    event TEXT,
                    ts TIMESTAMP,
                    details TEXT
                )
                """
            )

    # ------------------------------------------------------------------ MP
    def upsert_mp_entries(self, entries: Iterable[KnownEntry]) -> int:
        rows = []
        ts = datetime.utcnow()
        for e in entries:
            rows.append(
                (
                    e.identifier,
                    e.name,
                    "|".join(e.composition),
                    e.reaction,
                    json.dumps(e.properties),
                    e.citation,
                    ts,
                )
            )
        if not rows:
            return 0
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO mp_entries(
                    identifier, name, composition, reaction,
                    properties_json, citation, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identifier) DO UPDATE SET
                    name = EXCLUDED.name,
                    composition = EXCLUDED.composition,
                    reaction = EXCLUDED.reaction,
                    properties_json = EXCLUDED.properties_json,
                    citation = EXCLUDED.citation,
                    retrieved_at = EXCLUDED.retrieved_at
                """,
                rows,
            )
            con.execute(
                "INSERT INTO provenance(source, event, ts, details) VALUES (?, ?, ?, ?)",
                ("MP", "upsert", ts, f"rows={len(rows)}"),
            )
        return len(rows)

    def fetch_mp_by_reaction(self, reaction: str) -> list[KnownEntry]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT identifier, name, composition, reaction, properties_json, citation, retrieved_at "
                "FROM mp_entries WHERE reaction = ?",
                [reaction],
            ).fetchall()
        return [_row_to_mp_entry(r) for r in rows]

    def fetch_mp_by_composition(self, symbols: Sequence[str]) -> list[KnownEntry]:
        normalized = "|".join(sorted(set(symbols)))
        with self._connect() as con:
            rows = con.execute(
                "SELECT identifier, name, composition, reaction, properties_json, citation, retrieved_at "
                "FROM mp_entries"
            ).fetchall()
        out: list[KnownEntry] = []
        target = set(symbols)
        for r in rows:
            entry = _row_to_mp_entry(r)
            if target.issubset(set(entry.composition)):
                out.append(entry)
        return out

    # ----------------------------------------------------------------- OCP
    def upsert_ocp_entries(self, entries: Iterable[dict]) -> int:
        rows = []
        ts = datetime.utcnow()
        for e in entries:
            rows.append(
                (
                    e["identifier"],
                    "|".join(e["composition"]),
                    e["adsorbate"],
                    float(e["binding_energy_ev"]),
                    e.get("surface_termination", ""),
                    e.get("citation", ""),
                    ts,
                )
            )
        if not rows:
            return 0
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO ocp_entries(
                    identifier, composition, adsorbate, binding_energy_ev,
                    surface_termination, citation, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identifier) DO UPDATE SET
                    composition = EXCLUDED.composition,
                    adsorbate = EXCLUDED.adsorbate,
                    binding_energy_ev = EXCLUDED.binding_energy_ev,
                    surface_termination = EXCLUDED.surface_termination,
                    citation = EXCLUDED.citation,
                    retrieved_at = EXCLUDED.retrieved_at
                """,
                rows,
            )
            con.execute(
                "INSERT INTO provenance(source, event, ts, details) VALUES (?, ?, ?, ?)",
                ("OCP", "upsert", ts, f"rows={len(rows)}"),
            )
        return len(rows)

    def fetch_ocp_by_composition(self, symbols: Sequence[str]) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT identifier, composition, adsorbate, binding_energy_ev, "
                "       surface_termination, citation, retrieved_at "
                "FROM ocp_entries"
            ).fetchall()
        out: list[dict] = []
        target = set(symbols)
        for r in rows:
            comp = r[1].split("|") if r[1] else []
            if target.issubset(set(comp)):
                out.append(
                    {
                        "identifier": r[0],
                        "composition": comp,
                        "adsorbate": r[2],
                        "binding_energy_ev": r[3],
                        "surface_termination": r[4],
                        "citation": r[5],
                        "retrieved_at": r[6],
                    }
                )
        return out

    def list_provenance(self, limit: int = 20) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT source, event, ts, details FROM provenance ORDER BY ts DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [
            {"source": r[0], "event": r[1], "ts": r[2], "details": r[3]} for r in rows
        ]


def _row_to_mp_entry(row) -> KnownEntry:
    properties = json.loads(row[4]) if row[4] else {}
    composition = row[2].split("|") if row[2] else []
    return KnownEntry(
        source="MP",
        identifier=row[0],
        name=row[1],
        composition=composition,
        reaction=row[3],
        properties=properties,
        citation=row[5],
        provenance={"retrieved_at": row[6]},
    )
