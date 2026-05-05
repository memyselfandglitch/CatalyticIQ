"""DuckDB-backed feedback store for CatalyticIQ.

Two tables:

  experiments(
    candidate_id          TEXT,
    pseudo_smiles         TEXT,
    composition_view      TEXT,
    measured_sty          DOUBLE,
    measured_selectivity  DOUBLE,
    measured_stability_tos_h DOUBLE,
    conditions_json       TEXT,
    user                  TEXT,
    notes                 TEXT,
    logged_at             TIMESTAMP,
    model_version         TEXT
  )

  model_versions(
    version       TEXT PRIMARY KEY,
    parent        TEXT,
    delta_r2      DOUBLE,
    n_feedback_used INTEGER,
    psi           DOUBLE,
    created_at    TIMESTAMP,
    notes         TEXT
  )

The store is intentionally append-only. Every call to ``log_experiment``
inserts a new row even if a previous row exists for the same candidate id;
this preserves the experimental audit trail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb

_DEFAULT_DB = Path("cache") / "feedback.duckdb"


@dataclass
class ExperimentRecord:
    candidate_id: str
    pseudo_smiles: str
    composition_view: str
    measured_sty: float | None = None
    predicted_sty: float | None = None
    measured_selectivity: float | None = None
    predicted_selectivity: float | None = None
    measured_yield: float | None = None
    predicted_yield: float | None = None
    measured_stability_tos_h: float | None = None
    predicted_stability_tos_h: float | None = None
    measured_enzyme_activity: float | None = None
    predicted_enzyme_activity: float | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    user: str = "anonymous"
    notes: str = ""
    model_version: str = "current"

    def to_row(self) -> tuple:
        return (
            self.candidate_id,
            self.pseudo_smiles,
            self.composition_view,
            self.measured_sty,
            self.predicted_sty,
            self.measured_selectivity,
            self.predicted_selectivity,
            self.measured_yield,
            self.predicted_yield,
            self.measured_stability_tos_h,
            self.predicted_stability_tos_h,
            self.measured_enzyme_activity,
            self.predicted_enzyme_activity,
            json.dumps(self.conditions),
            self.user,
            self.notes,
            datetime.utcnow(),
            self.model_version,
        )


@dataclass
class ModelVersion:
    version: str
    parent: str | None = None
    delta_r2: float | None = None
    n_feedback_used: int = 0
    psi: float | None = None
    notes: str = ""

    def to_row(self) -> tuple:
        return (
            self.version,
            self.parent,
            self.delta_r2,
            self.n_feedback_used,
            self.psi,
            datetime.utcnow(),
            self.notes,
        )


class FeedbackStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments(
                    candidate_id          TEXT,
                    pseudo_smiles         TEXT,
                    composition_view      TEXT,
                    measured_sty          DOUBLE,
                    predicted_sty         DOUBLE,
                    measured_selectivity  DOUBLE,
                    predicted_selectivity DOUBLE,
                    measured_yield        DOUBLE,
                    predicted_yield       DOUBLE,
                    measured_stability_tos_h DOUBLE,
                    predicted_stability_tos_h DOUBLE,
                    measured_enzyme_activity DOUBLE,
                    predicted_enzyme_activity DOUBLE,
                    conditions_json       TEXT,
                    user                  TEXT,
                    notes                 TEXT,
                    logged_at             TIMESTAMP,
                    model_version         TEXT
                )
                """
            )
            self._ensure_experiment_columns(con)
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions(
                    version          TEXT PRIMARY KEY,
                    parent           TEXT,
                    delta_r2         DOUBLE,
                    n_feedback_used  INTEGER,
                    psi              DOUBLE,
                    created_at       TIMESTAMP,
                    notes            TEXT
                )
                """
            )

    def _ensure_experiment_columns(self, con: duckdb.DuckDBPyConnection) -> None:
        """Migrate older feedback.duckdb files without losing append-only history."""
        existing = {row[1] for row in con.execute("PRAGMA table_info('experiments')").fetchall()}
        for name, dtype in (
            ("predicted_sty", "DOUBLE"),
            ("predicted_selectivity", "DOUBLE"),
            ("measured_yield", "DOUBLE"),
            ("predicted_yield", "DOUBLE"),
            ("predicted_stability_tos_h", "DOUBLE"),
            ("measured_enzyme_activity", "DOUBLE"),
            ("predicted_enzyme_activity", "DOUBLE"),
        ):
            if name not in existing:
                con.execute(f"ALTER TABLE experiments ADD COLUMN {name} {dtype}")

    def log_experiment(self, record: ExperimentRecord) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO experiments(
                    candidate_id,
                    pseudo_smiles,
                    composition_view,
                    measured_sty,
                    predicted_sty,
                    measured_selectivity,
                    predicted_selectivity,
                    measured_yield,
                    predicted_yield,
                    measured_stability_tos_h,
                    predicted_stability_tos_h,
                    measured_enzyme_activity,
                    predicted_enzyme_activity,
                    conditions_json,
                    user,
                    notes,
                    logged_at,
                    model_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record.to_row(),
            )

    def log_model_version(self, version: ModelVersion) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                  parent = EXCLUDED.parent,
                  delta_r2 = EXCLUDED.delta_r2,
                  n_feedback_used = EXCLUDED.n_feedback_used,
                  psi = EXCLUDED.psi,
                  created_at = EXCLUDED.created_at,
                  notes = EXCLUDED.notes
                """,
                version.to_row(),
            )

    def list_experiments(self, limit: int = 200) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT candidate_id, pseudo_smiles, composition_view, measured_sty, "
                "       predicted_sty, measured_selectivity, predicted_selectivity, "
                "       measured_yield, predicted_yield, measured_stability_tos_h, "
                "       predicted_stability_tos_h, measured_enzyme_activity, "
                "       predicted_enzyme_activity, conditions_json, user, notes, "
                "       logged_at, model_version "
                "FROM experiments ORDER BY logged_at DESC LIMIT ?",
                [limit],
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                conds = json.loads(r[13]) if r[13] else {}
            except json.JSONDecodeError:
                conds = {}
            out.append(
                {
                    "candidate_id": r[0],
                    "pseudo_smiles": r[1],
                    "composition_view": r[2],
                    "measured_sty": r[3],
                    "predicted_sty": r[4],
                    "measured_selectivity": r[5],
                    "predicted_selectivity": r[6],
                    "measured_yield": r[7],
                    "predicted_yield": r[8],
                    "measured_stability_tos_h": r[9],
                    "predicted_stability_tos_h": r[10],
                    "measured_enzyme_activity": r[11],
                    "predicted_enzyme_activity": r[12],
                    "conditions": conds,
                    "user": r[14],
                    "notes": r[15],
                    "logged_at": r[16],
                    "model_version": r[17],
                }
            )
        return out

    def count_since_last_train(self, model_version: str = "current") -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM experiments WHERE model_version = ?",
                [model_version],
            ).fetchone()
        return int(row[0] if row else 0)

    def list_model_versions(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT version, parent, delta_r2, n_feedback_used, psi, created_at, notes "
                "FROM model_versions ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "version": r[0],
                "parent": r[1],
                "delta_r2": r[2],
                "n_feedback_used": r[3],
                "psi": r[4],
                "created_at": r[5],
                "notes": r[6],
            }
            for r in rows
        ]


__all__ = ["FeedbackStore", "ExperimentRecord", "ModelVersion"]
