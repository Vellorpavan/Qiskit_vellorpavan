"""
ShiftProof Database Schema

Defines DDL for the interactive workspace tables in SQLite.
Enforces foreign key relationships and integrity constraints.
"""

from __future__ import annotations

import sqlite3
from typing import Optional
from app.database.db import get_db_connection

SCHEMA_SQL = """
-- 1. Datasets Metadata
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    description         TEXT,
    source_type         TEXT NOT NULL,
    source_filename     TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count           INTEGER NOT NULL DEFAULT 0,
    col_count           INTEGER NOT NULL DEFAULT 0,
    is_synthetic        BOOLEAN NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'uploaded'
);

-- 2. Workers
CREATE TABLE IF NOT EXISTS workers (
    worker_id           TEXT NOT NULL,
    dataset_id          TEXT NOT NULL,
    external_id         TEXT,
    name                TEXT NOT NULL,
    skill               TEXT,
    max_shifts          INTEGER DEFAULT 1,
    worker_index        INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, worker_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

-- 3. Shifts
CREATE TABLE IF NOT EXISTS shifts (
    shift_id            TEXT NOT NULL,
    dataset_id          TEXT NOT NULL,
    external_id         TEXT,
    shift_name          TEXT NOT NULL,
    shift_date          TEXT,
    shift_type          TEXT,
    required_skill      TEXT,
    shift_index         INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, shift_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

-- 4. Eligibility
CREATE TABLE IF NOT EXISTS eligibility (
    dataset_id          TEXT NOT NULL,
    worker_id           TEXT NOT NULL,
    shift_id            TEXT NOT NULL,
    is_eligible         BOOLEAN NOT NULL DEFAULT 1,
    reason              TEXT,
    PRIMARY KEY (dataset_id, worker_id, shift_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, worker_id) REFERENCES workers(dataset_id, worker_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, shift_id) REFERENCES shifts(dataset_id, shift_id) ON DELETE CASCADE
);

-- 5. Costs
CREATE TABLE IF NOT EXISTS costs (
    dataset_id          TEXT NOT NULL,
    worker_id           TEXT NOT NULL,
    shift_id            TEXT NOT NULL,
    assignment_cost     REAL NOT NULL,
    PRIMARY KEY (dataset_id, worker_id, shift_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, worker_id) REFERENCES workers(dataset_id, worker_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id, shift_id) REFERENCES shifts(dataset_id, shift_id) ON DELETE CASCADE
);

-- 6. Constraints Configuration
CREATE TABLE IF NOT EXISTS constraints (
    constraint_id       TEXT PRIMARY KEY,
    dataset_id          TEXT NOT NULL,
    constraint_type     TEXT NOT NULL,
    parameters_json     TEXT NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

-- 7. Experiments / Optimization Runs
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    dataset_id          TEXT NOT NULL,
    solver_type         TEXT NOT NULL,
    status              TEXT NOT NULL,
    parameters_json     TEXT NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

-- 8. Solutions
CREATE TABLE IF NOT EXISTS solutions (
    solution_id         TEXT PRIMARY KEY,
    experiment_id       TEXT NOT NULL,
    dataset_id          TEXT NOT NULL,
    feasible            BOOLEAN NOT NULL,
    total_cost          REAL,
    runtime_seconds     REAL NOT NULL,
    assignment_json     TEXT NOT NULL,
    verification_json   TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

-- 9. QAOA Measurement Distribution
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id      TEXT PRIMARY KEY,
    experiment_id       TEXT NOT NULL,
    bitstring           TEXT NOT NULL,
    shot_count          INTEGER NOT NULL,
    probability         REAL NOT NULL,
    is_feasible         BOOLEAN NOT NULL,
    assignment_cost     REAL,
    qubo_energy         REAL NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);
"""


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Initialize database tables with foreign key enforcement."""
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
            # Automatic schema migration for source_filename
            try:
                conn.execute("ALTER TABLE datasets ADD COLUMN source_filename TEXT;")
            except Exception:
                pass
    finally:
        if close_after:
            conn.close()
