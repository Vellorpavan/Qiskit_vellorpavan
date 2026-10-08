"""
ShiftProof Database Seeder

Seeds the built-in "Synthetic Demo Dataset".
Explicitly labeled: SYNTHETIC · NOT REAL ORGANIZATION DATA.
Small 3x3 scheduling instance compatible with the frozen Instance model.
"""

from __future__ import annotations

import sqlite3
from typing import Optional
from app.database.db import get_db_connection
from app.database.schema import init_db

DEMO_DATASET_ID = "synthetic_demo_ward_3x3"
DEMO_NAME = "Synthetic Demo Dataset — Hospital Ward 3x3"
DEMO_DESC = "SYNTHETIC · NOT REAL ORGANIZATION DATA. 3 workers, 3 shifts, variable reduction from 9 down to 7 qubits."


def seed_synthetic_demo(conn: Optional[sqlite3.Connection] = None) -> str:
    """
    Seed the synthetic demo dataset into SQLite.
    Idempotent: removes and re-creates synthetic_demo_ward_3x3 if it exists.
    Returns dataset_id.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    init_db(conn)

    try:
        with conn:
            # Clean existing demo dataset if present
            conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (DEMO_DATASET_ID,))

            # 1. Insert dataset record
            conn.execute(
                """
                INSERT INTO datasets (dataset_id, name, description, source_type, row_count, col_count, is_synthetic, status)
                VALUES (?, ?, ?, 'synthetic_demo', 9, 4, 1, 'ready')
                """,
                (DEMO_DATASET_ID, DEMO_NAME, DEMO_DESC),
            )

            # 2. Insert workers
            workers = [
                ("W0", "RN-101", "Elena Rostova", "ICU_Certified", 1, 0),
                ("W1", "RN-102", "Marcus Chen", "General_Care", 1, 1),
                ("W2", "RN-103", "Sarah Miller", "Emergency_Care", 1, 2),
            ]
            conn.executemany(
                """
                INSERT INTO workers (worker_id, dataset_id, external_id, name, skill, max_shifts, worker_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [(w_id, DEMO_DATASET_ID, ext_id, name, skill, max_s, idx) for w_id, ext_id, name, skill, max_s, idx in workers],
            )

            # 3. Insert shifts
            shifts = [
                ("S0", "SHIFT-DAY", "Day Ward Shift", "2026-10-15", "Morning", "General_Care", 0),
                ("S1", "SHIFT-EVE", "Evening Ward Shift", "2026-10-15", "Afternoon", "General_Care", 1),
                ("S2", "SHIFT-NGT", "Night ICU Shift", "2026-10-15", "Night", "ICU_Certified", 2),
            ]
            conn.executemany(
                """
                INSERT INTO shifts (shift_id, dataset_id, external_id, shift_name, shift_date, shift_type, required_skill, shift_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(s_id, DEMO_DATASET_ID, ext_id, name, date, s_type, req_skill, idx) for s_id, ext_id, name, date, s_type, req_skill, idx in shifts],
            )

            # 4. Insert eligibility and costs
            # 7 eligible pairs, 2 ineligible pairs (W1 and W2 ineligible for Night ICU Shift S2)
            assignments = [
                ("W0", "S0", True, 42.0, "Eligible"),
                ("W0", "S1", True, 50.0, "Eligible"),
                ("W0", "S2", True, 65.0, "ICU Certified"),
                ("W1", "S0", True, 30.0, "Eligible"),
                ("W1", "S1", True, 35.0, "Eligible"),
                ("W1", "S2", False, 999.0, "Requires ICU certification"),
                ("W2", "S0", True, 38.0, "Eligible"),
                ("W2", "S1", True, 40.0, "Eligible"),
                ("W2", "S2", False, 999.0, "Requires ICU certification"),
            ]
            conn.executemany(
                """
                INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(DEMO_DATASET_ID, w_id, s_id, elig, reason) for w_id, s_id, elig, _, reason in assignments],
            )
            conn.executemany(
                """
                INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost)
                VALUES (?, ?, ?, ?)
                """,
                [(DEMO_DATASET_ID, w_id, s_id, cost) for w_id, s_id, _, cost, _ in assignments],
            )

            # 5. Insert default constraint configurations
            conn.execute(
                """
                INSERT INTO constraints (constraint_id, dataset_id, constraint_type, parameters_json, is_active)
                VALUES (?, ?, 'shift_coverage', '{"required_workers_per_shift": 1}', 1)
                """,
                (f"{DEMO_DATASET_ID}_c1", DEMO_DATASET_ID),
            )
            conn.execute(
                """
                INSERT INTO constraints (constraint_id, dataset_id, constraint_type, parameters_json, is_active)
                VALUES (?, ?, 'worker_at_most_one', '{"max_shifts_per_worker": 1}', 1)
                """,
                (f"{DEMO_DATASET_ID}_c2", DEMO_DATASET_ID),
            )
            conn.execute(
                """
                INSERT INTO constraints (constraint_id, dataset_id, constraint_type, parameters_json, is_active)
                VALUES (?, ?, 'eligibility_strict', '{"allow_ineligible": false}', 1)
                """,
                (f"{DEMO_DATASET_ID}_c3", DEMO_DATASET_ID),
            )

        return DEMO_DATASET_ID
    finally:
        if close_after:
            conn.close()
