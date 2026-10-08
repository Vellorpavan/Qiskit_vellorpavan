"""
ShiftProof — Scheduling Service & Database-to-Instance Adapter

Converts SQLite dataset representation into the frozen src.model.Instance.
Preserves deterministic ordering, performs validation, and formats schedule rosters.
ZERO modifications to src/model.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import sqlite3

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hashlib
import json
import logging
from src.model import Instance, check_constraints, assignment_cost
from app.database.db import get_db_connection

logger = logging.getLogger("shiftproof.scheduling")


def compute_dataset_fingerprint(dataset_id: str, conn: Optional[sqlite3.Connection] = None) -> str:
    """
    Computes a deterministic SHA-256 fingerprint for a dataset based on its
    normalized workers, shifts, costs, and eligibility in SQLite.
    Returns: 'fp_' + 16 hex characters.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT worker_id, worker_index, name FROM workers WHERE dataset_id = ? ORDER BY worker_index ASC",
            (dataset_id,),
        )
        workers = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT shift_id, shift_index, shift_name FROM shifts WHERE dataset_id = ? ORDER BY shift_index ASC",
            (dataset_id,),
        )
        shifts = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT worker_id, shift_id, assignment_cost FROM costs WHERE dataset_id = ? ORDER BY worker_id, shift_id",
            (dataset_id,),
        )
        costs = [(r["worker_id"], r["shift_id"], float(r["assignment_cost"])) for r in cur.fetchall()]

        cur.execute(
            "SELECT worker_id, shift_id, is_eligible FROM eligibility WHERE dataset_id = ? ORDER BY worker_id, shift_id",
            (dataset_id,),
        )
        elig = [(r["worker_id"], r["shift_id"], int(r["is_eligible"])) for r in cur.fetchall()]

        cur.execute(
            "SELECT constraint_id, constraint_type, parameters_json FROM constraints WHERE dataset_id = ? ORDER BY constraint_id",
            (dataset_id,),
        )
        constraints = [dict(r) for r in cur.fetchall()]

        payload = json.dumps(
            {
                "dataset_id": dataset_id,
                "workers": workers,
                "shifts": shifts,
                "costs": costs,
                "eligibility": elig,
                "constraints": constraints,
            },
            sort_keys=True,
        )

        return "fp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    finally:
        if close_after:
            conn.close()


def db_to_instance(dataset_id: str, conn: Optional[sqlite3.Connection] = None) -> Instance:
    """
    Constructs a frozen src.model.Instance directly from SQLite data.
    Preserves deterministic worker and shift ordering.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        # 1. Fetch workers ordered by worker_index
        cur = conn.cursor()
        cur.execute(
            """
            SELECT worker_id, worker_index, name 
            FROM workers 
            WHERE dataset_id = ? 
            ORDER BY worker_index ASC
            """,
            (dataset_id,),
        )
        worker_rows = cur.fetchall()
        if not worker_rows:
            raise ValueError(f"No workers found for dataset '{dataset_id}'")

        # 2. Fetch shifts ordered by shift_index
        cur.execute(
            """
            SELECT shift_id, shift_index, shift_name 
            FROM shifts 
            WHERE dataset_id = ? 
            ORDER BY shift_index ASC
            """,
            (dataset_id,),
        )
        shift_rows = cur.fetchall()
        if not shift_rows:
            raise ValueError(f"No shifts found for dataset '{dataset_id}'")

        n_workers = len(worker_rows)
        n_shifts = len(shift_rows)

        # Map IDs to indices
        worker_id_to_idx = {r["worker_id"]: r["worker_index"] for r in worker_rows}
        shift_id_to_idx = {r["shift_id"]: r["shift_index"] for r in shift_rows}

        # 3. Initialize dense matrices
        cost_matrix = np.zeros((n_workers, n_shifts), dtype=np.float64)
        eligibility = np.zeros((n_workers, n_shifts), dtype=bool)

        # 4. Populate costs
        cur.execute(
            """
            SELECT worker_id, shift_id, assignment_cost 
            FROM costs 
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        )
        for r in cur.fetchall():
            w_id, s_id, cost = r["worker_id"], r["shift_id"], r["assignment_cost"]
            if w_id in worker_id_to_idx and s_id in shift_id_to_idx:
                w_idx = worker_id_to_idx[w_id]
                s_idx = shift_id_to_idx[s_id]
                cost_matrix[w_idx, s_idx] = float(cost)

        # 5. Populate eligibility
        cur.execute(
            """
            SELECT worker_id, shift_id, is_eligible 
            FROM eligibility 
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        )
        for r in cur.fetchall():
            w_id, s_id, is_el = r["worker_id"], r["shift_id"], r["is_eligible"]
            if w_id in worker_id_to_idx and s_id in shift_id_to_idx:
                w_idx = worker_id_to_idx[w_id]
                s_idx = shift_id_to_idx[s_id]
                eligibility[w_idx, s_idx] = bool(is_el)

        # 6. Construct frozen Instance (built-in validation will trigger in __post_init__)
        instance = Instance(
            n_workers=n_workers,
            n_shifts=n_shifts,
            cost_matrix=cost_matrix,
            eligibility=eligibility,
            instance_id=dataset_id,
            seed=42,
        )

        fp = compute_dataset_fingerprint(dataset_id, conn)
        logger.info(
            f"[DATASET] dataset_id: {dataset_id}, fingerprint: {fp} | "
            f"[INSTANCE] worker_count: {n_workers}, shift_count: {n_shifts}, variable_count: {instance.n_vars}"
        )

        return instance
    finally:
        if close_after:
            conn.close()


def format_solution_roster(
    instance: Instance,
    assignment: dict,
    dataset_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """
    Format an assignment dictionary {(w, s): 1} into human-readable roster records
    with worker names, shift names, dates, and costs.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        cur = conn.cursor()
        cur.execute("SELECT worker_id, worker_index, name, skill FROM workers WHERE dataset_id = ?", (dataset_id,))
        workers_by_idx = {r["worker_index"]: dict(r) for r in cur.fetchall()}

        cur.execute("SELECT shift_id, shift_index, shift_name, shift_date, shift_type FROM shifts WHERE dataset_id = ?", (dataset_id,))
        shifts_by_idx = {r["shift_index"]: dict(r) for r in cur.fetchall()}

        roster = []
        for s_idx in range(instance.n_shifts):
            shift_info = shifts_by_idx.get(s_idx, {"shift_name": f"Shift {s_idx}", "shift_date": "N/A", "shift_type": "N/A"})
            # Find assigned worker
            assigned_w_idx = None
            for w_idx in range(instance.n_workers):
                if assignment.get((w_idx, s_idx), 0) == 1:
                    assigned_w_idx = w_idx
                    break

            if assigned_w_idx is not None:
                worker_info = workers_by_idx.get(assigned_w_idx, {"name": f"Worker {assigned_w_idx}", "skill": "N/A"})
                cost = instance.get_cost(assigned_w_idx, s_idx)
                is_el = instance.is_eligible(assigned_w_idx, s_idx)
                roster.append({
                    "Shift": shift_info.get("shift_name"),
                    "Date": shift_info.get("shift_date"),
                    "Type": shift_info.get("shift_type"),
                    "Assigned Worker": worker_info.get("name"),
                    "Skill": worker_info.get("skill"),
                    "Cost": cost,
                    "Eligible": "✓ Yes" if is_el else "✗ No",
                    "Status": "Assigned",
                })
            else:
                roster.append({
                    "Shift": shift_info.get("shift_name"),
                    "Date": shift_info.get("shift_date"),
                    "Type": shift_info.get("shift_type"),
                    "Assigned Worker": "UNASSIGNED",
                    "Skill": "-",
                    "Cost": 0.0,
                    "Eligible": "-",
                    "Status": "Violation (Uncovered)",
                })

        return roster
    finally:
        if close_after:
            conn.close()


def compute_dataframe_fingerprint(df_normalized: pd.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 fingerprint from a normalized DataFrame.
    Returns: 'fp_' + 16 hex characters.
    """
    df_sorted = df_normalized.sort_values(["worker_id", "shift_id"]).reset_index(drop=True)
    rows = []
    for _, r in df_sorted.iterrows():
        rows.append({
            "worker_id": str(r["worker_id"]),
            "shift_id": str(r["shift_id"]),
            "cost": float(r.get("assignment_cost", 0.0)),
            "eligible": bool(r.get("is_eligible", True)),
            "available": bool(r.get("is_available", True)),
        })
    payload = json.dumps(rows, sort_keys=True)
    return "fp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalized_to_instance(
    norm_res: Any,
    instance_id: str = "normalized_instance",
    seed: int = 42,
) -> Instance:
    """
    Directly converts a NormalizationResult (or normalized DataFrame) into a frozen src.model.Instance.
    Enables zero-DB transformation for notebooks, testing, and memory-only pipelines.
    """
    df = norm_res.df_normalized if hasattr(norm_res, "df_normalized") else norm_res

    # Order workers and shifts deterministically
    worker_ids = sorted(list(df["worker_id"].unique()))
    shift_ids = sorted(list(df["shift_id"].unique()))

    n_workers = len(worker_ids)
    n_shifts = len(shift_ids)

    w_map = {wid: i for i, wid in enumerate(worker_ids)}
    s_map = {sid: j for j, sid in enumerate(shift_ids)}

    cost_matrix = np.zeros((n_workers, n_shifts), dtype=np.float64)
    eligibility = np.zeros((n_workers, n_shifts), dtype=bool)

    for _, row in df.iterrows():
        w_id = row["worker_id"]
        s_id = row["shift_id"]
        if w_id in w_map and s_id in s_map:
            w_idx = w_map[w_id]
            s_idx = s_map[s_id]
            cost_matrix[w_idx, s_idx] = float(row.get("assignment_cost", 0.0))
            is_el = bool(row.get("is_eligible", True))
            is_av = bool(row.get("is_available", True))
            eligibility[w_idx, s_idx] = is_el and is_av

    return Instance(
        n_workers=n_workers,
        n_shifts=n_shifts,
        cost_matrix=cost_matrix,
        eligibility=eligibility,
        instance_id=instance_id,
        seed=seed,
    )
