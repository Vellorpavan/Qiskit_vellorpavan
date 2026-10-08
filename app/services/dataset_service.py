"""
ShiftProof — Dataset Ingestion Service

Handles file reading (CSV, XLSX, JSON), column profiling, candidate matching,
and schema normalization into SQLite.
Enforces security guardrails: 5MB size limit, parameterized SQL, no eval/pickle.
"""

from __future__ import annotations

import io
import json
import uuid
import re
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import sqlite3

from app.database.db import get_db_connection
from app.database.seed import DEMO_DATASET_ID

__all__ = [
    "DEMO_DATASET_ID",
    "MAX_FILE_SIZE_BYTES",
    "ALLOWED_EXTENSIONS",
    "validate_file_header",
    "parse_uploaded_file",
    "detect_candidate_columns",
    "profile_dataframe",
    "save_mapped_tabular_dataset",
    "list_datasets",
    "delete_dataset",
    "set_active_dataset",
    "get_active_dataset",
    "get_active_dataset_identity",
]


def set_active_dataset(dataset_id: Optional[str], conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Set authoritative active dataset and canonical session state:
    - active_dataset_id
    - active_dataset_name
    - active_dataset_fingerprint
    - active_dataset_validation_status
    Invalidates stale dataset-derived state, solver runs, and results.
    Never executes optimization automatically.
    """
    try:
        import streamlit as st
    except ImportError:
        st = None

    ds_name = None
    ds_fp = None
    ds_status = None

    if dataset_id:
        close_conn = False
        if conn is None:
            conn = get_db_connection()
            close_conn = True
        try:
            cur = conn.cursor()
            cur.execute("SELECT name, status FROM datasets WHERE dataset_id = ?", (dataset_id,))
            row = cur.fetchone()
            if row:
                ds_name = row["name"]
                ds_status = row["status"]
            try:
                from app.services.scheduling_service import compute_dataset_fingerprint
                ds_fp = compute_dataset_fingerprint(dataset_id, conn)
            except Exception:
                ds_fp = f"fp_{dataset_id}"
        except Exception:
            pass
        finally:
            if close_conn:
                conn.close()

    if st is not None:
        try:
            old_id = st.session_state.get("active_dataset_id")
            st.session_state["active_dataset_id"] = dataset_id
            st.session_state["active_dataset_name"] = ds_name
            st.session_state["active_dataset_fingerprint"] = ds_fp
            st.session_state["active_dataset_validation_status"] = ds_status
            st.session_state["workspace_dataset_id"] = dataset_id

            # Sync execution state & reset to NOT_RUN
            try:
                from app.services.execution_state import sync_execution_state, reset_to_ready_not_run
                sync_execution_state(dataset_id, ds_fp)
                reset_to_ready_not_run()
            except Exception:
                pass

            # Invalidate any cached execution artifacts
            st.session_state.pop("latest_experiment_id", None)
            st.session_state.pop("selected_run_id", None)
            st.session_state.pop("just_saved_dataset_name", None)
            st.session_state.pop("current_instance", None)
            st.session_state.pop("current_schedule", None)
            st.session_state.pop("current_classical_result", None)
            st.session_state.pop("current_exact_result", None)
            st.session_state.pop("current_qaoa_result", None)
            st.session_state.pop("measurement_result", None)

            # Invalidate Workspace local state
            st.session_state["ws_active_dataset_id"] = dataset_id
            st.session_state["ws_active_fingerprint"] = ds_fp
            st.session_state["ws_classical_state"] = "NOT_RUN"
            st.session_state["ws_classical_data"] = None
            st.session_state["ws_classical_error"] = None
            st.session_state["ws_qaoa_state"] = "NOT_RUN"
            st.session_state["ws_qaoa_data"] = None
            st.session_state["ws_qaoa_error"] = None

            for k in list(st.session_state.keys()):
                if (
                    k.startswith("run_selector_")
                    or k.startswith("opt_run_")
                    or k.startswith("res_")
                ):
                    st.session_state.pop(k, None)
        except Exception:
            pass


def get_active_dataset(default: Optional[str] = None) -> Optional[str]:
    """Get authoritative active dataset ID. Returns None if no dataset is active."""
    try:
        import streamlit as st
        if "active_dataset_id" in st.session_state and st.session_state["active_dataset_id"]:
            return st.session_state["active_dataset_id"]
        if "workspace_dataset_id" in st.session_state and st.session_state["workspace_dataset_id"]:
            st.session_state["active_dataset_id"] = st.session_state["workspace_dataset_id"]
            return st.session_state["active_dataset_id"]
        return default
    except Exception:
        return default


def get_active_dataset_identity(conn: Optional[sqlite3.Connection] = None) -> dict[str, Any]:
    """
    Retrieve canonical active dataset identity:
    {
        'dataset_id': ...,
        'name': ...,
        'fingerprint': ...,
        'validation_status': ...
    }
    Restores missing metadata from SQLite if session state is missing details (e.g. after refresh).
    """
    ds_id = get_active_dataset()
    if not ds_id:
        return {
            "dataset_id": None,
            "name": None,
            "fingerprint": None,
            "validation_status": None,
        }

    try:
        import streamlit as st
        name = st.session_state.get("active_dataset_name")
        fp = st.session_state.get("active_dataset_fingerprint")
        status = st.session_state.get("active_dataset_validation_status")
    except ImportError:
        st = None
        name, fp, status = None, None, None

    # If any canonical metadata is missing, hydrate from SQLite
    if not name or not fp or not status:
        close_conn = False
        if conn is None:
            conn = get_db_connection()
            close_conn = True
        try:
            cur = conn.cursor()
            cur.execute("SELECT name, status FROM datasets WHERE dataset_id = ?", (ds_id,))
            row = cur.fetchone()
            if row:
                name = name or row["name"]
                status = status or row["status"]
            if not fp:
                try:
                    from app.services.scheduling_service import compute_dataset_fingerprint
                    fp = compute_dataset_fingerprint(ds_id, conn)
                except Exception:
                    fp = f"fp_{ds_id}"
            if st is not None:
                st.session_state["active_dataset_name"] = name
                st.session_state["active_dataset_fingerprint"] = fp
                st.session_state["active_dataset_validation_status"] = status
        except Exception:
            pass
        finally:
            if close_conn:
                conn.close()

    return {
        "dataset_id": ds_id,
        "name": name or ds_id,
        "fingerprint": fp,
        "validation_status": status or "ready",
    }

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB safety ceiling
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

# Synonyms for auto-detecting candidate columns
CANDIDATE_PATTERNS = {
    "worker_id": [r"worker.*id", r"emp.*id", r"staff.*id", r"nurse.*id", r"employee", r"^worker$", r"^staff$", r"staff.*code", r"worker.*code"],
    "shift_id": [r"shift.*id", r"slot.*id", r"duty.*id", r"^shift$", r"^slot$", r"duty.*code", r"shift.*code", r"slot.*code", r"^duty$"],
    "cost": [r"cost", r"rate", r"hourly.*rate", r"wage", r"expense", r"penalty", r"price", r"hourly.*wage"],
    "eligible": [r"eligib", r"availab", r"can_work", r"qualified", r"active", r"status"],
    "worker_name": [r"worker.*name", r"emp.*name", r"staff.*name", r"^name$", r"full.*name"],
    "shift_name": [r"shift.*name", r"slot.*name", r"duty.*name", r"title"],
    "shift_date": [r"date", r"day", r"shift.*date"],
    "shift_type": [r"shift.*type", r"time.*of.*day", r"type"],
    "skill": [r"skill", r"role", r"cert", r"specialty"],
}


def validate_file_header(filename: str, file_bytes: bytes) -> None:
    """Ensure file size and extension obey safety rules."""
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size ({len(file_bytes) / 1024 / 1024:.2f} MB) exceeds maximum limit of 5 MB.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File extension '{ext}' not supported. Allowed: {sorted(ALLOWED_EXTENSIONS)}")


def parse_uploaded_file(file_bytes: bytes, filename: str) -> tuple[Union[pd.DataFrame, dict], str]:
    """
    Parse uploaded file into DataFrame (tabular) or dict (Format B relational JSON).
    Returns (data, format_type).
    """
    validate_file_header(filename, file_bytes)
    ext = Path(filename).suffix.lower()

    if ext == ".csv":
        # Safe CSV parse
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df, "tabular_csv"

    elif ext in (".xlsx", ".xls"):
        # Safe Excel parse
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df, "tabular_xlsx"

    elif ext == ".json":
        # Safe JSON parse
        raw_text = file_bytes.decode("utf-8")
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            # List of dicts -> DataFrame (Format A)
            df = pd.DataFrame(parsed)
            return df, "tabular_json"
        elif isinstance(parsed, dict) and "workers" in parsed and "shifts" in parsed:
            # Format B: Relational JSON
            return parsed, "relational_json"
        elif isinstance(parsed, dict) and "assignments" in parsed:
            df = pd.DataFrame(parsed["assignments"])
            return df, "tabular_json"
        else:
            raise ValueError(
                "JSON format not recognized. Supported: list of assignment records (Format A) "
                "or relational object with 'workers', 'shifts', and 'assignments' (Format B)."
            )

    raise ValueError(f"Unsupported format: {ext}")


def detect_candidate_columns(columns: list[str]) -> dict[str, Optional[str]]:
    """
    Detect likely matches for standard ShiftProof fields from column names.
    Returns dict mapping target field -> matched column name (or None).
    """
    mapping: dict[str, Optional[str]] = {k: None for k in CANDIDATE_PATTERNS}

    for target_field, regex_list in CANDIDATE_PATTERNS.items():
        for col in columns:
            clean_col = col.strip().lower()
            if any(re.search(pat, clean_col) for pat in regex_list):
                mapping[target_field] = col
                break

    return mapping


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate high-level profile of tabular data."""
    cols = list(df.columns)
    candidates = detect_candidate_columns(cols)
    null_counts = df.isnull().sum().to_dict()
    dtypes = {c: str(df[c].dtype) for c in cols}

    return {
        "row_count": len(df),
        "col_count": len(cols),
        "columns": cols,
        "null_counts": null_counts,
        "dtypes": dtypes,
        "candidate_mappings": candidates,
        "has_nulls": bool(df.isnull().any().any()),
    }


def save_mapped_tabular_dataset(
    dataset_id: str,
    name: str,
    description: str,
    df: pd.DataFrame,
    mapping: dict[str, str],
    conn: Optional[sqlite3.Connection] = None,
    source_filename: Optional[str] = None,
) -> str:
    """
    Normalize tabular DataFrame (Format A) into SQLite using user-confirmed column mapping.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        w_col = mapping["worker_id"]
        s_col = mapping["shift_id"]
        cost_col = mapping["cost"]
        elig_col = mapping.get("eligible")

        # Optional columns
        w_name_col = mapping.get("worker_name")
        s_name_col = mapping.get("shift_name")
        s_date_col = mapping.get("shift_date")
        s_type_col = mapping.get("shift_type")
        skill_col = mapping.get("skill")

        # Distinct workers and shifts
        raw_workers = df[w_col].dropna().unique().tolist()
        raw_shifts = df[s_col].dropna().unique().tolist()

        sorted_workers = sorted(str(w) for w in raw_workers)
        sorted_shifts = sorted(str(s) for s in raw_shifts)

        with conn:
            # Insert dataset record
            conn.execute(
                """
                INSERT INTO datasets (dataset_id, name, description, source_type, source_filename, row_count, col_count, is_synthetic, status)
                VALUES (?, ?, ?, 'upload', ?, ?, ?, 0, 'ready')
                """,
                (dataset_id, name, description, source_filename, len(df), len(df.columns)),
            )

            # Insert workers
            for idx, w_id in enumerate(sorted_workers):
                # Lookup display name if available
                w_name = w_id
                w_skill = "General"
                if w_name_col and w_name_col in df.columns:
                    match_rows = df[df[w_col].astype(str) == w_id]
                    if not match_rows.empty and pd.notna(match_rows.iloc[0][w_name_col]):
                        w_name = str(match_rows.iloc[0][w_name_col])
                if skill_col and skill_col in df.columns:
                    match_rows = df[df[w_col].astype(str) == w_id]
                    if not match_rows.empty and pd.notna(match_rows.iloc[0][skill_col]):
                        w_skill = str(match_rows.iloc[0][skill_col])

                conn.execute(
                    """
                    INSERT INTO workers (worker_id, dataset_id, external_id, name, skill, max_shifts, worker_index)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (w_id, dataset_id, w_id, w_name, w_skill, idx),
                )

            # Insert shifts
            for idx, s_id in enumerate(sorted_shifts):
                s_name = s_id
                s_date = "N/A"
                s_type = "Standard"
                if s_name_col and s_name_col in df.columns:
                    match_rows = df[df[s_col].astype(str) == s_id]
                    if not match_rows.empty and pd.notna(match_rows.iloc[0][s_name_col]):
                        s_name = str(match_rows.iloc[0][s_name_col])
                if s_date_col and s_date_col in df.columns:
                    match_rows = df[df[s_col].astype(str) == s_id]
                    if not match_rows.empty and pd.notna(match_rows.iloc[0][s_date_col]):
                        s_date = str(match_rows.iloc[0][s_date_col])
                if s_type_col and s_type_col in df.columns:
                    match_rows = df[df[s_col].astype(str) == s_id]
                    if not match_rows.empty and pd.notna(match_rows.iloc[0][s_type_col]):
                        s_type = str(match_rows.iloc[0][s_type_col])

                conn.execute(
                    """
                    INSERT INTO shifts (shift_id, dataset_id, external_id, shift_name, shift_date, shift_type, required_skill, shift_index)
                    VALUES (?, ?, ?, ?, ?, ?, 'Standard', ?)
                    """,
                    (s_id, dataset_id, s_id, s_name, s_date, s_type, idx),
                )

            # Insert costs & eligibility
            for _, row in df.iterrows():
                w_id = str(row[w_col])
                s_id = str(row[s_col])
                cost_val = float(row[cost_col])

                # Eligibility logic
                is_el = True
                if elig_col and elig_col in df.columns:
                    raw_el = row[elig_col]
                    if isinstance(raw_el, (bool, np.bool_)):
                        is_el = bool(raw_el)
                    elif isinstance(raw_el, (int, float, np.number)):
                        is_el = bool(raw_el != 0)
                    elif isinstance(raw_el, str):
                        is_el = raw_el.strip().lower() in ("true", "1", "yes", "y", "eligible", "available")

                conn.execute(
                    """
                    INSERT OR REPLACE INTO costs (dataset_id, worker_id, shift_id, assignment_cost)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dataset_id, w_id, s_id, cost_val),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eligibility (dataset_id, worker_id, shift_id, is_eligible, reason)
                    VALUES (?, ?, ?, ?, 'Uploaded flag')
                    """,
                    (dataset_id, w_id, s_id, is_el),
                )

        return dataset_id
    finally:
        if close_after:
            conn.close()


def list_datasets(conn: Optional[sqlite3.Connection] = None, include_synthetic: bool = False) -> list[dict]:
    """List all available datasets in the application workspace."""
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        cur = conn.cursor()
        where_clause = "" if include_synthetic else "WHERE d.is_synthetic = 0 AND d.source_type != 'synthetic_demo'"
        cur.execute(
            f"""
            SELECT d.dataset_id, d.name, d.description, d.source_type, d.source_filename, d.created_at,
                   d.row_count, d.col_count, d.is_synthetic, d.status,
                   COUNT(DISTINCT w.worker_id) as worker_count,
                   COUNT(DISTINCT s.shift_id) as shift_count
            FROM datasets d
            LEFT JOIN workers w ON d.dataset_id = w.dataset_id
            LEFT JOIN shifts s ON d.dataset_id = s.dataset_id
            {where_clause}
            GROUP BY d.dataset_id
            ORDER BY d.created_at DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close_after:
            conn.close()


def delete_dataset(dataset_id: str, conn: Optional[sqlite3.Connection] = None) -> bool:
    """
    Safely delete a dataset from SQLite workspace with strict dataset-scoped cleanup.
    Protected: Refuses to delete synthetic demo (DEMO_DATASET_ID) and cannot delete scientific benchmark.
    CRITICAL: Never deletes or touches application source code, package directories,
    or files under app/, src/, tests/.

    Explicitly cascades removal:
      - measurements (WHERE experiment_id IN (...))
      - solutions
      - experiments
      - costs
      - eligibility
      - constraints
      - shifts
      - workers
      - datasets (WHERE dataset_id = ?)

    If the deleted dataset is currently active:
      - clears canonical active dataset state
      - clears execution state and result state
    If another dataset is active:
      - leaves it untouched.
    """
    if not dataset_id:
        return False

    if dataset_id == DEMO_DATASET_ID:
        return False

    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        with conn:
            cur = conn.cursor()
            # 1. Measurements
            cur.execute(
                "DELETE FROM measurements WHERE experiment_id IN (SELECT experiment_id FROM experiments WHERE dataset_id = ?)",
                (dataset_id,)
            )
            # 2. Solutions
            cur.execute("DELETE FROM solutions WHERE dataset_id = ?", (dataset_id,))
            # 3. Experiments
            cur.execute("DELETE FROM experiments WHERE dataset_id = ?", (dataset_id,))
            # 4. Costs
            cur.execute("DELETE FROM costs WHERE dataset_id = ?", (dataset_id,))
            # 5. Eligibility
            cur.execute("DELETE FROM eligibility WHERE dataset_id = ?", (dataset_id,))
            # 6. Constraints
            cur.execute("DELETE FROM constraints WHERE dataset_id = ?", (dataset_id,))
            # 7. Shifts
            cur.execute("DELETE FROM shifts WHERE dataset_id = ?", (dataset_id,))
            # 8. Workers
            cur.execute("DELETE FROM workers WHERE dataset_id = ?", (dataset_id,))
            # 9. Datasets record
            cur.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            deleted = cur.rowcount > 0

        # State cleanup if deleted dataset was active
        try:
            active_id = get_active_dataset()
            if active_id == dataset_id:
                set_active_dataset(None)
        except Exception:
            pass

        return deleted
    finally:
        if close_after:
            conn.close()


def load_dataset_table(dataset_id: str, conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    """Retrieve full tabular representation of any stored dataset."""
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        query = """
        SELECT 
            w.worker_id AS 'Worker ID',
            w.name AS 'Worker Name',
            s.shift_id AS 'Shift ID',
            s.shift_name AS 'Shift Name',
            s.shift_date AS 'Shift Date',
            s.shift_type AS 'Shift Type',
            COALESCE(w.skill, 'General') AS 'Skill',
            COALESCE(s.required_skill, 'Any') AS 'Required Skill',
            COALESCE(c.assignment_cost, 0.0) AS 'Assignment Cost',
            CASE WHEN COALESCE(e.is_eligible, 1) = 1 THEN 'Eligible' ELSE 'Ineligible' END AS 'Eligibility',
            CASE WHEN COALESCE(e.is_eligible, 1) = 1 THEN 'Available' ELSE 'Unavailable' END AS 'Availability',
            COALESCE(w.max_shifts, 1) AS 'Capacity'
        FROM workers w
        JOIN shifts s ON w.dataset_id = s.dataset_id
        LEFT JOIN costs c ON w.dataset_id = c.dataset_id AND w.worker_id = c.worker_id AND s.shift_id = c.shift_id
        LEFT JOIN eligibility e ON w.dataset_id = e.dataset_id AND w.worker_id = e.worker_id AND s.shift_id = e.shift_id
        WHERE w.dataset_id = ?
        ORDER BY w.worker_index ASC, s.shift_index ASC
        """
        return pd.read_sql_query(query, conn, params=(dataset_id,))
    finally:
        if close_after:
            conn.close()
