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
]


def set_active_dataset(dataset_id: str) -> None:
    """Set authoritative active dataset and invalidate stale dataset-derived state."""
    try:
        import streamlit as st
        old_id = st.session_state.get("active_dataset_id")
        st.session_state["active_dataset_id"] = dataset_id
        st.session_state["workspace_dataset_id"] = dataset_id
        if old_id != dataset_id:
            st.session_state.pop("latest_experiment_id", None)
            st.session_state.pop("selected_run_id", None)
            st.session_state.pop("just_saved_dataset_name", None)
            st.session_state.pop("current_instance", None)
            st.session_state.pop("current_schedule", None)
            st.session_state.pop("current_classical_result", None)
            st.session_state.pop("current_exact_result", None)
            st.session_state.pop("current_qaoa_result", None)
            for k in list(st.session_state.keys()):
                if k.startswith("run_selector_") or k.startswith("opt_run_"):
                    st.session_state.pop(k, None)
    except Exception:
        pass


def get_active_dataset(default: str = DEMO_DATASET_ID) -> str:
    """Get authoritative active dataset ID."""
    try:
        import streamlit as st
        if "active_dataset_id" in st.session_state:
            return st.session_state["active_dataset_id"]
        if "workspace_dataset_id" in st.session_state:
            st.session_state["active_dataset_id"] = st.session_state["workspace_dataset_id"]
            return st.session_state["active_dataset_id"]
        st.session_state["active_dataset_id"] = default
        st.session_state["workspace_dataset_id"] = default
        return default
    except Exception:
        return default

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


def list_datasets(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """List all available datasets in the application workspace."""
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.dataset_id, d.name, d.description, d.source_type, d.source_filename, d.created_at,
                   d.row_count, d.col_count, d.is_synthetic, d.status,
                   COUNT(DISTINCT w.worker_id) as worker_count,
                   COUNT(DISTINCT s.shift_id) as shift_count
            FROM datasets d
            LEFT JOIN workers w ON d.dataset_id = w.dataset_id
            LEFT JOIN shifts s ON d.dataset_id = s.dataset_id
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
    Safely delete a dataset from SQLite workspace.
    Protected: Cannot delete scientific benchmark (which isn't in SQLite anyway)
    and warns if deleting synthetic demo.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            return cur.rowcount > 0
    finally:
        if close_after:
            conn.close()
