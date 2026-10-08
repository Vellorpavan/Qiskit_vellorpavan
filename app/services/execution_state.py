"""
ShiftProof — Execution State Model & Lifecycle Service

Enforces strict separation between DATASET ACTIVATION and EXPERIMENT EXECUTION.
Activating a dataset only registers and validates the dataset; it NEVER executes
optimization and NEVER automatically runs a solver.

Execution Lifecycle:
  NO_DATASET
      ↓ (User selects/activates dataset)
  DATASET_READY ("READY — NOT RUN")
      ↓ (User explicitly clicks "Run Classical" / "Run QAOA")
  CLASSICAL_RUNNING / QAOA_RUNNING
      ↓
  CLASSICAL_COMPLETE / QAOA_COMPLETE (Only now results are displayed)
"""

from __future__ import annotations

import json
import sqlite3
from enum import Enum
from typing import Any, Optional


class ExecutionState(str, Enum):
    NO_DATASET = "NO_DATASET"
    DATASET_READY = "DATASET_READY"  # READY — NOT RUN
    CLASSICAL_RUNNING = "CLASSICAL_RUNNING"
    CLASSICAL_COMPLETE = "CLASSICAL_COMPLETE"
    QAOA_RUNNING = "QAOA_RUNNING"
    QAOA_COMPLETE = "QAOA_COMPLETE"
    EXECUTION_FAILED = "EXECUTION_FAILED"


def sync_execution_state(
    active_dataset_id: Optional[str],
    active_fingerprint: Optional[str] = None,
) -> dict[str, Any]:
    """
    Synchronizes the execution state in Streamlit session_state with strict isolation:
    1. If active_dataset_id is None -> NO_DATASET.
    2. If dataset was switched, refreshed, or newly activated -> DATASET_READY ("READY — NOT RUN").
       Clears any prior experiment pointers. Previous results are NEVER automatically displayed.
    3. If an explicit execution occurred in the current session -> preserves current experiment.
    4. If the user explicitly chose to view a historical run -> preserves historical view pointer.
    """
    try:
        import streamlit as st
    except ImportError:
        # Fallback for headless testing
        st = None

    if st is None:
        if not active_dataset_id:
            return {"state": ExecutionState.NO_DATASET, "is_ready_not_run": False}
        return {"state": ExecutionState.DATASET_READY, "is_ready_not_run": True}

    # 1. No dataset loaded
    if not active_dataset_id:
        st.session_state["exec_active_dataset_id"] = None
        st.session_state["exec_active_fingerprint"] = None
        st.session_state["execution_state"] = ExecutionState.NO_DATASET
        st.session_state["current_classical_experiment_id"] = None
        st.session_state["current_qaoa_experiment_id"] = None
        st.session_state["current_ibm_experiment_id"] = None
        st.session_state["current_experiment_id"] = None
        st.session_state["current_solver_type"] = None
        st.session_state["viewing_historical_experiment_id"] = None
        st.session_state["navigation_target"] = None
        return {
            "state": ExecutionState.NO_DATASET,
            "is_ready_not_run": False,
            "current_experiment_id": None,
            "current_classical_experiment_id": None,
            "current_qaoa_experiment_id": None,
            "viewing_historical_experiment_id": None,
            "is_viewing_historical": False,
        }

    # 2. Dataset changed, switched, or refreshed without existing session context
    stored_ds_id = st.session_state.get("exec_active_dataset_id")
    stored_fp = st.session_state.get("exec_active_fingerprint")

    if stored_ds_id != active_dataset_id or (active_fingerprint and stored_fp != active_fingerprint):
        # Immediate invalidation upon activation or switch
        st.session_state["exec_active_dataset_id"] = active_dataset_id
        st.session_state["exec_active_fingerprint"] = active_fingerprint
        st.session_state["execution_state"] = ExecutionState.DATASET_READY
        st.session_state["current_classical_experiment_id"] = None
        st.session_state["current_qaoa_experiment_id"] = None
        st.session_state["current_ibm_experiment_id"] = None
        st.session_state["current_experiment_id"] = None
        st.session_state["current_solver_type"] = None
        st.session_state["viewing_historical_experiment_id"] = None
        st.session_state["navigation_target"] = None
        # Invalidate workspace in-memory execution caches
        st.session_state["ws_classical_state"] = "NOT_RUN"
        st.session_state["ws_classical_data"] = None
        st.session_state["ws_classical_error"] = None
        st.session_state["ws_qaoa_state"] = "NOT_RUN"
        st.session_state["ws_qaoa_data"] = None
        st.session_state["ws_qaoa_error"] = None

    # Retrieve current session state
    current_state = st.session_state.get("execution_state", ExecutionState.DATASET_READY)
    current_exp_id = st.session_state.get("current_experiment_id")
    hist_exp_id = st.session_state.get("viewing_historical_experiment_id")

    return {
        "state": current_state,
        "is_ready_not_run": (current_state == ExecutionState.DATASET_READY and current_exp_id is None and hist_exp_id is None),
        "current_classical_experiment_id": st.session_state.get("current_classical_experiment_id"),
        "current_qaoa_experiment_id": st.session_state.get("current_qaoa_experiment_id"),
        "current_experiment_id": current_exp_id,
        "current_solver_type": st.session_state.get("current_solver_type"),
        "viewing_historical_experiment_id": hist_exp_id,
        "is_viewing_historical": bool(hist_exp_id is not None),
    }


def record_classical_execution(experiment_id: str) -> None:
    """Record that an explicit Classical run just completed in the current session."""
    try:
        import streamlit as st
        st.session_state["current_classical_experiment_id"] = experiment_id
        st.session_state["current_experiment_id"] = experiment_id
        st.session_state["current_solver_type"] = "classical"
        st.session_state["execution_state"] = ExecutionState.CLASSICAL_COMPLETE
        st.session_state["viewing_historical_experiment_id"] = None
        st.session_state["navigation_target"] = "results"
    except Exception:
        pass


def record_qaoa_execution(experiment_id: str) -> None:
    """Record that an explicit QAOA run just completed in the current session."""
    try:
        import streamlit as st
        st.session_state["current_qaoa_experiment_id"] = experiment_id
        st.session_state["current_experiment_id"] = experiment_id
        st.session_state["current_solver_type"] = "qaoa_p1"
        st.session_state["execution_state"] = ExecutionState.QAOA_COMPLETE
        st.session_state["viewing_historical_experiment_id"] = None
        st.session_state["navigation_target"] = "quantum"
    except Exception:
        pass


def record_ibm_execution(experiment_id: str) -> None:
    """Record that an explicit IBM Quantum Hardware run just completed in the current session."""
    try:
        import streamlit as st
        st.session_state["current_ibm_experiment_id"] = experiment_id
        st.session_state["current_experiment_id"] = experiment_id
        st.session_state["current_solver_type"] = "ibm_quantum"
        st.session_state["viewing_historical_experiment_id"] = None
        st.session_state["navigation_target"] = "quantum"
    except Exception:
        pass


def set_navigation_target(target: str) -> None:
    """Set the explicit navigation target ('workspace', 'results', 'quantum')."""
    try:
        import streamlit as st
        st.session_state["navigation_target"] = target
    except Exception:
        pass


def get_navigation_target() -> Optional[str]:
    """Retrieve the pending navigation target if set."""
    try:
        import streamlit as st
        return st.session_state.get("navigation_target")
    except Exception:
        return None


def clear_navigation_target() -> None:
    """Clear the pending navigation target."""
    try:
        import streamlit as st
        st.session_state["navigation_target"] = None
    except Exception:
        pass


def set_viewing_historical(experiment_id: Optional[str]) -> None:
    """Set or clear inspection of a historical experiment."""
    try:
        import streamlit as st
        st.session_state["viewing_historical_experiment_id"] = experiment_id
    except Exception:
        pass


def reset_to_ready_not_run() -> None:
    """Reset the execution state to DATASET_READY (NOT_RUN) for the active dataset."""
    try:
        import streamlit as st
        st.session_state["execution_state"] = ExecutionState.DATASET_READY
        st.session_state["current_classical_experiment_id"] = None
        st.session_state["current_qaoa_experiment_id"] = None
        st.session_state["current_ibm_experiment_id"] = None
        st.session_state["current_experiment_id"] = None
        st.session_state["current_solver_type"] = None
        st.session_state["viewing_historical_experiment_id"] = None
        st.session_state["navigation_target"] = None
    except Exception:
        pass


def get_completed_qaoa_experiment(
    dataset_id: str,
    active_fingerprint: str,
    conn: sqlite3.Connection,
    instance_fingerprint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Look up the most recent completed QAOA experiment matching this exact dataset and fingerprint.
    Guarantees strict dataset isolation and cryptographic verification.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.experiment_id, e.dataset_id, e.solver_type, e.status, e.parameters_json, e.created_at,
               s.solution_id, s.feasible, s.total_cost, s.runtime_seconds, s.assignment_json, s.verification_json
        FROM experiments e
        JOIN solutions s ON e.experiment_id = s.experiment_id
        WHERE e.dataset_id = ? AND e.solver_type = 'qaoa_p1' AND e.status = 'completed'
        ORDER BY e.created_at DESC
        """,
        (dataset_id,),
    )
    for row in cur.fetchall():
        row_dict = dict(row)
        try:
            params = json.loads(row_dict.get("parameters_json") or "{}")
        except Exception:
            params = {}
        if params.get("dataset_fingerprint") == active_fingerprint:
            if instance_fingerprint and params.get("instance_fingerprint"):
                if params.get("instance_fingerprint") != instance_fingerprint:
                    continue
            return row_dict
    return None


def get_completed_classical_experiment(
    dataset_id: str,
    active_fingerprint: str,
    conn: sqlite3.Connection,
    instance_fingerprint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Look up the most recent completed Classical experiment matching this exact dataset and fingerprint.
    Guarantees strict dataset isolation and cryptographic verification.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.experiment_id, e.dataset_id, e.solver_type, e.status, e.parameters_json, e.created_at,
               s.solution_id, s.feasible, s.total_cost, s.runtime_seconds, s.assignment_json, s.verification_json
        FROM experiments e
        JOIN solutions s ON e.experiment_id = s.experiment_id
        WHERE e.dataset_id = ? AND e.solver_type IN ('classical', 'classical_exact', 'classical_greedy') AND e.status = 'completed'
        ORDER BY e.created_at DESC
        """,
        (dataset_id,),
    )
    for row in cur.fetchall():
        row_dict = dict(row)
        try:
            params = json.loads(row_dict.get("parameters_json") or "{}")
        except Exception:
            params = {}
        if params.get("dataset_fingerprint") == active_fingerprint:
            if instance_fingerprint and params.get("instance_fingerprint"):
                if params.get("instance_fingerprint") != instance_fingerprint:
                    continue
            return row_dict
    return None


def get_completed_ibm_experiment(
    dataset_id: str,
    active_fingerprint: str,
    conn: sqlite3.Connection,
    instance_fingerprint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Look up the most recent completed IBM Quantum Hardware experiment matching this exact dataset and fingerprint.
    Guarantees strict dataset isolation and cryptographic verification.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.experiment_id, e.dataset_id, e.solver_type, e.status, e.parameters_json, e.created_at,
               s.solution_id, s.feasible, s.total_cost, s.runtime_seconds, s.assignment_json, s.verification_json
        FROM experiments e
        JOIN solutions s ON e.experiment_id = s.experiment_id
        WHERE e.dataset_id = ? AND e.solver_type IN ('ibm_quantum', 'qaoa_ibm', 'qaoa_hardware') AND e.status = 'completed'
        ORDER BY e.created_at DESC
        """,
        (dataset_id,),
    )
    for row in cur.fetchall():
        row_dict = dict(row)
        try:
            params = json.loads(row_dict.get("parameters_json") or "{}")
        except Exception:
            params = {}
        if params.get("dataset_fingerprint") == active_fingerprint:
            if instance_fingerprint and params.get("instance_fingerprint"):
                if params.get("instance_fingerprint") != instance_fingerprint:
                    continue
            return row_dict
    return None

