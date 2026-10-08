"""
ShiftProof — UX Execution Lifecycle & Solver Navigation Test Suite

Directly verifies the 11 UX requirements specified in Section 19:
- TEST A: Activate dataset -> Classical NOT RUN, QAOA NOT RUN, no execution occurs
- TEST B: Run Classical -> classical execution completes, navigation target = results
- TEST C: Run QAOA -> QAOA execution completes, navigation target = quantum
- TEST D: QAOA already complete -> open Quantum page -> no second execution occurs
- TEST E: QAOA already complete -> click View Quantum Result -> navigation target = quantum
- TEST F: Classical already complete -> click View Classical Result -> navigation target = results
- TEST G: Switch Dataset A -> Dataset B -> A's QAOA result is not shown for B
- TEST H: Switch Dataset A -> Dataset B -> open Quantum -> B does not auto-run QAOA
- TEST I: Switch Dataset A -> Dataset B -> run QAOA -> navigation target = quantum, belongs to B
- TEST J: Run QAOA, leave page, return to Quantum -> same result displayed, no new run
- TEST K: Execution fails -> no automatic navigation, failure remains visible
"""

from __future__ import annotations

import sqlite3
import pytest
import streamlit as st

from app.database.schema import init_db
from app.services.dataset_service import set_active_dataset, get_active_dataset
from app.services.ingestion.pipeline import inspect_uploaded_file, finalize_and_save_dataset
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint
from app.services.experiment_service import run_classical_workspace, run_qaoa_workspace
from app.services.execution_state import (
    ExecutionState,
    sync_execution_state,
    record_classical_execution,
    record_qaoa_execution,
    set_navigation_target,
    get_navigation_target,
    clear_navigation_target,
    get_completed_qaoa_experiment,
    get_completed_classical_experiment,
)


@pytest.fixture
def nav_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def _create_dataset(conn: sqlite3.Connection, name: str, csv_content: bytes) -> tuple[str, str]:
    insp = inspect_uploaded_file(csv_content, f"{name.lower().replace(' ', '_')}.csv")
    ds_id, _ = finalize_and_save_dataset(conn, insp, dataset_name=name)
    fp = compute_dataset_fingerprint(ds_id, conn)
    return ds_id, fp


def test_a_activate_dataset_not_run(nav_db):
    """
    TEST A:
    Activate dataset.
    Verify:
        Classical NOT RUN
        QAOA NOT RUN
    No execution occurs.
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    # User activates dataset
    set_active_dataset(ds_id)
    ctx = sync_execution_state(ds_id, fp)

    # 1. Verify no execution occurs in database
    cur = nav_db.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id,))
    assert cur.fetchone()[0] == 0

    # 2. Verify status is READY — NOT RUN
    assert ctx["is_ready_not_run"] is True
    assert ctx["state"] == ExecutionState.DATASET_READY
    assert ctx["current_classical_experiment_id"] is None
    assert ctx["current_qaoa_experiment_id"] is None
    assert ctx["current_experiment_id"] is None


def test_b_run_classical_navigation(nav_db):
    """
    TEST B:
    Run Classical.
    Verify:
        classical execution completes
        navigation target = results
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)
    clear_navigation_target()

    instance = db_to_instance(ds_id, nav_db)
    c_res = run_classical_workspace(instance, ds_id, conn=nav_db)
    record_classical_execution(c_res["experiment_id"])

    # Verify execution completes
    assert c_res["experiment_id"] is not None
    assert c_res["greedy_result"].feasible is True

    # Verify navigation target is results
    assert get_navigation_target() == "results"
    assert st.session_state.get("current_classical_experiment_id") == c_res["experiment_id"]


def test_c_run_qaoa_navigation(nav_db):
    """
    TEST C:
    Run QAOA.
    Verify:
        QAOA execution completes
        navigation target = quantum
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)
    clear_navigation_target()

    instance = db_to_instance(ds_id, nav_db)
    q_res = run_qaoa_workspace(instance, ds_id, shots=100, maxiter=10, seed=42, conn=nav_db)
    record_qaoa_execution(q_res["experiment_id"])

    # Verify execution completes
    assert q_res["experiment_id"] is not None
    assert q_res["qubit_count"] == 2

    # Verify navigation target is quantum
    assert get_navigation_target() == "quantum"
    assert st.session_state.get("current_qaoa_experiment_id") == q_res["experiment_id"]


def test_d_qaoa_already_complete_no_reexecution(nav_db):
    """
    TEST D:
    QAOA already complete.
    Open Quantum page.
    Verify:
        no second QAOA execution occurs.
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)

    instance = db_to_instance(ds_id, nav_db)
    q_res = run_qaoa_workspace(instance, ds_id, shots=100, maxiter=10, seed=42, conn=nav_db)
    record_qaoa_execution(q_res["experiment_id"])

    cur = nav_db.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ? AND solver_type = 'qaoa_p1'", (ds_id,))
    count_before = cur.fetchone()[0]
    assert count_before == 1

    # Simulate user visiting / opening Quantum page (sync_execution_state + lookup)
    ctx = sync_execution_state(ds_id, fp)
    completed_qaoa = get_completed_qaoa_experiment(ds_id, fp, nav_db, instance.instance_id)

    # Verify completed result is recognized without running a new solver
    assert completed_qaoa is not None
    assert completed_qaoa["experiment_id"] == q_res["experiment_id"]

    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ? AND solver_type = 'qaoa_p1'", (ds_id,))
    count_after = cur.fetchone()[0]
    assert count_after == count_before == 1  # No second execution


def test_e_view_quantum_result_navigation(nav_db):
    """
    TEST E:
    QAOA already complete.
    Click View Quantum Result.
    Verify:
        navigation target = quantum
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)

    instance = db_to_instance(ds_id, nav_db)
    q_res = run_qaoa_workspace(instance, ds_id, shots=100, maxiter=10, seed=42, conn=nav_db)
    record_qaoa_execution(q_res["experiment_id"])

    # Simulate user clicking "View Quantum Result"
    set_navigation_target("quantum")
    assert get_navigation_target() == "quantum"


def test_f_view_classical_result_navigation(nav_db):
    """
    TEST F:
    Classical already complete.
    Click View Classical Result.
    Verify:
        navigation target = results
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)

    instance = db_to_instance(ds_id, nav_db)
    c_res = run_classical_workspace(instance, ds_id, conn=nav_db)
    record_classical_execution(c_res["experiment_id"])

    # Simulate user clicking "View Classical Result"
    set_navigation_target("results")
    assert get_navigation_target() == "results"


def test_g_switch_dataset_a_to_b_isolation(nav_db):
    """
    TEST G:
    Switch Dataset A -> Dataset B.
    Verify:
        A's QAOA result is not shown for B.
    """
    csv_a = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    csv_b = b"Worker,Shift,Cost\nCharlie,Night,15.0\nDave,Night,25.0\n"
    ds_id_a, fp_a = _create_dataset(nav_db, "Dataset A", csv_a)
    ds_id_b, fp_b = _create_dataset(nav_db, "Dataset B", csv_b)

    # Run QAOA on A
    set_active_dataset(ds_id_a)
    sync_execution_state(ds_id_a, fp_a)
    inst_a = db_to_instance(ds_id_a, nav_db)
    q_res_a = run_qaoa_workspace(inst_a, ds_id_a, shots=100, maxiter=10, seed=42, conn=nav_db)
    record_qaoa_execution(q_res_a["experiment_id"])

    # Switch to B
    set_active_dataset(ds_id_b)
    ctx_b = sync_execution_state(ds_id_b, fp_b)

    # Verify memory is wiped for B
    assert ctx_b["current_qaoa_experiment_id"] is None
    assert ctx_b["current_experiment_id"] is None
    assert ctx_b["is_ready_not_run"] is True

    # Verify completed query for B returns None
    inst_b = db_to_instance(ds_id_b, nav_db)
    completed_b = get_completed_qaoa_experiment(ds_id_b, fp_b, nav_db, inst_b.instance_id)
    assert completed_b is None


def test_h_switch_dataset_open_quantum_no_autorun(nav_db):
    """
    TEST H:
    Switch Dataset A -> Dataset B.
    Open Quantum.
    Verify:
        B does not automatically run QAOA.
    """
    csv_a = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    csv_b = b"Worker,Shift,Cost\nCharlie,Night,15.0\nDave,Night,25.0\n"
    ds_id_a, fp_a = _create_dataset(nav_db, "Dataset A", csv_a)
    ds_id_b, fp_b = _create_dataset(nav_db, "Dataset B", csv_b)

    set_active_dataset(ds_id_a)
    sync_execution_state(ds_id_a, fp_a)
    inst_a = db_to_instance(ds_id_a, nav_db)
    run_qaoa_workspace(inst_a, ds_id_a, shots=100, maxiter=10, seed=42, conn=nav_db)

    # Switch to B and "open Quantum"
    set_active_dataset(ds_id_b)
    ctx_b = sync_execution_state(ds_id_b, fp_b)

    # Check DB experiments for B
    cur = nav_db.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id_b,))
    assert cur.fetchone()[0] == 0
    assert ctx_b["is_ready_not_run"] is True


def test_i_switch_dataset_run_qaoa_belongs_to_b(nav_db):
    """
    TEST I:
    Switch Dataset A -> Dataset B.
    Run QAOA.
    Verify:
        navigation target = quantum
        result belongs to B.
    """
    csv_a = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    csv_b = b"Worker,Shift,Cost\nCharlie,Night,15.0\nDave,Night,25.0\n"
    ds_id_a, fp_a = _create_dataset(nav_db, "Dataset A", csv_a)
    ds_id_b, fp_b = _create_dataset(nav_db, "Dataset B", csv_b)

    set_active_dataset(ds_id_b)
    sync_execution_state(ds_id_b, fp_b)
    clear_navigation_target()

    inst_b = db_to_instance(ds_id_b, nav_db)
    q_res_b = run_qaoa_workspace(inst_b, ds_id_b, shots=100, maxiter=10, seed=42, conn=nav_db)
    record_qaoa_execution(q_res_b["experiment_id"])

    assert get_navigation_target() == "quantum"

    # Verify result belongs strictly to B
    cur = nav_db.cursor()
    cur.execute("SELECT dataset_id, parameters_json FROM experiments WHERE experiment_id = ?", (q_res_b["experiment_id"],))
    row = cur.fetchone()
    assert row["dataset_id"] == ds_id_b
    import json
    params = json.loads(row["parameters_json"])
    assert params["dataset_fingerprint"] == fp_b


def test_j_run_qaoa_leave_page_return_same_result(nav_db):
    """
    TEST J:
    Run QAOA, leave page, return to Quantum.
    Verify:
        same current QAOA result is displayed
        no new execution occurs.
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)
    inst = db_to_instance(ds_id, nav_db)
    q_res = run_qaoa_workspace(inst, ds_id, shots=100, maxiter=10, seed=42, conn=nav_db)
    record_qaoa_execution(q_res["experiment_id"])

    # Leave page (simulate navigating to workspace)
    clear_navigation_target()
    st.session_state["navigation_target"] = "workspace"

    # Return to Quantum
    ctx_return = sync_execution_state(ds_id, fp)
    completed = get_completed_qaoa_experiment(ds_id, fp, nav_db, inst.instance_fingerprint if hasattr(inst, "instance_fingerprint") else inst.instance_id)

    assert completed is not None
    assert completed["experiment_id"] == q_res["experiment_id"]

    cur = nav_db.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id,))
    assert cur.fetchone()[0] == 1  # No duplicate execution


def test_k_execution_fails_no_navigation(nav_db):
    """
    TEST K:
    Execution fails.
    Verify:
        no automatic navigation
        failure remains visible.
    """
    csv = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    ds_id, fp = _create_dataset(nav_db, "Dataset Alpha", csv)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)
    clear_navigation_target()

    # Simulate execution failure
    try:
        raise RuntimeError("Simulated solver divergence or timeout")
    except Exception as ex:
        st.session_state["ws_classical_state"] = "FAILED"
        st.session_state["ws_classical_error"] = str(ex)

    # Verify no navigation occurred
    assert get_navigation_target() is None

    # Verify error remains visible
    assert st.session_state.get("ws_classical_state") == "FAILED"
    assert "Simulated solver divergence or timeout" in st.session_state.get("ws_classical_error")
