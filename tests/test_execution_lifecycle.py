"""
ShiftProof — Execution Lifecycle & State Separation Test Suite

Verifies strict architectural separation:
DATASET ACTIVATION ≠ EXPERIMENT EXECUTION

Covers:
TEST A: Activate dataset -> NOT RUN (no auto-execution, no results displayed)
TEST B: Activate dataset -> Explicit Run Classical -> executes and belongs to current dataset
TEST C: Activate dataset -> Explicit Run QAOA -> executes, AerSimulator backend, measurement counts exist
TEST D: Activate A, run QAOA, switch to B -> A's result does not bleed, B is NOT_RUN
TEST E: Refresh after activating dataset -> restores identity, state is NOT_RUN, no silent reuse
TEST F: Historical experiment exists -> activating dataset does NOT automatically load it as current
TEST G: No current experiment exists -> no demo, no benchmark, no global latest result, no fabrication
"""

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
    set_viewing_historical,
)


@pytest.fixture
def lifecycle_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def test_a_activate_dataset_not_run(lifecycle_db):
    """
    TEST A:
    Activate dataset.
    Assert:
    - no experiment executed
    - no classical result displayed
    - no QAOA result displayed
    - status is READY — NOT RUN
    """
    csv_bytes = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    insp = inspect_uploaded_file(csv_bytes, "team_shifts.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Team Shifts")
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)

    # User activates dataset
    set_active_dataset(ds_id)
    ctx = sync_execution_state(ds_id, fp)

    # Assert no experiment executed in DB
    cur = lifecycle_db.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id,))
    assert cur.fetchone()[0] == 0

    # Assert execution state is READY — NOT RUN
    assert ctx["is_ready_not_run"] is True
    assert ctx["state"] == ExecutionState.DATASET_READY
    assert ctx["current_experiment_id"] is None
    assert ctx["current_classical_experiment_id"] is None
    assert ctx["current_qaoa_experiment_id"] is None
    assert ctx["is_viewing_historical"] is False


def test_b_explicit_run_classical(lifecycle_db):
    """
    TEST B:
    Activate dataset, then click Run Classical.
    Assert:
    - classical experiment executes
    - result belongs to current dataset
    """
    csv_bytes = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    insp = inspect_uploaded_file(csv_bytes, "team_shifts.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Team Shifts")
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)

    # User explicitly clicks Run Classical
    instance = db_to_instance(ds_id, lifecycle_db)
    c_res = run_classical_workspace(instance, ds_id, conn=lifecycle_db)
    exp_id = c_res["experiment_id"]
    record_classical_execution(exp_id)

    ctx = sync_execution_state(ds_id, fp)
    assert ctx["is_ready_not_run"] is False
    assert ctx["current_classical_experiment_id"] == exp_id
    assert ctx["current_experiment_id"] == exp_id
    assert ctx["state"] == ExecutionState.CLASSICAL_COMPLETE

    # Assert experiment and solution belong strictly to ds_id
    cur = lifecycle_db.cursor()
    cur.execute("SELECT dataset_id, solver_type FROM experiments WHERE experiment_id = ?", (exp_id,))
    exp_row = cur.fetchone()
    assert exp_row[0] == ds_id
    assert exp_row[1] == "classical"


def test_c_explicit_run_qaoa(lifecycle_db):
    """
    TEST C:
    Activate dataset, then click Run QAOA.
    Assert:
    - QAOA actually executes
    - result belongs to current dataset
    - Qiskit AerSimulator was used
    - actual measurement counts exist
    """
    csv_bytes = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    insp = inspect_uploaded_file(csv_bytes, "team_shifts.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Team Shifts")
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)

    set_active_dataset(ds_id)
    sync_execution_state(ds_id, fp)

    # User explicitly clicks Run QAOA
    instance = db_to_instance(ds_id, lifecycle_db)
    q_res = run_qaoa_workspace(instance, ds_id, shots=64, maxiter=5, seed=42, conn=lifecycle_db)
    exp_id = q_res["experiment_id"]
    record_qaoa_execution(exp_id)

    ctx = sync_execution_state(ds_id, fp)
    assert ctx["is_ready_not_run"] is False
    assert ctx["current_qaoa_experiment_id"] == exp_id
    assert ctx["current_experiment_id"] == exp_id
    assert ctx["state"] == ExecutionState.QAOA_COMPLETE

    # Verify backend and measurement counts
    assert q_res["backend"] == "Qiskit AerSimulator"
    cur = lifecycle_db.cursor()
    cur.execute("SELECT COUNT(*), SUM(shot_count) FROM measurements WHERE experiment_id = ?", (exp_id,))
    row = cur.fetchone()
    assert row[0] > 0
    assert row[1] == 64


def test_d_dataset_switching_isolation(lifecycle_db):
    """
    TEST D:
    Activate Dataset A.
    Run QAOA on A.
    Switch to Dataset B.
    Assert:
    - Dataset A result is not displayed as Dataset B result
    - Dataset B = NOT_RUN
    """
    csv_a = b"Worker,Shift,Cost\nAlpha,Morning,10.0\nBeta,Morning,20.0\n"
    insp_a = inspect_uploaded_file(csv_a, "dataset_a.csv")
    ds_id_a, _ = finalize_and_save_dataset(lifecycle_db, insp_a, dataset_name="Dataset A")
    fp_a = compute_dataset_fingerprint(ds_id_a, lifecycle_db)

    csv_b = b"Worker,Shift,Cost\nGamma,Morning,50.0\nDelta,Morning,60.0\n"
    insp_b = inspect_uploaded_file(csv_b, "dataset_b.csv")
    ds_id_b, _ = finalize_and_save_dataset(lifecycle_db, insp_b, dataset_name="Dataset B")
    fp_b = compute_dataset_fingerprint(ds_id_b, lifecycle_db)

    # 1. Activate A & Run QAOA
    set_active_dataset(ds_id_a)
    sync_execution_state(ds_id_a, fp_a)
    inst_a = db_to_instance(ds_id_a, lifecycle_db)
    q_res_a = run_qaoa_workspace(inst_a, ds_id_a, shots=64, maxiter=3, seed=42, conn=lifecycle_db)
    record_qaoa_execution(q_res_a["experiment_id"])

    ctx_a = sync_execution_state(ds_id_a, fp_a)
    assert ctx_a["current_qaoa_experiment_id"] == q_res_a["experiment_id"]

    # 2. Switch to Dataset B
    set_active_dataset(ds_id_b)
    ctx_b = sync_execution_state(ds_id_b, fp_b)

    # Dataset B MUST be NOT_RUN
    assert ctx_b["is_ready_not_run"] is True
    assert ctx_b["current_experiment_id"] is None
    assert ctx_b["current_qaoa_experiment_id"] is None
    assert ctx_b["state"] == ExecutionState.DATASET_READY


def test_e_refresh_after_activating_dataset(lifecycle_db):
    """
    TEST E:
    Refresh after only activating a dataset.
    Assert:
    - dataset remains active
    - no new execution occurs
    - no historical result appears as a new result
    """
    csv_bytes = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    insp = inspect_uploaded_file(csv_bytes, "team_shifts.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Team Shifts")
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)

    set_active_dataset(ds_id)

    # Simulate fresh browser session (refresh clears st.session_state)
    st.session_state.clear()
    st.session_state["active_dataset_id"] = ds_id

    ctx = sync_execution_state(ds_id, fp)
    assert ctx["is_ready_not_run"] is True
    assert ctx["current_experiment_id"] is None
    assert ctx["state"] == ExecutionState.DATASET_READY

    # Assert no experiments created
    cur = lifecycle_db.cursor()
    cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id,))
    assert cur.fetchone()[0] == 0


def test_f_historical_experiment_not_automatically_current(lifecycle_db):
    """
    TEST F:
    Historical experiment exists.
    Activate dataset.
    Assert:
    - historical result is not automatically presented as current
    - user must explicitly choose View Historical Experiment
    """
    csv_bytes = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    insp = inspect_uploaded_file(csv_bytes, "team_shifts.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Team Shifts")
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)

    # Pre-populate historical experiment in DB
    inst = db_to_instance(ds_id, lifecycle_db)
    c_res = run_classical_workspace(inst, ds_id, conn=lifecycle_db)
    hist_exp_id = c_res["experiment_id"]

    # Now simulate a user activating the dataset on Results page or refreshing
    st.session_state.clear()
    set_active_dataset(ds_id)
    ctx = sync_execution_state(ds_id, fp)

    # MUST be NOT_RUN for the current session!
    assert ctx["is_ready_not_run"] is True
    assert ctx["current_experiment_id"] is None
    assert ctx["is_viewing_historical"] is False

    # Now user explicitly chooses to view historical experiment
    set_viewing_historical(hist_exp_id)
    ctx_hist = sync_execution_state(ds_id, fp)
    assert ctx_hist["is_viewing_historical"] is True
    assert ctx_hist["viewing_historical_experiment_id"] == hist_exp_id


def test_g_no_current_experiment_no_fallbacks(lifecycle_db):
    """
    TEST G:
    No current experiment exists.
    Assert:
    - no demo result
    - no benchmark result
    - no global latest result
    - no fabricated result
    """
    csv_bytes = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
    insp = inspect_uploaded_file(csv_bytes, "custom_user_data.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Custom User Data")
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)

    set_active_dataset(ds_id)
    ctx = sync_execution_state(ds_id, fp)

    # No current experiment ID
    assert ctx["current_experiment_id"] is None
    assert ctx["current_classical_experiment_id"] is None
    assert ctx["current_qaoa_experiment_id"] is None
    assert ctx["is_ready_not_run"] is True

    # Check database: no results exist for this dataset
    cur = lifecycle_db.cursor()
    cur.execute("SELECT COUNT(*) FROM solutions WHERE dataset_id = ?", (ds_id,))
    assert cur.fetchone()[0] == 0
