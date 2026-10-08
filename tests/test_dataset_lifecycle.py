"""
ShiftProof — Dataset Lifecycle & Deletion Safety Regression Test Suite

Verifies:
TEST 1: Activating dataset does NOT execute classical or QAOA and displays no result.
TEST 2: "Go to Workspace" retains active dataset in DATASET_READY (NOT_RUN) state.
TEST 3: Deleting inactive dataset leaves active dataset intact, keeps Data page & app.services usable.
TEST 4: Deleting active dataset clears active dataset identity & execution state safely.
TEST 5: Dataset switching (A -> B) displays B in NOT_RUN state without leaking A's results.
TEST 6: Historical QAOA experiment is NOT automatically promoted to current execution.
TEST 7: Browser refresh restores dataset identity but keeps state at NOT_RUN.
TEST 8: Explicit Run QAOA actually executes Qiskit circuit on AerSimulator with counts.
TEST 9: Importing all application modules after dataset deletion succeeds without KeyError.
TEST 10: Attempt deleting arbitrary dataset never deletes or alters app/, src/, tests/.
"""

import sqlite3
import pytest
import importlib
import sys
from pathlib import Path

from app.database.schema import init_db
from app.services.dataset_service import (
    set_active_dataset,
    get_active_dataset,
    get_active_dataset_identity,
    delete_dataset,
    list_datasets,
)
from app.services.ingestion.pipeline import inspect_uploaded_file, finalize_and_save_dataset
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint
from app.services.experiment_service import run_classical_workspace, run_qaoa_workspace
from app.services.execution_state import (
    ExecutionState,
    sync_execution_state,
    reset_to_ready_not_run,
    record_classical_execution,
    record_qaoa_execution,
    set_viewing_historical,
)


_CSV_3X3 = """worker_id,worker_name,shift_id,shift_name,cost,eligible
W0,Alice,S0,Morning,10,1
W0,Alice,S1,Evening,12,1
W0,Alice,S2,Night,15,1
W1,Bob,S0,Morning,11,1
W1,Bob,S1,Evening,10,1
W1,Bob,S2,Night,14,1
W2,Charlie,S0,Morning,13,1
W2,Charlie,S1,Evening,11,1
W2,Charlie,S2,Night,10,1
"""

_CSV_3X3_B = """worker_id,worker_name,shift_id,shift_name,cost,eligible
W0,Diana,S0,ShiftA,20,1
W0,Diana,S1,ShiftB,22,1
W0,Diana,S2,ShiftC,25,1
W1,Edward,S0,ShiftA,21,1
W1,Edward,S1,ShiftB,20,1
W1,Edward,S2,ShiftC,24,1
W2,Fiona,S0,ShiftA,23,1
W2,Fiona,S1,ShiftB,21,1
W2,Fiona,S2,ShiftC,20,1
"""


@pytest.fixture
def lifecycle_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_db(conn)
    yield conn
    conn.close()


def test_1_activate_dataset_no_execution_no_result(lifecycle_db):
    """TEST 1: Upload/register dataset. Activate dataset. Assert no execution, no result."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "test1.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Test 1 Dataset")

    # Activate
    set_active_dataset(ds_id, lifecycle_db)
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)
    state_info = sync_execution_state(ds_id, fp)

    assert get_active_dataset() == ds_id
    assert state_info["state"] == ExecutionState.DATASET_READY
    assert state_info["is_ready_not_run"] is True
    assert state_info["current_classical_experiment_id"] is None
    assert state_info["current_qaoa_experiment_id"] is None
    assert state_info["current_experiment_id"] is None


def test_2_activate_and_navigate_workspace_lifecycle(lifecycle_db):
    """TEST 2: Activate dataset. Click Go to Workspace. Assert state is DATASET_READY, NOT RUN."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "test2.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Test 2 Dataset")

    set_active_dataset(ds_id, lifecycle_db)
    ident = get_active_dataset_identity(lifecycle_db)

    assert ident["dataset_id"] == ds_id
    assert ident["name"] == "Test 2 Dataset"
    assert ident["fingerprint"] is not None

    # Workspace contract check
    fp = ident["fingerprint"]
    state_info = sync_execution_state(ds_id, fp)
    assert state_info["is_ready_not_run"] is True
    assert state_info["current_experiment_id"] is None


def test_3_delete_inactive_dataset_safety(lifecycle_db):
    """TEST 3: Delete inactive dataset. Assert active dataset is unaffected, app.services importable."""
    insp_a = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "a.csv")
    ds_a, _ = finalize_and_save_dataset(lifecycle_db, insp_a, dataset_name="Dataset A")

    insp_b = inspect_uploaded_file(_CSV_3X3_B.encode("utf-8"), "b.csv")
    ds_b, _ = finalize_and_save_dataset(lifecycle_db, insp_b, dataset_name="Dataset B")

    # Activate A
    set_active_dataset(ds_a, lifecycle_db)
    assert get_active_dataset() == ds_a

    # Delete inactive B
    deleted = delete_dataset(ds_b, lifecycle_db)
    assert deleted is True

    # Active dataset A must NOT be affected
    assert get_active_dataset() == ds_a

    # Verify B is gone from DB
    cur = lifecycle_db.cursor()
    cur.execute("SELECT count(*) FROM datasets WHERE dataset_id = ?", (ds_b,))
    assert cur.fetchone()[0] == 0

    # Verify imports
    import app.services
    import app.services.dataset_service
    assert hasattr(app.services.dataset_service, "list_datasets")


def test_4_delete_active_dataset(lifecycle_db):
    """TEST 4: Delete active dataset. Assert dataset removed, active dataset cleared, execution cleared."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "del_active.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Active To Delete")

    set_active_dataset(ds_id, lifecycle_db)
    assert get_active_dataset() == ds_id

    # Delete the active dataset
    deleted = delete_dataset(ds_id, lifecycle_db)
    assert deleted is True

    # Active dataset must now be None
    assert get_active_dataset() is None

    # Execution state must be NO_DATASET
    state_info = sync_execution_state(None)
    assert state_info["state"] == ExecutionState.NO_DATASET

    # Verify DB cleanup
    cur = lifecycle_db.cursor()
    cur.execute("SELECT count(*) FROM datasets WHERE dataset_id = ?", (ds_id,))
    assert cur.fetchone()[0] == 0


def test_5_dataset_switching_workspace_isolation(lifecycle_db):
    """TEST 5: Dataset A active with run. Switch to Dataset B. Assert B = NOT_RUN, A's results isolated."""
    insp_a = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "switch_a.csv")
    ds_a, _ = finalize_and_save_dataset(lifecycle_db, insp_a, dataset_name="Switch A")

    insp_b = inspect_uploaded_file(_CSV_3X3_B.encode("utf-8"), "switch_b.csv")
    ds_b, _ = finalize_and_save_dataset(lifecycle_db, insp_b, dataset_name="Switch B")

    # Run classical on A
    set_active_dataset(ds_a, lifecycle_db)
    inst_a = db_to_instance(ds_a, lifecycle_db)
    res_a = run_classical_workspace(inst_a, ds_a, conn=lifecycle_db)
    record_classical_execution(res_a["experiment_id"])

    # Switch to B
    set_active_dataset(ds_b, lifecycle_db)
    fp_b = compute_dataset_fingerprint(ds_b, lifecycle_db)
    state_info_b = sync_execution_state(ds_b, fp_b)

    assert get_active_dataset() == ds_b
    assert state_info_b["is_ready_not_run"] is True
    assert state_info_b["current_experiment_id"] is None
    assert state_info_b["current_classical_experiment_id"] is None


def test_6_historical_qaoa_not_automatically_current(lifecycle_db):
    """TEST 6: Historical QAOA experiment exists. Activate dataset. Assert historical is not automatically current."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "hist.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Historical QAOA Test")

    inst = db_to_instance(ds_id, lifecycle_db)
    q_res = run_qaoa_workspace(inst, ds_id, shots=128, maxiter=4, seed=42, conn=lifecycle_db)
    hist_exp_id = q_res["experiment_id"]

    # Now activate dataset afresh
    set_active_dataset(ds_id, lifecycle_db)
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)
    state_info = sync_execution_state(ds_id, fp)

    # Must be NOT RUN
    assert state_info["is_ready_not_run"] is True
    assert state_info["current_qaoa_experiment_id"] is None
    assert state_info["current_experiment_id"] is None

    # Only when user explicitly chooses to view historical
    set_viewing_historical(hist_exp_id)
    view_state = sync_execution_state(ds_id, fp)
    assert view_state["is_viewing_historical"] is True
    assert view_state["viewing_historical_experiment_id"] == hist_exp_id


def test_7_refresh_after_activation(lifecycle_db):
    """TEST 7: Refresh after activation. Assert dataset identity is restored, state remains NOT_RUN."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "refresh.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Refresh Test")

    set_active_dataset(ds_id, lifecycle_db)

    # Simulate refresh: read identity from DB
    ident = get_active_dataset_identity(lifecycle_db)
    assert ident["dataset_id"] == ds_id
    assert ident["name"] == "Refresh Test"

    state_info = sync_execution_state(ident["dataset_id"], ident["fingerprint"])
    assert state_info["is_ready_not_run"] is True
    assert state_info["current_experiment_id"] is None


def test_8_explicit_run_qaoa_aer_simulator(lifecycle_db):
    """TEST 8: Explicit Run QAOA. Assert AerSimulator invoked, QuantumCircuit created, measurement counts exist."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "qaoa_run.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="QAOA Execution Test")

    set_active_dataset(ds_id, lifecycle_db)
    inst = db_to_instance(ds_id, lifecycle_db)

    # Explicit QAOA run
    q_res = run_qaoa_workspace(inst, ds_id, shots=256, maxiter=5, seed=42, conn=lifecycle_db)
    record_qaoa_execution(q_res["experiment_id"])

    assert q_res["backend"] == "Qiskit AerSimulator"
    assert q_res["circuit_depth"] > 0
    assert q_res["two_qubit_gate_count"] > 0
    assert len(q_res["qaoa_result"].samples) > 0
    assert q_res["dataset_id"] == ds_id

    # Verify state reflects QAOA_COMPLETE
    fp = compute_dataset_fingerprint(ds_id, lifecycle_db)
    state_info = sync_execution_state(ds_id, fp)
    assert state_info["current_qaoa_experiment_id"] == q_res["experiment_id"]
    assert state_info["current_solver_type"] == "qaoa_p1"


def test_9_import_app_services_after_deletion(lifecycle_db):
    """TEST 9: Import all application modules after dataset deletion. Assert no KeyError: 'app.services'."""
    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "import_check.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Import Test")

    delete_dataset(ds_id, lifecycle_db)

    # Invalidate import caches to test fresh lookup
    importlib.invalidate_caches()

    # Re-import
    import app
    import app.services
    import app.services.dataset_service
    import app.services.execution_state
    import app.services.scheduling_service
    import app.services.experiment_service

    assert "app.services" in sys.modules
    assert "app.services.dataset_service" in sys.modules
    assert hasattr(app.services.dataset_service, "list_datasets")


def test_10_directory_integrity_on_dataset_deletion(lifecycle_db):
    """TEST 10: Attempt deleting arbitrary dataset. Assert no path under app/, src/, tests/ is deleted/altered."""
    root = Path(__file__).parents[1]
    app_dir = root / "app"
    src_dir = root / "src"
    tests_dir = root / "tests"

    app_files_before = set(p.relative_to(root) for p in app_dir.rglob("*.py"))
    src_files_before = set(p.relative_to(root) for p in src_dir.rglob("*.py"))
    tests_files_before = set(p.relative_to(root) for p in tests_dir.rglob("*.py"))

    insp = inspect_uploaded_file(_CSV_3X3.encode("utf-8"), "integ.csv")
    ds_id, _ = finalize_and_save_dataset(lifecycle_db, insp, dataset_name="Integrity Check")

    delete_dataset(ds_id, lifecycle_db)

    app_files_after = set(p.relative_to(root) for p in app_dir.rglob("*.py"))
    src_files_after = set(p.relative_to(root) for p in src_dir.rglob("*.py"))
    tests_files_after = set(p.relative_to(root) for p in tests_dir.rglob("*.py"))

    # Assert no files were deleted
    assert app_files_before == app_files_after
    assert src_files_before == src_files_after
    assert tests_files_before == tests_files_after
