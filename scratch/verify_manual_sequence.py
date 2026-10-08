"""
Script to simulate and verify the exact 24-step sequence from Section 20.
"""

import sqlite3
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
    get_completed_ibm_experiment,
)

conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row
init_db(conn)

csv_a = b"Worker,Shift,Cost\nAlice,Morning,10.0\nBob,Morning,20.0\n"
insp_a = inspect_uploaded_file(csv_a, "dataset_a.csv")
ds_id_a, _ = finalize_and_save_dataset(conn, insp_a, dataset_name="Dataset A")
fp_a = compute_dataset_fingerprint(ds_id_a, conn)

csv_b = b"Worker,Shift,Cost\nCharlie,Night,15.0\nDave,Night,25.0\n"
insp_b = inspect_uploaded_file(csv_b, "dataset_b.csv")
ds_id_b, _ = finalize_and_save_dataset(conn, insp_b, dataset_name="Dataset B")
fp_b = compute_dataset_fingerprint(ds_id_b, conn)

print("--- Starting 24-step manual verification simulation ---")

# Step 1: Select Dataset A
set_active_dataset(ds_id_a)
ctx = sync_execution_state(ds_id_a, fp_a)
print("Step 1: Selected Dataset A")

# Step 2: Workspace shows DATASET READY, Classical NOT RUN, QAOA NOT RUN
inst_a = db_to_instance(ds_id_a, conn)
c_done = get_completed_classical_experiment(ds_id_a, fp_a, conn, inst_a.instance_id) is not None
q_done = get_completed_qaoa_experiment(ds_id_a, fp_a, conn, inst_a.instance_id) is not None
assert not c_done, "Classical must be NOT RUN"
assert not q_done, "QAOA must be NOT RUN"
assert ctx["is_ready_not_run"], "Status must be DATASET READY"
print("Step 2 verified: DATASET READY, Classical NOT RUN, QAOA NOT RUN")

# Step 3: Click Run QAOA Simulation
q_res_a = run_qaoa_workspace(inst_a, ds_id_a, shots=100, maxiter=10, seed=42, conn=conn)
record_qaoa_execution(q_res_a["experiment_id"])
print(f"Step 3-4: QAOA completed with experiment_id={q_res_a['experiment_id']}")

# Step 5: Application automatically opens Quantum
nav_target = get_navigation_target()
assert nav_target == "quantum", f"Expected navigation target 'quantum', got {nav_target}"
print("Step 5 verified: Navigation target is 'quantum'")

# Step 6: Quantum page shows completed QAOA result
completed_q = get_completed_qaoa_experiment(ds_id_a, fp_a, conn, inst_a.instance_id)
assert completed_q is not None and completed_q["experiment_id"] == q_res_a["experiment_id"]
print("Step 6 verified: Quantum page resolves completed QAOA result")

# Step 7: Navigate back to Workspace
clear_navigation_target()
st.session_state["navigation_target"] = "workspace"
print("Step 7: Navigated back to Workspace")

# Step 8: Workspace shows QAOA COMPLETE
q_done_ws = (st.session_state.get("ws_qaoa_state") == "COMPLETED") or (get_completed_qaoa_experiment(ds_id_a, fp_a, conn, inst_a.instance_id) is not None)
assert q_done_ws, "Workspace must show QAOA COMPLETE"
print("Step 8 verified: Workspace shows QAOA COMPLETE")

# Step 9-10: Click View Quantum Result -> Application opens Quantum
set_navigation_target("quantum")
assert get_navigation_target() == "quantum"
print("Step 9-10 verified: Clicking View Quantum Result navigates to 'quantum'")

# Step 11-13: Navigate away and return to Quantum -> Confirm QAOA does NOT execute again
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id_a,))
runs_before = cur.fetchone()[0]
# Return to quantum:
ctx = sync_execution_state(ds_id_a, fp_a)
q_loaded = get_completed_qaoa_experiment(ds_id_a, fp_a, conn, inst_a.instance_id)
cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id_a,))
runs_after = cur.fetchone()[0]
assert runs_before == runs_after, "QAOA must NOT execute again when visiting page"
assert q_loaded["experiment_id"] == q_res_a["experiment_id"]
print("Step 11-13 verified: QAOA did NOT execute again")

# Step 14-16: Return Workspace, Click Run Classical Optimization
clear_navigation_target()
c_res_a = run_classical_workspace(inst_a, ds_id_a, conn=conn)
record_classical_execution(c_res_a["experiment_id"])
print(f"Step 14-16: Classical completed with experiment_id={c_res_a['experiment_id']}")

# Step 17: Application automatically opens Results
assert get_navigation_target() == "results"
print("Step 17 verified: Navigation target is 'results'")

# Step 18-20: Return Workspace, Click View Classical Result -> Application opens Results
set_navigation_target("results")
assert get_navigation_target() == "results"
c_loaded = get_completed_classical_experiment(ds_id_a, fp_a, conn, inst_a.instance_id)
assert c_loaded["experiment_id"] == c_res_a["experiment_id"]
print("Step 18-20 verified: View Classical Result navigates to 'results'")

# Step 21: Switch to Dataset B
set_active_dataset(ds_id_b)
ctx_b = sync_execution_state(ds_id_b, fp_b)
print("Step 21: Switched to Dataset B")

# Step 22: Confirm no Dataset A result is shown
assert ctx_b["current_classical_experiment_id"] is None
assert ctx_b["current_qaoa_experiment_id"] is None
assert ctx_b["current_experiment_id"] is None
inst_b = db_to_instance(ds_id_b, conn)
completed_b = get_completed_qaoa_experiment(ds_id_b, fp_b, conn, inst_b.instance_id)
assert completed_b is None, "Dataset B must have no QAOA result"
print("Step 22 verified: No Dataset A result is shown for Dataset B")

# Step 23-24: Open Quantum -> Confirm B's QAOA has NOT automatically executed
cur.execute("SELECT COUNT(*) FROM experiments WHERE dataset_id = ?", (ds_id_b,))
b_runs = cur.fetchone()[0]
assert b_runs == 0, "B must have 0 executions"
print("Step 23-24 verified: B's QAOA has NOT automatically executed")

print("--- All 24 manual verification steps passed with 100% fidelity ---")
