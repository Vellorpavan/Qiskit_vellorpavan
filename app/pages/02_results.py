"""
ShiftProof — Page 02: Results & Schedule Verification

Displays actual optimization results, verified assignments, constraint audits,
and cross-solver comparisons for the active dataset.

Strict Architectural Separation:
DATASET ACTIVATION ≠ EXPERIMENT EXECUTION
Activating or refreshing a dataset NEVER automatically executes anything and
NEVER automatically loads historical results as current results.
Results only display upon explicit execution or explicit historical inspection.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px

from app.database.db import get_db_connection
from app.services.dataset_service import list_datasets, get_active_dataset, set_active_dataset
from app.services.scheduling_service import db_to_instance, format_solution_roster, compute_dataset_fingerprint
from app.services.experiment_service import list_dataset_experiments, run_classical_workspace, run_qaoa_workspace
from app.services.execution_state import (
    ExecutionState,
    sync_execution_state,
    record_classical_execution,
    record_qaoa_execution,
    set_viewing_historical,
    get_completed_classical_experiment,
)
from src.model import check_constraints

try:
    st.set_page_config(page_title="Results — ShiftProof", page_icon="📋", layout="wide")
except Exception:
    pass

_CSS = Path(__file__).parents[1] / "styles" / "theme.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

conn = get_db_connection()
datasets = list_datasets(conn)
dataset_ids = [d["dataset_id"] for d in datasets]

active_id = get_active_dataset()
if active_id and active_id not in dataset_ids:
    active_id = None
    set_active_dataset(None)

st.markdown("# 📋 Results & Schedule Verification")
st.caption("Inspect verified schedules, constraint satisfaction audits, and cross-solver benchmarks.")
st.divider()

# ── CASE 1: NO DATASET LOADED ─────────────────────────────────────────────────
if not active_id:
    sync_execution_state(None)
    st.warning("No dataset loaded.")
    st.markdown("Upload or select a workforce dataset to inspect optimization results.")
    col_c1, col_c2 = st.columns([1.5, 3])
    with col_c1:
        if st.button("📁 Go to Data Ingestion", type="primary"):
            st.switch_page("pages/01_data.py")
    if datasets:
        with col_c2:
            choice = st.selectbox(
                "Or activate an existing dataset:",
                options=dataset_ids,
                format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                key="res_no_ds_select",
            )
            if st.button("Activate Dataset", key="res_btn_activate"):
                set_active_dataset(choice)
                st.rerun()
    st.stop()

# ── CASE 2: ACTIVE DATASET LOADED ─────────────────────────────────────────────
cur = conn.cursor()
cur.execute("SELECT * FROM datasets WHERE dataset_id = ?", (active_id,))
ds_record = cur.fetchone()
ds_meta = dict(ds_record) if ds_record else {}
active_name = ds_meta.get("name", active_id)

try:
    instance = db_to_instance(active_id, conn)
    active_fp = compute_dataset_fingerprint(active_id, conn)
    n_vars = instance.n_vars
    qaoa_eligible = (n_vars <= 9)
    opt_mode = ds_meta.get("optimization_mode") or "COST_OPTIMIZATION"
except Exception as e:
    st.error(f"Error loading model for dataset '{active_id}': {e}")
    st.stop()

# Synchronize execution state with strict dataset isolation
exec_ctx = sync_execution_state(active_id, active_fp)

# Active Target Bar
c_top1, c_top2 = st.columns([3, 1])
with c_top1:
    st.markdown(f"### Active Target: **{active_name}**")
    st.caption(f"Dataset ID: `{active_id}` | SHA-256 Fingerprint: `{active_fp}` | Mode: `{opt_mode}`")
with c_top2:
    if len(datasets) > 1:
        switched = st.selectbox(
            "Switch Target",
            options=dataset_ids,
            index=dataset_ids.index(active_id) if active_id in dataset_ids else 0,
            format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
            label_visibility="collapsed",
            key="res_ds_switcher",
        )
        if switched != active_id:
            set_active_dataset(switched)
            st.rerun()

# Fetch all historical experiments for active dataset
all_experiments = list_dataset_experiments(active_id, conn)

# ── 1. DETERMINE TARGET EXPERIMENT TO DISPLAY ─────────────────────────────────
target_exp_id = None
is_historical = False

if exec_ctx["is_viewing_historical"]:
    target_exp_id = exec_ctx["viewing_historical_experiment_id"]
    is_historical = True
elif exec_ctx["current_classical_experiment_id"]:
    target_exp_id = exec_ctx["current_classical_experiment_id"]
    is_historical = False
elif exec_ctx["current_experiment_id"] and exec_ctx.get("current_solver_type") in ("classical", "classical_exact", "classical_greedy"):
    target_exp_id = exec_ctx["current_experiment_id"]
    is_historical = False
else:
    # Check if a completed classical run already exists for this exact dataset & fingerprint
    completed_c = get_completed_classical_experiment(active_id, active_fp, conn, instance.instance_id)
    if completed_c:
        target_exp_id = completed_c["experiment_id"]
        is_historical = False
        st.session_state["current_classical_experiment_id"] = target_exp_id
        st.session_state["current_experiment_id"] = target_exp_id
        st.session_state["ws_classical_state"] = "COMPLETED"

# If NO classical execution result exists for this dataset
if not target_exp_id:
    st.divider()
    st.subheader("EXECUTION STATUS")
    st.info("No classical execution result for this dataset yet.")

    es_c1, es_c2, es_c3 = st.columns(3)
    es_c1.metric("Execution Status", "READY — NOT RUN")
    es_c2.metric("Classical", "Not run")
    es_c3.metric("QAOA", "Not run")

    st.markdown("#### Run Classical Optimization on Active Dataset")
    st.caption("Click to trigger explicit classical optimization using the active dataset.")
    if st.button("▶ Run Classical Optimization", type="primary", use_container_width=True):
        with st.spinner("Executing Classical Optimization (Greedy & Exact)..."):
            try:
                c_res = run_classical_workspace(instance, active_id, conn=conn)
                record_classical_execution(c_res["experiment_id"])
                st.session_state["ws_classical_state"] = "COMPLETED"
                st.rerun()
            except Exception as ex:
                st.error(f"Classical optimization error: {ex}")

    # Show Historical Executions if any exist
    if all_experiments:
        st.divider()
        st.subheader(f"📜 HISTORICAL EXECUTIONS ({len(all_experiments)} past runs recorded)")
        st.caption("Past experiments recorded in database for this dataset. Click to view a historical run.")

        hist_df_rows = []
        for e in all_experiments:
            sol_id = e.get("solution_id", "")
            method_lbl = "Classical Greedy" if "greedy" in sol_id else ("Classical Exact" if "exact" in sol_id else ("Quantum QAOA" if "qaoa" in sol_id else e.get("solver_type", "")))
            c_val = f"${e.get('total_cost'):.2f}" if e.get("total_cost") is not None else ("Feasible" if e.get("feasible") else "Infeasible")
            hist_df_rows.append({
                "Experiment ID": e.get("experiment_id"),
                "Method": method_lbl,
                "Result": c_val,
                "Status": "✓ Feasible" if e.get("feasible") else "✗ Infeasible",
                "Runtime": f"{e.get('runtime_seconds', 0.0)*1000:.1f} ms",
                "Created At": e.get("created_at"),
            })
        st.dataframe(pd.DataFrame(hist_df_rows), use_container_width=True, hide_index=True)

        hist_choices = [e["experiment_id"] for e in all_experiments]
        selected_hist = st.selectbox(
            "Select a historical experiment to inspect:",
            options=hist_choices,
            format_func=lambda eid: f"{eid} — {next((r['Method'] + ' | ' + r['Result'] for r in hist_df_rows if r['Experiment ID'] == eid), eid)}",
            key="hist_select_box",
        )
        if st.button("🔍 View Selected Historical Experiment", type="secondary"):
            set_viewing_historical(selected_hist)
            st.rerun()

    st.stop()

# Fetch the experiment from database strictly scoped to active_id and target_exp_id
cur.execute(
    """
    SELECT e.experiment_id, e.dataset_id, e.solver_type, e.status, e.parameters_json, e.created_at,
           s.solution_id, s.feasible, s.total_cost, s.runtime_seconds, s.assignment_json, s.verification_json
    FROM experiments e
    JOIN solutions s ON e.experiment_id = s.experiment_id
    WHERE e.dataset_id = ? AND e.experiment_id = ?
    ORDER BY s.rowid ASC LIMIT 1
    """,
    (active_id, target_exp_id),
)
selected_exp_row = cur.fetchone()

if not selected_exp_row:
    st.warning("Experiment record not found. Resetting to Ready state.")
    sync_execution_state(active_id, active_fp)
    st.rerun()

selected_exp = dict(selected_exp_row)
exp_id = selected_exp["experiment_id"]

# Verify dataset isolation
if selected_exp.get("dataset_id") != active_id:
    st.error("⚠️ Selected run does not belong to the active dataset.")
    set_active_dataset(active_id)
    st.rerun()

# Extract parameters & verify fingerprint
exp_params = {}
try:
    exp_params = json.loads(selected_exp.get("parameters_json") or "{}")
except Exception:
    exp_params = {}

exp_fp = exp_params.get("dataset_fingerprint")
if exp_fp and exp_fp != active_fp:
    st.error(
        f"⚠️ Result does not belong to the current dataset.\n\n"
        f"• **Dataset Fingerprint:** `{active_fp}`\n"
        f"• **Experiment Fingerprint:** `{exp_fp}`\n\n"
        "The underlying workforce scheduling data has changed since this experiment was executed. "
        "Please rerun optimization."
    )
    st.stop()

# Parse assignment
raw_asgn = selected_exp.get("assignment_json")
assignment_dict = {}
if raw_asgn:
    try:
        parsed = json.loads(raw_asgn)
        for k, v in parsed.items():
            if "," in str(k):
                p = str(k).split(",")
                assignment_dict[(int(p[0]), int(p[1]))] = int(v)
    except Exception:
        assignment_dict = {}

is_feasible, details = check_constraints(instance, assignment_dict)
total_cost = selected_exp.get("total_cost")
runtime_sec = selected_exp.get("runtime_seconds", 0.0)
sol_id = selected_exp.get("solution_id", "")
if "greedy" in sol_id:
    method_str = "Classical Greedy Heuristic"
elif "exact" in sol_id:
    method_str = "Classical Exact Solver"
elif "qaoa" in sol_id:
    method_str = "QAOA Simulation — Qiskit AerSimulator"
else:
    method_str = selected_exp.get("solver_type", "").title()

st.divider()

# ── 3. EXECUTION CONTEXT BANNER ───────────────────────────────────────────────
if is_historical:
    st.warning(
        f"📜 **HISTORICAL EXECUTION RECORD** — Viewing past run `{exp_id}` executed on {selected_exp['created_at']}. "
        "NOTE: This is a historical record. It is NOT the current execution."
    )
    if st.button("⬅️ Return to Current Session View (READY — NOT RUN)", type="secondary"):
        set_viewing_historical(None)
        st.rerun()
else:
    st.success(
        f"✓ **CURRENT SESSION EXECUTION** — Result generated by explicit run `{exp_id}` ({method_str})."
    )

# ── 4. RESULT SUMMARY ─────────────────────────────────────────────────────────
st.subheader("RESULT SUMMARY")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Dataset", active_name)
k2.metric("Method", method_str)
k3.metric("Feasibility", "✓ Feasible" if is_feasible else "✗ Infeasible")
if opt_mode == "FEASIBILITY_ONLY":
    k4.metric("Objective", "Feasibility Only (No Cost)")
else:
    k4.metric("Total Cost", f"${total_cost:.2f}" if total_cost is not None else "Infeasible")

st.caption(
    f"Experiment ID: `{exp_id}` | Dataset Fingerprint: `{active_fp}` | "
    f"Runtime: `{runtime_sec*1000:.2f} ms` ({runtime_sec:.4f} s) | Solution ID: `{sol_id}`"
)
if "qaoa" in sol_id or "qaoa" in selected_exp.get("solver_type", ""):
    st.caption("Hardware Status: IBM Quantum Hardware: Not connected (Simulation executed on Qiskit AerSimulator)")

st.divider()

# ── 5. FINAL SCHEDULE ─────────────────────────────────────────────────────────
st.subheader("FINAL SCHEDULE")
roster_rows = format_solution_roster(instance, assignment_dict, active_id, conn)
df_roster = pd.DataFrame(roster_rows)

assigned_workers = {w for (w, s), val in assignment_dict.items() if val == 1}
n_assigned = len(assigned_workers)
n_unassigned = instance.n_workers - n_assigned

st.caption(
    f"Assigned Workers: **{n_assigned}** | Unassigned Workers: **{n_unassigned}** | "
    f"Total Staff: **{instance.n_workers}** | Shifts Filled: **{n_assigned}/{instance.n_shifts}**"
)

st.dataframe(df_roster, height=280, use_container_width=True, hide_index=True)

c_dl1, c_dl2 = st.columns(2)
with c_dl1:
    csv_bytes = df_roster.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Schedule to CSV",
        data=csv_bytes,
        file_name=f"shiftproof_schedule_{active_id}_{exp_id[:8]}.csv",
        mime="text/csv",
    )
with c_dl2:
    cert = {
        "dataset_id": active_id,
        "dataset_name": active_name,
        "fingerprint": active_fp,
        "experiment_id": exp_id,
        "method": method_str,
        "feasible": is_feasible,
        "total_cost": total_cost,
        "runtime_seconds": runtime_sec,
        "schedule": [r for r in roster_rows if r.get("Status") == "Assigned"],
    }
    st.download_button(
        "📜 Export Audit Certificate (JSON)",
        data=json.dumps(cert, indent=2).encode("utf-8"),
        file_name=f"shiftproof_audit_{active_id}_{exp_id[:8]}.json",
        mime="application/json",
    )

st.divider()

# ── 6. CONSTRAINT AUDIT ───────────────────────────────────────────────────────
st.subheader("CONSTRAINT AUDIT")
cov_ok = all(details.get("shift_coverage", {}).values()) if details else False
cap_ok = all(details.get("worker_at_most_one", {}).values()) if details else False
elig_ok = all(details.get("eligibility", {}).values()) if details else False

ca1, ca2 = st.columns(2)
with ca1:
    st.write("✓ **Every required shift assigned:** Verified" if cov_ok else "✗ **Every required shift assigned:** Violated")
    st.write("✓ **No worker double-booked:** Verified" if cap_ok else "✗ **No worker double-booked:** Violated")
    st.write("✓ **Eligibility respected:** Verified" if elig_ok else "✗ **Eligibility respected:** Violated")
with ca2:
    st.write("✓ **Availability respected:** Verified" if elig_ok else "✗ **Availability respected:** Violated")
    st.write("✓ **Required skills satisfied:** Verified" if elig_ok else "✗ **Required skills satisfied:** Violated")
    st.write("✓ **Capacity respected:** Verified" if cap_ok else "✗ **Capacity respected:** Violated")

st.divider()

# ── 7. METHOD COMPARISON ──────────────────────────────────────────────────────
st.subheader("METHOD COMPARISON")

# Query best runs for active dataset
cur.execute(
    """
    SELECT s.total_cost, s.feasible, s.runtime_seconds, e.experiment_id
    FROM solutions s
    JOIN experiments e ON s.experiment_id = e.experiment_id
    WHERE s.dataset_id = ? AND s.solution_id LIKE 'sol_greedy_%'
    ORDER BY e.created_at DESC LIMIT 1
    """,
    (active_id,),
)
greedy_row = cur.fetchone()

cur.execute(
    """
    SELECT s.total_cost, s.feasible, s.runtime_seconds, e.experiment_id
    FROM solutions s
    JOIN experiments e ON s.experiment_id = e.experiment_id
    WHERE s.dataset_id = ? AND s.solution_id LIKE 'sol_exact_%'
    ORDER BY e.created_at DESC LIMIT 1
    """,
    (active_id,),
)
exact_row = cur.fetchone()

cur.execute(
    """
    SELECT s.total_cost, s.feasible, s.runtime_seconds, e.experiment_id
    FROM solutions s
    JOIN experiments e ON s.experiment_id = e.experiment_id
    WHERE s.dataset_id = ? AND s.solution_id LIKE 'sol_qaoa_%'
    ORDER BY e.created_at DESC LIMIT 1
    """,
    (active_id,),
)
qaoa_row = cur.fetchone()

comp_cols = st.columns(3)

with comp_cols[0]:
    st.markdown("#### Classical Greedy")
    if greedy_row:
        if opt_mode == "FEASIBILITY_ONLY":
            st.write("• **Objective:** Feasibility Only")
        else:
            st.write(f"• **Cost:** `${greedy_row['total_cost']:.2f}`")
        st.write(f"• **Status:** {'✓ Feasible' if greedy_row['feasible'] else '✗ Infeasible'}")
        st.write(f"• **Runtime:** `{greedy_row['runtime_seconds']*1000:.2f} ms`")
    else:
        st.info("Not run")

with comp_cols[1]:
    st.markdown("#### Classical Exact")
    if exact_row:
        if opt_mode == "FEASIBILITY_ONLY":
            st.write("• **Objective:** Feasibility Only")
        else:
            st.write(f"• **Optimal Cost:** `${exact_row['total_cost']:.2f}`")
        st.write(f"• **Status:** {'✓ Feasible' if exact_row['feasible'] else '✗ Infeasible'}")
        st.write(f"• **Runtime:** `{exact_row['runtime_seconds']*1000:.2f} ms`")
    elif instance.n_vars > 16:
        st.caption("Not available (exceeds 16 variable safety limit)")
    else:
        st.info("Not run")

with comp_cols[2]:
    st.markdown("#### Quantum QAOA (p=1)")
    if qaoa_row:
        if qaoa_row["feasible"]:
            if opt_mode == "FEASIBILITY_ONLY":
                st.write("• **Objective:** Feasibility Only")
            else:
                c_str = f"${qaoa_row['total_cost']:.2f}"
                gap_str = ""
                if exact_row and exact_row["total_cost"] is not None:
                    gap = qaoa_row["total_cost"] - exact_row["total_cost"]
                    gap_str = f" (Gap: ${gap:+.2f})"
                st.write(f"• **Best Cost:** `{c_str}`{gap_str}")
            st.write("• **Status:** ✓ Feasible sample observed")
        else:
            st.write("• **Status:** ✗ Infeasible")
            st.write("• **Sample:** No feasible quantum sample observed.")
        st.write(f"• **Runtime:** `{qaoa_row['runtime_seconds']:.3f} s`")
        st.caption("Hardware Status: IBM Quantum Hardware: Not connected (Simulation executed on Qiskit AerSimulator)")
    elif instance.n_vars > 9:
        st.caption("Not available (exceeds 9 variable QAOA demonstrator scale)")
    else:
        st.info("Not run")

# ── 7. ACTIONS ─────────────────────────────────────────────────────────────────
st.divider()
c_act1, c_act2 = st.columns(2)
with c_act1:
    if st.button("↻ Run Classical Again", type="secondary", use_container_width=True):
        with st.spinner("Executing Classical Solvers (Greedy + Exact)..."):
            try:
                c_res = run_classical_workspace(instance, active_id, conn=conn)
                record_classical_execution(c_res["experiment_id"])
                st.session_state["ws_classical_state"] = "COMPLETED"
                st.success("✓ Classical optimization completed!")
                st.rerun()
            except Exception as ex:
                st.error(f"Classical optimization error: {ex}")
with c_act2:
    if st.button("⚛️ Go to Quantum Optimization (QAOA)", type="primary", use_container_width=True):
        st.session_state["navigation_target"] = "quantum"
        st.switch_page("pages/03_quantum.py")

# ── 8. HISTORICAL RUNS EXPANDER ───────────────────────────────────────────────
if all_experiments and len(all_experiments) > 1:
    st.divider()
    with st.expander(f"📜 Other Historical Executions for {active_name} ({len(all_experiments)} total)"):
        other_exp_ids = [e["experiment_id"] for e in all_experiments if e["experiment_id"] != exp_id]
        if other_exp_ids:
            pick_other = st.selectbox(
                "Switch to view another historical execution:",
                options=other_exp_ids,
                key="res_other_hist_pick",
            )
            if st.button("Switch View", key="res_btn_switch_other"):
                set_viewing_historical(pick_other)
                st.rerun()
