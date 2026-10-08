"""
ShiftProof — Page 00: Interactive Scheduling Workspace

True data-driven scheduling optimization application.
Guides the user from Data Selection -> Understanding -> Validation -> Explicit Optimization (Classical & QAOA)
-> Independent Verification -> Results.

Zero automatic execution on dataset switch. Zero hardcoded results. Zero cross-dataset contamination.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
import streamlit as st

from app.database.db import get_db_connection
from app.services.dataset_service import list_datasets, get_active_dataset, set_active_dataset
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint, format_solution_roster
from app.services.experiment_service import run_classical_workspace, run_qaoa_workspace
from app.services.execution_state import (
    sync_execution_state,
    record_classical_execution,
    record_qaoa_execution,
    set_navigation_target,
    get_completed_classical_experiment,
    get_completed_qaoa_experiment,
)
from app.services.ibm_quantum_service import get_connection_status
from src.model import check_constraints

try:
    st.set_page_config(page_title="ShiftProof — Quantum Scheduling", page_icon="⚡", layout="wide")
except Exception:
    pass

_CSS = Path(__file__).parents[1] / "styles" / "theme.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

conn = get_db_connection()
datasets = list_datasets(conn)
dataset_ids = [d["dataset_id"] for d in datasets]

selected_d_id = get_active_dataset()
if selected_d_id and selected_d_id not in dataset_ids:
    selected_d_id = None
    set_active_dataset(None)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ShiftProof")
st.caption("Constraint-Verified Quantum Scheduling")

# ── 1. NO DEFAULT DATA (Startup State) ────────────────────────────────────────
if not selected_d_id:
    st.divider()
    st.warning("No dataset loaded")
    st.markdown("Upload a workforce scheduling dataset to begin.")

    col_cta1, col_cta2 = st.columns([1.5, 3])
    with col_cta1:
        if st.button("📁 Upload Dataset", type="primary", use_container_width=True):
            st.switch_page("pages/01_data.py")

    if datasets:
        with col_cta2:
            with st.expander("Or select an existing workspace dataset"):
                choice_ds = st.selectbox(
                    "Choose dataset to activate:",
                    options=dataset_ids,
                    format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                    key="ws_startup_select",
                )
                if st.button("Activate Dataset", key="ws_btn_startup_activate"):
                    set_active_dataset(choice_ds)
                    st.rerun()

    st.stop()

# ── 2. ACTIVE DATASET LOADED & VALIDATED ──────────────────────────────────────
cur = conn.cursor()
cur.execute("SELECT * FROM datasets WHERE dataset_id = ?", (selected_d_id,))
ds_record = cur.fetchone()
ds_meta = dict(ds_record) if ds_record else {}
active_name = ds_meta.get("name", selected_d_id)

try:
    instance = db_to_instance(selected_d_id, conn)
    n_workers = instance.n_workers
    n_shifts = instance.n_shifts
    potential_pairs = n_workers * n_shifts
    n_vars = instance.n_vars
    qaoa_eligible = (n_vars <= 9)
    exact_eligible = (n_vars <= 16)
    active_fp = compute_dataset_fingerprint(selected_d_id, conn)
    opt_mode = ds_meta.get("optimization_mode") or ("FEASIBILITY_ONLY" if np.all(instance.cost_matrix == 0.0) else "COST_OPTIMIZATION")
except Exception as e:
    st.error(f"Error loading model for dataset '{selected_d_id}': {e}")
    if st.button("Reset Dataset Selection"):
        set_active_dataset(None)
        st.rerun()
    st.stop()

# ── STATE MANAGEMENT: INVALIDATE STALE RESULTS UPON DATASET / FINGERPRINT CHANGE ──
sync_execution_state(selected_d_id, active_fp)

if (
    st.session_state.get("ws_active_dataset_id") != selected_d_id
    or st.session_state.get("ws_active_fingerprint") != active_fp
):
    st.session_state["ws_active_dataset_id"] = selected_d_id
    st.session_state["ws_active_fingerprint"] = active_fp
    st.session_state["ws_classical_state"] = "NOT_RUN"  # NOT_RUN, RUNNING, COMPLETED, FAILED
    st.session_state["ws_classical_data"] = None
    st.session_state["ws_classical_error"] = None
    st.session_state["ws_qaoa_state"] = "NOT_RUN"       # NOT_RUN, RUNNING, COMPLETED, FAILED, UNAVAILABLE
    st.session_state["ws_qaoa_data"] = None
    st.session_state["ws_qaoa_error"] = None

# Top bar with dataset info & switcher
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    src_label = ds_meta.get("source_filename") or ("Synthetic Demo" if ds_meta.get("is_synthetic") else ds_meta.get("source_type", "upload"))
    st.markdown(f"### Active Dataset: **{active_name}**")
    st.caption(f"Dataset ID: `{selected_d_id}` | SHA-256 Fingerprint: `{active_fp}` | Source: `{src_label}`")
with top_col2:
    if len(datasets) > 1:
        new_active = st.selectbox(
            "Switch Dataset",
            options=dataset_ids,
            index=dataset_ids.index(selected_d_id) if selected_d_id in dataset_ids else 0,
            format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
            label_visibility="collapsed",
            key="ws_active_picker",
        )
        if new_active != selected_d_id:
            set_active_dataset(new_active)
            st.rerun()
    if st.button("📁 Upload Another Dataset", use_container_width=True):
        st.switch_page("pages/01_data.py")

st.divider()

# ── 3. DATASET UNDERSTANDING & PROBLEM SPECIFICATION ─────────────────────────
st.subheader("DATASET UNDERSTANDING")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Workers", n_workers)
m2.metric("Shifts", n_shifts)
m3.metric("Candidate Assignments", potential_pairs)
m4.metric("Variables (Qubits)", n_vars)
m5.metric("Optimization Mode", "Feasibility Only" if opt_mode == "FEASIBILITY_ONLY" else "Cost Optimization")

st.caption(
    f"Pruned **{potential_pairs - n_vars}** ineligible assignments. "
    f"Scheduling model dynamically constructed with **{n_vars}** binary decision variables. "
    f"Mode: **{opt_mode}**."
)

st.divider()

# ── 4. MATHEMATICAL SCHEDULING FORMULATION (I6 MODEL) ─────────────────────────
st.subheader("DETECTED SCHEDULING MODEL")

col_form1, col_form2 = st.columns(2)
with col_form1:
    st.markdown("#### Decision Variable")
    st.markdown(
        "• **$x[w, s] = 1$** if worker $w$ is assigned to shift $s$.\n"
        "• **$x[w, s] = 0$** otherwise."
    )
    st.markdown("#### Objective Function")
    if opt_mode == "FEASIBILITY_ONLY":
        st.markdown(
            "• **Constraint Feasibility Only:** Satisfy all shift coverage and worker capacity limits.\n"
            "• *Mode:* `FEASIBILITY ONLY` (No cost data provided; zero monetary costs fabricated)."
        )
    else:
        st.markdown("• **Minimize Cost:** $C(x) = \\sum_{(w, s)} c[w, s] \\cdot x[w, s]$")

with col_form2:
    st.markdown("#### Constraints Enforced")
    st.markdown(
        "• ✓ **Shift Coverage:** Every required shift receives exactly one worker ($\\sum_w x[w, s] = 1$).\n"
        "• ✓ **Worker Capacity:** No worker assigned to more than 1 shift ($\\sum_s x[w, s] \\le 1$).\n"
        "• ✓ **Eligibility:** Ineligible assignments cannot be selected.\n"
        "• ✓ **Availability:** Unavailable assignments cannot be selected.\n"
        "• ✓ **Required Skills:** Worker qualification must satisfy shift requirements."
    )

st.divider()

# ── 5. OPTIMIZATION ACTIONS (NO AUTOMATIC RUNS) ────────────────────────────────
st.subheader("OPTIMIZATION")

completed_c = get_completed_classical_experiment(selected_d_id, active_fp, conn, instance.instance_id)
completed_q = get_completed_qaoa_experiment(selected_d_id, active_fp, conn, instance.instance_id)
completed_ibm = get_completed_ibm_experiment(selected_d_id, active_fp, conn, instance.instance_id)

c_is_complete = (st.session_state.get("ws_classical_state") == "COMPLETED") or (completed_c is not None)
q_is_complete = (st.session_state.get("ws_qaoa_state") == "COMPLETED") or (completed_q is not None)
ibm_is_complete = (st.session_state.get("current_ibm_experiment_id") is not None) or (completed_ibm is not None)

opt_c1, opt_c2, opt_c3 = st.columns(3)

with opt_c1:
    st.markdown("#### Classical Optimization")
    st.caption("Greedy priority heuristic & exact branch-and-bound solver.")
    if c_is_complete:
        st.success("✓ Classical: **COMPLETE**")
        c_exp_target = (
            st.session_state.get("ws_classical_data", {}).get("experiment_id")
            if st.session_state.get("ws_classical_data")
            else (completed_c["experiment_id"] if completed_c else None)
        )
        if st.button("📋 View Classical Result", type="primary", use_container_width=True):
            if c_exp_target:
                st.session_state["current_classical_experiment_id"] = c_exp_target
                st.session_state["current_experiment_id"] = c_exp_target
            set_navigation_target("results")
            st.switch_page("pages/02_results.py")
        if st.button("↻ Run Classical Again", type="secondary", use_container_width=True):
            st.session_state["ws_classical_state"] = "RUNNING"
            with st.spinner("Executing Classical Solvers (Greedy + Exact)..."):
                try:
                    c_res = run_classical_workspace(instance, selected_d_id, conn=conn)
                    record_classical_execution(c_res["experiment_id"])
                    st.session_state["ws_classical_state"] = "COMPLETED"
                    st.session_state["ws_classical_data"] = {
                        "experiment_id": c_res["experiment_id"],
                        "dataset_fingerprint": active_fp,
                        "greedy_cost": c_res["greedy_result"].optimal_cost if c_res["greedy_result"] else None,
                        "greedy_feasible": c_res["greedy_result"].feasible if c_res["greedy_result"] else False,
                        "greedy_runtime": c_res["greedy_runtime"],
                        "exact_cost": c_res["exact_result"].optimal_cost if c_res.get("exact_result") else None,
                        "exact_feasible": c_res["exact_result"].feasible if c_res.get("exact_result") else False,
                        "exact_runtime": c_res.get("exact_runtime"),
                        "assignment": c_res["greedy_result"].optimal_assignment if c_res["greedy_result"] else {},
                    }
                    st.session_state["ws_classical_error"] = None
                    st.session_state["latest_experiment_id"] = c_res["experiment_id"]
                    st.switch_page("pages/02_results.py")
                except Exception as ex:
                    st.session_state["ws_classical_state"] = "FAILED"
                    st.session_state["ws_classical_error"] = str(ex)
                    st.error(f"Classical execution failed: {ex}")
    else:
        st.info("Classical: **NOT RUN**")
        if st.button("▶ Run Classical Optimization", type="primary", use_container_width=True):
            st.session_state["ws_classical_state"] = "RUNNING"
            with st.spinner("Executing Classical Solvers (Greedy + Exact)..."):
                try:
                    c_res = run_classical_workspace(instance, selected_d_id, conn=conn)
                    record_classical_execution(c_res["experiment_id"])
                    st.session_state["ws_classical_state"] = "COMPLETED"
                    st.session_state["ws_classical_data"] = {
                        "experiment_id": c_res["experiment_id"],
                        "dataset_fingerprint": active_fp,
                        "greedy_cost": c_res["greedy_result"].optimal_cost if c_res["greedy_result"] else None,
                        "greedy_feasible": c_res["greedy_result"].feasible if c_res["greedy_result"] else False,
                        "greedy_runtime": c_res["greedy_runtime"],
                        "exact_cost": c_res["exact_result"].optimal_cost if c_res.get("exact_result") else None,
                        "exact_feasible": c_res["exact_result"].feasible if c_res.get("exact_result") else False,
                        "exact_runtime": c_res.get("exact_runtime"),
                        "assignment": c_res["greedy_result"].optimal_assignment if c_res["greedy_result"] else {},
                    }
                    st.session_state["ws_classical_error"] = None
                    st.session_state["latest_experiment_id"] = c_res["experiment_id"]
                    st.switch_page("pages/02_results.py")
                except Exception as ex:
                    st.session_state["ws_classical_state"] = "FAILED"
                    st.session_state["ws_classical_error"] = str(ex)
                    st.error(f"Classical execution failed: {ex}")

with opt_c2:
    st.markdown("#### QAOA Simulation")
    st.caption("QAOA Simulation — Qiskit AerSimulator (1,024 shots, COBYLA optimizer).")
    if q_is_complete:
        st.success("✓ QAOA Simulation: **COMPLETE**")
        q_exp_target = (
            st.session_state.get("ws_qaoa_data", {}).get("experiment_id")
            if st.session_state.get("ws_qaoa_data")
            else (completed_q["experiment_id"] if completed_q else None)
        )
        if st.button("⚛️ View Quantum Result", type="primary", use_container_width=True):
            if q_exp_target:
                st.session_state["current_qaoa_experiment_id"] = q_exp_target
                st.session_state["current_experiment_id"] = q_exp_target
            set_navigation_target("quantum")
            st.switch_page("pages/03_quantum.py")
        if qaoa_eligible:
            if st.button("↻ Run QAOA Again", type="secondary", use_container_width=True):
                st.session_state["ws_qaoa_state"] = "RUNNING"
                with st.spinner("Executing Qiskit QAOA Pipeline (QUBO → Ising → AerSimulator)..."):
                    try:
                        q_res = run_qaoa_workspace(instance, selected_d_id, shots=1024, maxiter=100, seed=42, conn=conn)
                        record_qaoa_execution(q_res["experiment_id"])
                        st.session_state["ws_qaoa_state"] = "COMPLETED"
                        st.session_state["ws_qaoa_data"] = {
                            "experiment_id": q_res["experiment_id"],
                            "dataset_id": selected_d_id,
                            "dataset_fingerprint": active_fp,
                            "instance_fingerprint": instance.instance_id,
                            "feasible": q_res["feasible"],
                            "best_cost": q_res["best_feasible_cost"],
                            "runtime": q_res["runtime_seconds"],
                            "qubit_count": q_res["qubit_count"],
                            "shots": q_res["shots"],
                            "feasible_rate": q_res["feasible_rate"],
                            "optimal_solution_probability": q_res["optimal_solution_probability"],
                            "circuit_depth": q_res["circuit_depth"],
                            "two_qubit_gate_count": q_res["two_qubit_gate_count"],
                            "backend": "Qiskit AerSimulator",
                            "samples": q_res["qaoa_result"].samples,
                        }
                        st.session_state["ws_qaoa_error"] = None
                        st.session_state["latest_experiment_id"] = q_res["experiment_id"]
                        st.switch_page("pages/03_quantum.py")
                    except Exception as ex:
                        st.session_state["ws_qaoa_state"] = "FAILED"
                        st.session_state["ws_qaoa_error"] = str(ex)
                        st.error(f"QAOA simulation failed: {ex}")
    else:
        st.info("QAOA Simulation: **NOT RUN**")
        if qaoa_eligible:
            if st.button("⚡ Run QAOA Simulation", type="primary", use_container_width=True):
                st.session_state["ws_qaoa_state"] = "RUNNING"
                with st.spinner("Executing Qiskit QAOA Pipeline (QUBO → Ising → AerSimulator)..."):
                    try:
                        q_res = run_qaoa_workspace(instance, selected_d_id, shots=1024, maxiter=100, seed=42, conn=conn)
                        record_qaoa_execution(q_res["experiment_id"])
                        st.session_state["ws_qaoa_state"] = "COMPLETED"
                        st.session_state["ws_qaoa_data"] = {
                            "experiment_id": q_res["experiment_id"],
                            "dataset_id": selected_d_id,
                            "dataset_fingerprint": active_fp,
                            "instance_fingerprint": instance.instance_id,
                            "feasible": q_res["feasible"],
                            "best_cost": q_res["best_feasible_cost"],
                            "runtime": q_res["runtime_seconds"],
                            "qubit_count": q_res["qubit_count"],
                            "shots": q_res["shots"],
                            "feasible_rate": q_res["feasible_rate"],
                            "optimal_solution_probability": q_res["optimal_solution_probability"],
                            "circuit_depth": q_res["circuit_depth"],
                            "two_qubit_gate_count": q_res["two_qubit_gate_count"],
                            "backend": "Qiskit AerSimulator",
                            "samples": q_res["qaoa_result"].samples,
                        }
                        st.session_state["ws_qaoa_error"] = None
                        st.session_state["latest_experiment_id"] = q_res["experiment_id"]
                        st.switch_page("pages/03_quantum.py")
                    except Exception as ex:
                        st.session_state["ws_qaoa_state"] = "FAILED"
                        st.session_state["ws_qaoa_error"] = str(ex)
                        st.error(f"QAOA simulation failed: {ex}")
        else:
            st.button("🔒 Run QAOA Simulation (Scale Gate Locked)", disabled=True, use_container_width=True)
            st.caption("QAOA simulation unavailable for this dataset size in the validated demonstrator (> 9 variables). Classical optimization remains available.")

with opt_c3:
    st.markdown("#### IBM Quantum Hardware")
    st.caption("Real quantum processing unit execution via IBM Quantum Platform.")
    if ibm_is_complete:
        st.success("✓ IBM Quantum Hardware: **COMPLETE**")
        if st.button("⚛️ View Quantum Hardware Result", type="primary", use_container_width=True):
            set_navigation_target("quantum")
            st.switch_page("pages/03_quantum.py")
    else:
        st.info("IBM Quantum Hardware: **NOT RUN**")
        if st.button("🌐 Run on IBM Quantum Hardware", use_container_width=True):
            set_navigation_target("quantum")
            st.switch_page("pages/03_quantum.py")

st.divider()

# ── 6. EXECUTION STATUS & RESULTS (DATASET-SCOPED) ──────────────────────────────
st.subheader("EXECUTION STATUS & RESULTS")

c_state = "COMPLETED" if c_is_complete else st.session_state.get("ws_classical_state", "NOT_RUN")
q_state = "COMPLETED" if q_is_complete else st.session_state.get("ws_qaoa_state", "NOT_RUN")
ibm_state = "COMPLETED" if ibm_is_complete else "NOT_RUN"

# Status metrics bar
col_nr1, col_nr2, col_nr3, col_nr4, col_nr5 = st.columns(5)
col_nr1.metric("Dataset", active_name)
col_nr2.metric(
    "Status",
    "DATASET READY" if (not c_is_complete and not q_is_complete and not ibm_is_complete) else "OPTIMIZATION COMPLETE",
)
col_nr3.metric("Classical", "COMPLETE" if c_is_complete else "NOT RUN")
col_nr4.metric("QAOA Simulation", "COMPLETE" if q_is_complete else "NOT RUN")
col_nr5.metric("IBM Hardware", "COMPLETE" if ibm_is_complete else "NOT RUN")

# Initial state message
if not c_is_complete and not q_is_complete and not ibm_is_complete:
    st.info("No execution result for this dataset yet. Dataset status: **DATASET READY**.")

# Errors if failed
if st.session_state.get("ws_classical_state") == "FAILED":
    st.error(f"Classical optimization failed: {st.session_state.get('ws_classical_error')}")
if st.session_state.get("ws_qaoa_state") == "FAILED":
    st.error(f"QAOA simulation failed: {st.session_state.get('ws_qaoa_error')}")

# Display Classical results if completed
if c_is_complete:
    c_data = st.session_state.get("ws_classical_data")
    if not c_data and completed_c:
        # Load from completed_c
        try:
            c_asgn_json = json.loads(completed_c.get("assignment_json") or "{}")
        except Exception:
            c_asgn_json = {}
        c_data = {
            "experiment_id": completed_c["experiment_id"],
            "dataset_fingerprint": active_fp,
            "greedy_cost": completed_c["total_cost"],
            "greedy_feasible": bool(completed_c["feasible"]),
            "greedy_runtime": completed_c.get("runtime_seconds", 0.0),
            "exact_cost": completed_c["total_cost"],
            "exact_feasible": bool(completed_c["feasible"]),
            "exact_runtime": completed_c.get("runtime_seconds", 0.0),
            "assignment": {(int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in c_asgn_json.items() if "," in k},
        }

    if c_data:
        st.markdown("#### Classical Optimization Result")
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            if opt_mode == "FEASIBILITY_ONLY":
                st.metric("Greedy Status", "✓ Feasible" if c_data['greedy_feasible'] else "✗ Infeasible")
                st.write("• **Objective:** Feasibility Only (No Cost)")
            else:
                st.metric("Greedy Total Cost", f"${c_data['greedy_cost']:.2f}" if c_data['greedy_cost'] is not None else "Infeasible")
                st.write(f"• **Status:** {'✓ Feasible' if c_data['greedy_feasible'] else '✗ Infeasible'}")
            st.write(f"• **Runtime:** `{c_data['greedy_runtime']*1000:.2f} ms`")
        with res_c2:
            if c_data.get('exact_cost') is not None:
                if opt_mode == "FEASIBILITY_ONLY":
                    st.metric("Exact Status", "✓ Feasible" if c_data['exact_feasible'] else "✗ Infeasible")
                    st.write("• **Objective:** Feasibility Only (No Cost)")
                else:
                    st.metric("Exact Optimal Cost", f"${c_data['exact_cost']:.2f}")
                    st.write(f"• **Status:** {'✓ Feasible' if c_data['exact_feasible'] else '✗ Infeasible'}")
                st.write(f"• **Runtime:** `{c_data['exact_runtime']*1000:.2f} ms`")
            else:
                st.metric("Exact Solver", "Skipped (>16 vars)")
                st.write("• Exact safety limit: <= 16 vars")
        with res_c3:
            st.metric("Dataset Fingerprint", f"`{active_fp[:10]}...`")
            st.write(f"• **Experiment:** `{c_data['experiment_id'][:12]}`")
            st.write(f"• **Mode:** `{opt_mode}`")

        # Formatted Roster
        st.markdown("##### Final Schedule Roster")
        roster_rows = format_solution_roster(instance, c_data.get("assignment", {}), selected_d_id, conn)
        if roster_rows:
            st.dataframe(pd.DataFrame(roster_rows), height=180, use_container_width=True, hide_index=True)
        st.markdown("")

# Display QAOA results if completed
if q_is_complete:
    q_data = st.session_state.get("ws_qaoa_data")
    if not q_data and completed_q:
        q_params = {}
        try:
            q_params = json.loads(completed_q.get("parameters_json") or "{}")
        except Exception:
            q_params = {}
        q_data = {
            "experiment_id": completed_q["experiment_id"],
            "dataset_id": selected_d_id,
            "dataset_fingerprint": active_fp,
            "instance_fingerprint": instance.instance_id,
            "feasible": bool(completed_q["feasible"]),
            "best_cost": completed_q["total_cost"],
            "runtime": completed_q.get("runtime_seconds", 0.0),
            "qubit_count": q_params.get("variable_count", instance.n_vars),
            "shots": q_params.get("shots", 1024),
            "feasible_rate": q_params.get("feasible_rate", 0.0),
            "optimal_solution_probability": q_params.get("optimal_solution_probability"),
            "circuit_depth": q_params.get("circuit_depth"),
            "two_qubit_gate_count": q_params.get("two_qubit_gate_count"),
            "backend": q_params.get("backend", "Qiskit AerSimulator"),
        }

    if q_data:
        st.markdown("#### QAOA Simulation Result — Qiskit AerSimulator")
        st.caption("Backend: Qiskit AerSimulator · Classical simulation of quantum circuit (Not real hardware)")

        # 1. QAOA Telemetry
        qm1, qm2, qm3, qm4 = st.columns(4)
        with qm1:
            st.metric("Backend", q_data.get("backend", "Qiskit AerSimulator"))
            st.write(f"• **Qubits:** `{q_data['qubit_count']}`")
            st.write(f"• **Shots:** `{q_data['shots']}`")
        with qm2:
            if q_data["feasible"]:
                if opt_mode == "FEASIBILITY_ONLY":
                    st.metric("QAOA Solution", "✓ Feasible Sample")
                    st.write("• **Status:** ✓ Feasible sample observed")
                else:
                    st.metric("Best Feasible Cost", f"${q_data['best_cost']:.2f}" if q_data['best_cost'] is not None else "None")
                    st.write("• **Status:** ✓ Feasible sample observed")
            else:
                st.metric("QAOA Solution", "Infeasible")
                st.write("• **Status:** No feasible quantum sample observed.")
            st.write(f"• **Feasibility Rate:** `{q_data.get('feasible_rate', 0.0)*100:.1f}%`")
        with qm3:
            opt_prob = q_data.get("optimal_solution_probability")
            st.metric(
                "Optimal Solution Prob",
                f"{opt_prob*100:.2f}%" if opt_prob is not None else "N/A"
            )
            st.write(f"• **Circuit Depth:** `{q_data.get('circuit_depth', 'N/A')}`")
            st.write(f"• **Two-Qubit Gates:** `{q_data.get('two_qubit_gate_count', 'N/A')}`")
        with qm4:
            st.metric("Runtime", f"{q_data['runtime']:.3f} s")
            st.write(f"• **Experiment:** `{q_data['experiment_id'][:12]}`")
            st.write(f"• **Dataset Fingerprint:** `{q_data['dataset_fingerprint'][:10]}...`")

        # 2. Comparison with Classical Result
        st.markdown("##### Classical vs QAOA Comparison")
        if opt_mode == "FEASIBILITY_ONLY":
            cc1, cc2, cc3, cc4 = st.columns(4)
            c_feas = (c_is_complete and c_data and (c_data.get('exact_feasible') or c_data.get('greedy_feasible')))
            cc1.metric("Classical", "✓ Feasible" if c_feas else ("Not run" if not c_is_complete else "✗ Infeasible"))
            cc2.metric("QAOA Simulation", "✓ Feasible" if q_data["feasible"] else "✗ Infeasible")
            cc3.metric("Feasibility Rate", f"{q_data.get('feasible_rate', 0.0)*100:.1f}%")
            cc4.metric("Mode", "FEASIBILITY ONLY")
            if c_feas and q_data["feasible"]:
                st.info("ℹ️ **Both Classical and QAOA satisfied all shift coverage and capacity constraints.** (FEASIBILITY ONLY mode · Zero monetary costs fabricated)")
            elif not q_data["feasible"]:
                st.error("❌ **No feasible quantum sample observed.** (No cost fabricated.)")
        else:
            classical_best_cost = None
            if c_data:
                classical_best_cost = c_data.get("exact_cost") if c_data.get("exact_cost") is not None else c_data.get("greedy_cost")

            if classical_best_cost is not None:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Classical Best Cost", f"${classical_best_cost:.2f}")

                if q_data["feasible"] and q_data["best_cost"] is not None:
                    q_cost = q_data["best_cost"]
                    cc2.metric("QAOA Best Feasible Cost", f"${q_cost:.2f}")
                    abs_gap = q_cost - classical_best_cost
                    rel_gap = (abs_gap / classical_best_cost * 100.0) if classical_best_cost > 0 else 0.0
                    cc3.metric("Absolute Gap", f"${abs_gap:+.2f}")
                    cc4.metric("Relative Gap", f"{rel_gap:+.1f}%")

                    if abs_gap <= 1e-4:
                        st.info("ℹ️ **QAOA observed the classical optimum in this simulation.** (Simulation on AerSimulator; no quantum advantage claimed.)")
                    else:
                        st.warning("⚠️ **QAOA found a feasible solution but did not reach the classical optimum in this run.**")
                else:
                    cc2.metric("QAOA Best Feasible Cost", "None")
                    cc3.metric("Absolute Gap", "N/A")
                    cc4.metric("Relative Gap", "N/A")
                    st.error("❌ **No feasible quantum sample observed.** (No cost fabricated.)")
            else:
                st.info("ℹ️ Run Classical Optimization in the workspace to evaluate side-by-side solution quality and optimality gap.")

        st.caption("IBM Quantum Hardware: Distinct physical quantum backend driver.")
        st.markdown("")

# Solver-specific bottom navigation actions
if c_is_complete or q_is_complete or ibm_is_complete:
    st.divider()
    nav_b1, nav_b2 = st.columns(2)
    with nav_b1:
        if c_is_complete:
            if st.button("📋 Open Results Page (View Classical Result & Audit)", type="primary", use_container_width=True):
                set_navigation_target("results")
                st.switch_page("pages/02_results.py")
    with nav_b2:
        if q_is_complete:
            if st.button("⚛️ Open Quantum Page (View QAOA Simulation Dashboard)", type="primary", use_container_width=True):
                set_navigation_target("quantum")
                st.switch_page("pages/03_quantum.py")

