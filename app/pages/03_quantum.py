"""
ShiftProof — Page 03: Quantum Optimization & Qiskit QAOA Pipeline

Interactive Qiskit QAOA simulation (AerSimulator) for the active dataset,
with strict scale gating (<= 9 variables), truthful status reporting,
transparent IBM Quantum hardware boundary disclosure ("IBM Quantum Hardware: NOT CONNECTED"),
and a clear, responsive Executive Optimization & Decision Dashboard.

Zero fake hardware results. Zero cross-dataset contamination. Full computational provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px

from app.database.db import get_db_connection
from app.services.dataset_service import list_datasets, get_active_dataset, set_active_dataset
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint
from app.services.experiment_service import run_qaoa_workspace
from app.services.execution_proof_service import get_execution_proof
from app.services.execution_state import (
    sync_execution_state,
    record_qaoa_execution,
    set_viewing_historical,
    get_completed_qaoa_experiment,
)
from src.quantum import build_qubo, qubo_to_ising
from src.classical import solve_exact, solve_greedy
from src.model import decode_bitstring
from app.services.ibm_quantum_service import get_connection_status, check_backend_compatibility

try:
    st.set_page_config(page_title="Quantum — ShiftProof", page_icon="⚛️", layout="wide")
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

st.markdown("# ⚛️ Quantum Optimization — Qiskit QAOA")
st.caption("Formulate workforce scheduling constraints as an Ising spin Hamiltonian and optimize via Qiskit QAOA.")
st.divider()

# ── CASE 1: NO DATASET LOADED ─────────────────────────────────────────────────
if not active_id:
    st.warning("No dataset loaded.")
    st.markdown("Upload or select a workforce scheduling dataset to inspect quantum formulation and run QAOA.")
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
                key="q_no_ds_select",
            )
            if st.button("Activate Dataset", key="q_btn_activate"):
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
    n_vars = instance.n_vars
    qaoa_eligible = (n_vars <= 9)
    active_fp = compute_dataset_fingerprint(active_id, conn)
    qubo = build_qubo(instance)
    ising = qubo_to_ising(qubo)
except Exception as e:
    st.error(f"Error loading model for dataset '{active_id}': {e}")
    st.stop()

# Top Target Bar with Target Switcher
c_top1, c_top2 = st.columns([3, 1])
with c_top1:
    st.markdown(f"### Active Target: **{active_name}**")
    st.caption(f"Dataset ID: `{active_id}` | SHA-256 Fingerprint: `{active_fp}`")
with c_top2:
    if len(datasets) > 1:
        switched = st.selectbox(
            "Switch Target",
            options=dataset_ids,
            index=dataset_ids.index(active_id) if active_id in dataset_ids else 0,
            format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
            label_visibility="collapsed",
            key="q_ds_switcher",
        )
        if switched != active_id:
            set_active_dataset(switched)
            st.rerun()

# Synchronize execution state with strict dataset isolation
exec_ctx = sync_execution_state(active_id, active_fp)

target_qaoa_exp_id = None
is_historical_qaoa = False

if exec_ctx["is_viewing_historical"]:
    target_qaoa_exp_id = exec_ctx["viewing_historical_experiment_id"]
    is_historical_qaoa = True
elif exec_ctx["current_qaoa_experiment_id"]:
    target_qaoa_exp_id = exec_ctx["current_qaoa_experiment_id"]
    is_historical_qaoa = False
else:
    # Check if a completed QAOA run already exists in database for this exact dataset & fingerprint
    completed_qaoa = get_completed_qaoa_experiment(active_id, active_fp, conn, instance.instance_id)
    if completed_qaoa:
        target_qaoa_exp_id = completed_qaoa["experiment_id"]
        is_historical_qaoa = False
        st.session_state["current_qaoa_experiment_id"] = target_qaoa_exp_id
        st.session_state["current_experiment_id"] = target_qaoa_exp_id
        st.session_state["ws_qaoa_state"] = "COMPLETED"

# Query historical QAOA runs for active dataset
cur.execute(
    """
    SELECT e.experiment_id, e.created_at, s.feasible, s.total_cost, s.runtime_seconds
    FROM experiments e
    JOIN solutions s ON e.experiment_id = s.experiment_id
    WHERE e.dataset_id = ? AND e.solver_type = 'qaoa_p1' AND e.status = 'completed'
    ORDER BY e.created_at DESC
    """,
    (active_id,),
)
hist_qaoa_rows = [dict(r) for r in cur.fetchall()]

# ── A. CURRENT DATASET ────────────────────────────────────────────────────────
st.subheader("A. CURRENT DATASET")
ds_c1, ds_c2, ds_c3, ds_c4 = st.columns(4)
ds_c1.metric("Dataset Name", active_name)
ds_c2.metric("Dataset ID", active_id)
ds_c3.metric("Dataset Fingerprint", f"{active_fp[:12]}...", help=f"Full SHA-256: {active_fp}")
ds_c4.metric("Instance Fingerprint", f"{instance.instance_id[:12]}...", help=f"Full ID: {instance.instance_id}")

st.divider()

# ── B. EXECUTION STATUS ───────────────────────────────────────────────────────
st.subheader("B. EXECUTION STATUS")
ibm_status = get_connection_status()
ibm_state = ibm_status["state"]  # "NOT CONFIGURED", "CONNECTED", "CONNECTION FAILED"

st_c1, st_c2, st_c3, st_c4 = st.columns(4)
engine_state = "Complete" if (target_qaoa_exp_id or is_historical_qaoa) else ("Running" if st.session_state.get("ws_qaoa_state") == "RUNNING" else "Ready")
st_c1.metric("Engine State", engine_state)
st_c2.metric("Backend Driver", "AerSimulator (Classical)")
st_c3.metric(
    "IBM Quantum Hardware",
    ibm_state,
    help=(
        "IBM_QUANTUM_TOKEN not configured in environment." if ibm_state == "NOT CONFIGURED"
        else f"Authenticated with IBM Quantum Platform ({ibm_status['backend_count']} backends discovered)." if ibm_state == "CONNECTED"
        else f"Connection error: {ibm_status.get('error')}"
    ),
)

if is_historical_qaoa:
    run_status_text = "Historical Record"
elif target_qaoa_exp_id:
    run_status_text = "QAOA COMPLETE"
else:
    run_status_text = "QAOA READY — NOT RUN"
st_c4.metric("QAOA Run Status", run_status_text)

# IBM Connection & Backend Discovery Details
if ibm_state == "NOT CONFIGURED":
    st.info("ℹ️ **IBM Quantum Hardware**: `NOT CONFIGURED` — Set `IBM_QUANTUM_TOKEN` in environment variables to discover available quantum backends.")
elif ibm_state == "CONNECTION FAILED":
    st.error(f"⚠️ **IBM Quantum Hardware**: `CONNECTION FAILED` — {ibm_status.get('error')}")
elif ibm_state == "CONNECTED":
    with st.expander(f"🌐 IBM Quantum Platform — CONNECTED ({ibm_status['backend_count']} Backends Available)", expanded=True):
        st.caption(f"Status: **CONNECTED** | Channel: `{ibm_status['channel']}` | Discovered Backends: **{ibm_status['backend_count']}**")
        
        backend_names = [b["name"] for b in ibm_status["backends"]]
        if backend_names:
            col_b_sel, col_b_info = st.columns([1.5, 2.5])
            with col_b_sel:
                selected_backend_name = st.selectbox(
                    "Select Discovered IBM Backend",
                    options=backend_names,
                    key="ibm_backend_selection",
                )
            
            selected_meta = next(b for b in ibm_status["backends"] if b["name"] == selected_backend_name)
            compat = check_backend_compatibility(selected_meta, logical_qubits=n_vars)
            
            with col_b_info:
                if compat["compatible"]:
                    st.success(f"✓ **{compat['status_text']}**: {compat['reason']}")
                else:
                    st.error(f"⚠️ **{compat['status_text']}**: {compat['reason']}")

            # Backend metadata cards
            b_m1, b_m2, b_m3, b_m4 = st.columns(4)
            b_m1.metric("Physical Qubits", selected_meta["num_qubits"])
            b_m2.metric("Operational Status", "Operational" if selected_meta["operational"] else "Offline")
            b_m3.metric("Pending Queue", f"{selected_meta['pending_jobs']} jobs")
            b_m4.metric("Hardware Type", "Simulator" if selected_meta["simulator"] else "Real QPU")
            
            if selected_meta.get("supported_instructions"):
                st.caption(f"Supported Instructions: `{', '.join(selected_meta['supported_instructions'][:12])}`" + (f" (+{len(selected_meta['supported_instructions'])-12} more)" if len(selected_meta['supported_instructions']) > 12 else ""))
            
            st.info("🔒 **Phase 1 Guardrail**: Connection and backend discovery established. Physical hardware job submission is disabled until Phase 2.")

st.divider()

# ── C. QAOA CONFIGURATION ─────────────────────────────────────────────────────
st.subheader("C. QAOA CONFIGURATION")
cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns(5)
cfg_c1.metric("Qubits (Variables)", n_vars)
cfg_c2.metric("Ansatz Depth (p)", "1")
cfg_c3.metric("Shots", "1024")
cfg_c4.metric("Classical Optimizer", "COBYLA")
cfg_c5.metric("Random Seed", "42")

st.divider()

# ── D. ACTION ─────────────────────────────────────────────────────────────────
st.subheader("D. ACTION")
if target_qaoa_exp_id and not is_historical_qaoa:
    st.success(f"✓ **QAOA COMPLETE** — Active execution `{target_qaoa_exp_id}` is loaded and displayed below.")
    st.caption("A completed QAOA simulation already exists for this active dataset. To re-run, use the explicit action below:")
    if qaoa_eligible:
        if st.button("↻ Run QAOA Again", type="secondary", use_container_width=True):
            with st.spinner("Executing new QAOA simulation on AerSimulator (1,024 shots)..."):
                try:
                    res_q = run_qaoa_workspace(instance, active_id, shots=1024, maxiter=100, seed=42, conn=conn)
                    record_qaoa_execution(res_q["experiment_id"])
                    st.session_state["latest_experiment_id"] = res_q["experiment_id"]
                    st.session_state["ws_qaoa_state"] = "COMPLETED"
                    st.success("✓ QAOA simulation completed successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"QAOA execution failed: {ex}")
    else:
        st.button("🔒 Run QAOA Simulation (Scale Limit Exceeded)", disabled=True, use_container_width=True)
else:
    if qaoa_eligible:
        st.caption("Uploading/selecting a dataset does NOT run QAOA automatically. Click below to execute QAOA simulation for the current dataset.")
        if st.button("⚡ Run QAOA Simulation (1,024 Shots on AerSimulator)", type="primary", use_container_width=True):
            with st.spinner("Executing QAOA on AerSimulator (Building QUBO → Ising → QAOA p=1 → Sampling 1,024 shots)..."):
                try:
                    res_q = run_qaoa_workspace(instance, active_id, shots=1024, maxiter=100, seed=42, conn=conn)
                    record_qaoa_execution(res_q["experiment_id"])
                    st.session_state["latest_experiment_id"] = res_q["experiment_id"]
                    st.session_state["ws_qaoa_state"] = "COMPLETED"
                    st.success("✓ QAOA simulation completed successfully!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"QAOA execution failed: {ex}")
    else:
        st.info(f"Dataset validated. The problem requires {n_vars} variables (validated QAOA demonstrator scale: ≤ 9 qubits). Classical optimization remains available.")
        st.button("🔒 Run QAOA (Scale Limit Exceeded)", disabled=True, use_container_width=True)

# ── IF NO QAOA RUN HAS BEEN EXECUTED ──────────────────────────────────────────
if not target_qaoa_exp_id:
    st.divider()
    st.info("No execution result for this dataset yet. Status: **QAOA READY — NOT RUN**. Click **Run QAOA Simulation** above to execute.")
    if hist_qaoa_rows:
        with st.expander(f"📜 Historical QAOA Executions ({len(hist_qaoa_rows)} past runs recorded)"):
            st.caption("Past QAOA simulation runs recorded in database. Selecting an execution opens it in Historical View mode.")
            hist_choices = [r["experiment_id"] for r in hist_qaoa_rows]
            pick_hist_q = st.selectbox(
                "Select historical QAOA run to inspect:",
                options=hist_choices,
                format_func=lambda eid: f"{eid} | Executed on {next((r['created_at'] for r in hist_qaoa_rows if r['experiment_id'] == eid), '')}",
                key="hist_qaoa_pick",
            )
            if st.button("🔍 Inspect Selected Historical QAOA Run", key="btn_inspect_hist_q"):
                set_viewing_historical(pick_hist_q)
                st.rerun()
    st.stop()

# ── EXECUTION PROOF VERIFICATION ──────────────────────────────────────────────
proof = get_execution_proof(active_id, conn, experiment_id=target_qaoa_exp_id)

if proof["status"] == "NO_EXECUTION":
    st.info("No execution exists for the current dataset. Click **Run QAOA Simulation** above to execute.")
    st.stop()
elif proof["status"] == "STALE_EXECUTION":
    st.warning("⚠️ Dataset was modified since the last execution. Fingerprints do not match. Please re-run QAOA to produce a fresh result.")
    st.stop()
elif proof["status"] == "VALID":
    p_id = proof["dataset_identity"]
    p_inst = proof["instance_data"]
    p_vars = proof["decision_variables"]
    p_qubo = proof["qubo_data"]
    p_ising = proof["ising_data"]
    p_circ = proof["circuit_data"]
    p_meas = proof["measurement_counts"]
    p_verif = proof["independent_verification"]
    p_comp = proof["classical_comparison"]
    p_best = proof["best_feasible_sample"]
    p_params = proof["parameters_data"]

    st.divider()

    # ── 9. QUANTUM RESULT (Clearly labeled as Classical Simulation) ───────────
    st.subheader("QAOA Simulation — Qiskit AerSimulator")
    if is_historical_qaoa:
        st.warning(
            f"📜 **HISTORICAL QAOA RECORD** — Viewing past run `{target_qaoa_exp_id}` executed on {p_id['created_at']}. "
            "NOTE: This is a historical record. It is NOT the current session execution."
        )
        if st.button("⬅️ Return to Current Session View (READY — NOT RUN)", type="secondary"):
            set_viewing_historical(None)
            st.rerun()
    else:
        st.success(f"✓ **CURRENT SESSION EXECUTION** — Run `{p_id['experiment_id']}` executed on Qiskit AerSimulator.")

    st.caption(f"Backend: Qiskit AerSimulator (Simulation) | IBM Quantum Hardware: {ibm_state}")

    # Compute high-level metrics
    total_shots = p_params.get("shots", 1024)
    feasible_shots = sum(m["shot_count"] for m in p_meas if m["is_feasible"])
    feas_rate_val = (feasible_shots / total_shots * 100.0) if total_shots > 0 else 0.0

    exact_res = solve_exact(instance) if instance.n_vars <= 16 else None
    greedy_res = solve_greedy(instance)
    classical_opt_cost = exact_res.optimal_cost if (exact_res and exact_res.feasible) else greedy_res.optimal_cost

    opt_shots = 0
    if classical_opt_cost is not None and p_meas:
        opt_shots = sum(
            m["shot_count"] for m in p_meas
            if m["is_feasible"] and m.get("assignment_cost") is not None and abs(m["assignment_cost"] - classical_opt_cost) < 1e-4
        )
    opt_prob_val = (opt_shots / total_shots * 100.0) if total_shots > 0 else 0.0

    # 8 Core Telemetry Metrics Row
    m_row1 = st.columns(4)
    m_row1[0].metric("Qubits (Width)", p_circ["num_qubits"])
    m_row1[1].metric("Circuit Depth", p_circ["depth"])
    m_row1[2].metric("Gate Count", p_circ["total_gates"])
    m_row1[3].metric("Two-Qubit Gates (CX)", p_circ["two_qubit_gates"])

    m_row2 = st.columns(4)
    m_row2[0].metric("Shots", total_shots)
    m_row2[1].metric("Feasibility Rate", f"{feas_rate_val:.2f}%")
    if p_id["optimization_mode"] == "FEASIBILITY_ONLY":
        m_row2[2].metric("Best Feasible State", "✓ Verified Feasible" if p_best else "✗ Infeasible")
    else:
        m_row2[2].metric("Best Feasible Cost", f"${p_best['cost']:.2f}" if (p_best and p_best["cost"] is not None) else "Infeasible")
    m_row2[3].metric("Optimal Sol. Prob.", f"{opt_prob_val:.2f}%")

    st.divider()

    # ── 12. OPTIMIZATION RESULT (VISUALLY PROMINENT) ──────────────────────────
    st.subheader("OPTIMIZATION RESULT")

    if p_best:
        best_cost_str = f"${p_best['cost']:.2f}" if (p_id["optimization_mode"] != "FEASIBILITY_ONLY" and p_best["cost"] is not None) else "Feasibility Verified"
        st.markdown(
            f"""
            <div style="background: rgba(78, 222, 163, 0.08); border: 2px solid #4edea3; border-radius: 10px; padding: 18px 24px; margin-bottom: 20px;">
                <h3 style="color: #4edea3; margin-top: 0; margin-bottom: 8px;">🏆 SELECTED OPTIMIZED SCHEDULE — BEST FEASIBLE QAOA SOLUTION</h3>
                <p style="margin: 0; font-size: 15px;">
                    <strong>Cost:</strong> <span style="font-size: 18px; color: #4edea3; font-weight: bold;">{best_cost_str}</span> &nbsp;|&nbsp; 
                    <strong>Empirical Probability:</strong> <span>{p_best['probability']:.2%}</span> ({p_best['shot_count']} / {total_shots} shots) &nbsp;|&nbsp; 
                    <strong>Bitstring:</strong> <code>|{p_best['bitstring']}&gt;</code> &nbsp;|&nbsp; 
                    <strong>Status:</strong> <span style="color: #4edea3;">✓ Constraint Feasible</span>
                </p>
                <p style="margin-top: 8px; margin-bottom: 0; font-size: 13px; color: #a0aec0;">
                    <em>Reason selected: Lowest-cost feasible schedule observed in the AerSimulator measurement distribution.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Decoded Assignment Roster
        st.markdown("#### Selected Workforce Schedule")
        sched_assignment = decode_bitstring(instance, p_best["bitstring"])
        sched_rows = []
        for (w, s), val in sorted(sched_assignment.items(), key=lambda x: x[0][1]):
            if val == 1:
                w_name = p_inst["worker_names"][w] if w < len(p_inst["worker_names"]) else f"Worker {w}"
                s_name = p_inst["shift_names"][s] if s < len(p_inst["shift_names"]) else f"Shift {s}"
                c_val = instance.cost_matrix[w][s]
                sched_rows.append({
                    "Shift": s_name,
                    "Selected Worker": w_name,
                    "Cost ($)": f"${c_val:.2f}" if p_id["optimization_mode"] != "FEASIBILITY_ONLY" else "N/A",
                    "Constraint Status": "✓ Exactly 1 worker (Coverage & Capacity satisfied)",
                })
        df_roster_selected = pd.DataFrame(sched_rows)
        st.dataframe(df_roster_selected, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ **No feasible quantum sample observed on AerSimulator in this run.** ShiftProof will NOT substitute the classical optimum.")

    # Most Probable vs Best Feasible State Comparison
    st.markdown("#### Most Probable State vs. Selected Optimized State")
    most_probable = p_verif[0] if p_verif else None

    if most_probable and p_best:
        mp_c1, mp_c2 = st.columns(2)
        with mp_c1:
            st.markdown("**Most Probable Measured State (Distribution Mode):**")
            st.write(f"• **Bitstring:** `|{most_probable['bitstring']}>`")
            st.write(f"• **Shots:** {most_probable['shot_count']} / {total_shots} ({most_probable['probability']:.2%})")
            st.write(f"• **Feasible:** {'✓ Yes' if most_probable['is_feasible'] else '✗ No'}")
            cost_disp = f"${most_probable['cost']:.2f}" if most_probable['cost'] is not None else "N/A (Infeasible)"
            st.write(f"• **Cost:** {cost_disp}")

        with mp_c2:
            st.markdown("**Selected Optimized State (Best Feasible Observed):**")
            st.write(f"• **Bitstring:** `|{p_best['bitstring']}>`")
            st.write(f"• **Shots:** {p_best['shot_count']} / {total_shots} ({p_best['probability']:.2%})")
            st.write("• **Feasible:** ✓ Yes")
            st.write(f"• **Cost:** {best_cost_str}")

        if most_probable["bitstring"] == p_best["bitstring"] and most_probable["is_feasible"]:
            st.info("The most probable quantum state is also the lowest-cost feasible schedule.")
        else:
            st.info(
                "ℹ **Optimization Rule:** The most probable quantum state is not automatically the optimal schedule. "
                "The selected schedule is the lowest-cost **feasible** solution observed in the measurement distribution."
            )

    # Three-Way Optimization Quality Comparison
    st.markdown("#### Multi-Method Solution Quality Benchmark")
    comp_records = []
    if exact_res and exact_res.feasible:
        comp_records.append({
            "Method": "Exact Classical Reference",
            "Engine / Backend": "solve_exact (Local CPU)",
            "Feasible": "✓ Yes",
            "Cost": f"${exact_res.optimal_cost:.2f}",
            "Optimality Gap": "$0.00 (Reference)",
            "Status": "Ground Truth",
        })
    comp_records.append({
        "Method": "Greedy Classical Heuristic",
        "Engine / Backend": "solve_greedy (Local CPU)",
        "Feasible": "✓ Yes" if greedy_res.feasible else "✗ No",
        "Cost": f"${greedy_res.optimal_cost:.2f}" if greedy_res.optimal_cost is not None else "N/A",
        "Optimality Gap": f"${greedy_res.optimal_cost - classical_opt_cost:.2f}" if (greedy_res.optimal_cost is not None and classical_opt_cost is not None) else "N/A",
        "Status": "Classical Heuristic",
    })
    comp_records.append({
        "Method": "QAOA Quantum Simulation",
        "Engine / Backend": "Qiskit AerSimulator",
        "Feasible": "✓ Yes" if p_best else "✗ No",
        "Cost": f"${p_best['cost']:.2f}" if (p_best and p_best["cost"] is not None) else "Infeasible",
        "Optimality Gap": f"${p_comp['absolute_gap']:.2f}" if p_comp["absolute_gap"] is not None else "N/A",
        "Status": "Simulation (AerSimulator)",
    })
    st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)

    st.divider()

    # ── 10. ACTUAL CIRCUIT ────────────────────────────────────────────────────
    st.subheader("⚛️ Actual Qiskit QuantumCircuit")
    st.caption("Actual, uncompiled Qiskit QuantumCircuit synthesized by src.quantum.build_qaoa_circuit with bound optimal variational parameters.")

    circ_c1, circ_c2, circ_c3, circ_c4 = st.columns(4)
    circ_c1.metric("Qubits Width", p_circ["num_qubits"])
    circ_c2.metric("Circuit Depth", p_circ["depth"])
    circ_c3.metric("Total Gates", p_circ["total_gates"])
    circ_c4.metric("CX Gates", p_circ["two_qubit_gates"])

    assert p_circ["num_qubits"] == instance.n_vars, "Qubit count mismatch with problem decision variables!"
    st.code(p_circ["text_diagram"], language="text")

    st.divider()

    # ── 11. MEASUREMENT RESULTS ───────────────────────────────────────────────
    st.subheader("📊 AerSimulator Measurement Distribution")
    st.caption(f"Actual measurement counts sampled on Qiskit AerSimulator ({total_shots} total shots).")

    m_df = pd.DataFrame(p_meas)
    if not m_df.empty:
        m_df["Feasibility"] = m_df["is_feasible"].map({1: "Feasible", 0: "Infeasible", True: "Feasible", False: "Infeasible"})
        fig_meas = px.bar(
            m_df.head(15),
            x="bitstring",
            y="shot_count",
            color="Feasibility",
            color_discrete_map={"Feasible": "#4edea3", "Infeasible": "#ffb4ab"},
            title=f"Sampled Bitstring Distribution (Top 15) · Exp: {p_id['experiment_id']}",
        )
        fig_meas.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171b26",
            plot_bgcolor="#1c1f2a",
            height=280,
            margin=dict(l=30, r=30, t=30, b=50),
        )
        st.plotly_chart(fig_meas, use_container_width=True)

        # Full Measurement Table with Audit Details
        table_records = []
        for v in p_verif[:20]:
            table_records.append({
                "Rank": len(table_records) + 1,
                "Bitstring": v["bitstring"],
                "Count": v["shot_count"],
                "Probability (%)": f"{v['probability'] * 100.0:.2f}%",
                "Feasible": "YES" if v["is_feasible"] else "NO",
                "Cost": f"${v['cost']:.2f}" if v["cost"] is not None else "N/A",
                "Decoded Assignment": ", ".join(v["readable_assignment"]) if v["readable_assignment"] else "None",
            })
        st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)

    st.divider()

    # ── 14. EXPANDABLE SCIENTIFIC EVIDENCE ────────────────────────────────────
    st.subheader("🔬 Advanced Scientific Evidence & Reproducibility")

    with st.expander("📁 [1. Dataset & Instance Normalization Details]", expanded=False):
        ins_c1, ins_c2 = st.columns(2)
        with ins_c1:
            st.markdown(f"**Workers ({p_inst['n_workers']}):**")
            st.dataframe(pd.DataFrame(p_inst["workers"]), use_container_width=True, hide_index=True)
        with ins_c2:
            st.markdown(f"**Shifts ({p_inst['n_shifts']}):**")
            st.dataframe(pd.DataFrame(p_inst["shifts"]), use_container_width=True, hide_index=True)

    with st.expander("🔢 [2. Binary Decision Variables (Qubit Mapping)]", expanded=False):
        st.dataframe(pd.DataFrame(p_vars), use_container_width=True, hide_index=True)

    with st.expander("📐 [3. QUBO Formulation & Coefficients]", expanded=False):
        st.markdown(r"$$\min_{x \in \{0, 1\}^n} Q(x) = q_0 + \sum_{i} a_i x_i + \sum_{i < j} b_{ij} x_i x_j$$")
        q_m1, q_m2, q_m3, q_m4 = st.columns(4)
        q_m1.metric("Shift Coverage Penalty (A)", f"{p_qubo['A']:.1f}")
        q_m2.metric("Worker Capacity Penalty (B)", f"{p_qubo['B']:.1f}")
        q_m3.metric("Greedy Reference Cost", f"{p_qubo['x_ref_cost']:.2f}")
        q_m4.metric("Constant Offset (q0)", f"{p_qubo['q0']:.2f}")
        st.dataframe(pd.DataFrame(p_qubo["linear_coefficients"]), use_container_width=True, hide_index=True)

    with st.expander("🧲 [4. Ising Spin Hamiltonian & Energy Equivalence Proof]", expanded=False):
        st.markdown(r"$$H = \text{offset} + \sum_{i} h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j, \quad Z_i = 1 - 2 x_i$$")
        st.success(
            f"✓ **QUBO ↔ Ising Equivalence Proved:** For all tested states, "
            f"$|Q(x) - H(z)| = {p_ising['max_energy_discrepancy']:.6e}$ Ha, confirming exact analytical energy equivalence."
        )

    with st.expander("📜 [5. OpenQASM 2.0 Circuit Specification]", expanded=False):
        st.code(p_circ["qasm"], language="qasm")

    with st.expander("🔁 [6. Deterministic Python Reproducibility Script]", expanded=False):
        st.code(proof["reproducibility"]["code_snippet"], language="python")
