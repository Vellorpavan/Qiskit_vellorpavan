"""
ShiftProof — Page 02: Results & Verification

Primary output dashboard for ShiftProof.
Displays optimization results and verified schedules for the active dataset.
Includes rigorous Result Provenance, cross-solver benchmarks,
independent constraint verification, and live measurement distribution.
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
from app.database.seed import DEMO_DATASET_ID
from app.services.dataset_service import list_datasets, get_active_dataset, set_active_dataset
from app.services.scheduling_service import db_to_instance, format_solution_roster, compute_dataset_fingerprint
from app.services.experiment_service import list_dataset_experiments
from src.model import check_constraints
from src.quantum import build_qubo, qubo_to_ising

try:
    st.set_page_config(page_title="Results — ShiftProof", page_icon="📋", layout="wide")
except Exception:
    pass

_CSS = Path(__file__).parents[1] / "styles" / "theme.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

st.markdown("# 📋 Scheduling Results & Verification")
st.markdown(
    "Final assignment roster, cross-solver optimization comparison, "
    "independent constraint audits, and QAOA solution distribution."
)
st.divider()

conn = get_db_connection()
datasets = list_datasets(conn)
if not datasets:
    from app.database.seed import seed_synthetic_demo
    seed_synthetic_demo(conn)
    datasets = list_datasets(conn)

active_id = get_active_dataset()
dataset_ids = [d["dataset_id"] for d in datasets]
if active_id not in dataset_ids and dataset_ids:
    active_id = dataset_ids[0]
    set_active_dataset(active_id)

active_fp = compute_dataset_fingerprint(active_id, conn)

# Fetch active dataset record directly from SQLite for provenance
cur = conn.cursor()
cur.execute("SELECT * FROM datasets WHERE dataset_id = ?", (active_id,))
ds_record = cur.fetchone()
ds_meta = dict(ds_record) if ds_record else {}

# Dataset header & switch
c_top1, c_top2 = st.columns([3, 1])
with c_top1:
    st.markdown(f"### Active Dataset: **{ds_meta.get('name', active_id)}**")
    if ds_meta.get("is_synthetic"):
        st.markdown(
            "<span style='background:#1e293b; color:#94a3b8; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.75rem; border:1px solid #334155;'>"
            "SYNTHETIC DEMO DATASET · NOT REAL ORGANIZATION DATA"
            "</span>",
            unsafe_allow_html=True,
        )
    src_info = ds_meta.get('source_filename') or ds_meta.get('source_type', 'upload')
    st.caption(f"Source: `{src_info}` | Dataset ID: `{active_id}` | Fingerprint: `{active_fp}`")

with c_top2:
    switched_d = st.selectbox(
        "Active Dataset",
        options=dataset_ids,
        index=dataset_ids.index(active_id) if active_id in dataset_ids else 0,
        format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
        label_visibility="collapsed",
    )
    if switched_d != active_id:
        set_active_dataset(switched_d)
        st.rerun()

try:
    instance = db_to_instance(active_id, conn)
    n_workers = instance.n_workers
    n_shifts = instance.n_shifts
    potential_pairs = n_workers * n_shifts
    n_vars = instance.n_vars
    n_qubits = n_vars
except Exception as e:
    st.error(f"Error loading model for dataset '{active_id}': {e}")
    st.stop()

# Dataset Dimensions (Section 11 requirement: clearly show 23 workers, 3 shifts, 69 possible, 9 eligible, 9 qubits)
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Total Workers", n_workers, help="Total workforce registered in this dataset")
d2.metric("Shifts to Cover", n_shifts, help="Total distinct duty slots to schedule")
d3.metric("Possible Assignments", potential_pairs, help=f"{n_workers} workers × {n_shifts} shifts")
d4.metric("Eligible Assignments", n_vars, help="Variables after variable reduction (ineligible pairs pruned)")
d5.metric("Qubits Required", n_qubits)

st.divider()

experiments = list_dataset_experiments(active_id, conn)

if not experiments:
    st.info(
        f"No optimization runs have been executed for `{active_id}` yet. "
        "Go to **Workspace** to run Classical or QAOA optimization."
    )
    if st.button("⚡ Go to Workspace"):
        st.switch_page("pages/00_workspace.py")
else:
    # Cross-solver summaries for active dataset
    cur.execute(
        """
        SELECT s.total_cost, s.feasible, s.runtime_seconds
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
        SELECT s.total_cost, s.feasible, s.runtime_seconds
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
    qaoa_data = None
    if qaoa_row:
        q_exp_id = qaoa_row["experiment_id"]
        cur.execute(
            """
            SELECT SUM(CASE WHEN is_feasible = 1 THEN shot_count ELSE 0 END) * 1.0 / SUM(shot_count) as feas_rate,
                   SUM(CASE WHEN is_feasible = 1 THEN assignment_cost * shot_count ELSE 0 END) * 1.0 / NULLIF(SUM(CASE WHEN is_feasible = 1 THEN shot_count ELSE 0 END), 0) as mean_feas_cost
            FROM measurements
            WHERE experiment_id = ?
            """,
            (q_exp_id,),
        )
        rate_res = cur.fetchone()
        feas_rate = rate_res["feas_rate"] if rate_res and rate_res["feas_rate"] is not None else 0.0
        mean_feas_cost = rate_res["mean_feas_cost"] if rate_res and rate_res["mean_feas_cost"] is not None else None

        opt_status = "Infeasible Sample"
        opt_gap = None
        if qaoa_row["feasible"]:
            if exact_row and exact_row["feasible"] and exact_row["total_cost"] is not None:
                opt_gap = qaoa_row["total_cost"] - exact_row["total_cost"]
                if abs(opt_gap) < 1e-4:
                    opt_status = "Optimal (Matches Exact Minimum)"
                else:
                    opt_status = f"Suboptimal (+{opt_gap:.2f})"
            else:
                opt_status = "Feasible Candidate"

        qaoa_data = {
            "cost": qaoa_row["total_cost"],
            "feasible": bool(qaoa_row["feasible"]),
            "runtime": qaoa_row["runtime_seconds"],
            "feasibility_rate": feas_rate,
            "mean_feasible_cost": mean_feas_cost,
            "optimality_status": opt_status,
            "optimality_gap": opt_gap,
            "experiment_id": q_exp_id,
        }

    # Cross-Solver Result Summary
    st.markdown("### Result Summary")
    num_cols = 1 + (1 if qaoa_data else 0) + (1 if exact_row else 0)
    res_cols = st.columns(num_cols)
    col_idx = 0

    if greedy_row:
        with res_cols[col_idx]:
            st.markdown(
                f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem;">
  <strong style="color:#58a6ff; font-size:1.05rem;">🏛 Classical Result (Greedy)</strong><br>
  <div style="margin-top:0.6rem; font-size:0.9rem; line-height:1.7;">
    • <strong>Cost:</strong> {greedy_row['total_cost']:.2f}<br>
    • <strong>Feasible:</strong> {'<span style="color:#4ade80;">✓ Yes</span>' if greedy_row['feasible'] else '<span style="color:#f87171;">✗ No</span>'}<br>
    • <strong>Runtime:</strong> {greedy_row['runtime_seconds']:.4f}s
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        col_idx += 1

    if qaoa_data:
        qaoa_best_cost = f"{qaoa_data['cost']:.2f}" if qaoa_data['cost'] is not None else "N/A"
        with res_cols[col_idx]:
            gap_str = f"<br>• <strong>Optimality Gap:</strong> {qaoa_data['optimality_gap']:.2f}" if qaoa_data['optimality_gap'] is not None else ""
            mean_c_str = f"<br>• <strong>Mean Feasible Cost:</strong> {qaoa_data['mean_feasible_cost']:.2f}" if qaoa_data['mean_feasible_cost'] is not None else ""
            st.markdown(
                f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem;">
  <strong style="color:#a371f7; font-size:1.05rem;">⚛ QAOA Result (AerSimulator)</strong><br>
  <div style="margin-top:0.6rem; font-size:0.9rem; line-height:1.7;">
    • <strong>Best Feasible Cost:</strong> {qaoa_best_cost}<br>
    • <strong>Feasible:</strong> {'<span style="color:#4ade80;">✓ Yes</span>' if qaoa_data['feasible'] else '<span style="color:#f87171;">✗ No</span>'}<br>
    • <strong>Runtime:</strong> {qaoa_data['runtime']:.3f}s<br>
    • <strong>Feasibility Rate:</strong> {qaoa_data['feasibility_rate']*100:.1f}%{mean_c_str}{gap_str}<br>
    • <strong>Status:</strong> {qaoa_data['optimality_status']}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        col_idx += 1

    if exact_row:
        with res_cols[col_idx]:
            st.markdown(
                f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem;">
  <strong style="color:#3fb950; font-size:1.05rem;">🎯 Exact Result (Exhaustive)</strong><br>
  <div style="margin-top:0.6rem; font-size:0.9rem; line-height:1.7;">
    • <strong>Cost:</strong> {exact_row['total_cost']:.2f}<br>
    • <strong>Feasible:</strong> {'<span style="color:#4ade80;">✓ Yes</span>' if exact_row['feasible'] else '<span style="color:#f87171;">✗ No</span>'}<br>
    • <strong>Runtime:</strong> {exact_row['runtime_seconds']:.4f}s
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.divider()

    # Active run detail selector
    def _format_label(e):
        sol_id = e.get("solution_id", "")
        if "greedy" in sol_id:
            s_name = "Classical (Greedy Priority)"
        elif "exact" in sol_id:
            s_name = "Classical (Exact Enumeration)"
        elif "qaoa" in sol_id:
            s_name = "Quantum QAOA (AerSimulator)"
        else:
            s_name = e["solver_type"].title()
        
        c = e.get("total_cost")
        c_str = f"Cost: {c:.2f}" if c is not None else "Cost: N/A"
        feas_str = "✓ Feasible" if e.get("feasible") else "✗ Infeasible"
        ts = str(e.get("created_at", ""))[:19]
        exp_short = e.get("experiment_id", "")
        return f"{s_name} | {c_str} | {feas_str} | {ts} [{exp_short}]"

    run_options = [_format_label(e) for e in experiments]
    chosen_label = st.selectbox(
        "Select Optimization Run to Inspect:",
        run_options,
        index=0,
        key=f"results_run_selector_{active_id}",
    )
    active_exp = experiments[run_options.index(chosen_label)]
    exp_id = active_exp["experiment_id"]
    solver_type = active_exp["solver_type"]

    # SECTION 4: PREVENT WRONG DATASET RESULTS
    if active_exp.get("dataset_id") != active_id:
        st.error(f"⚠️ Result belongs to dataset '{active_exp.get('dataset_id')}', but active dataset is '{active_id}'. Please rerun optimization.")
        st.stop()

    exp_params = json.loads(active_exp.get("parameters_json") or "{}")
    exp_fp = exp_params.get("dataset_fingerprint", "N/A")
    if exp_fp != "N/A" and exp_fp != active_fp:
        st.warning(f"⚠️ Notice: The active dataset has changed since this optimization was run (Recorded Fingerprint: `{exp_fp}`, Current: `{active_fp}`). Please rerun optimization for this dataset.")

    raw_asgn = active_exp.get("assignment_json")
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

    feasible, details = check_constraints(instance, assignment_dict)

    # Worker assignment counts (Section 6 requirement: assigned vs unassigned)
    assigned_worker_indices = {w for (w, s), val in assignment_dict.items() if val == 1}
    n_assigned = len(assigned_worker_indices)
    n_unassigned = n_workers - n_assigned

    # ── SECTION 3: RESULT PROVENANCE (Database-backed) ─────────────────────────
    st.markdown("### Result Provenance")
    src_type = ds_meta.get("source_type", "upload")
    if ds_meta.get("is_synthetic"):
        src_label = "Synthetic Demo"
    elif src_type == "upload":
        src_label = "Uploaded Dataset (CSV / XLSX / JSON)"
    else:
        src_label = f"Uploaded {src_type.upper()}"

    fp_match_badge = '<span style="color:#4ade80;">✓ Verified Match</span>' if (exp_fp != 'N/A' and exp_fp == active_fp) else ''

    st.markdown(
        f"""
<div style="background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:1.2rem; margin-bottom:1.5rem;">
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:0.9rem; font-size:0.88rem;">
    <div><span style="color:#8b949e;">Dataset:</span> <strong style="color:#f0f6fc;">{ds_meta.get('name', active_id)}</strong><br><small style="color:#58a6ff;">(ID: {active_id})</small></div>
    <div><span style="color:#8b949e;">Source:</span> <strong style="color:#f0f6fc;">{src_label}</strong></div>
    <div><span style="color:#8b949e;">Dataset Fingerprint:</span> <code style="color:#58a6ff;">{active_fp}</code></div>
    <div><span style="color:#8b949e;">Experiment Fingerprint:</span> <code style="color:#f0f6fc;">{exp_fp}</code> {fp_match_badge}</div>
    <div><span style="color:#8b949e;">Workforce Count:</span> <strong style="color:#f0f6fc;">{n_workers} Workers</strong> <span style="color:#8b949e;">(Assigned: {n_assigned}, Unassigned: {n_unassigned})</span></div>
    <div><span style="color:#8b949e;">Shift Count:</span> <strong style="color:#f0f6fc;">{n_shifts} Shifts</strong> <span style="color:#8b949e;">(Coverage: {n_assigned}/{n_shifts})</span></div>
    <div><span style="color:#8b949e;">Potential Pairs:</span> <strong style="color:#f0f6fc;">{potential_pairs}</strong></div>
    <div><span style="color:#8b949e;">Eligible Pairs / Variables:</span> <strong style="color:#f0f6fc;">{n_vars}</strong></div>
    <div><span style="color:#8b949e;">Required Qubits:</span> <strong style="color:#f0f6fc;">{n_qubits}</strong></div>
    <div><span style="color:#8b949e;">Experiment ID:</span> <code style="color:#f0f6fc;">{exp_id}</code></div>
    <div><span style="color:#8b949e;">Solver:</span> <strong style="color:#a371f7;">{'QAOA Simulator (p=1)' if solver_type == 'qaoa_p1' else 'Classical Solvers'}</strong></div>
    <div><span style="color:#8b949e;">Backend:</span> <strong style="color:#38bdf8;">{'IDEAL SIMULATOR — AerSimulator' if solver_type == 'qaoa_p1' else 'Classical Python'}</strong></div>
    <div><span style="color:#8b949e;">Status:</span> <span style="background:#1e3a1e; color:#4ade80; padding:2px 8px; border-radius:4px; font-weight:700;">LIVE WORKSPACE RESULT</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Independent Verification Section (Section 10 requirement) ─────────────
    st.markdown("### Independent Constraint Verification")
    v_col1, v_col2 = st.columns([1.5, 3])
    with v_col1:
        if feasible:
            st.markdown(
                """
<div style="background:#0d2818; border:2px solid #1e5c32; border-radius:8px; padding:1.2rem; text-align:center;">
  <span style="color:#4ade80; font-weight:800; font-size:1.2rem;">✓ VERIFIED FEASIBLE SOLUTION</span>
  <p style="color:#86efac; margin:0.3rem 0 0 0; font-size:0.85rem;">All scheduling constraints independently verified by checker.</p>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div style="background:#2d1215; border:2px solid #6b1f24; border-radius:8px; padding:1.2rem; text-align:center;">
  <span style="color:#f87171; font-weight:800; font-size:1.2rem;">✗ INFEASIBLE SOLUTION</span>
  <p style="color:#fca5a5; margin:0.3rem 0 0 0; font-size:0.85rem;">One or more hard constraints violated.</p>
</div>
""",
                unsafe_allow_html=True,
            )

    with v_col2:
        cov_ok = all(details["shift_coverage"].values())
        cap_ok = all(details["worker_at_most_one"].values())
        elig_ok = all(details["eligibility"].values())

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("✓ Shift Coverage", "Pass" if cov_ok else "Fail", help="Each shift has exactly 1 worker assigned")
        vc2.metric("✓ Worker Capacity", "Pass" if cap_ok else "Fail", help=f"All {n_workers} workers assigned to at most 1 shift")
        vc3.metric("✓ Eligibility", "Pass" if elig_ok else "Fail", help="Only eligible worker-shift pairs assigned")

    st.markdown("")

    # ── Final Schedule Table (Section 6 requirement: assigned shifts & counts) ─
    st.markdown("### Final Schedule Roster")
    st.caption(
        f"Assigned Workers: **{n_assigned}** | Unassigned Workers: **{n_unassigned}** | "
        f"Total Workforce in Dataset: **{n_workers}** | Shifts Filled: **{n_assigned}/{n_shifts}**"
    )

    roster_rows = format_solution_roster(instance, assignment_dict, active_id, conn)
    df_roster = pd.DataFrame(roster_rows)
    st.dataframe(df_roster, use_container_width=True, hide_index=True)

    csv_bytes = df_roster.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Final Schedule to CSV",
        data=csv_bytes,
        file_name=f"shiftproof_schedule_{active_id}_{exp_id}.csv",
        mime="text/csv",
    )

    # ── QAOA Measurement Visualizations (Section 9 requirement) ───────────────
    if solver_type == "qaoa_p1":
        st.divider()
        st.markdown("### QAOA Measurement Analysis (Live AerSimulator Execution)")
        cur.execute(
            """
            SELECT bitstring, shot_count, probability, is_feasible, assignment_cost, qubo_energy
            FROM measurements
            WHERE experiment_id = ?
            ORDER BY shot_count DESC
            LIMIT 20
            """,
            (exp_id,),
        )
        meas_rows = [dict(r) for r in cur.fetchall()]

        if meas_rows:
            df_meas = pd.DataFrame(meas_rows)
            df_meas["Feasibility"] = df_meas["is_feasible"].map({1: "Feasible", 0: "Infeasible", True: "Feasible", False: "Infeasible"})

            fig_meas = px.bar(
                df_meas,
                x="bitstring",
                y="shot_count",
                color="Feasibility",
                color_discrete_map={"Feasible": "#4ade80", "Infeasible": "#f87171"},
                text="shot_count",
                labels={"bitstring": "Measured Bitstring", "shot_count": "Counts (1024 Shots)"},
                title=f"Top 20 Measured Bitstrings for '{ds_meta.get('name', active_id)}' (Green = Feasible, Red = Infeasible)",
            )
            fig_meas.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                xaxis_tickangle=-45,
                margin=dict(l=40, r=40, t=50, b=80),
                height=360,
            )
            st.plotly_chart(fig_meas, use_container_width=True)

            st.markdown("##### Detailed Measurement Distribution")
            st.dataframe(
                df_meas[[
                    "bitstring", "shot_count", "probability",
                    "Feasibility", "assignment_cost", "qubo_energy"
                ]],
                use_container_width=True,
                hide_index=True,
            )

    # Mathematical & Circuit Transparency Expandable
    st.markdown("")
    with st.expander("🔬 How did Qiskit solve this? (Mathematical & Circuit Transparency)"):
        try:
            import numpy as np
            qubo = build_qubo(instance)
            qubo_mat = np.copy(qubo.b)
            np.fill_diagonal(qubo_mat, qubo.a)
            qubo_const = float(qubo.q0)
            ising = qubo_to_ising(qubo)
            pauli_op = ising.to_sparse_pauli_op()

            st.markdown("##### 1. QUBO Formulation")
            st.markdown(
                f"• **Binary Variables:** {instance.n_vars}<br>"
                f"• **Coverage Penalty ($A$):** {qubo.A:.1f}<br>"
                f"• **Capacity Penalty ($B$):** {qubo.B:.1f}<br>"
                f"• **QUBO Constant Offset:** {qubo_const:.2f}",
                unsafe_allow_html=True,
            )

            st.markdown("##### 2. Ising Spin Hamiltonian ($Z$-Basis)")
            st.markdown(
                f"• **Total Pauli Terms:** {len(pauli_op)}<br>"
                f"• **Ising Energy Offset:** {ising.offset:.4f}",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Could not render QUBO/Ising transparency: {e}")

        st.markdown("##### 3. QAOA Circuit Structure ($p=1$)")
        st.markdown(
            "• **Initial State:** Uniform superposition via Hadamard gates $H^{\\otimes n}$<br>"
            "• **Cost Layer:** Problem unitary $U(C, \\gamma) = e^{-i \\gamma H_C}$ using $R_{ZZ}$ rotations<br>"
            "• **Mixer Layer:** Transverse field mixer $U(B, \\beta) = e^{-i \\beta \\sum X_i}$ using $R_X$ rotations<br>"
            "• **Parameter Optimization:** COBYLA (maxiter=100) running on Qiskit AerSimulator (1024 shots).",
            unsafe_allow_html=True,
        )
