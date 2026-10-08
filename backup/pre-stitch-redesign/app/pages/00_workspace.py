"""
ShiftProof — Page 00: Interactive Scheduling Workspace

Primary landing and operational workspace for ShiftProof.
Guides the user from Data Selection -> Validation -> Optimization (Classical & QAOA)
-> Independent Verification -> Final Schedule Roster.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.database.db import get_db_connection
from app.database.seed import seed_synthetic_demo, DEMO_DATASET_ID
from app.services.dataset_service import list_datasets, get_active_dataset, set_active_dataset
from app.services.scheduling_service import db_to_instance, format_solution_roster, compute_dataset_fingerprint
from app.services.experiment_service import (
    run_classical_workspace,
    run_qaoa_workspace,
    list_dataset_experiments,
)
from src.model import decode_bitstring, check_constraints, assignment_cost

try:
    st.set_page_config(page_title="ShiftProof — Quantum Scheduling", page_icon="⚡", layout="wide")
except Exception:
    pass

_CSS = Path(__file__).parents[1] / "styles" / "theme.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

# Database connection & dataset discovery
conn = get_db_connection()
datasets = list_datasets(conn)
if not datasets:
    seed_synthetic_demo(conn)
    datasets = list_datasets(conn)

dataset_ids = [d["dataset_id"] for d in datasets]

# Authoritative session state management
selected_d_id = get_active_dataset()
if selected_d_id not in dataset_ids and dataset_ids:
    selected_d_id = dataset_ids[0]
    set_active_dataset(selected_d_id)

current_dataset_meta = next((d for d in datasets if d["dataset_id"] == selected_d_id), datasets[0])

# ── Primary Hero Header ───────────────────────────────────────────────────────
st.markdown("# ⚡ ShiftProof")
st.markdown("### Constraint-Verified Quantum Scheduling")
st.markdown(
    "Upload a workforce scheduling dataset. ShiftProof analyzes the data, validates scheduling "
    "constraints, constructs the optimization model, compares classical and quantum solutions, "
    "and independently verifies the result."
)

# ── 6-Question System Architecture Card ───────────────────────────────────────
st.markdown(
    """
<div style="background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:1.1rem 1.3rem; margin:1rem 0 1.25rem 0;">
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:1rem; font-size:0.875rem; color:#c9d1d9;">
    <div><strong style="color:#58a6ff;">• What is ShiftProof?</strong> Constraint-verified workforce scheduling comparing classical and quantum optimization.</div>
    <div><strong style="color:#58a6ff;">• What data can I upload?</strong> Tabular CSV, XLSX, or JSON containing workers, shifts, wages/costs, and eligibility.</div>
    <div><strong style="color:#58a6ff;">• What happens after upload?</strong> Automatic column profiling, candidate schema mapping, and 7-point mathematical constraint validation.</div>
    <div><strong style="color:#58a6ff;">• Can I run classical optimization?</strong> Yes. Greedy baseline and exact branch-and-bound (for small instances) run in milliseconds.</div>
    <div><strong style="color:#58a6ff;">• Can I run QAOA?</strong> Yes, on Qiskit AerSimulator for datasets up to 9 binary variables (demonstrator limit).</div>
    <div><strong style="color:#58a6ff;">• Can I later run IBM Quantum?</strong> Yes. The validated formulation is prepared to submit to physical QPUs via Qiskit Runtime.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Initial Quick-Action Bar ──────────────────────────────────────────────────
col_hero_1, col_hero_2, col_hero_3 = st.columns([1.5, 1.5, 3])

with col_hero_1:
    if st.button("📁 Upload Dataset", use_container_width=True, type="primary"):
        st.switch_page("pages/01_data.py")

with col_hero_2:
    if st.button("⚡ Try Synthetic Demo", use_container_width=True):
        set_active_dataset(DEMO_DATASET_ID)
        st.success("Loaded Synthetic Demo Dataset.")
        st.rerun()

with col_hero_3:
    st.markdown(
        "<div style='padding:0.4rem 0.8rem; background:#161b22; border-radius:6px; border:1px solid #30363d; font-size:0.85rem; color:#8b949e;'>"
        "Supported formats: <strong>CSV</strong> · <strong>XLSX</strong> · <strong>JSON</strong> &nbsp;|&nbsp; "
        "Max scale for QAOA demonstrator: <strong>≤ 9 binary variables</strong>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Active Dataset & Dynamic Sizing ───────────────────────────────────────────
st.markdown("### Active Scheduling Dataset")

try:
    instance = db_to_instance(selected_d_id, conn)
    n_workers = instance.n_workers
    n_shifts = instance.n_shifts
    potential_pairs = n_workers * n_shifts
    n_vars = instance.n_vars
    n_qubits = n_vars
    active_fp = compute_dataset_fingerprint(selected_d_id, conn)

    # Demonstrator scale rules
    qaoa_eligible = (n_vars <= 9)
    exact_eligible = (n_vars <= 16)
except Exception as e:
    st.error(f"Error loading model for dataset '{selected_d_id}': {e}")
    st.stop()

col_meta, col_switch = st.columns([3, 1])

with col_meta:
    st.markdown(f"**{current_dataset_meta['name']}**")
    if current_dataset_meta.get("is_synthetic"):
        st.markdown(
            "<span style='background:#1e293b; color:#94a3b8; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.75rem; border:1px solid #334155;'>"
            "SYNTHETIC DEMO DATASET · NOT REAL ORGANIZATION DATA"
            "</span>",
            unsafe_allow_html=True,
        )
    st.caption(f"{current_dataset_meta.get('description', '')} | Dataset ID: `{selected_d_id}` | Fingerprint: `{active_fp}`")

with col_switch:
    st.markdown("**Switch Dataset:**")
    switch_choice = st.selectbox(
        "Active Dataset",
        options=dataset_ids,
        index=dataset_ids.index(selected_d_id) if selected_d_id in dataset_ids else 0,
        format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
        label_visibility="collapsed",
    )
    if switch_choice != selected_d_id:
        set_active_dataset(switch_choice)
        st.rerun()

# Dynamic Problem Sizing Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Workers", n_workers)
m2.metric("Shifts", n_shifts)
m3.metric("Potential Pairs", potential_pairs, help=f"{n_workers} workers × {n_shifts} shifts")
m4.metric("Eligible Pairs", n_vars, help="Variables after variable reduction (ineligible pairs pruned)")
m5.metric("Required Qubits", n_qubits)

# Scale Status Banner
if qaoa_eligible:
    st.markdown(
        f"<div style='background:#0d2818; border:1px solid #1e5c32; border-radius:6px; padding:0.6rem 1rem; margin-top:0.5rem;'>"
        f"<strong style='color:#4ade80;'>✓ Quantum QAOA Available:</strong> "
        f"<span style='color:#d1fae5;'>Problem requires {n_vars} variables/qubits, within the validated QAOA demonstrator scale (≤ 9 variables).</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div style='background:#1c1917; border:1px solid #78350f; border-radius:6px; padding:0.6rem 1rem; margin-top:0.5rem;'>"
        f"<strong style='color:#fbbf24;'>🔒 Quantum QAOA Unavailable for this dataset:</strong> "
        f"<span style='color:#d6d3d1;'>This dataset requires <strong>{n_vars} variables/qubits</strong>, which exceeds the currently validated QAOA demonstrator scale of <strong>9 binary variables</strong>. "
        f"Classical optimization remains available.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ── Optimization Control Area ─────────────────────────────────────────────────
st.markdown("### Optimization Controls")

opt_c1, opt_c2 = st.columns(2)

with opt_c1:
    st.markdown("#### Classical Solvers")
    st.caption("Greedy priority heuristic & exact exhaustive enumeration (where computationally practical).")
    run_classical = st.button("▶ Run Classical Optimization", use_container_width=True)

with opt_c2:
    st.markdown("#### Quantum QAOA (p=1)")
    st.caption("Parametric QAOA on Qiskit AerSimulator (1024 shots, COBYLA maxiter=100, seed=42).")
    if qaoa_eligible:
        run_qaoa = st.button("⚡ Run Quantum QAOA (p=1)", use_container_width=True, type="primary")
    else:
        run_qaoa = False
        st.button(
            "🔒 Run QAOA (Exceeds 9-Variable Limit)",
            disabled=True,
            use_container_width=True,
            help="Dataset requires > 9 variables. Demonstrator limit is strictly <= 9 variables.",
        )

# Execution Handlers with Honest Live Status
if run_classical:
    status_p = st.empty()
    status_p.markdown(
        """
<div style='background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.8rem 1rem; margin:0.8rem 0;'>
  <strong style='color:#58a6ff;'>RUNNING CLASSICAL OPTIMIZATION...</strong><br>
  <span style='color:#8b949e; font-size:0.85rem;'>Preparing model ↓ Running greedy heuristic ↓ Executing exact enumeration (if ≤ 16 variables) ↓ Verifying constraints ↓ Completed</span>
</div>
""",
        unsafe_allow_html=True,
    )
    res = run_classical_workspace(instance, selected_d_id, conn)
    status_p.empty()
    st.session_state["latest_experiment_id"] = res["experiment_id"]
    st.success("✓ Classical optimization completed successfully.")
    st.rerun()

if run_qaoa:
    status_p = st.empty()
    status_p.markdown(
        """
<div style='background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.8rem 1rem; margin:0.8rem 0;'>
  <strong style='color:#58a6ff;'>RUNNING QAOA SIMULATION...</strong><br>
  <span style='color:#8b949e; font-size:0.85rem;'>Preparing model ↓ Building QUBO ↓ Mapping to Ising ↓ Building QAOA circuit ↓ Optimizing parameters (COBYLA) ↓ Running measurements (1024 shots) ↓ Verifying solutions ↓ Completed</span>
</div>
""",
        unsafe_allow_html=True,
    )
    try:
        res_q = run_qaoa_workspace(instance, selected_d_id, shots=1024, maxiter=100, seed=42, conn=conn)
        status_p.empty()
        st.session_state["latest_experiment_id"] = res_q["experiment_id"]
        st.success("✓ Quantum QAOA simulation completed successfully.")
        st.rerun()
    except Exception as ex:
        status_p.empty()
        st.error(f"QAOA execution encountered an issue: {ex}")

st.divider()

# ── Results & Final Schedule ─────────────────────────────────────────────────
st.markdown("### Final Schedule & Results")

# Query dataset results across all solvers
cur = conn.cursor()
cur.execute(
    """
    SELECT s.total_cost, s.feasible, s.runtime_seconds
    FROM solutions s
    JOIN experiments e ON s.experiment_id = e.experiment_id
    WHERE s.dataset_id = ? AND s.solution_id LIKE 'sol_greedy_%'
    ORDER BY e.created_at DESC LIMIT 1
    """,
    (selected_d_id,),
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
    (selected_d_id,),
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
    (selected_d_id,),
)
qaoa_row = cur.fetchone()
qaoa_summary = None
if qaoa_row:
    exp_id = qaoa_row["experiment_id"]
    cur.execute(
        """
        SELECT SUM(CASE WHEN is_feasible = 1 THEN shot_count ELSE 0 END) * 1.0 / SUM(shot_count) as feas_rate
        FROM measurements
        WHERE experiment_id = ?
        """,
        (exp_id,),
    )
    rate_res = cur.fetchone()
    feas_rate = rate_res["feas_rate"] if rate_res and rate_res["feas_rate"] is not None else 0.0

    opt_status = "Infeasible Sample"
    if qaoa_row["feasible"]:
        if exact_row and exact_row["feasible"] and exact_row["total_cost"] is not None:
            if abs(qaoa_row["total_cost"] - exact_row["total_cost"]) < 1e-4:
                opt_status = "Optimal (Matches Exact)"
            else:
                opt_status = f"Suboptimal (+{qaoa_row['total_cost'] - exact_row['total_cost']:.2f})"
        else:
            opt_status = "Feasible Candidate"

    qaoa_summary = {
        "cost": qaoa_row["total_cost"],
        "feasible": bool(qaoa_row["feasible"]),
        "runtime": qaoa_row["runtime_seconds"],
        "feasibility_rate": feas_rate,
        "optimality_status": opt_status,
    }

# Render Result Summary Section (Section 12 requirement)
if greedy_row or qaoa_summary:
    st.markdown("#### Result Summary")
    num_cols = 1 + (1 if qaoa_summary else 0) + (1 if exact_row else 0)
    summary_cols = st.columns(num_cols)
    col_idx = 0

    # Classical Column
    if greedy_row:
        with summary_cols[col_idx]:
            st.markdown(
                f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.8rem; text-align:left;">
  <strong style="color:#58a6ff; font-size:0.95rem;">🏛 Classical (Greedy)</strong><br>
  <div style="margin-top:0.4rem; font-size:0.85rem; line-height:1.6;">
    • <strong>Cost:</strong> {greedy_row['total_cost']:.2f}<br>
    • <strong>Feasible:</strong> {'<span style="color:#4ade80;">✓ Yes</span>' if greedy_row['feasible'] else '<span style="color:#f87171;">✗ No</span>'}<br>
    • <strong>Runtime:</strong> {greedy_row['runtime_seconds']:.4f}s
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        col_idx += 1

    # QAOA Column
    if qaoa_summary:
        qaoa_cost_str = f"{qaoa_summary['cost']:.2f}" if qaoa_summary['cost'] is not None else "N/A"
        with summary_cols[col_idx]:
            st.markdown(
                f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.8rem; text-align:left;">
  <strong style="color:#a371f7; font-size:0.95rem;">⚛ QAOA (AerSimulator)</strong><br>
  <div style="margin-top:0.4rem; font-size:0.85rem; line-height:1.6;">
    • <strong>Cost:</strong> {qaoa_cost_str}<br>
    • <strong>Feasible:</strong> {'<span style="color:#4ade80;">✓ Yes</span>' if qaoa_summary['feasible'] else '<span style="color:#f87171;">✗ No</span>'}<br>
    • <strong>Runtime:</strong> {qaoa_summary['runtime']:.3f}s<br>
    • <strong>Feasibility Rate:</strong> {qaoa_summary['feasibility_rate']*100:.1f}%<br>
    • <strong>Status:</strong> {qaoa_summary['optimality_status']}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        col_idx += 1

    # Exact Column (only if executed!)
    if exact_row:
        with summary_cols[col_idx]:
            st.markdown(
                f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.8rem; text-align:left;">
  <strong style="color:#3fb950; font-size:0.95rem;">🎯 Exact (Exhaustive)</strong><br>
  <div style="margin-top:0.4rem; font-size:0.85rem; line-height:1.6;">
    • <strong>Cost:</strong> {exact_row['total_cost']:.2f}<br>
    • <strong>Feasible:</strong> {'<span style="color:#4ade80;">✓ Yes</span>' if exact_row['feasible'] else '<span style="color:#f87171;">✗ No</span>'}<br>
    • <strong>Runtime:</strong> {exact_row['runtime_seconds']:.4f}s
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
    st.markdown("")

experiments = list_dataset_experiments(selected_d_id, conn)

if not experiments:
    st.info("No optimization runs executed yet for this dataset. Click **'Run Classical Optimization'** or **'Run Quantum QAOA'** above to generate schedules.")
else:
    def _format_run_label(e):
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

    solver_types = [_format_run_label(e) for e in experiments]
    selected_run_label = st.selectbox(
        "Select Optimization Run to Inspect:",
        solver_types,
        index=0,
        key=f"ws_run_selector_{selected_d_id}",
    )
    selected_idx = solver_types.index(selected_run_label)
    active_exp = experiments[selected_idx]

    exp_id = active_exp["experiment_id"]
    solver_type = active_exp["solver_type"]

    # SECTION 4: PREVENT WRONG DATASET RESULTS
    if active_exp.get("dataset_id") != selected_d_id:
        st.error(f"⚠️ Result belongs to dataset '{active_exp.get('dataset_id')}', but active dataset is '{selected_d_id}'. Please rerun optimization.")
        st.stop()

    exp_params = json.loads(active_exp.get("parameters_json") or "{}")
    exp_fp = exp_params.get("dataset_fingerprint", "N/A")
    if exp_fp != "N/A" and exp_fp != active_fp:
        st.warning(f"⚠️ Notice: The active dataset has changed since this optimization was run (Recorded Fingerprint: `{exp_fp}`, Current: `{active_fp}`). Please rerun optimization for this dataset.")

    # Decode assignment
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

    # Verification audit
    feasible, details = check_constraints(instance, assignment_dict)
    cost = active_exp.get("total_cost")
    runtime = active_exp.get("runtime_seconds", 0.0)

    # Worker assignment counts (Section 6 requirement: assigned vs unassigned)
    assigned_worker_indices = {w for (w, s), val in assignment_dict.items() if val == 1}
    n_assigned = len(assigned_worker_indices)
    n_unassigned = n_workers - n_assigned

    # ── SECTION 3: RESULT PROVENANCE (Database-backed) ─────────────────────────
    st.markdown("#### Result Provenance")
    src_type = current_dataset_meta.get("source_type", "upload")
    if current_dataset_meta.get("is_synthetic"):
        src_label = "Synthetic Demo"
    elif src_type == "upload":
        src_label = "Uploaded Dataset (CSV / XLSX / JSON)"
    else:
        src_label = f"Uploaded {src_type.upper()}"

    fp_match_badge = '<span style="color:#4ade80;">✓ Verified Match</span>' if (exp_fp != 'N/A' and exp_fp == active_fp) else ''

    st.markdown(
        f"""
<div style="background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:1.1rem; margin-bottom:1.2rem;">
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:0.8rem; font-size:0.875rem;">
    <div><span style="color:#8b949e;">Dataset:</span> <strong style="color:#f0f6fc;">{current_dataset_meta.get('name', selected_d_id)}</strong><br><small style="color:#58a6ff;">(ID: {selected_d_id})</small></div>
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

    # Result Summary KPIs
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        if feasible:
            st.markdown(
                '<div style="background:#0d2818;border:1px solid #1e5c32;border-radius:6px;padding:0.7rem;text-align:center;">'
                '<span style="color:#4ade80;font-weight:700;font-size:1.1rem;">✓ VERIFIED FEASIBLE</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#2d1215;border:1px solid #6b1f24;border-radius:6px;padding:0.7rem;text-align:center;">'
                '<span style="color:#f87171;font-weight:700;font-size:1.1rem;">✗ INFEASIBLE</span>'
                '</div>',
                unsafe_allow_html=True,
            )
    with r2:
        st.metric("Total Assignment Cost", f"{cost:.2f}" if cost is not None else "None")
    with r3:
        st.metric("Solver Used", "Classical" if solver_type == "classical" else "QAOA Simulator")
    with r4:
        st.metric("Runtime", f"{runtime:.3f}s")

    st.markdown("")

    # Final Schedule Table
    st.markdown("#### Final Schedule Roster")
    st.caption(
        f"Assigned Workers: **{n_assigned}** | Unassigned Workers: **{n_unassigned}** | "
        f"Total Workforce in Dataset: **{n_workers}** | Shifts Filled: **{n_assigned}/{n_shifts}**"
    )
    roster_rows = format_solution_roster(instance, assignment_dict, selected_d_id, conn)
    df_roster = pd.DataFrame(roster_rows)
    st.dataframe(df_roster, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_bytes = df_roster.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Final Schedule to CSV",
        data=csv_bytes,
        file_name=f"shiftproof_schedule_{selected_d_id}_{exp_id}.csv",
        mime="text/csv",
    )

    # Independent Constraint Verification Breakdown
    st.markdown("")
    st.markdown("#### Independent Constraint Audit")
    cov_ok = all(details["shift_coverage"].values())
    cap_ok = all(details["worker_at_most_one"].values())
    elig_ok = all(details["eligibility"].values())

    v1, v2, v3 = st.columns(3)
    v1.metric("Shift Coverage (Each Shift 1 Worker)", "✓ Pass" if cov_ok else "✗ Fail")
    v2.metric("Worker Capacity (Max 1 Shift/Worker)", "✓ Pass" if cap_ok else "✗ Fail")
    v3.metric("Eligibility & Qualifications", "✓ Pass" if elig_ok else "✗ Fail")

    # QAOA Specific Measurement Distribution
    if solver_type == "qaoa_p1":
        st.markdown("")
        st.markdown("#### QAOA Measurement Distribution")
        cur = conn.cursor()
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
                title="Top 20 Measured Bitstrings (Green = Feasible, Red = Infeasible)",
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

    with st.expander("🔬 How did Qiskit solve this? (Mathematical & Circuit Transparency)"):
        try:
            from src.quantum import build_qubo, qubo_to_ising
            import numpy as np
            qubo = build_qubo(instance)
            qubo_mat = np.copy(qubo.b)
            np.fill_diagonal(qubo_mat, qubo.a)
            qubo_const = float(qubo.q0)
            ising = qubo_to_ising(qubo)
            pauli_op = ising.to_sparse_pauli_op()

            st.markdown(
                f"• **QUBO Variables:** {instance.n_vars}<br>"
                f"• **Coverage Penalty ($A$):** {qubo.A:.1f}<br>"
                f"• **Capacity Penalty ($B$):** {qubo.B:.1f}<br>"
                f"• **Ising Pauli Terms:** {len(pauli_op)}<br>"
                f"• **Ising Energy Offset:** {ising.offset:.4f}<br>"
                "• **QAOA Parameterization:** $p=1$ depth with $R_{ZZ}$ problem unitary and $R_X$ mixer unitary, solved with COBYLA on AerSimulator.",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(f"Could not render QUBO/Ising transparency: {e}")

st.divider()

st.markdown(
    "<div class='no-advantage-banner'>"
    "Notice: ShiftProof supports workforce/resource scheduling datasets that can be mapped to its scheduling model. "
    "Results represent proof-of-concept experimentation under ideal noiseless AerSimulator simulation. No quantum advantage is claimed."
    "</div>",
    unsafe_allow_html=True,
)
