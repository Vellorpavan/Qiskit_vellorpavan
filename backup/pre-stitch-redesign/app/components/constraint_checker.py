"""
ShiftProof — Constraint Checker Component

Helper functions to format and display constraint verification results.
Calls src.model.check_constraints() directly.
"""

from __future__ import annotations

import streamlit as st
from src.model import Instance, check_constraints, assignment_cost


def render_constraint_summary(instance: Instance, assignment: dict) -> tuple[bool, dict]:
    """
    Check constraints and render a formatted visual summary.
    Returns (feasible, details).
    """
    feasible, details = check_constraints(instance, assignment)
    cost = assignment_cost(instance, assignment) if feasible else None

    shift_cov = details["shift_coverage"]
    worker_lim = details["worker_at_most_one"]
    elig = details["eligibility"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if feasible:
            st.markdown(
                '<div style="background:#0d2818;border:1px solid #1e5c32;border-radius:6px;padding:0.6rem;text-align:center;">'
                '<span style="color:#4ade80;font-weight:600;font-size:1.1rem;">✓ FEASIBLE</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#2d1215;border:1px solid #6b1f24;border-radius:6px;padding:0.6rem;text-align:center;">'
                '<span style="color:#f87171;font-weight:600;font-size:1.1rem;">✗ INFEASIBLE</span>'
                '</div>',
                unsafe_allow_html=True,
            )
    with c2:
        cov_ok = all(shift_cov.values())
        st.metric("Shift Coverage", f"{sum(shift_cov.values())}/{len(shift_cov)}", delta="Pass" if cov_ok else "Fail", delta_color="normal" if cov_ok else "inverse")
    with c3:
        w_ok = all(worker_lim.values())
        st.metric("Worker Limits", f"{sum(worker_lim.values())}/{len(worker_lim)}", delta="Pass" if w_ok else "Fail", delta_color="normal" if w_ok else "inverse")
    with c4:
        e_ok = all(elig.values())
        st.metric("Eligibility", f"{sum(elig.values())}/{len(elig)}", delta="Pass" if e_ok else "Fail", delta_color="normal" if e_ok else "inverse")

    return feasible, details
