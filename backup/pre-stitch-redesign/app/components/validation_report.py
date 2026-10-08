"""
ShiftProof — Validation Report Component

Renders the visual validation report in Streamlit.
Provides a concise 7-point mathematical constraint audit and clear status banner:
READY TO OPTIMIZE or BLOCKED.
"""

from __future__ import annotations

import streamlit as st
from app.services.validation_service import ValidationReport


def render_validation_report(report: ValidationReport) -> None:
    """Render concise 7-point validation audit in the UI."""
    st.markdown("### Dataset Validation")

    # Concise 7-point check badges (Section 8 requirement)
    check_mapping = [
        ("V1_schema", "Schema"),
        ("V2_nulls", "Missing values"),
        ("V3_costs", "Cost validity"),
        ("V4_duplicates", "Duplicate pairs"),
        ("V5_coverage", "Shift coverage"),
        ("V6_capacity", "Worker capacity"),
        ("V7_scale", "QAOA scale"),
    ]

    cols = st.columns(len(check_mapping))
    for i, (key, label) in enumerate(check_mapping):
        chk = report.checks.get(key, {})
        passed = chk.get("passed", False)
        icon = "✓" if passed else "✗"
        color = "#4ade80" if passed else "#f87171"
        with cols[i]:
            st.markdown(
                f"<div style='background:#161b22; border:1px solid #30363d; border-radius:6px; padding:0.6rem 0.2rem; text-align:center;'>"
                f"<span style='color:{color}; font-weight:700; font-size:1.1rem;'>{icon}</span><br>"
                f"<span style='font-size:0.8rem; color:#c9d1d9;'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Status Banner
    if report.status == "READY TO OPTIMIZE":
        st.markdown(
            '<div style="background:#0d2818; border:1px solid #1e5c32; border-radius:8px; padding:0.8rem 1rem; text-align:center; margin-bottom:1rem;">'
            '<span style="color:#4ade80; font-weight:700; font-size:1.2rem;">READY TO OPTIMIZE</span>'
            '<p style="color:#86efac; margin:0.2rem 0 0 0; font-size:0.85rem;">Dataset satisfies all mathematical constraints and is within the validated QAOA demonstrator scale (≤ 9 variables).</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif report.status == "READY FOR CLASSICAL OPTIMIZATION (QAOA LOCKED)":
        st.markdown(
            '<div style="background:#2d2106; border:1px solid #78350f; border-radius:8px; padding:0.8rem 1rem; text-align:center; margin-bottom:1rem;">'
            '<span style="color:#fbbf24; font-weight:700; font-size:1.2rem;">READY TO OPTIMIZE (CLASSICAL ONLY — QAOA LOCKED)</span>'
            '<p style="color:#fde68a; margin:0.2rem 0 0 0; font-size:0.85rem;">Valid scheduling model, but scale exceeds the validated QAOA demonstrator scale (≤ 9 variables).</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#2d1215; border:1px solid #6b1f24; border-radius:8px; padding:0.8rem 1rem; text-align:center; margin-bottom:1rem;">'
            '<span style="color:#f87171; font-weight:700; font-size:1.2rem;">BLOCKED</span>'
            '<p style="color:#fca5a5; margin:0.2rem 0 0 0; font-size:0.85rem;">Dataset cannot be mapped to a valid scheduling model. Correct the blocking issues before proceeding.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Blocking Issues Breakdown (if any)
    if report.errors:
        st.markdown("#### Blocking Issues:")
        for err in report.errors:
            st.error(f"• {err}")

    # Operational Notices Breakdown (if any)
    if report.warnings:
        with st.expander("ℹ Operational Notices"):
            for warn in report.warnings:
                st.caption(f"• {warn}")
