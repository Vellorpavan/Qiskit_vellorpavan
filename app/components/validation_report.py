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
        icon = "✓ Pass" if passed else "✗ Fail"
        with cols[i]:
            st.metric(label, icon)

    st.markdown("")

    # Status Banner
    if report.status == "READY TO OPTIMIZE":
        st.success("✓ READY TO OPTIMIZE: Dataset satisfies all mathematical constraints and is within the validated QAOA demonstrator scale (≤ 9 variables).")
    elif "CLASSICAL" in report.status:
        st.warning("⚠️ READY TO OPTIMIZE (CLASSICAL ONLY — QAOA LOCKED): Valid scheduling model, but scale exceeds the validated QAOA demonstrator scale (≤ 9 variables).")
    else:
        st.error("❌ BLOCKED: Dataset cannot be mapped to a valid scheduling model. Correct the blocking issues before proceeding.")

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
