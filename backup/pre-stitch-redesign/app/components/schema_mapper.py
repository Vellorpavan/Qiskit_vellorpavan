"""
ShiftProof — Schema Mapper Component

Interactive UI for column candidate confirmation and schema mapping.
Automatically accepts unambiguous candidate matches.
Only presents manual mapping selectors if required columns are missing or overridden.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Optional


def render_schema_mapper(df: pd.DataFrame, initial_guesses: dict[str, Optional[str]]) -> Optional[dict[str, str]]:
    """
    Render column mapping selectors for a dataframe.
    Automatically accepts confident mappings without forcing manual re-selection.
    Returns confirmed mapping dict if valid, else None.
    """
    columns = ["(None / Skip)"] + list(df.columns)

    w_det = initial_guesses.get("worker_id")
    s_det = initial_guesses.get("shift_id")
    c_det = initial_guesses.get("cost")
    e_det = initial_guesses.get("eligible")

    # Check if all mandatory fields are detected unambiguously
    is_confident = (
        bool(w_det) and w_det in df.columns
        and bool(s_det) and s_det in df.columns
        and bool(c_det) and c_det in df.columns
    )

    if is_confident:
        st.markdown("### Automatic Column Detection")
        e_display = f"`{e_det}`" if (e_det and e_det in df.columns) else "Default (All Eligible)"

        st.markdown(
            f"""
<div style="background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.8rem;">
  <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:0.6rem; font-size:0.875rem;">
    <div><span style="color:#4ade80; font-weight:700;">✓ Worker ID</span> &nbsp;→&nbsp; <strong style="color:#f0f6fc;">`{w_det}`</strong></div>
    <div><span style="color:#4ade80; font-weight:700;">✓ Shift ID</span> &nbsp;→&nbsp; <strong style="color:#f0f6fc;">`{s_det}`</strong></div>
    <div><span style="color:#4ade80; font-weight:700;">✓ Assignment Cost</span> &nbsp;→&nbsp; <strong style="color:#f0f6fc;">`{c_det}`</strong></div>
    <div><span style="color:#4ade80; font-weight:700;">✓ Eligibility</span> &nbsp;→&nbsp; <strong style="color:#f0f6fc;">{e_display}</strong></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ Override Column Mapping (Optional)"):
            st.caption("Auto-detection was confident. You may manually reassign columns below if necessary.")
            return _render_manual_selectors(df, columns, initial_guesses)

        # Build confident auto-mapping
        auto_map: dict[str, str] = {
            "worker_id": w_det,
            "shift_id": s_det,
            "cost": c_det,
        }
        if e_det and e_det in df.columns:
            auto_map["eligible"] = e_det
        for opt_key in ["worker_name", "shift_name", "shift_date", "shift_type", "skill"]:
            val = initial_guesses.get(opt_key)
            if val and val in df.columns:
                auto_map[opt_key] = val

        return auto_map

    else:
        # Manual selection required
        missing = []
        if not w_det or w_det not in df.columns:
            missing.append("Worker ID")
        if not s_det or s_det not in df.columns:
            missing.append("Shift ID")
        if not c_det or c_det not in df.columns:
            missing.append("Assignment Cost")

        st.warning(
            f"⚠️ **Manual Mapping Required:** Could not automatically detect: `{', '.join(missing)}`. "
            "Please select the corresponding columns below."
        )
        return _render_manual_selectors(df, columns, initial_guesses)


def _render_manual_selectors(
    df: pd.DataFrame,
    columns: list[str],
    initial_guesses: dict[str, Optional[str]],
) -> Optional[dict[str, str]]:
    """Render manual column dropdowns."""
    mapping: dict[str, str] = {}
    missing_mandatory = []

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Mandatory Fields")

        # Worker ID
        default_w = initial_guesses.get("worker_id")
        idx_w = columns.index(default_w) if default_w in columns else 0
        sel_w = st.selectbox("Worker ID *", options=columns, index=idx_w, key="map_worker_id")
        if sel_w != "(None / Skip)":
            mapping["worker_id"] = sel_w
        else:
            missing_mandatory.append("Worker ID")

        # Shift ID
        default_s = initial_guesses.get("shift_id")
        idx_s = columns.index(default_s) if default_s in columns else 0
        sel_s = st.selectbox("Shift ID *", options=columns, index=idx_s, key="map_shift_id")
        if sel_s != "(None / Skip)":
            mapping["shift_id"] = sel_s
        else:
            missing_mandatory.append("Shift ID")

        # Assignment Cost
        default_c = initial_guesses.get("cost")
        idx_c = columns.index(default_c) if default_c in columns else 0
        sel_c = st.selectbox("Assignment Cost / Wage Rate *", options=columns, index=idx_c, key="map_cost")
        if sel_c != "(None / Skip)":
            mapping["cost"] = sel_c
        else:
            missing_mandatory.append("Assignment Cost")

        # Eligibility
        default_el = initial_guesses.get("eligible")
        idx_el = columns.index(default_el) if default_el in columns else 0
        sel_el = st.selectbox("Eligibility Flag (Optional)", options=columns, index=idx_el, key="map_eligible")
        if sel_el != "(None / Skip)":
            mapping["eligible"] = sel_el

    with c2:
        st.markdown("#### Optional Descriptive Fields")
        for field, label in [
            ("worker_name", "Worker Name"),
            ("shift_name", "Shift Name"),
            ("shift_date", "Shift Date"),
            ("shift_type", "Shift Type"),
            ("skill", "Skill / Role"),
        ]:
            default_val = initial_guesses.get(field)
            idx_val = columns.index(default_val) if default_val in columns else 0
            sel_val = st.selectbox(f"{label} (Optional)", options=columns, index=idx_val, key=f"map_{field}")
            if sel_val != "(None / Skip)":
                mapping[field] = sel_val

    if missing_mandatory:
        st.error(f"Missing mandatory field(s): {', '.join(missing_mandatory)}.")
        return None

    return mapping
