"""
ShiftProof — Page 01: Workforce Dataset Management

Upload, auto-detect, preview, validate, and save workforce scheduling datasets.
Supports CSV, Excel (XLSX), and JSON.
Strictly checks schema completeness and mathematical constraints before admitting data.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.database.db import get_db_connection
from app.database.seed import DEMO_DATASET_ID
from app.services.dataset_service import (
    list_datasets,
    parse_uploaded_file,
    profile_dataframe,
    save_mapped_tabular_dataset,
    delete_dataset,
    get_active_dataset,
    set_active_dataset,
)
from app.components.schema_mapper import render_schema_mapper
from app.services.validation_service import validate_mapped_dataframe
from app.components.validation_report import render_validation_report

try:
    st.set_page_config(page_title="Data — ShiftProof", page_icon="📁", layout="wide")
except Exception:
    pass

_CSS = Path(__file__).parents[1] / "styles" / "theme.css"
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

st.markdown("# 📁 Workforce Dataset Management")
st.markdown(
    "Import, inspect, and validate workforce scheduling datasets. "
    "Only validated scheduling datasets can enter the classical and quantum optimization pipeline."
)
st.divider()

conn = get_db_connection()
datasets = list_datasets(conn)

tab_upload, tab_manage = st.tabs(["📤 Upload New Dataset", "📋 Workspace Datasets"])

# ── Tab 1: Upload New Dataset ─────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload Workforce Scheduling Dataset")
    st.markdown(
        "Upload a **CSV**, **Excel (XLSX)**, or **JSON** file containing worker assignments, duties, and costs. "
        "Max file size: **5 MB**."
    )

    # Clean Supported Model Notice
    st.markdown(
        """
<div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:1.2rem; font-size:0.875rem;">
  <strong style="color:#38bdf8; font-size:0.92rem;">Workforce & Resource Scheduling Data Model</strong><br>
  <span style="color:#cbd5e1; line-height:1.5;">
    ShiftProof requires data mapping to staff, shifts/duties, and assignment costs.
    Arbitrary tabular datasets (e.g. sales logs, time-series, weather) cannot be mapped to the scheduling model.
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose file to upload",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload tabular assignment records (CSV/XLSX/JSON)",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        # Stable session-state management for user-editable dataset display name
        default_name = Path(filename).stem
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != filename:
            st.session_state["last_uploaded_file"] = filename
            st.session_state["dataset_display_name"] = default_name
            st.session_state["dataset_description"] = f"Workforce schedule imported from {filename}."

        try:
            parsed_data, format_type = parse_uploaded_file(file_bytes, filename)

            if isinstance(parsed_data, pd.DataFrame):
                df_upload = parsed_data
                profile = profile_dataframe(df_upload)
                candidate_mappings = profile["candidate_mappings"]

                # ── Dynamic Calculation for Preview Summary (Section 6) ──────
                w_col = candidate_mappings.get("worker_id")
                s_col = candidate_mappings.get("shift_id")
                e_col = candidate_mappings.get("eligible")

                calc_workers = int(df_upload[w_col].dropna().nunique()) if (w_col and w_col in df_upload.columns) else len(df_upload)
                calc_shifts = int(df_upload[s_col].dropna().nunique()) if (s_col and s_col in df_upload.columns) else len(df_upload)
                calc_records = len(df_upload)
                calc_missing = int(df_upload.isnull().sum().sum())

                if e_col and e_col in df_upload.columns:
                    calc_eligible = int((df_upload[e_col].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "t", "y"])).sum())
                else:
                    calc_eligible = calc_records

                calc_qaoa_vars = calc_eligible

                # ── Preview Summary Bar (Section 6) ──────────────────────────
                st.markdown("#### Data Summary")
                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Workers", calc_workers)
                s2.metric("Shifts", calc_shifts)
                s3.metric("Records", calc_records)
                s4.metric("Eligible", calc_eligible)
                s5.metric("Missing", calc_missing)
                s6.metric("QAOA Variables", calc_qaoa_vars)

                st.markdown("")

                # ── Bounded Full-Data Preview (Section 5) ─────────────────────
                st.markdown("#### Data Preview")
                st.caption(f"Showing **{calc_records} of {calc_records} records** (scrollable table)")
                st.dataframe(df_upload, height=340, use_container_width=True)

                st.divider()

                # ── Automatic Column Mapping (Section 7) ──────────────────────
                confirmed_mapping = render_schema_mapper(df_upload, candidate_mappings)

                if confirmed_mapping is not None:
                    st.divider()

                    # ── Concise 7-Point Validation (Section 8) ─────────────────
                    report = validate_mapped_dataframe(df_upload, confirmed_mapping, dataset_id="upload_preview")
                    render_validation_report(report)

                    if report.is_valid_model:
                        st.divider()

                        # ── Simple Save Dataset UI (Section 9 & 10) ────────────
                        st.markdown("### Save Dataset")

                        c_save1, c_save2 = st.columns([2, 1])
                        with c_save1:
                            user_entered_name = st.text_input(
                                "Dataset Name",
                                key="dataset_display_name",
                                help="User-friendly display name. Preserved across all reruns and pages.",
                            )
                            user_entered_desc = st.text_area(
                                "Description (Optional)",
                                key="dataset_description",
                            )
                        with c_save2:
                            st.markdown("**File Source:**")
                            st.markdown(f"`{filename}`")
                            st.markdown("**Format:**")
                            st.markdown(f"`{format_type.upper()}`")
                            st.markdown("**Total Records:**")
                            st.markdown(f"**{calc_records}** records")

                        if st.button("💾 Save Dataset", type="primary", use_container_width=False):
                            saved_display_name = user_entered_name.strip() or default_name
                            new_id = f"ds_{uuid.uuid4().hex[:8]}"

                            save_mapped_tabular_dataset(
                                dataset_id=new_id,
                                name=saved_display_name,
                                description=user_entered_desc.strip(),
                                df=df_upload,
                                mapping=confirmed_mapping,
                                conn=conn,
                                source_filename=filename,
                            )

                            set_active_dataset(new_id)
                            st.session_state["just_saved_dataset_name"] = saved_display_name
                            st.success(f"✓ Dataset saved: **{saved_display_name}**")
                            st.rerun()

                        # If dataset was just saved, offer direct continue button (Section 11)
                        if st.session_state.get("workspace_dataset_id") and st.session_state.get("just_saved_dataset_name"):
                            curr_name = st.session_state.get("just_saved_dataset_name")
                            st.markdown(
                                f"""
<div style="background:#0d2818; border:1px solid #1e5c32; border-radius:8px; padding:0.9rem 1.1rem; margin-top:1rem;">
  <strong style="color:#4ade80;">✓ Dataset saved:</strong> <span style="color:#f0f6fc;"><strong>{curr_name}</strong></span><br>
  <span style="color:#86efac; font-size:0.88rem;">Ready for classical and quantum optimization.</span>
</div>
""",
                                unsafe_allow_html=True,
                            )
                            st.markdown("")
                            if st.button("🚀 Continue to Optimization", type="primary"):
                                st.switch_page("pages/00_workspace.py")

                    else:
                        st.error("❌ Dataset cannot be admitted to optimization pipeline due to validation errors above.")

            elif isinstance(parsed_data, dict):
                st.info("Format B (Relational JSON) detected. Feature coming in future release.")

        except Exception as e:
            st.error(f"❌ Failed to parse or process file: {e}")

# ── Tab 2: Existing Workspace Datasets ────────────────────────────────────────
with tab_manage:
    st.markdown("### Registered Workspace Datasets")
    if not datasets:
        st.info("No datasets currently registered.")
    else:
        # Prepare clean display table
        display_rows = []
        for d in datasets:
            src = d.get("source_filename") or ("Synthetic Demo" if d.get("is_synthetic") else d.get("source_type", "upload"))
            display_rows.append({
                "Dataset Name": d["name"],
                "Source": src,
                "Workers": d["worker_count"],
                "Shifts": d["shift_count"],
                "Records": d["row_count"],
                "Status": d["status"].upper(),
                "Registered At": str(d["created_at"])[:19],
                "_id": d["dataset_id"],
            })

        df_display = pd.DataFrame(display_rows)
        st.dataframe(
            df_display[[
                "Dataset Name", "Source", "Workers",
                "Shifts", "Records", "Status", "Registered At"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### Switch Active Dataset")
            dataset_ids = [d["dataset_id"] for d in datasets]
            current_active = get_active_dataset()
            active_idx = dataset_ids.index(current_active) if current_active in dataset_ids else 0

            selected_to_activate = st.selectbox(
                "Choose active dataset",
                options=dataset_ids,
                index=active_idx,
                format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                label_visibility="collapsed",
            )
            if st.button("Activate Selected Dataset"):
                set_active_dataset(selected_to_activate)
                chosen_meta = next((d for d in datasets if d["dataset_id"] == selected_to_activate), None)
                chosen_name = chosen_meta["name"] if chosen_meta else selected_to_activate
                st.success(f"✓ Active dataset switched to: **{chosen_name}**")
                st.rerun()

        with col_m2:
            st.markdown("#### Delete Dataset")
            deletable = [d["dataset_id"] for d in datasets if not d.get("is_synthetic") and d["dataset_id"] != DEMO_DATASET_ID]
            if deletable:
                del_choice = st.selectbox(
                    "Select dataset to delete",
                    options=deletable,
                    format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                    label_visibility="collapsed",
                )
                if st.button("Delete Selected Dataset", type="primary"):
                    if delete_dataset(del_choice, conn):
                        st.success("✓ Dataset removed from workspace.")
                        if get_active_dataset() == del_choice:
                            set_active_dataset(DEMO_DATASET_ID)
                        st.rerun()
            else:
                st.caption("Only user-uploaded datasets can be deleted. The synthetic demo is protected.")
