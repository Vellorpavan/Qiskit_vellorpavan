"""
ShiftProof — Page 01: Universal Workforce Dataset Management

Upload, inspect, understand, classify, preview, validate, and activate workforce scheduling datasets.
Supports CSV, TSV, XLSX, XLS, JSON, XML, and PDF.
Deterministic local parsing by default, with optional Gemini document adapter.
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
from app.services.dataset_service import (
    list_datasets,
    delete_dataset,
    get_active_dataset,
    set_active_dataset,
    get_active_dataset_identity,
    load_dataset_table,
)
from app.services.ingestion import (
    inspect_uploaded_file,
    finalize_and_save_dataset,
    is_gemini_configured,
    ClassificationStatus,
    FileCategory,
    FileType,
)
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint

try:
    st.set_page_config(page_title="Data — ShiftProof", page_icon="📁", layout="wide")
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

st.markdown("# 📁 Workforce Dataset Management")
st.caption("Universal data ingestion, semantic understanding, validation, and normalization for Qiskit optimization.")

# ── Canonical Active Dataset Identity & Workspace Transition Bar ───────────
if active_id:
    active_ident = get_active_dataset_identity(conn)
    act_name = active_ident["name"]
    act_fp = active_ident["fingerprint"] or "computed"
    col_bar1, col_bar2 = st.columns([3, 1])
    with col_bar1:
        st.success(
            f"🟢 **Active Dataset:** **{act_name}** (`{active_id}`) | "
            f"Fingerprint: `{act_fp[:16]}...` | **READY — NOT RUN**"
        )
    with col_bar2:
        if st.button("⚡ Go to Workspace", type="primary", key="btn_top_go_workspace", use_container_width=True):
            st.switch_page("pages/00_workspace.py")
else:
    col_bar1, col_bar2 = st.columns([3, 1])
    with col_bar1:
        st.info("⚪ **No dataset currently active.** Upload a file below or activate a registered dataset.")
    with col_bar2:
        if st.button("⚡ Go to Workspace", key="btn_top_go_workspace_disabled", use_container_width=True):
            st.warning("No active dataset. Return to Data and activate a dataset first.")

st.divider()

tab_upload, tab_active, tab_registry = st.tabs([
    "📤 Upload New Dataset",
    "🔍 Active Dataset View & Preview",
    "📋 Registered Datasets",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD NEW DATASET (Universal Ingestion Engine)
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Upload Workforce Scheduling Dataset")
    st.markdown(
        "Upload a workforce scheduling file. Supports **CSV**, **TSV**, **Excel (XLSX/XLS)**, **JSON**, **XML**, or **PDF** roster documents. "
        "ShiftProof automatically inspects file structure, classifies the problem, and extracts canonical scheduling concepts."
    )

    # Status of optional intelligent adapter
    gem_status = "Configured" if is_gemini_configured() else "Not configured (Local deterministic ingestion active)"
    st.caption(f"Intelligent Document Understanding Adapter: **{gem_status}**")

    uploaded_file = st.file_uploader(
        "Choose scheduling dataset file",
        type=["csv", "tsv", "xlsx", "xls", "json", "xml", "pdf"],
        help="Upload tabular assignment records or roster documents",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        default_stem = Path(filename).stem

        # Multi-sheet Excel support
        sheet_choice = None
        if filename.lower().endswith((".xlsx", ".xls")):
            try:
                import io
                excel_obj = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
                if len(excel_obj.sheet_names) > 1:
                    sheet_choice = st.selectbox(
                        "Workbook contains multiple sheets. Select target sheet to inspect:",
                        options=excel_obj.sheet_names,
                        key="ws_sheet_selector",
                    )
            except Exception:
                pass

        try:
            # 1. Run Universal Ingestion Inspection
            inspection = inspect_uploaded_file(file_bytes, filename, sheet_name=sheet_choice)
            df_raw = inspection.parsed_df
            raw_cols = list(df_raw.columns)
            classification = inspection.classification

            # 2. Ingestion Provenance & Metadata Bar
            st.divider()
            st.subheader("INGESTION & STRUCTURE ANALYSIS")
            prov_c1, prov_c2, prov_c3, prov_c4 = st.columns(4)
            prov_c1.metric("Detected Format", inspection.file_type.value.upper())
            prov_c2.metric("Category", inspection.file_category.value.title())
            prov_c3.metric("Raw Rows", len(df_raw))
            prov_c4.metric("Raw Columns", len(df_raw.columns))

            # 3. Scheduling Problem Classification Diagnosis
            st.markdown("#### Classification Diagnosis")
            if classification.status in (ClassificationStatus.VALID_SCHEDULING_DATASET, ClassificationStatus.SCHEDULING_DATASET_NEEDS_MAPPING):
                if classification.status == ClassificationStatus.VALID_SCHEDULING_DATASET:
                    st.success("✓ **Scheduling dataset detected**")
                else:
                    st.warning("⚠️ **Scheduling dataset detected (Confirmation required)**")

                diag_c1, diag_c2 = st.columns(2)
                with diag_c1:
                    if classification.optimization_mode == "FEASIBILITY_ONLY":
                        st.info("🎯 **Optimization mode:**\n**FEASIBILITY ONLY**")
                    else:
                        st.success("🎯 **Optimization mode:**\n**COST OPTIMIZATION**")
                with diag_c2:
                    if classification.optimization_mode == "FEASIBILITY_ONLY":
                        st.warning("💰 **Cost / Wage:**\n**Not provided**")
                    else:
                        st.info(f"💰 **Cost / Wage:**\n`{classification.detected_mapping.get('cost')}`")

                st.caption(f"**Explanation:** {classification.explanation}")
            elif classification.status == ClassificationStatus.POTENTIAL_SHIFT_MATRIX:
                st.info("ℹ️ **POTENTIAL SHIFT MATRIX STRUCTURE DETECTED**")
                st.caption(classification.explanation)
            elif classification.status == ClassificationStatus.SCHEDULING_DATASET_MISSING_REQUIRED_CONCEPT:
                st.error("⚠️ **Scheduling problem not detected.**")
                st.markdown(classification.explanation)
                st.markdown("**Diagnosis:**")
                st.write(f"• **Detected concepts:** {', '.join(classification.detected_concepts) if classification.detected_concepts else 'None'}")
                st.write(f"• **Missing required concepts:** {', '.join(classification.missing_concepts)}")
                st.info("💡 **Suggested action:** Use the manual mapping controls below to map an existing column to the missing concept.")
            elif classification.status == ClassificationStatus.NOT_A_SCHEDULING_DATASET:
                st.error("⚠️ **This file does not appear to contain a workforce scheduling problem.**")
                st.caption(classification.explanation)
                st.write(f"• **Missing required concepts:** {', '.join(classification.missing_concepts)}")

            # 4. Canonical Concept Mapping & Confirmation
            st.divider()
            st.subheader("CANONICAL CONCEPT MAPPING")

            is_matrix_mode = classification.is_matrix
            matrix_cell_type = "cost"

            if is_matrix_mode:
                st.markdown("##### Matrix Structure Interpretation")
                st.write(f"• **Worker Entity Column:** `{classification.detected_mapping.get('worker', raw_cols[0])}`")
                st.write(f"• **Shift Slot Columns:** `{', '.join(classification.matrix_shift_columns)}`")
                matrix_meaning = st.radio(
                    "What do cell values in these shift columns represent?",
                    options=["Assignment Cost ($) per shift", "Worker Availability / Eligibility (Yes/No)"],
                    key="matrix_meaning_radio",
                )
                matrix_cell_type = "cost" if "Cost" in matrix_meaning else "availability"

                user_mapping = {"worker": classification.detected_mapping.get("worker", raw_cols[0])}
            else:
                col_m1, col_m2 = st.columns(2)
                user_mapping = {}

                with col_m1:
                    # Worker Column
                    worker_default = classification.detected_mapping.get("worker")
                    w_idx = raw_cols.index(worker_default) if (worker_default and worker_default in raw_cols) else 0
                    user_mapping["worker"] = st.selectbox(
                        "Worker / Employee Identity Column (Required):",
                        options=raw_cols,
                        index=w_idx,
                        key="map_worker_col",
                        help="Identifies the workforce staff member or resource.",
                    )

                    # Shift Column
                    shift_default = classification.detected_mapping.get("shift")
                    s_idx = raw_cols.index(shift_default) if (shift_default and shift_default in raw_cols) else (1 if len(raw_cols) > 1 else 0)
                    user_mapping["shift"] = st.selectbox(
                        "Shift / Duty / Slot Column (Required):",
                        options=raw_cols,
                        index=s_idx,
                        key="map_shift_col",
                        help="Identifies the shift, duty, or time block to be filled.",
                    )

                with col_m2:
                    # Cost Column
                    cost_options = ["None (Feasibility Only — No Monetary Costs)"] + raw_cols
                    cost_default = classification.detected_mapping.get("cost")
                    c_idx = (raw_cols.index(cost_default) + 1) if (cost_default and cost_default in raw_cols) else 0
                    chosen_cost_opt = st.selectbox(
                        "Assignment Cost / Wage Column (Optional):",
                        options=cost_options,
                        index=c_idx,
                        key="map_cost_col",
                        help="When omitted, ShiftProof optimizes constraint feasibility without fabricating monetary costs.",
                    )
                    user_mapping["cost"] = chosen_cost_opt if chosen_cost_opt != "None (Feasibility Only — No Monetary Costs)" else None

                    # Availability / Eligibility Column
                    avail_options = ["Inferred from assignment rows"] + raw_cols
                    avail_default = classification.detected_mapping.get("availability")
                    a_idx = (raw_cols.index(avail_default) + 1) if (avail_default and avail_default in raw_cols) else 0
                    chosen_avail_opt = st.selectbox(
                        "Availability / Eligibility Flag Column (Optional):",
                        options=avail_options,
                        index=a_idx,
                        key="map_avail_col",
                        help="Explicit boolean flag indicating if worker is available for shift.",
                    )
                    user_mapping["availability"] = chosen_avail_opt if chosen_avail_opt != "Inferred from assignment rows" else None

            # 5. Original Data Preview (Bounded scrollable container)
            st.divider()
            st.subheader("ORIGINAL DATA PREVIEW")
            st.caption(f"Bounded view of raw uploaded data ({len(df_raw)} rows × {len(df_raw.columns)} columns)")
            st.dataframe(df_raw, height=300, use_container_width=True)

            # 6. Save & Activate Section (No auto-optimization)
            if classification.status != ClassificationStatus.NOT_A_SCHEDULING_DATASET:
                st.divider()
                st.subheader("Save & Activate Workforce Dataset")

                c_save1, c_save2 = st.columns([2, 1])
                with c_save1:
                    user_entered_name = st.text_input(
                        "Dataset Name",
                        value=st.session_state.get("dataset_display_name", default_stem),
                        key="ws_ds_name_input",
                        help="User-friendly name preserved across runs.",
                    )
                    user_entered_desc = st.text_area(
                        "Clinical / Operational Scope",
                        value=st.session_state.get("dataset_description", f"Imported from {filename}."),
                        key="ws_ds_desc_input",
                        height=70,
                    )
                with c_save2:
                    st.write(f"**Source File:** `{filename}`")
                    st.write(f"**Format:** `{inspection.file_type.value.upper()}`")
                    st.write(f"**Raw Records:** `{len(df_raw)}`")

                if st.button("💾 Save and Activate Dataset", type="primary", use_container_width=True, key="btn_save_activate_upload"):
                    with st.spinner("Normalizing data and validating mathematical feasibility..."):
                        try:
                            saved_display_name = user_entered_name.strip() or default_stem

                            new_id, norm_res = finalize_and_save_dataset(
                                conn=conn,
                                inspection=inspection,
                                user_mapping=user_mapping,
                                dataset_name=saved_display_name,
                                is_matrix=is_matrix_mode,
                                matrix_cell_type=matrix_cell_type,
                            )

                            set_active_dataset(new_id, conn)
                            st.session_state["dataset_just_saved_id"] = new_id
                            st.session_state["just_saved_dataset_name"] = saved_display_name
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Normalization & Validation error: {ex}")

        except Exception as e:
            st.error(f"❌ Failed to parse or inspect file '{filename}': {e}")

    # Show post-activation confirmation and persistent Go to Workspace button
    if st.session_state.get("dataset_just_saved_id") and st.session_state.get("dataset_just_saved_id") == active_id:
        st.divider()
        st.success(f"✓ Dataset validated and active: **{st.session_state.get('just_saved_dataset_name', active_id)}**")
        st.info("Dataset loaded. Run an experiment to generate results.")
        col_gows1, col_gows2 = st.columns([1, 2])
        with col_gows1:
            if st.button("⚡ Go to Workspace to Optimize", type="primary", key="btn_go_ws_saved", use_container_width=True):
                st.switch_page("pages/00_workspace.py")
        with col_gows2:
            st.page_link("pages/00_workspace.py", label="Open Workspace in Navigation", icon="⚡")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ACTIVE DATASET VIEW & PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_active:
    if not active_id:
        st.info("No dataset is currently active. Select one below or upload on the Upload tab.")
        if datasets:
            pick_to_view = st.selectbox(
                "Choose dataset to activate and inspect:",
                options=dataset_ids,
                format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                key="data_tab2_select",
            )
            if st.button("Activate and View Data", key="data_tab2_btn"):
                set_active_dataset(pick_to_view)
                st.rerun()
    else:
        cur = conn.cursor()
        cur.execute("SELECT * FROM datasets WHERE dataset_id = ?", (active_id,))
        rec = cur.fetchone()
        cur_meta = dict(rec) if rec else {}
        cur_name = cur_meta.get("name", active_id)
        cur_desc = cur_meta.get("description", "No description provided.")
        cur_source = cur_meta.get("source_filename") or cur_meta.get("source_type", "upload")

        try:
            active_inst = db_to_instance(active_id, conn)
            active_fp = compute_dataset_fingerprint(active_id, conn)
            n_w = active_inst.n_workers
            n_s = active_inst.n_shifts
            n_v = active_inst.n_vars
            q_elig = (n_v <= 9)
        except Exception as ex:
            st.error(f"Error loading active dataset: {ex}")
            st.stop()

        df_active = load_dataset_table(active_id, conn)
        n_rows = len(df_active)
        n_cols = len(df_active.columns)

        # DATASET SUMMARY
        st.subheader("DATASET SUMMARY")
        ds1, ds2, ds3, ds4, ds5 = st.columns(5)
        ds1.metric("Workers", n_w)
        ds2.metric("Shifts", n_s)
        ds3.metric("Row Count", n_rows)
        ds4.metric("Column Count", n_cols)
        ds5.metric("Eligible Assignments", n_v)

        dsa1, dsa2, dsa3, dsa4 = st.columns(4)
        dsa1.metric("QAOA Variables", n_v)
        dsa2.metric("QAOA Eligibility", "Eligible (<=9)" if q_elig else f"Locked ({n_v}>9)")
        dsa3.metric("Source Format", cur_source)
        dsa4.metric("Validation Status", "Verified")

        st.write(f"• **Name:** **{cur_name}** | **Dataset ID:** `{active_id}`")
        st.write(f"• **Fingerprint:** `{active_fp}`")
        st.write(f"• **Clinical / Operational Scope:** {cur_desc}")

        st.divider()

        # SCHEMA
        st.subheader("CANONICAL NORMALIZED SCHEMA (12 Columns)")
        sm_c1, sm_c2, sm_c3 = st.columns(3)
        with sm_c1:
            st.write("• **Worker ID:** `worker_id`")
            st.write("• **Worker Name:** `worker_name`")
            st.write("• **Shift ID:** `shift_id`")
            st.write("• **Shift Name:** `shift_name`")
        with sm_c2:
            st.write("• **Date:** `shift_date`")
            st.write("• **Shift Type:** `shift_type`")
            st.write("• **Skill:** `skill`")
            st.write("• **Required Skill:** `required_skill`")
        with sm_c3:
            st.write("• **Assignment Cost:** `assignment_cost`")
            st.write("• **Eligibility:** `is_eligible`")
            st.write("• **Availability:** `is_available`")
            st.write("• **Capacity:** `max_shifts`")

        st.divider()

        # DATA PREVIEW (Bounded container, full table)
        st.subheader("NORMALIZED DATA PREVIEW")
        st.caption(f"Bounded container with horizontal and vertical scrolling ({n_rows} rows × {n_cols} columns)")
        st.dataframe(df_active, height=360, use_container_width=True)

        st.markdown("")
        col_act1, col_act2 = st.columns([1, 2])
        with col_act1:
            if st.button("⚡ Go to Workspace to Optimize", type="primary", use_container_width=True, key="btn_go_workspace_tab2"):
                st.switch_page("pages/00_workspace.py")
        with col_act2:
            st.page_link("pages/00_workspace.py", label="Open Workspace in Navigation", icon="⚡")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: REGISTERED DATASETS
# ══════════════════════════════════════════════════════════════════════════════
with tab_registry:
    st.subheader("Workspace Datasets Registry")

    if not datasets:
        st.info("No datasets currently registered.")
    else:
        display_rows = []
        for d in datasets:
            src = d.get("source_filename") or ("Synthetic Demo" if d.get("is_synthetic") else d.get("source_type", "upload"))
            is_active = (d["dataset_id"] == active_id)
            display_rows.append({
                "Status": "ACTIVE" if is_active else "Standby",
                "Dataset Name": d["name"],
                "Source": src,
                "Workers": d["worker_count"],
                "Shifts": d["shift_count"],
                "Records": d["row_count"],
                "Dataset ID": d["dataset_id"],
                "Registered": str(d.get("created_at", ""))[:19],
            })

        df_reg = pd.DataFrame(display_rows)
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

        st.divider()
        c_sw, c_del = st.columns(2)

        with c_sw:
            st.markdown("#### Switch Active Dataset")
            target_switch = st.selectbox(
                "Select dataset to make active",
                options=dataset_ids,
                index=dataset_ids.index(active_id) if active_id in dataset_ids else 0,
                format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                label_visibility="collapsed",
                key="reg_switch_select",
            )
            col_sw1, col_sw2 = st.columns(2)
            with col_sw1:
                if st.button("Activate Selected Dataset", key="reg_btn_activate", use_container_width=True):
                    set_active_dataset(target_switch, conn)
                    st.success("✓ Active dataset switched.")
                    st.rerun()
            with col_sw2:
                if st.button("⚡ Go to Workspace", key="reg_btn_go_ws", use_container_width=True):
                    if get_active_dataset():
                        st.switch_page("pages/00_workspace.py")
                    else:
                        st.error("No active dataset. Return to Data and activate a dataset first.")

        with c_del:
            st.markdown("#### Remove Dataset")
            deletable_ids = [d["dataset_id"] for d in datasets if not d.get("is_synthetic")]
            if deletable_ids:
                target_del = st.selectbox(
                    "Select dataset to remove",
                    options=deletable_ids,
                    format_func=lambda did: next((d["name"] for d in datasets if d["dataset_id"] == did), did),
                    label_visibility="collapsed",
                    key="reg_del_select",
                )
                if st.button("Delete Dataset", type="primary", key="reg_btn_del", use_container_width=True):
                    if delete_dataset(target_del, conn):
                        st.session_state.pop("reg_del_select", None)
                        st.session_state.pop("reg_switch_select", None)
                        st.success("✓ Dataset removed from workspace.")
                        st.rerun()
            else:
                st.caption("No user datasets available to delete.")
