"""
ShiftProof — Application Router & Navigation Hub

Multi-section Streamlit navigation separating the interactive application workspace,
scientific validation benchmark, research evidence, and challenge compliance.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from app.database.db import get_db_connection
from app.database.seed import seed_synthetic_demo, DEMO_DATASET_ID

# Set global page configuration (must precede st.navigation)
st.set_page_config(
    page_title="ShiftProof — Quantum Scheduling Platform",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state defaults — NO default dataset on startup
if "active_dataset_id" not in st.session_state:
    st.session_state["active_dataset_id"] = None
if "selected_instance_id" not in st.session_state:
    st.session_state["selected_instance_id"] = None

# Inject CSS
_CSS_FILE = Path(__file__).parent / "styles" / "theme.css"
if _CSS_FILE.exists():
    with open(_CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

nav_pages = {
    "ShiftProof": [
        st.Page("pages/00_workspace.py", title="Workspace", icon="⚡", default=True),
        st.Page("pages/01_data.py", title="Data", icon="📁"),
        st.Page("pages/02_results.py", title="Results", icon="📋"),
        st.Page("pages/03_quantum.py", title="Quantum", icon="⚛️"),
    ]
}

pg = st.navigation(nav_pages)
pg.run()
