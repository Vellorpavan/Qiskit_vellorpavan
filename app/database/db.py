"""
ShiftProof Database Connection Helper

Manages SQLite connection for the interactive application workspace (data/shiftproof.db).
Enforces foreign keys and ensures parameterized queries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = _PROJECT_ROOT / "data" / "shiftproof.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Open connection to SQLite database with foreign keys enabled and row factory set.
    """
    target = db_path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
