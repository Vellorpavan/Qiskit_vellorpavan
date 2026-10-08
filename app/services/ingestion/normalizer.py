"""
ShiftProof — Ingestion: Normalizer & Validation Engine

Converts parsed, classified scheduling data into the canonical 12-column
ShiftProof representation:
[worker_id, worker_name, shift_id, shift_name, shift_date, shift_type,
 skill, required_skill, assignment_cost, is_eligible, is_available, max_shifts]

Handles:
- Row-based assignments (infers eligibility from row presence)
- Explicit availability/eligibility boolean normalization
- Matrix unpivoting (melting worker x shift columns)
- Scientific feasibility validation for src.model.Instance
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

import numpy as np
import pandas as pd


class NormalizationResult:
    """Canonical normalized workforce dataset ready for database persistence and modeling."""

    def __init__(
        self,
        df_normalized: pd.DataFrame,
        worker_count: int,
        shift_count: int,
        row_count: int,
        eligible_pairs_count: int,
        is_feasible: bool,
        validation_errors: list[str],
        validation_warnings: list[str],
        assumptions: list[str],
        optimization_mode: str = "COST_OPTIMIZATION",
    ):
        self.df_normalized = df_normalized
        self.worker_count = worker_count
        self.shift_count = shift_count
        self.row_count = row_count
        self.eligible_pairs_count = eligible_pairs_count
        self.is_feasible = is_feasible
        self.validation_errors = validation_errors
        self.validation_warnings = validation_warnings
        self.assumptions = assumptions
        self.optimization_mode = optimization_mode

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0 and self.is_feasible

    @property
    def normalized_df(self) -> pd.DataFrame:
        return self.df_normalized


def normalize_dataset(
    df: pd.DataFrame,
    mapping: dict[str, str],
    is_matrix: bool = False,
    matrix_shift_columns: Optional[list[str]] = None,
    matrix_cell_type: str = "cost",  # "cost" | "availability"
    default_cost: float = 10.0,
) -> NormalizationResult:
    """
    Normalizes arbitrary parsed data into canonical workforce scheduling table.
    """
    assumptions = []
    errors = []
    warnings = []

    # 1. Handle Matrix Unpivoting if applicable
    if is_matrix and matrix_shift_columns:
        worker_col = mapping.get("worker", df.columns[0])
        melted_df = df.melt(
            id_vars=[worker_col],
            value_vars=matrix_shift_columns,
            var_name="__shift_name__",
            value_name="__cell_value__",
        )
        assumptions.append(f"Unpivoted {len(matrix_shift_columns)} shift columns into row assignments.")

        if matrix_cell_type == "cost":
            melted_df["__cost__"] = pd.to_numeric(melted_df["__cell_value__"], errors="coerce").fillna(0.0)
            melted_df["__available__"] = 1
            melted_df["__eligible__"] = 1
            opt_mode = "COST_OPTIMIZATION"
        else:
            melted_df["__available__"] = melted_df["__cell_value__"].apply(_parse_boolean_value)
            melted_df["__eligible__"] = melted_df["__available__"]
            melted_df["__cost__"] = 0.0
            opt_mode = "FEASIBILITY_ONLY"
            assumptions.append("Matrix cells interpreted as availability. Mode: FEASIBILITY_ONLY (no monetary costs fabricated).")

        working_df = melted_df
        w_col = worker_col
        s_col = "__shift_name__"
        cost_col = "__cost__"
        avail_col = "__available__"
        elig_col = "__eligible__"
        skill_col = None
        cap_col = None
    else:
        working_df = df.copy()
        w_col = mapping.get("worker")
        s_col = mapping.get("shift")
        cost_col = mapping.get("cost")
        avail_col = mapping.get("availability")
        elig_col = mapping.get("eligibility")
        skill_col = mapping.get("skill")
        cap_col = mapping.get("capacity")

        if cost_col and cost_col in working_df.columns:
            opt_mode = "COST_OPTIMIZATION"
        else:
            opt_mode = "FEASIBILITY_ONLY"
            cost_col = None
            assumptions.append("Optimization Mode: FEASIBILITY_ONLY. Objective cost coefficients set to 0.0 (no monetary costs fabricated).")

    if not w_col or w_col not in working_df.columns:
        errors.append("Missing required Worker column mapping.")
    if not s_col or s_col not in working_df.columns:
        errors.append("Missing required Shift column mapping.")

    # 2. Extract and Normalize Canonical Fields
    canonical_rows = []
    has_explicit_elig = elig_col and elig_col in working_df.columns
    has_explicit_avail = avail_col and avail_col in working_df.columns

    if not has_explicit_elig and not has_explicit_avail:
        assumptions.append("Eligibility inferred from assignment rows (all present rows treated as eligible).")

    for _, r in working_df.iterrows():
        raw_w = str(r[w_col]).strip()
        raw_s = str(r[s_col]).strip()
        if not raw_w or raw_w.lower() in {"nan", "none", ""}:
            continue
        if not raw_s or raw_s.lower() in {"nan", "none", ""}:
            continue

        # Worker
        w_id = _slugify(raw_w)
        w_name = raw_w

        # Shift
        s_id = _slugify(raw_s)
        s_name = raw_s
        s_type = _infer_shift_type(s_name)

        # Cost
        cost_val = 0.0
        if cost_col and cost_col in working_df.columns:
            try:
                raw_c = str(r[cost_col]).replace("$", "").replace("€", "").replace(",", "").strip()
                cost_val = float(raw_c) if raw_c else 0.0
            except Exception:
                cost_val = 0.0
        else:
            cost_val = 0.0

        if cost_val < 0:
            warnings.append(f"Negative cost detected for worker {w_name} on shift {s_name}. Clamped to 0.0.")
            cost_val = 0.0

        # Eligibility & Availability
        is_avail = 1
        if has_explicit_avail:
            is_avail = _parse_boolean_value(r[avail_col])

        is_elig = 1
        if has_explicit_elig:
            is_elig = _parse_boolean_value(r[elig_col])
        elif has_explicit_avail:
            is_elig = is_avail

        # Skill
        skill = str(r[skill_col]).strip() if (skill_col and skill_col in working_df.columns and pd.notna(r[skill_col])) else "Standard"
        req_skill = "Standard"

        # Capacity
        cap = 1
        if cap_col and cap_col in working_df.columns and pd.notna(r[cap_col]):
            try:
                cap = max(1, int(float(r[cap_col])))
            except Exception:
                cap = 1

        canonical_rows.append({
            "worker_id": w_id,
            "worker_name": w_name,
            "shift_id": s_id,
            "shift_name": s_name,
            "shift_date": "2026-10-15",
            "shift_type": s_type,
            "skill": skill,
            "required_skill": req_skill,
            "assignment_cost": cost_val,
            "is_eligible": int(is_elig),
            "is_available": int(is_avail),
            "max_shifts": cap,
        })

    if not canonical_rows:
        raise ValueError("Normalization produced 0 valid rows from source data.")

    norm_df = pd.DataFrame(canonical_rows)

    # 3. Validation & Scientific Feasibility Check
    unique_workers = norm_df["worker_id"].unique()
    unique_shifts = norm_df["shift_id"].unique()
    n_workers = len(unique_workers)
    n_shifts = len(unique_shifts)

    eligible_mask = (norm_df["is_eligible"] == 1) & (norm_df["is_available"] == 1)
    eligible_pairs = int(eligible_mask.sum())

    if n_workers < n_shifts:
        warnings.append(f"Worker count ({n_workers}) is less than shift count ({n_shifts}). Problem may be mathematically overconstrained.")

    # Check shift coverage feasibility: every shift must have at least 1 eligible worker
    eligible_df = norm_df[eligible_mask]
    shifts_with_eligible = set(eligible_df["shift_id"].unique())
    missing_shifts = set(unique_shifts) - shifts_with_eligible
    if missing_shifts:
        missing_names = norm_df[norm_df["shift_id"].isin(missing_shifts)]["shift_name"].unique()
        errors.append(f"Shifts without any eligible worker: {', '.join(missing_names)}. Infeasible scheduling problem.")

    is_feasible = (len(errors) == 0) and (eligible_pairs > 0)

    return NormalizationResult(
        df_normalized=norm_df,
        worker_count=n_workers,
        shift_count=n_shifts,
        row_count=len(norm_df),
        eligible_pairs_count=eligible_pairs,
        is_feasible=is_feasible,
        validation_errors=errors,
        validation_warnings=warnings,
        assumptions=assumptions,
        optimization_mode=opt_mode,
    )


def _slugify(val: str) -> str:
    """Generate deterministic identifier slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", str(val).strip()).strip("_")
    return clean[:32] if clean else "entity_" + uuid.uuid4().hex[:6]


def _infer_shift_type(name: str) -> str:
    """Infer Morning/Afternoon/Night/Standard from shift label."""
    n = name.lower()
    if "night" in n or "graveyard" in n or "late" in n:
        return "Night"
    elif "afternoon" in n or "evening" in n or "mid" in n:
        return "Afternoon"
    elif "morning" in n or "day" in n or "early" in n:
        return "Morning"
    return "Standard"


def _parse_boolean_value(val: Any) -> int:
    """Strict normalization of boolean/availability tokens."""
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    true_tokens = {"1", "1.0", "true", "yes", "y", "t", "available", "eligible", "active"}
    false_tokens = {"0", "0.0", "false", "no", "n", "f", "unavailable", "ineligible", "inactive"}
    if s in true_tokens:
        return 1
    elif s in false_tokens:
        return 0
    return 1 if s else 0
