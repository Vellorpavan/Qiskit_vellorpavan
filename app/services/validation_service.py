"""
ShiftProof — Dataset Validation Service

Implements strict 7-point validation pipeline:
V1: Schema Completeness
V2: Null / Empty Values
V3: Non-Negative Costs
V4: Duplicate Worker-Shift Pairs
V5: Every Shift Has At Least One Eligible Worker
V6: Worker Capacity Feasibility
V7: Computational Scale Check (<=9 variables for QAOA demonstrator)

Applies all scientific corrections:
- QAOA scale limit: strictly <= 9 variables / qubits.
- Exact enumeration: application safety limit (<= 16 variables).
- Greedy: scalable classical fallback subject to resource limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class ValidationReport:
    """Full dataset validation report."""
    dataset_id: str
    is_valid_model: bool
    status: str                         # 'READY TO OPTIMIZE', 'BLOCKED', 'READY FOR CLASSICAL ONLY'
    qaoa_eligible: bool
    exact_eligible: bool
    greedy_eligible: bool
    n_workers: int = 0
    n_shifts: int = 0
    n_vars: int = 0                     # Number of eligible (w,s) variables
    n_qubits: int = 0
    checks: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_mapped_dataframe(
    df: pd.DataFrame,
    mapping: dict[str, str],
    dataset_id: str = "upload",
) -> ValidationReport:
    """
    Execute V1 through V7 validation rules on a mapped DataFrame.
    Never silently modifies data.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, dict] = {}

    w_col = mapping.get("worker_id")
    s_col = mapping.get("shift_id")
    c_col = mapping.get("cost")
    el_col = mapping.get("eligible")

    # ── V1: Schema Completeness ───────────────────────────────────────────────
    v1_ok = bool(w_col and s_col and c_col and w_col in df.columns and s_col in df.columns and c_col in df.columns)
    if not v1_ok:
        errors.append("V1 Schema Incomplete: One or more mandatory fields are not mapped.")
    checks["V1_schema"] = {
        "title": "V1: Schema Completeness",
        "passed": v1_ok,
        "detail": "All mandatory columns mapped" if v1_ok else "Missing mandatory columns",
    }
    if not v1_ok:
        return ValidationReport(
            dataset_id=dataset_id,
            is_valid_model=False,
            status="BLOCKED",
            qaoa_eligible=False,
            exact_eligible=False,
            greedy_eligible=False,
            checks=checks,
            errors=errors,
        )

    # ── V2: Null / Empty Values ───────────────────────────────────────────────
    null_w = df[w_col].isnull().sum()
    null_s = df[s_col].isnull().sum()
    null_c = df[c_col].isnull().sum()
    v2_ok = (null_w == 0) and (null_s == 0) and (null_c == 0)
    if not v2_ok:
        errors.append(f"V2 Null Values Detected: {null_w} null workers, {null_s} null shifts, {null_c} null costs.")
    checks["V2_nulls"] = {
        "title": "V2: Null / Empty Checks",
        "passed": v2_ok,
        "detail": "Zero nulls in primary fields" if v2_ok else f"Found null entries (W:{null_w}, S:{null_s}, C:{null_c})",
    }

    # ── V3: Non-Negative Costs ────────────────────────────────────────────────
    numeric_costs = pd.to_numeric(df[c_col], errors="coerce")
    nan_costs = numeric_costs.isnull().sum()
    neg_costs = (numeric_costs < 0).sum() if nan_costs == 0 else 0
    v3_ok = (nan_costs == 0) and (neg_costs == 0)
    if nan_costs > 0:
        errors.append(f"V3 Invalid Cost Values: {nan_costs} cost values could not be parsed as numeric.")
    if neg_costs > 0:
        errors.append(f"V3 Negative Costs: {neg_costs} cost entries are negative. ShiftProof requires non-negative costs.")
    checks["V3_costs"] = {
        "title": "V3: Non-Negative Costs",
        "passed": v3_ok,
        "detail": "All costs numeric and non-negative" if v3_ok else f"Invalid costs (non-numeric:{nan_costs}, negative:{neg_costs})",
    }

    # ── V4: Duplicate Worker-Shift Pairs ──────────────────────────────────────
    dup_count = df.duplicated(subset=[w_col, s_col]).sum()
    v4_ok = (dup_count == 0)
    if not v4_ok:
        errors.append(f"V4 Duplicate Pairs: Found {dup_count} duplicate (worker, shift) pairs. Each assignment candidate must be unique.")
    checks["V4_duplicates"] = {
        "title": "V4: Unique Assignment Pairs",
        "passed": v4_ok,
        "detail": "Zero duplicate (worker, shift) pairs" if v4_ok else f"Found {dup_count} duplicates",
    }

    # Extract distinct workers & shifts
    workers = df[w_col].dropna().unique().tolist()
    shifts = df[s_col].dropna().unique().tolist()
    n_workers = len(workers)
    n_shifts = len(shifts)

    # ── V5: Shift Coverage Feasibility ────────────────────────────────────────
    # Determine eligibility for each row
    if el_col and el_col in df.columns:
        is_eligible_series = df[el_col].apply(
            lambda x: (
                bool(x) if isinstance(x, (bool, np.bool_))
                else bool(x != 0) if isinstance(x, (int, float, np.number))
                else str(x).strip().lower() in ("true", "1", "yes", "y", "eligible", "available")
            )
        )
    else:
        is_eligible_series = pd.Series([True] * len(df), index=df.index)

    df_eligible = df[is_eligible_series]
    covered_shifts = df_eligible[s_col].dropna().unique().tolist()
    uncovered_shifts = [s for s in shifts if s not in covered_shifts]
    v5_ok = (len(uncovered_shifts) == 0)
    if not v5_ok:
        errors.append(f"V5 Shift Coverage Impossible: The following shift(s) have ZERO eligible workers: {uncovered_shifts}")
    checks["V5_coverage"] = {
        "title": "V5: Shift Coverage Eligibility",
        "passed": v5_ok,
        "detail": "Every shift has ≥1 eligible worker" if v5_ok else f"{len(uncovered_shifts)} shift(s) have 0 eligible workers",
    }

    # ── V6: Worker Capacity Feasibility ───────────────────────────────────────
    # Under current model: each worker assigned to at most 1 shift; each shift needs 1 worker
    # Therefore, n_workers >= n_shifts is necessary for a full schedule
    v6_ok = (n_workers >= n_shifts)
    if not v6_ok:
        errors.append(f"V6 Worker Capacity Violation: Total workers ({n_workers}) < total shifts ({n_shifts}). Under the 1-shift-per-worker rule, not all shifts can be covered.")
    checks["V6_capacity"] = {
        "title": "V6: Worker Capacity Feasibility",
        "passed": v6_ok,
        "detail": f"{n_workers} workers available for {n_shifts} shifts" if v6_ok else f"Insufficient workers ({n_workers} < {n_shifts})",
    }

    # Basic model validity: V1-V6 must all pass
    is_valid_model = v1_ok and v2_ok and v3_ok and v4_ok and v5_ok and v6_ok

    # Calculate variable count (only eligible pairs receive a binary variable/qubit)
    n_vars = len(df_eligible) if is_valid_model else 0
    n_qubits = n_vars

    # ── V7: Computational Scale Check ─────────────────────────────────────────
    # CORRECTION 1: QAOA demonstrator scale is strictly <= 9 variables / qubits.
    qaoa_eligible = is_valid_model and (n_vars <= 9)

    # CORRECTION 2: Exact enumeration application safety limit: <= 16 variables.
    exact_eligible = is_valid_model and (n_vars <= 16)

    # CORRECTION 3: Greedy heuristic is the scalable classical fallback.
    greedy_eligible = is_valid_model

    scale_passed = is_valid_model and qaoa_eligible
    if is_valid_model:
        if not qaoa_eligible:
            warnings.append(
                f"QAOA Scale Locked: Dataset requires {n_vars} variables/qubits. "
                f"Exceeds the currently validated QAOA demonstrator limit (<= 9 variables). "
                f"Classical optimization remains available."
            )
        if not exact_eligible:
            warnings.append(
                f"Exact Enumeration Bypassed: Dataset requires {n_vars} variables (2^{n_vars} = {2**n_vars:,} states). "
                f"Exceeds the application safety limit (<= 16 variables). Greedy heuristic available."
            )

    checks["V7_scale"] = {
        "title": "V7: Computational Scale Check",
        "passed": scale_passed,
        "detail": (
            f"QAOA Eligible ({n_vars} vars <= 9 limit)"
            if qaoa_eligible
            else f"QAOA Locked ({n_vars} vars > 9 limit)"
            if is_valid_model
            else "Evaluation blocked by model errors"
        ),
    }

    # Final overall status
    if not is_valid_model:
        status = "BLOCKED"
    elif qaoa_eligible:
        status = "READY TO OPTIMIZE"
    else:
        status = "READY FOR CLASSICAL OPTIMIZATION (QAOA LOCKED)"

    return ValidationReport(
        dataset_id=dataset_id,
        is_valid_model=is_valid_model,
        status=status,
        qaoa_eligible=qaoa_eligible,
        exact_eligible=exact_eligible,
        greedy_eligible=greedy_eligible,
        n_workers=n_workers,
        n_shifts=n_shifts,
        n_vars=n_vars,
        n_qubits=n_qubits,
        checks=checks,
        errors=errors,
        warnings=warnings,
    )
