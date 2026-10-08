"""
ShiftProof UI — Metrics Loader

Loads results/metrics.csv with correct dtypes.
READ-ONLY. Never modifies the benchmark file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import pandas as pd
import numpy as np
import streamlit as st

METRICS_CSV = _ROOT / "results" / "metrics.csv"

_BOOL_COLS = ["exact_feasible", "greedy_feasible", "qaoa_feasible", "greedy_failed"]
_INT_COLS = [
    "seed", "n_workers", "n_shifts", "n_vars", "n_qubits",
    "qaoa_feasible_samples", "qaoa_shots", "qaoa_p", "qaoa_maxiter", "qaoa_seed",
    "qaoa_circuit_depth", "qaoa_two_qubit_gate_count", "qaoa_optimizer_evaluations",
]
_FLOAT_COLS = [
    "exact_optimal_cost", "exact_runtime",
    "greedy_cost", "greedy_runtime",
    "qaoa_optimal_energy", "qaoa_feasible_rate", "qaoa_optimal_solution_probability",
    "qaoa_best_feasible_cost", "qaoa_mean_feasible_cost",
    "qaoa_assignment_cost_best", "qaoa_qubo_energy_best",
    "qaoa_runtime", "absolute_gap", "relative_gap",
]
_JSON_COLS = ["qaoa_optimal_params", "exact_optimal_assignment", "greedy_assignment"]


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load and parse metrics.csv. Returns a clean DataFrame (100 rows)."""
    df = pd.read_csv(METRICS_CSV, dtype=str)

    # Parse booleans
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(
                {"true": True, "false": False}
            )

    # Parse integers
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse floats (handles 'nan', empty strings, None)
    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: float("nan") if str(x).strip() in ("nan", "None", "") else float(x)
                if str(x).replace(".", "").replace("-", "").replace("e", "").isdigit()
                else float("nan")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse JSON-encoded columns
    def _parse_json(x):
        if pd.isna(x) or str(x).strip() in ("", "nan", "None"):
            return None
        try:
            return json.loads(str(x))
        except Exception:
            return None

    for col in _JSON_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_json)

    if len(df) != 100:
        raise ValueError(f"Expected 100 metric rows, got {len(df)}")

    return df


def compute_aggregate_stats(df: pd.DataFrame) -> dict:
    """
    Compute verified aggregate benchmark statistics.

    All denominators are explicit and preserved exactly as in the frozen benchmark.
    This must reproduce:
      QAOA feasible:        97/100
      Greedy feasible:      95/100
      Exact feasible:      100/100
      QAOA optimum:         80/97   (of QAOA-feasible)
      Greedy optimum:       52/95   (of greedy-feasible)
      Mutually feasible:    92/100
      QAOA better:          38/92
      Greedy better:         7/92
      Same:                 47/92
    """
    total = len(df)

    qaoa_feas = df[df["qaoa_feasible"] == True]
    greedy_feas = df[df["greedy_feasible"] == True]
    exact_feas = df[df["exact_feasible"] == True]
    mutual = df[(df["qaoa_feasible"] == True) & (df["greedy_feasible"] == True)]

    # QAOA optimality: qaoa_feasible rows where optimal_solution_probability > 0
    qaoa_optimal = int((qaoa_feas["qaoa_optimal_solution_probability"] > 0).sum())

    # Greedy optimality: greedy_feasible rows where greedy_cost ≈ exact_optimal_cost
    greedy_optimal = 0
    for _, row in greedy_feas.iterrows():
        gc = row["greedy_cost"]
        ec = row["exact_optimal_cost"]
        if pd.notna(gc) and pd.notna(ec) and abs(float(gc) - float(ec)) < 1e-6:
            greedy_optimal += 1

    # Mutual-feasible comparison
    qaoa_better = greedy_better = same_cost = 0
    for _, row in mutual.iterrows():
        qc = row["qaoa_best_feasible_cost"]
        gc = row["greedy_cost"]
        if pd.notna(qc) and pd.notna(gc):
            diff = float(qc) - float(gc)
            if diff < -1e-6:
                qaoa_better += 1
            elif diff > 1e-6:
                greedy_better += 1
            else:
                same_cost += 1

    # Per-category breakdown
    cat_stats: dict[str, dict] = {}
    for cat in ["easy", "constraint_heavy", "cost_conflict"]:
        cdf = df[df["category"] == cat]
        cq = cdf[cdf["qaoa_feasible"] == True]
        cg = cdf[cdf["greedy_feasible"] == True]
        cq_opt = int((cq["qaoa_optimal_solution_probability"] > 0).sum())
        cg_opt = 0
        for _, row in cg.iterrows():
            gc = row["greedy_cost"]
            ec = row["exact_optimal_cost"]
            if pd.notna(gc) and pd.notna(ec) and abs(float(gc) - float(ec)) < 1e-6:
                cg_opt += 1
        cat_stats[cat] = {
            "total": len(cdf),
            "qaoa_feasible": len(cq),
            "greedy_feasible": len(cg),
            "qaoa_optimal": cq_opt,
            "greedy_optimal": cg_opt,
            "mean_feasible_rate": float(cdf["qaoa_feasible_rate"].mean()),
        }

    # Feasibility / gap rates (across all 100)
    all_feas_rates = df["qaoa_feasible_rate"].dropna()
    all_gaps = df["absolute_gap"].dropna()
    qaoa_runtimes = qaoa_feas["qaoa_runtime"].dropna()

    return {
        "total": total,
        "qaoa_feasible": len(qaoa_feas),
        "greedy_feasible": len(greedy_feas),
        "exact_feasible": len(exact_feas),
        "mutual_feasible": len(mutual),
        "qaoa_optimal": qaoa_optimal,
        "greedy_optimal": greedy_optimal,
        "qaoa_better": qaoa_better,
        "greedy_better": greedy_better,
        "same_cost": same_cost,
        "mean_qaoa_feasible_rate": float(all_feas_rates.mean()),
        "median_qaoa_feasible_rate": float(all_feas_rates.median()),
        "median_absolute_gap": float(all_gaps.median()),
        "mean_absolute_gap": float(all_gaps.mean()),
        "mean_qaoa_runtime": float(qaoa_runtimes.mean()) if len(qaoa_runtimes) > 0 else 0.0,
        "category_stats": cat_stats,
    }


def get_metric_row(df: pd.DataFrame, instance_id: str) -> dict | None:
    """Return metric dict for a specific instance_id, or None."""
    rows = df[df["instance_id"] == instance_id]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def verify_denominators(stats: dict) -> list[str]:
    """
    Verify computed stats match the frozen benchmark denominators.
    Returns list of any mismatches (empty = all good).
    """
    expected = {
        "qaoa_feasible": 97,
        "greedy_feasible": 95,
        "exact_feasible": 100,
        "mutual_feasible": 92,
        "qaoa_optimal": 80,
        "greedy_optimal": 52,
        "qaoa_better": 38,
        "greedy_better": 7,
        "same_cost": 47,
    }
    issues = []
    for key, exp in expected.items():
        got = stats.get(key)
        if got != exp:
            issues.append(f"{key}: expected {exp}, got {got}")
    return issues
