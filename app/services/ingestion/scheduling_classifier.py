"""
ShiftProof — Ingestion: Scheduling Problem Classifier

Explicitly classifies datasets into:
- VALID_SCHEDULING_DATASET
- SCHEDULING_DATASET_NEEDS_MAPPING
- SCHEDULING_DATASET_MISSING_REQUIRED_CONCEPT
- NOT_A_SCHEDULING_DATASET
- POTENTIAL_SHIFT_MATRIX

Identifies row-based assignments, explicit availability formats, and unpivotable matrix schedules.
Never fabricates missing data or invents costs.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

import pandas as pd
from app.services.ingestion.schema_understanding import ConceptCandidate, understand_schema


class ClassificationStatus(str, Enum):
    VALID_SCHEDULING_DATASET = "VALID_SCHEDULING_DATASET"
    SCHEDULING_DATASET_NEEDS_MAPPING = "SCHEDULING_DATASET_NEEDS_MAPPING"
    SCHEDULING_DATASET_MISSING_REQUIRED_CONCEPT = "SCHEDULING_DATASET_MISSING_REQUIRED_CONCEPT"
    NOT_A_SCHEDULING_DATASET = "NOT_A_SCHEDULING_DATASET"
    POTENTIAL_SHIFT_MATRIX = "POTENTIAL_SHIFT_MATRIX"


class ClassificationResult:
    """Detailed diagnosis of a workforce scheduling dataset classification."""

    def __init__(
        self,
        status: ClassificationStatus,
        confidence: float,
        detected_mapping: dict[str, Optional[str]],
        missing_concepts: list[str],
        detected_concepts: list[str],
        ambiguous_concepts: list[str],
        explanation: str,
        is_matrix: bool = False,
        matrix_shift_columns: Optional[list[str]] = None,
        matrix_cell_type: Optional[str] = None,  # "cost" | "availability" | "unknown"
        optimization_mode: str = "COST_OPTIMIZATION",  # "COST_OPTIMIZATION" | "FEASIBILITY_ONLY"
    ):
        self.status = status
        self.confidence = round(confidence, 2)
        self.detected_mapping = detected_mapping
        self.missing_concepts = missing_concepts
        self.detected_concepts = detected_concepts
        self.ambiguous_concepts = ambiguous_concepts
        self.explanation = explanation
        self.is_matrix = is_matrix
        self.matrix_shift_columns = matrix_shift_columns or []
        self.matrix_cell_type = matrix_cell_type
        self.optimization_mode = optimization_mode

    @property
    def is_valid(self) -> bool:
        return self.status == ClassificationStatus.VALID_SCHEDULING_DATASET

    @property
    def has_cost(self) -> bool:
        return self.optimization_mode == "COST_OPTIMIZATION" and self.detected_mapping.get("cost") is not None


def classify_dataset(
    df: pd.DataFrame,
    candidates: Optional[dict[str, list[ConceptCandidate]]] = None,
) -> ClassificationResult:
    """
    Classifies whether the DataFrame can represent an I6 workforce scheduling problem.
    """
    if candidates is None:
        candidates = understand_schema(df)

    # 1. Check for Matrix / Pivot Format (e.g. Employee | Morning | Evening | Night)
    matrix_res = _check_matrix_format(df, candidates)
    if matrix_res is not None:
        return matrix_res

    # 2. Check Row-Based Formats
    worker_cands = candidates.get("worker", [])
    shift_cands = candidates.get("shift", [])
    cost_cands = candidates.get("cost", [])
    avail_cands = candidates.get("availability", [])
    skill_cands = candidates.get("skill", [])
    cap_cands = candidates.get("capacity", [])

    detected_concepts = []
    missing_concepts = []
    ambiguous_concepts = []
    chosen_mapping: dict[str, Optional[str]] = {}

    # Worker evaluation
    if worker_cands and worker_cands[0].confidence >= 0.45:
        chosen_mapping["worker"] = worker_cands[0].column_name
        detected_concepts.append(f"Worker ({worker_cands[0].column_name})")
        if len(worker_cands) > 1 and worker_cands[1].confidence >= 0.70:
            ambiguous_concepts.append("worker")
    else:
        chosen_mapping["worker"] = None
        missing_concepts.append("Worker / Employee identity")

    # Shift evaluation
    if shift_cands and shift_cands[0].confidence >= 0.45:
        chosen_mapping["shift"] = shift_cands[0].column_name
        detected_concepts.append(f"Shift / Duty ({shift_cands[0].column_name})")
        if len(shift_cands) > 1 and shift_cands[1].confidence >= 0.70:
            ambiguous_concepts.append("shift")
    else:
        chosen_mapping["shift"] = None
        missing_concepts.append("Shift / Duty specification")

    # Cost evaluation
    # Cost evaluation (Optional objective)
    if cost_cands and cost_cands[0].confidence >= 0.45:
        chosen_mapping["cost"] = cost_cands[0].column_name
        detected_concepts.append(f"Assignment Cost ({cost_cands[0].column_name})")
        if len(cost_cands) > 1 and cost_cands[1].confidence >= 0.70:
            ambiguous_concepts.append("cost")
    else:
        chosen_mapping["cost"] = None

    # Optional concepts
    chosen_mapping["availability"] = avail_cands[0].column_name if (avail_cands and avail_cands[0].confidence >= 0.5) else None
    chosen_mapping["skill"] = skill_cands[0].column_name if (skill_cands and skill_cands[0].confidence >= 0.5) else None
    chosen_mapping["capacity"] = cap_cands[0].column_name if (cap_cands and cap_cands[0].confidence >= 0.5) else None

    # Decision tree
    has_worker = chosen_mapping["worker"] is not None
    has_shift = chosen_mapping["shift"] is not None
    has_cost = chosen_mapping["cost"] is not None

    if has_worker and has_shift:
        mode = "COST_OPTIMIZATION" if has_cost else "FEASIBILITY_ONLY"
        if ambiguous_concepts:
            return ClassificationResult(
                status=ClassificationStatus.SCHEDULING_DATASET_NEEDS_MAPPING,
                confidence=0.85,
                detected_mapping=chosen_mapping,
                missing_concepts=[],
                detected_concepts=detected_concepts,
                ambiguous_concepts=ambiguous_concepts,
                explanation=(
                    f"Scheduling dataset detected with multiple competing candidate columns for: {', '.join(ambiguous_concepts)}. "
                    "Confirmation required before normalization."
                ),
                optimization_mode=mode,
            )
        else:
            if has_cost:
                expl = "Valid workforce scheduling problem detected with unambiguous canonical column mappings."
            else:
                expl = (
                    "Workforce scheduling structure detected (Worker + Shift/Duty). "
                    "No assignment cost/wage data provided. ShiftProof can optimize constraint feasibility (FEASIBILITY ONLY mode) "
                    "without fabricating monetary costs."
                )
            return ClassificationResult(
                status=ClassificationStatus.VALID_SCHEDULING_DATASET,
                confidence=0.95,
                detected_mapping=chosen_mapping,
                missing_concepts=[],
                detected_concepts=detected_concepts,
                ambiguous_concepts=[],
                explanation=expl,
                optimization_mode=mode,
            )

    elif has_worker and not has_shift:
        return ClassificationResult(
            status=ClassificationStatus.SCHEDULING_DATASET_MISSING_REQUIRED_CONCEPT,
            confidence=0.50,
            detected_mapping=chosen_mapping,
            missing_concepts=["Shift / Duty specification"],
            detected_concepts=detected_concepts,
            ambiguous_concepts=ambiguous_concepts,
            explanation=(
                "Workforce data detected but missing required Shift/Duty specification. "
                "ShiftProof cannot optimize without defined shift slots."
            ),
            optimization_mode="FEASIBILITY_ONLY",
        )

    elif has_shift and not has_worker:
        return ClassificationResult(
            status=ClassificationStatus.SCHEDULING_DATASET_MISSING_REQUIRED_CONCEPT,
            confidence=0.50,
            detected_mapping=chosen_mapping,
            missing_concepts=["Worker / Personnel entity"],
            detected_concepts=detected_concepts,
            ambiguous_concepts=ambiguous_concepts,
            explanation=(
                "Shift structure detected but missing worker identity column. "
                "ShiftProof requires worker entities to assign."
            ),
            optimization_mode="FEASIBILITY_ONLY",
        )

    else:
        return ClassificationResult(
            status=ClassificationStatus.NOT_A_SCHEDULING_DATASET,
            confidence=0.10,
            detected_mapping=chosen_mapping,
            missing_concepts=["Worker", "Shift"],
            detected_concepts=detected_concepts,
            ambiguous_concepts=[],
            explanation="This file does not appear to contain a workforce scheduling problem.",
            optimization_mode="FEASIBILITY_ONLY",
        )


def _check_matrix_format(
    df: pd.DataFrame,
    candidates: dict[str, list[ConceptCandidate]],
) -> Optional[ClassificationResult]:
    """
    Check if dataset is organized as a matrix:
    Column 0 is worker entity, and columns 1..N represent shift slots (e.g. Morning, Evening, Night).
    """
    if len(df.columns) < 3:
        return None

    worker_cands = candidates.get("worker", [])
    first_col = df.columns[0]

    # First column should be worker-like
    is_first_col_worker = any(c.column_name == first_col for c in worker_cands) or "emp" in first_col.lower() or "worker" in first_col.lower()
    if not is_first_col_worker:
        return None
    other_cols = list(df.columns[1:])


    # If a dedicated cost column is clearly present among other_cols, it's a row-based dataset, not a matrix
    cost_cands = candidates.get("cost", [])
    has_explicit_cost_col = any(c.confidence >= 0.5 for c in cost_cands if c.column_name in other_cols)
    if has_explicit_cost_col:
        return None


    # Specific shift slot names that represent matrix column headers
    slot_keywords = {
        "morning", "evening", "night", "day", "afternoon", "graveyard",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "shift_1", "shift_2", "shift_3", "slot_1", "slot_2", "slot_3"
    }

    shift_cols = []
    for col in other_cols:
        col_tokens = set(re.findall(r"\b\w+\b", col.lower()))
        clean_c = col.lower().replace(" ", "_")
        if col_tokens.intersection(slot_keywords) or clean_c in slot_keywords:
            shift_cols.append(col)

    # A shift matrix requires at least 2 distinct shift columns
    if len(shift_cols) >= 2:
        # Inspect cell values
        sample_vals = df[shift_cols].dropna().values.flatten()[:30]
        sample_str = [str(v).lower().strip() for v in sample_vals]
        bool_tokens = {"true", "false", "yes", "no", "y", "n", "1", "0", "available", "unavailable"}

        if set(sample_str).issubset(bool_tokens):
            cell_type = "availability"
        elif all(re.match(r"^\d+(\.\d+)?$", s) for s in sample_str):
            cell_type = "cost"
        else:
            cell_type = "unknown"

        return ClassificationResult(
            status=ClassificationStatus.POTENTIAL_SHIFT_MATRIX,
            confidence=0.85,
            detected_mapping={"worker": first_col},
            missing_concepts=["Explicit Cost"] if cell_type == "availability" else [],
            detected_concepts=[f"Worker entity ({first_col})", f"Shift columns: {', '.join(shift_cols)}"],
            ambiguous_concepts=["matrix_cell_meaning"],
            explanation=(
                f"Detected potential Shift Matrix with {len(shift_cols)} shift columns ({', '.join(shift_cols)}). "
                f"Cell values appear to represent {cell_type}. User interpretation required."
            ),
            is_matrix=True,
            matrix_shift_columns=shift_cols,
            matrix_cell_type=cell_type,
        )

    return None

