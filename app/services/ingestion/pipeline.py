"""
ShiftProof — Ingestion: Universal Ingestion Pipeline

Coordinates:
File Type Detection -> Parser (Structured or Document) -> Schema Understanding
-> Scheduling Classifier -> Normalizer & Feasibility Validation -> DB Persistence
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Optional

import pandas as pd
from app.services.ingestion.file_detector import FileCategory, FileType, detect_file_type
from app.services.ingestion.structured_parser import ParsedStructure, parse_structured_file
from app.services.ingestion.document_parser import is_gemini_configured, parse_document_file
from app.services.ingestion.schema_understanding import ConceptCandidate, understand_schema
from app.services.ingestion.scheduling_classifier import ClassificationResult, ClassificationStatus, classify_dataset
from app.services.ingestion.normalizer import NormalizationResult, normalize_dataset
from app.services.scheduling_service import compute_dataset_fingerprint


class IngestionInspection:
    """Pre-normalization inspection result presented to the user before optimization."""

    def __init__(
        self,
        filename: str,
        file_bytes: bytes,
        file_type: FileType,
        file_category: FileCategory,
        parsed_structure: ParsedStructure,
        candidates: dict[str, list[ConceptCandidate]],
        classification: ClassificationResult,
    ):
        self.filename = filename
        self.file_bytes = file_bytes
        self.file_type = file_type
        self.file_category = file_category
        self.parsed_structure = parsed_structure
        self.parsed_df = parsed_structure.df
        self.candidates = candidates
        self.classification = classification
        self.sheet_names = parsed_structure.sheet_names
        self.selected_sheet = parsed_structure.selected_sheet


def inspect_uploaded_file(
    file_bytes: bytes,
    filename: str,
    sheet_name: Optional[str] = None,
) -> IngestionInspection:
    """
    Main entry point for inspecting any uploaded workforce scheduling file.
    Does NOT modify the database or run optimization.
    """
    # 1. Detect file format
    file_type, file_cat = detect_file_type(filename, file_bytes)

    # 2. Parse into DataFrame
    if file_cat == FileCategory.STRUCTURED:
        parsed = parse_structured_file(file_bytes, filename, file_type, sheet_name=sheet_name)
    elif file_cat == FileCategory.DOCUMENT:
        parsed = parse_document_file(file_bytes, filename, file_type)
    else:
        raise ValueError(f"Unsupported file format '{filename}'. Supported: CSV, XLSX, XLS, JSON, XML, PDF.")

    # 3. Semantic schema understanding
    candidates = understand_schema(parsed.df)

    # 4. Scheduling problem classification
    classification = classify_dataset(parsed.df, candidates)

    return IngestionInspection(
        filename=filename,
        file_bytes=file_bytes,
        file_type=file_type,
        file_category=file_cat,
        parsed_structure=parsed,
        candidates=candidates,
        classification=classification,
    )


def finalize_and_save_dataset(
    conn: sqlite3.Connection,
    inspection: IngestionInspection,
    user_mapping: Optional[dict[str, str]] = None,
    dataset_name: Optional[str] = None,
    is_matrix: bool = False,
    matrix_cell_type: str = "cost",
    default_cost: float = 10.0,
) -> tuple[str, NormalizationResult]:
    """
    Normalizes inspected data with confirmed mappings, persists to SQLite,
    and returns (dataset_id, NormalizationResult).
    """
    mapping = user_mapping or inspection.classification.detected_mapping

    norm_res = normalize_dataset(
        df=inspection.parsed_df,
        mapping=mapping,
        is_matrix=is_matrix or inspection.classification.is_matrix,
        matrix_shift_columns=inspection.classification.matrix_shift_columns,
        matrix_cell_type=matrix_cell_type,
        default_cost=default_cost,
    )

    if not norm_res.is_feasible:
        err_msg = "; ".join(norm_res.validation_errors)
        raise ValueError(f"Feasibility validation failed: {err_msg}")

    # Generate dataset ID
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    clean_name = dataset_name or inspection.filename.rsplit(".", 1)[0].replace("_", " ").title()

    # Ingestion method description
    ingestion_method = f"{inspection.file_type.value.upper()} → deterministic parser"
    if inspection.file_category == FileCategory.DOCUMENT:
        ext_meta = inspection.parsed_structure.metadata.get("extractor", "document_parser")
        ingestion_method = f"Document ({inspection.file_type.value.upper()}) → {ext_meta} → local validation"

    with conn:
        # 1. Insert datasets metadata
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, description, source_type, source_filename, row_count, col_count, is_synthetic, status, optimization_mode)
            VALUES (?, ?, ?, 'upload', ?, ?, ?, 0, 'ready', ?)
            """,
            (
                dataset_id,
                clean_name,
                f"Ingested via {ingestion_method}. Mode: {norm_res.optimization_mode}. {len(norm_res.assumptions)} normalization assumptions applied.",
                inspection.filename,
                norm_res.row_count,
                12,
                norm_res.optimization_mode,
            ),
        )

        # 2. Insert workers
        worker_map = {}
        for w_idx, w_id in enumerate(norm_res.df_normalized["worker_id"].unique()):
            worker_map[w_id] = w_idx
            w_rows = norm_res.df_normalized[norm_res.df_normalized["worker_id"] == w_id]
            w_name = w_rows["worker_name"].iloc[0]
            w_skill = w_rows["skill"].iloc[0]
            w_cap = int(w_rows["max_shifts"].iloc[0])

            conn.execute(
                """
                INSERT INTO workers (worker_id, dataset_id, external_id, name, skill, max_shifts, worker_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (w_id, dataset_id, w_id, w_name, w_skill, w_cap, w_idx),
            )

        # 3. Insert shifts
        shift_map = {}
        for s_idx, s_id in enumerate(norm_res.df_normalized["shift_id"].unique()):
            shift_map[s_id] = s_idx
            s_rows = norm_res.df_normalized[norm_res.df_normalized["shift_id"] == s_id]
            s_name = s_rows["shift_name"].iloc[0]
            s_date = s_rows["shift_date"].iloc[0]
            s_type = s_rows["shift_type"].iloc[0]
            s_skill = s_rows["required_skill"].iloc[0]

            conn.execute(
                """
                INSERT INTO shifts (shift_id, dataset_id, external_id, shift_name, shift_date, shift_type, required_skill, shift_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (s_id, dataset_id, s_id, s_name, s_date, s_type, s_skill, s_idx),
            )

        # 4. Insert costs and eligibility
        for _, r in norm_res.df_normalized.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO costs (dataset_id, worker_id, shift_id, assignment_cost)
                VALUES (?, ?, ?, ?)
                """,
                (dataset_id, r["worker_id"], r["shift_id"], float(r["assignment_cost"])),
            )

            conn.execute(
                """
                INSERT OR REPLACE INTO eligibility (dataset_id, worker_id, shift_id, is_eligible, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    r["worker_id"],
                    r["shift_id"],
                    int(r["is_eligible"]) if int(r["is_available"]) == 1 else 0,
                    "Inferred from ingestion",
                ),
            )

        # 5. Insert default constraints
        conn.execute(
            """
            INSERT INTO constraints (constraint_id, dataset_id, constraint_type, parameters_json, is_active)
            VALUES (?, ?, 'shift_coverage', '{"required_count": 1}', 1)
            """,
            (f"{dataset_id}_cov", dataset_id),
        )
        conn.execute(
            """
            INSERT INTO constraints (constraint_id, dataset_id, constraint_type, parameters_json, is_active)
            VALUES (?, ?, 'worker_capacity', '{"max_shifts": 1}', 1)
            """,
            (f"{dataset_id}_cap", dataset_id),
        )

    return dataset_id, norm_res
