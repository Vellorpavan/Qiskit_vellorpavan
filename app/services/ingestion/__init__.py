"""
ShiftProof — Universal Ingestion Package
"""

from app.services.ingestion.file_detector import FileCategory, FileType, detect_file_type
from app.services.ingestion.structured_parser import ParsedStructure, parse_structured_file
from app.services.ingestion.document_parser import is_gemini_configured, parse_document_file
from app.services.ingestion.schema_understanding import ConceptCandidate, ColumnProfile, understand_schema
from app.services.ingestion.scheduling_classifier import ClassificationResult, ClassificationStatus, classify_dataset
from app.services.ingestion.normalizer import NormalizationResult, normalize_dataset
from app.services.ingestion.pipeline import IngestionInspection, inspect_uploaded_file, finalize_and_save_dataset

__all__ = [
    "FileType",
    "FileCategory",
    "detect_file_type",
    "ParsedStructure",
    "parse_structured_file",
    "is_gemini_configured",
    "parse_document_file",
    "ConceptCandidate",
    "ColumnProfile",
    "understand_schema",
    "ClassificationResult",
    "ClassificationStatus",
    "classify_dataset",
    "NormalizationResult",
    "normalize_dataset",
    "IngestionInspection",
    "inspect_uploaded_file",
    "finalize_and_save_dataset",
]
