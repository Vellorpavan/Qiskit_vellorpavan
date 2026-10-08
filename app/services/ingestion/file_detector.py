"""
ShiftProof — Ingestion: File Type Detector

Detects file format and category (Structured vs Document/Unstructured).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import BinaryIO, Union


class FileType(str, Enum):
    CSV = "csv"
    TSV = "tsv"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"
    XML = "xml"
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


class FileCategory(str, Enum):
    STRUCTURED = "structured"
    DOCUMENT = "document"
    UNSUPPORTED = "unsupported"


def detect_file_type(filename: str, file_bytes: bytes = b"") -> tuple[FileType, FileCategory]:
    """
    Detect file format and categorization from filename extension and magic bytes.
    """
    ext = Path(filename).suffix.lower()

    if ext in {".csv"}:
        return FileType.CSV, FileCategory.STRUCTURED
    elif ext in {".tsv"}:
        return FileType.TSV, FileCategory.STRUCTURED
    elif ext in {".xlsx"}:
        return FileType.XLSX, FileCategory.STRUCTURED
    elif ext in {".xls"}:
        return FileType.XLS, FileCategory.STRUCTURED
    elif ext in {".json"}:
        return FileType.JSON, FileCategory.STRUCTURED
    elif ext in {".xml"}:
        return FileType.XML, FileCategory.STRUCTURED
    elif ext in {".pdf"}:
        return FileType.PDF, FileCategory.DOCUMENT
    elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        return FileType.IMAGE, FileCategory.DOCUMENT

    # Magic byte inspection if extension is ambiguous or missing
    if file_bytes.startswith(b"%PDF"):
        return FileType.PDF, FileCategory.DOCUMENT
    elif file_bytes.startswith(b"PK\x03\x04"):
        return FileType.XLSX, FileCategory.STRUCTURED
    elif file_bytes.startswith(b"{\n") or file_bytes.startswith(b"[\n") or file_bytes.startswith(b"{\"") or file_bytes.startswith(b"[{\""):
        return FileType.JSON, FileCategory.STRUCTURED
    elif file_bytes.startswith(b"<?xml") or file_bytes.startswith(b"<"):
        return FileType.XML, FileCategory.STRUCTURED

    return FileType.UNKNOWN, FileCategory.UNSUPPORTED
