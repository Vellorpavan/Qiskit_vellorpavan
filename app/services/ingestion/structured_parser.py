"""
ShiftProof — Ingestion: Deterministic Structured Parsers

Supports CSV, TSV, XLSX, XLS, JSON, XML via local deterministic inspection.
Never calls external LLMs. Inspects sheets, headers, datatypes, and layouts.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
from typing import Any, Optional

import pandas as pd
from app.services.ingestion.file_detector import FileType


class ParsedStructure:
    """Represents the raw parsed structure extracted from a file."""

    def __init__(
        self,
        df: pd.DataFrame,
        file_type: FileType,
        sheet_names: Optional[list[str]] = None,
        selected_sheet: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.df = df
        self.file_type = file_type
        self.sheet_names = sheet_names or []
        self.selected_sheet = selected_sheet
        self.metadata = metadata or {}
        self.row_count = len(df)
        self.col_count = len(df.columns)
        self.columns = list(df.columns)


def parse_csv_or_tsv(file_bytes: bytes, filename: str) -> ParsedStructure:
    """Parse CSV or TSV with delimiter detection and multi-encoding fallback."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    text = ""
    for enc in encodings:
        try:
            text = file_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if not text:
        text = file_bytes.decode("utf-8", errors="replace")

    # Delimiter detection
    sep = ","
    sample_lines = text.strip().splitlines()[:5]
    if sample_lines:
        first_line = sample_lines[0]
        if "\t" in first_line:
            sep = "\t"
        elif ";" in first_line and first_line.count(";") > first_line.count(","):
            sep = ";"
        elif "|" in first_line and first_line.count("|") > first_line.count(","):
            sep = "|"

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    except Exception:
        # Fallback to standard comma
        df = pd.read_csv(io.StringIO(text), engine="python")

    df = _clean_dataframe(df)
    return ParsedStructure(df=df, file_type=FileType.CSV, metadata={"delimiter": sep})


def parse_excel(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> ParsedStructure:
    """Parse Excel (XLSX / XLS) with sheet detection and header banner detection."""
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    sheets = excel_file.sheet_names

    if sheet_name and sheet_name in sheets:
        chosen_sheet = sheet_name
    elif len(sheets) == 1:
        chosen_sheet = sheets[0]
    else:
        # If no sheet specified, pick the sheet with the most data rows
        best_sheet = sheets[0]
        max_rows = -1
        for s in sheets:
            try:
                sdf = pd.read_excel(excel_file, sheet_name=s)
                if len(sdf) > max_rows:
                    max_rows = len(sdf)
                    best_sheet = s
            except Exception:
                pass
        chosen_sheet = best_sheet

    # First attempt: read sheet directly
    df = pd.read_excel(excel_file, sheet_name=chosen_sheet)

    # Check if row 0 was an empty/title banner and headers are on row 1
    if len(df) > 1 and df.columns[0].startswith("Unnamed:"):
        # Check first non-empty row
        for skip_idx in range(1, min(5, len(df))):
            cand_df = pd.read_excel(excel_file, sheet_name=chosen_sheet, skiprows=skip_idx)
            unnamed_ratio = sum(1 for c in cand_df.columns if str(c).startswith("Unnamed:")) / max(1, len(cand_df.columns))
            if unnamed_ratio < 0.5:
                df = cand_df
                break

    df = _clean_dataframe(df)
    return ParsedStructure(
        df=df,
        file_type=FileType.XLSX,
        sheet_names=sheets,
        selected_sheet=chosen_sheet,
        metadata={"total_sheets": len(sheets)},
    )


def parse_json(file_bytes: bytes, filename: str) -> ParsedStructure:
    """Parse JSON supporting record arrays, nested objects, and columnar schemas."""
    text = file_bytes.decode("utf-8", errors="replace")
    data = json.loads(text)

    if isinstance(data, list):
        # Array of records
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        # Check common container keys
        for key in ["records", "assignments", "schedule", "data", "rows", "items"]:
            if key in data and isinstance(data[key], list):
                df = pd.json_normalize(data[key])
                df = _clean_dataframe(df)
                return ParsedStructure(df=df, file_type=FileType.JSON, metadata={"root_key": key})

        # Check if it has multi-table workforce schema: {"workers": [...], "shifts": [...], "costs": [...]}
        if "workers" in data and "shifts" in data:
            df = _reconstruct_multi_table_json(data)
            return ParsedStructure(df=df, file_type=FileType.JSON, metadata={"schema": "multi_table"})

        # Try columnar dict or single object
        try:
            df = pd.DataFrame(data)
        except Exception:
            df = pd.json_normalize([data])
    else:
        raise ValueError("Unsupported JSON root element (must be list or object)")

    df = _clean_dataframe(df)
    return ParsedStructure(df=df, file_type=FileType.JSON)


def parse_xml(file_bytes: bytes, filename: str) -> ParsedStructure:
    """Parse XML by locating repeating record nodes and extracting fields."""
    root = ET.fromstring(file_bytes)

    # Search for repeating child elements
    tag_counts = {}
    for elem in root.iter():
        tag_counts[elem.tag] = tag_counts.get(elem.tag, 0) + 1

    # Find the tag with highest repetition > 1
    repeated_tags = [(tag, count) for tag, count in tag_counts.items() if count > 1]
    repeated_tags.sort(key=lambda x: x[1], reverse=True)

    records = []
    if repeated_tags:
        target_tag = repeated_tags[0][0]
        for elem in root.iter(target_tag):
            record = dict(elem.attrib)
            for child in elem:
                if child.text:
                    record[child.tag] = child.text.strip()
            if record:
                records.append(record)

    if not records:
        # Fallback: flatten immediate children of root
        for child in root:
            record = dict(child.attrib)
            for sub in child:
                if sub.text:
                    record[sub.tag] = sub.text.strip()
            if record:
                records.append(record)

    if not records:
        raise ValueError("Could not extract repeating records from XML document.")

    df = pd.DataFrame(records)
    df = _clean_dataframe(df)
    return ParsedStructure(df=df, file_type=FileType.XML)


def parse_structured_file(file_bytes: bytes, filename: str, file_type: FileType, sheet_name: Optional[str] = None) -> ParsedStructure:
    """Universal structured parsing entry point."""
    if file_type in {FileType.CSV, FileType.TSV}:
        return parse_csv_or_tsv(file_bytes, filename)
    elif file_type in {FileType.XLSX, FileType.XLS}:
        return parse_excel(file_bytes, filename, sheet_name)
    elif file_type == FileType.JSON:
        return parse_json(file_bytes, filename)
    elif file_type == FileType.XML:
        return parse_xml(file_bytes, filename)
    else:
        raise ValueError(f"File type '{file_type.value}' is not supported by the deterministic structured parser.")


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean unhelpful columns and normalize whitespace in column headers."""
    # Drop columns that are completely empty
    df = df.dropna(how="all", axis=1)
    # Drop rows that are completely empty
    df = df.dropna(how="all", axis=0)

    # Stringify and strip column names
    clean_cols = []
    for i, col in enumerate(df.columns):
        c_str = str(col).strip()
        if not c_str or c_str.startswith("Unnamed:"):
            c_str = f"column_{i+1}"
        clean_cols.append(c_str)
    df.columns = clean_cols
    return df.reset_index(drop=True)


def _reconstruct_multi_table_json(data: dict) -> pd.DataFrame:
    """Unroll separate workers/shifts/costs arrays into a unified assignment DataFrame."""
    workers = data.get("workers", [])
    shifts = data.get("shifts", [])
    costs = data.get("costs", [])
    elig = data.get("eligibility", [])

    rows = []
    # If explicit cost matrix or assignment list
    cost_map = {}
    if isinstance(costs, list):
        for c in costs:
            if isinstance(c, dict):
                w_id = c.get("worker_id") or c.get("worker")
                s_id = c.get("shift_id") or c.get("shift")
                cost_val = c.get("cost") or c.get("assignment_cost") or c.get("rate")
                if w_id and s_id:
                    cost_map[(str(w_id), str(s_id))] = float(cost_val or 0.0)

    # Build Cartesian product or assignment list
    for w in workers:
        w_id = str(w.get("worker_id", w.get("id", ""))) if isinstance(w, dict) else str(w)
        w_name = w.get("name", w_id) if isinstance(w, dict) else str(w)
        w_skill = w.get("skill", "General") if isinstance(w, dict) else "General"

        for s in shifts:
            s_id = str(s.get("shift_id", s.get("id", ""))) if isinstance(s, dict) else str(s)
            s_name = s.get("name", s_id) if isinstance(s, dict) else str(s)

            cost_val = cost_map.get((w_id, s_id), 0.0)
            rows.append({
                "worker_id": w_id,
                "worker_name": w_name,
                "skill": w_skill,
                "shift_id": s_id,
                "shift_name": s_name,
                "assignment_cost": cost_val,
                "is_eligible": 1,
            })

    return pd.DataFrame(rows)
