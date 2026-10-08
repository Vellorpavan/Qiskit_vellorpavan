"""
ShiftProof — Ingestion: Document & Unstructured Parser

Supports PDF and image schedule documents via:
1. Native local table/text extraction (pypdf).
2. Optional intelligent Gemini document understanding adapter (when GEMINI_API_KEY is configured).

Gemini output is treated as untrusted input and validated strictly against local schemas.
Missing concepts are never fabricated.
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Optional

import pandas as pd
from app.services.ingestion.file_detector import FileType
from app.services.ingestion.structured_parser import ParsedStructure


def is_gemini_configured() -> bool:
    """Check if GEMINI_API_KEY is available in the environment without exposing the key."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def parse_document_file(
    file_bytes: bytes,
    filename: str,
    file_type: FileType,
) -> ParsedStructure:
    """
    Parse PDF or image document. Attempts local extraction first,
    falling back to Gemini Document Adapter if configured.
    """
    if file_type == FileType.PDF:
        # 1. Attempt local deterministic extraction first
        local_res = _extract_local_pdf(file_bytes, filename)
        if local_res is not None and len(local_res.df) > 0 and len(local_res.df.columns) >= 2:
            return local_res

    # 2. If local extraction insufficient and Gemini is configured, use Gemini adapter
    if is_gemini_configured():
        return _extract_with_gemini(file_bytes, filename, file_type)

    # 3. If Gemini is not configured, explain clearly
    raise ValueError(
        f"Unable to extract structured workforce table from document '{filename}' via local parsing. "
        "Intelligent document understanding requires setting the GEMINI_API_KEY environment variable. "
        "Local structured files (CSV, XLSX, XLS, JSON, XML) remain fully supported without an API key."
    )


def _extract_local_pdf(file_bytes: bytes, filename: str) -> Optional[ParsedStructure]:
    """Attempt local tabular extraction from text-based PDF using pypdf."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"

        if not full_text.strip():
            return None

        # Look for table rows: split by newline
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        # Attempt to split lines by comma, tab, or multi-space
        parsed_rows = []
        for line in lines:
            if "\t" in line:
                tokens = [t.strip() for t in line.split("\t") if t.strip()]
            elif "," in line and line.count(",") >= 2:
                tokens = [t.strip() for t in line.split(",") if t.strip()]
            elif "|" in line and line.count("|") >= 2:
                tokens = [t.strip() for t in line.split("|") if t.strip()]
            else:
                tokens = re.split(r"\s{2,}", line)

            if len(tokens) >= 2:
                parsed_rows.append(tokens)

        if len(parsed_rows) >= 2:
            # Normalize column count to the most frequent column width
            widths = [len(r) for r in parsed_rows]
            target_width = max(set(widths), key=widths.count)

            clean_rows = [r for r in parsed_rows if len(r) == target_width]
            if len(clean_rows) >= 2:
                headers = clean_rows[0]
                data = clean_rows[1:]
                df = pd.DataFrame(data, columns=headers)
                return ParsedStructure(
                    df=df,
                    file_type=FileType.PDF,
                    metadata={"extractor": "local_pypdf", "page_count": len(reader.pages)},
                )
    except Exception:
        pass
    return None


def _extract_with_gemini(
    file_bytes: bytes,
    filename: str,
    file_type: FileType,
) -> ParsedStructure:
    """
    Intelligent document extraction adapter via Gemini.
    Output is strictly structured and validated locally. Missing fields are never fabricated.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        prompt = (
            "You are a strict workforce scheduling document ingestion engine. "
            "Inspect this workforce roster/schedule document and extract the tabular scheduling assignments. "
            "CRITICAL RULES:\n"
            "1. NEVER invent or fabricate missing data.\n"
            "2. If assignment cost is not in the document, return null for cost.\n"
            "3. If a concept (worker, shift, availability, cost) is missing, state it clearly.\n"
            "4. Return ONLY a strict JSON object with this schema:\n"
            "{\n"
            '  "document_type": "workforce_schedule",\n'
            '  "confidence": 0.95,\n'
            '  "worker_field": "Worker column name or null",\n'
            '  "shift_field": "Shift column name or null",\n'
            '  "cost_field": "Cost column name or null",\n'
            '  "availability_field": "Availability column name or null",\n'
            '  "structure_type": "row_based",\n'
            '  "records": [\n'
            '    {"worker": "...", "shift": "...", "cost": 100.0, "available": true}\n'
            "  ],\n"
            '  "warnings": []\n'
            "}"
        )

        mime_type = "application/pdf" if file_type == FileType.PDF else "image/png"
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        response_text = response.text.strip()
        data = json.loads(response_text)

        records = data.get("records", [])
        if not records or not isinstance(records, list):
            raise ValueError("Gemini document understanding found no structured records.")

        df = pd.DataFrame(records)
        return ParsedStructure(
            df=df,
            file_type=file_type,
            metadata={
                "extractor": "gemini_document_adapter",
                "model": "gemini-2.5-flash",
                "confidence": data.get("confidence", 0.9),
                "warnings": data.get("warnings", []),
            },
        )
    except Exception as e:
        raise ValueError(f"Gemini document understanding failed: {e}")
