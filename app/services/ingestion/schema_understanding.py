"""
ShiftProof — Ingestion: Semantic Schema Understanding Layer

Evaluates columns across multiple signals:
- Normalized column names & semantic tokens
- Datatypes and numeric characteristics
- Value patterns, sample distributions, and boolean tokens
- Cardinality and uniqueness relative to row count
- Relationships to other candidate columns

Produces scored candidate mappings with confidence and explainability.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd


class ColumnProfile:
    """Statistical and semantic profile of a single dataframe column."""

    def __init__(self, name: str, series: pd.Series):
        self.name = str(name).strip()
        self.clean_name = re.sub(r"[_\-\s\.]+", " ", self.name.lower()).strip()
        self.tokens = set(re.findall(r"\b\w+\b", self.clean_name))
        self.total_count = len(series)
        self.null_count = int(series.isna().sum())
        self.null_rate = self.null_count / max(1, self.total_count)
        self.non_null_series = series.dropna()
        self.unique_count = int(self.non_null_series.nunique())
        self.cardinality_ratio = self.unique_count / max(1, len(self.non_null_series))
        self.dtype = str(series.dtype)

        # Value inspection
        sample_vals = self.non_null_series.head(20).tolist()
        self.sample_strings = [str(v).strip() for v in sample_vals]
        self.is_numeric = pd.api.types.is_numeric_dtype(series)

        # Boolean-like detection
        self.is_boolean_like = self._check_boolean_like()

        # Shift keyword match in values
        self.shift_value_hits = self._check_shift_keywords()


    def _check_boolean_like(self) -> bool:
        """Check if column values represent booleans or availability flags."""
        if pd.api.types.is_bool_dtype(self.non_null_series):
            return True
        unique_lower = {s.lower() for s in self.sample_strings[:15]}
        bool_tokens = {"true", "false", "yes", "no", "y", "n", "1", "0", "1.0", "0.0", "available", "unavailable", "eligible", "ineligible"}
        if unique_lower and unique_lower.issubset(bool_tokens) and len(unique_lower) <= 3:
            return True
        return False

    def _check_shift_keywords(self) -> int:
        """Count shift-related keywords appearing inside cell values."""
        shift_words = {"morning", "evening", "night", "day", "afternoon", "graveyard", "on-call", "icu", "er", "rot1", "rot2"}
        hits = 0
        for s in self.sample_strings:
            words = set(re.findall(r"\b\w+\b", s.lower()))
            if words.intersection(shift_words):
                hits += 1
        return hits


class ConceptCandidate:
    """A scored mapping from a source column to a target canonical concept."""

    def __init__(self, concept: str, column_name: str, confidence: float, reason: str):
        self.concept = concept
        self.column_name = column_name
        self.confidence = round(confidence, 3)
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "column_name": self.column_name,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def understand_schema(df: pd.DataFrame) -> dict[str, list[ConceptCandidate]]:
    """
    Analyzes DataFrame columns across all canonical workforce concepts.
    Returns sorted lists of candidates for:
    'worker_id', 'worker_name', 'shift_id', 'shift_name', 'assignment_cost',
    'availability', 'eligibility', 'skill', 'capacity'.
    """
    profiles = [ColumnProfile(c, df[c]) for c in df.columns]
    results: dict[str, list[ConceptCandidate]] = {
        "worker": [],
        "shift": [],
        "cost": [],
        "availability": [],
        "skill": [],
        "capacity": [],
    }

    for p in profiles:
        # 1. Evaluate Worker concept
        score_w, reason_w = _score_worker(p)
        if score_w > 0.3:
            results["worker"].append(ConceptCandidate("worker", p.name, score_w, reason_w))

        # 2. Evaluate Shift concept
        score_s, reason_s = _score_shift(p)
        if score_s > 0.3:
            results["shift"].append(ConceptCandidate("shift", p.name, score_s, reason_s))

        # 3. Evaluate Cost concept
        score_c, reason_c = _score_cost(p)
        if score_c > 0.3:
            results["cost"].append(ConceptCandidate("cost", p.name, score_c, reason_c))

        # 4. Evaluate Availability / Eligibility concept
        score_a, reason_a = _score_availability(p)
        if score_a > 0.3:
            results["availability"].append(ConceptCandidate("availability", p.name, score_a, reason_a))

        # 5. Evaluate Skill concept
        score_sk, reason_sk = _score_skill(p)
        if score_sk > 0.3:
            results["skill"].append(ConceptCandidate("skill", p.name, score_sk, reason_sk))

        # 6. Evaluate Capacity concept
        score_cap, reason_cap = _score_capacity(p)
        if score_cap > 0.3:
            results["capacity"].append(ConceptCandidate("capacity", p.name, score_cap, reason_cap))

    # Sort each candidate list descending by confidence
    for k in results:
        results[k].sort(key=lambda x: x.confidence, reverse=True)

    return results


def _score_worker(p: ColumnProfile) -> tuple[float, str]:
    score = 0.0
    reasons = []

    # Semantic token match
    worker_tokens = {
        "worker", "employee", "staff", "personnel", "associate", "resource",
        "technician", "nurse", "doctor", "physician", "member", "name", "id",
        "emp", "crew", "candidate", "driver", "operator", "person"
    }
    overlap = p.tokens.intersection(worker_tokens)
    if "worker" in p.tokens or "employee" in p.tokens or "staff" in p.tokens or "personnel" in p.tokens:
        score += 0.55
        reasons.append("Strong worker name tokens")
    elif overlap:
        score += 0.35
        reasons.append(f"Worker-related tokens: {overlap}")

    # ID or Name indicators
    if "id" in p.tokens or "number" in p.tokens or "code" in p.tokens:
        score += 0.20
        reasons.append("ID indicator")
    elif "name" in p.tokens:
        score += 0.25
        reasons.append("Name indicator")

    # Cardinality check: workers should have > 1 unique value and reasonable cardinality
    if not p.is_numeric and 2 <= p.unique_count:
        score += 0.15
        reasons.append(f"Text column with {p.unique_count} distinct entities")

    # Negative signals
    if p.shift_value_hits > 5:
        score -= 0.40
        reasons.append("Negative: Contains shift duty keywords")
    if p.is_boolean_like:
        score -= 0.50
        reasons.append("Negative: Boolean values")

    return max(0.0, min(1.0, score)), "; ".join(reasons)


def _score_shift(p: ColumnProfile) -> tuple[float, str]:
    score = 0.0
    reasons = []

    shift_tokens = {"shift", "duty", "slot", "rotation", "time", "timing", "session", "period", "schedule", "block"}
    if "shift" in p.tokens or "duty" in p.tokens or "slot" in p.tokens:
        score += 0.55
        reasons.append("Direct shift/duty token match")
    elif p.tokens.intersection(shift_tokens):
        score += 0.35
        reasons.append(f"Time/schedule tokens: {p.tokens.intersection(shift_tokens)}")

    # Shift values inspection (Morning, Evening, Night, Day)
    if p.shift_value_hits >= 2:
        score += 0.35
        reasons.append("Values match standard shift types (Morning/Evening/Night/etc.)")

    # Cardinality check: shifts usually have small cardinality (1 to 15 unique shifts)
    if 1 <= p.unique_count <= 15:
        score += 0.15
        reasons.append(f"Low cardinality ({p.unique_count} distinct shifts)")

    # Negative signals
    if "worker" in p.tokens or "employee" in p.tokens or "salary" in p.tokens:
        score -= 0.30
    if p.is_numeric and p.unique_count > 20:
        score -= 0.50
        reasons.append("Negative: High-cardinality numeric")

    return max(0.0, min(1.0, score)), "; ".join(reasons)


def _score_cost(p: ColumnProfile) -> tuple[float, str]:
    score = 0.0
    reasons = []

    cost_tokens = {
        "cost", "wage", "rate", "hourly", "pay", "price", "expense",
        "fee", "charge", "compensation", "salary", "billing", "amount", "overtime"
    }
    overlap = p.tokens.intersection(cost_tokens)
    if "cost" in p.tokens or "wage" in p.tokens or "rate" in p.tokens or "hourly" in p.tokens:
        score += 0.55
        reasons.append("Direct cost/wage/rate token match")
    elif overlap:
        score += 0.35
        reasons.append(f"Monetary tokens: {overlap}")

    # Numeric check
    if p.is_numeric:
        score += 0.30
        # Positive values
        try:
            min_val = float(p.non_null_series.min())
            if min_val >= 0:
                score += 0.10
                reasons.append("Non-negative numeric values")
        except Exception:
            pass
    elif any("$" in s or "€" in s or "£" in s for s in p.sample_strings):
        score += 0.35
        reasons.append("Currency symbols in values")

    # Negative signals
    if p.is_boolean_like:
        score -= 0.70
        reasons.append("Negative: Boolean column")
    if "id" in p.tokens or "index" in p.tokens:
        score -= 0.40
        reasons.append("Negative: ID / index token")

    return max(0.0, min(1.0, score)), "; ".join(reasons)


def _score_availability(p: ColumnProfile) -> tuple[float, str]:
    score = 0.0
    reasons = []

    avail_tokens = {"available", "availability", "eligible", "eligibility", "active", "status", "present", "can_work"}
    overlap = p.tokens.intersection(avail_tokens)
    if "available" in p.tokens or "availability" in p.tokens or "eligible" in p.tokens or "eligibility" in p.tokens:
        score += 0.55
        reasons.append("Direct availability/eligibility token match")
    elif overlap:
        score += 0.30
        reasons.append(f"Status tokens: {overlap}")

    if p.is_boolean_like:
        score += 0.40
        reasons.append("Boolean or binary flag values")

    return max(0.0, min(1.0, score)), "; ".join(reasons)


def _score_skill(p: ColumnProfile) -> tuple[float, str]:
    score = 0.0
    reasons = []

    skill_tokens = {"skill", "qualification", "cert", "certification", "role", "specialty", "department", "rank"}
    overlap = p.tokens.intersection(skill_tokens)
    if "skill" in p.tokens or "qualification" in p.tokens or "specialty" in p.tokens:
        score += 0.60
        reasons.append("Skill/qualification token match")
    elif overlap:
        score += 0.35
        reasons.append(f"Role/specialty tokens: {overlap}")

    if not p.is_numeric and 2 <= p.unique_count <= 20:
        score += 0.15
        reasons.append(f"Categorical text ({p.unique_count} distinct skills)")

    return max(0.0, min(1.0, score)), "; ".join(reasons)


def _score_capacity(p: ColumnProfile) -> tuple[float, str]:
    score = 0.0
    reasons = []

    cap_tokens = {"capacity", "max", "max_shifts", "max_hours", "limit", "quota"}
    if p.tokens.intersection(cap_tokens):
        score += 0.60
        reasons.append("Capacity/max quota tokens")

    if p.is_numeric and 1 <= p.unique_count <= 5:
        score += 0.20
        reasons.append("Small integer capacity values")

    return max(0.0, min(1.0, score)), "; ".join(reasons)
