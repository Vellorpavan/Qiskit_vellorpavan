"""
ShiftProof UI — Instance Loader

Loads results/instances.json and provides src.model.Instance objects.
READ-ONLY. Never regenerates or overwrites stored instances.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import streamlit as st

from src.model import Instance

INSTANCES_JSON = _ROOT / "results" / "instances.json"

CATEGORY_LABELS = {
    "easy": "Easy",
    "constraint_heavy": "Constraint-heavy",
    "cost_conflict": "Cost-conflict",
}

CATEGORY_DESCRIPTIONS = {
    "easy": "Wide eligibility, little cost conflict.",
    "constraint_heavy": "Sparse eligibility, tight constrained shifts.",
    "cost_conflict": "Multiple shifts competing for the same cheap worker.",
}


@st.cache_data
def load_instances_json() -> list[dict]:
    """Load instances.json and return the raw list of instance dicts (100 items)."""
    with open(INSTANCES_JSON, "r") as f:
        data = json.load(f)
    instances = data["instances"]
    if len(instances) != 100:
        raise ValueError(f"Expected 100 instances, got {len(instances)}")
    return instances


@st.cache_data
def get_instance_ids() -> list[str]:
    """Return sorted list of all 100 instance IDs."""
    return [inst["instance_id"] for inst in load_instances_json()]


@st.cache_data
def get_instance_ids_by_category() -> dict[str, list[str]]:
    """Return dict: category → sorted list of instance IDs."""
    result: dict[str, list[str]] = {
        "easy": [], "constraint_heavy": [], "cost_conflict": []
    }
    for inst in load_instances_json():
        cat = inst.get("category", "unknown")
        if cat in result:
            result[cat].append(inst["instance_id"])
    return result


def get_instance_dict(instance_id: str) -> dict | None:
    """Return raw JSON dict for a specific instance_id, or None."""
    for inst in load_instances_json():
        if inst["instance_id"] == instance_id:
            return inst
    return None


@st.cache_data
def get_instance_object(instance_id: str) -> Instance | None:
    """
    Return a src.model.Instance object for the given instance_id.

    Reconstructed from the frozen instances.json data.
    Does NOT call generate_*() — uses stored data only.
    """
    raw = get_instance_dict(instance_id)
    if raw is None:
        return None
    return Instance(
        n_workers=raw["n_workers"],
        n_shifts=raw["n_shifts"],
        cost_matrix=np.array(raw["cost_matrix"]),
        eligibility=np.array(raw["eligibility"]),
        instance_id=raw["instance_id"],
        seed=raw["seed"],
    )


def verify_instance_counts() -> dict:
    """Verify instance counts match the frozen benchmark (40/30/30)."""
    instances = load_instances_json()
    cats: dict[str, int] = {}
    for inst in instances:
        cat = inst.get("category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
    return {
        "total": len(instances),
        "category_counts": cats,
        "valid": (
            len(instances) == 100
            and cats.get("easy", 0) == 40
            and cats.get("constraint_heavy", 0) == 30
            and cats.get("cost_conflict", 0) == 30
        ),
    }
