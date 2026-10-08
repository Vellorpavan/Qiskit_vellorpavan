"""
ShiftProof — Synthetic Benchmark Dataset (Phase 7)

Generates 100 reproducible synthetic instances for benchmarking.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import json

from src.model import Instance


@dataclass(frozen=True)
class InstanceRecord:
    """Serializable record of an instance."""
    instance_id: str
    seed: int
    category: str  # "easy", "constraint_heavy", "cost_conflict"
    n_workers: int
    n_shifts: int
    cost_matrix: list  # nested list
    eligibility: list  # nested list of bool
    planted_optimal_cost: Optional[float] = None
    planted_optimal_assignment: Optional[dict] = None


def generate_easy_instance(seed: int, instance_id: str) -> Instance:
    """
    Easy instance: wide eligibility, little cost conflict.
    Most workers eligible for most shifts, costs well separated.
    """
    rng = np.random.default_rng(seed)
    n_workers = 3
    n_shifts = 3

    # Base costs with clear separation
    base_costs = rng.uniform(1, 5, size=(n_workers, n_shifts))
    # Add small noise
    cost_matrix = base_costs + rng.uniform(0, 0.5, size=(n_workers, n_shifts))

    # High eligibility - most pairs eligible
    eligibility = rng.random((n_workers, n_shifts)) > 0.1  # 90% eligible
    # Ensure each shift has at least one eligible worker
    for s in range(n_shifts):
        if not np.any(eligibility[:, s]):
            eligibility[rng.integers(n_workers), s] = True

    return Instance(n_workers, n_shifts, cost_matrix, eligibility, instance_id, seed)


def generate_constraint_heavy_instance(seed: int, instance_id: str) -> Instance:
    """
    Constraint-heavy: sparse eligibility, tight constrained shifts.
    Few eligible workers per shift, many ineligible pairs.
    """
    rng = np.random.default_rng(seed)
    n_workers = 3
    n_shifts = 3

    # Costs don't matter much - constraints dominate
    cost_matrix = rng.uniform(1, 10, size=(n_workers, n_shifts))

    # Low eligibility - only ~40% eligible
    eligibility = rng.random((n_workers, n_shifts)) > 0.6

    # Ensure each shift has at least one eligible worker
    for s in range(n_shifts):
        if not np.any(eligibility[:, s]):
            eligibility[rng.integers(n_workers), s] = True

    # Ensure at least one feasible solution exists by construction
    # Force a valid permutation
    workers = list(range(n_workers))
    rng.shuffle(workers)
    for s in range(min(n_shifts, n_workers)):
        eligibility[workers[s], s] = True

    return Instance(n_workers, n_shifts, cost_matrix, eligibility, instance_id, seed)


def generate_cost_conflict_instance(seed: int, instance_id: str) -> Instance:
    """
    Cost-conflict: several shifts compete for the same cheap worker.
    One or two very cheap workers that multiple shifts want.
    """
    rng = np.random.default_rng(seed)
    n_workers = 3
    n_shifts = 3

    # Create cost matrix with conflicts
    cost_matrix = rng.uniform(5, 10, size=(n_workers, n_shifts))

    # Make worker 0 very cheap for shifts 0 and 1
    cost_matrix[0, 0] = rng.uniform(0.5, 1.5)
    cost_matrix[0, 1] = rng.uniform(0.5, 1.5)
    # Make worker 1 very cheap for shifts 1 and 2
    cost_matrix[1, 1] = rng.uniform(0.5, 1.5)
    cost_matrix[1, 2] = rng.uniform(0.5, 1.5)

    # High eligibility
    eligibility = np.ones((n_workers, n_shifts), dtype=bool)
    # But make some ineligible to create constraints
    if rng.random() > 0.5:
        eligibility[0, 2] = False
    if rng.random() > 0.5:
        eligibility[2, 0] = False

    return Instance(n_workers, n_shifts, cost_matrix, eligibility, instance_id, seed)


def generate_all_instances() -> list[Instance]:
    """
    Generate all 100 synthetic instances:
    - 40 Easy
    - 30 Constraint-heavy
    - 30 Cost-conflict
    """
    instances = []

    # 40 Easy instances (seeds 1000-1039)
    for i in range(40):
        seed = 1000 + i
        inst = generate_easy_instance(seed, f"easy_{i:03d}")
        instances.append(inst)

    # 30 Constraint-heavy instances (seeds 2000-2029)
    for i in range(30):
        seed = 2000 + i
        inst = generate_constraint_heavy_instance(seed, f"constraint_heavy_{i:03d}")
        instances.append(inst)

    # 30 Cost-conflict instances (seeds 3000-3029)
    for i in range(30):
        seed = 3000 + i
        inst = generate_cost_conflict_instance(seed, f"cost_conflict_{i:03d}")
        instances.append(inst)

    return instances


def save_instances(instances: list[Instance], filepath: str = "results/instances.json"):
    """Save instances to JSON file."""
    records = []
    for inst in instances:
        # Determine category from instance_id
        if inst.instance_id.startswith("easy"):
            category = "easy"
        elif inst.instance_id.startswith("constraint_heavy"):
            category = "constraint_heavy"
        else:
            category = "cost_conflict"

        record = InstanceRecord(
            instance_id=inst.instance_id,
            seed=inst.seed,
            category=category,
            n_workers=inst.n_workers,
            n_shifts=inst.n_shifts,
            cost_matrix=inst.cost_matrix.tolist(),
            eligibility=inst.eligibility.tolist()
        )
        records.append(record)

    # Convert to dict for JSON serialization
    data = {
        "num_instances": len(records),
        "categories": {
            "easy": sum(1 for r in records if r.category == "easy"),
            "constraint_heavy": sum(1 for r in records if r.category == "constraint_heavy"),
            "cost_conflict": sum(1 for r in records if r.category == "cost_conflict")
        },
        "instances": [
            {
                "instance_id": r.instance_id,
                "seed": r.seed,
                "category": r.category,
                "n_workers": r.n_workers,
                "n_shifts": r.n_shifts,
                "cost_matrix": r.cost_matrix,
                "eligibility": r.eligibility
            }
            for r in records
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(records)} instances to {filepath}")


def load_instances(filepath: str = "results/instances.json") -> list[Instance]:
    """Load instances from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    instances = []
    for rec in data["instances"]:
        inst = Instance(
            n_workers=rec["n_workers"],
            n_shifts=rec["n_shifts"],
            cost_matrix=np.array(rec["cost_matrix"]),
            eligibility=np.array(rec["eligibility"]),
            instance_id=rec["instance_id"],
            seed=rec["seed"]
        )
        instances.append(inst)

    return instances


def print_instance_summary(instances: list[Instance]):
    """Print summary of instance categories."""
    categories = {}
    for inst in instances:
        if inst.instance_id.startswith("easy"):
            cat = "easy"
        elif inst.instance_id.startswith("constraint_heavy"):
            cat = "constraint_heavy"
        else:
            cat = "cost_conflict"
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nTotal instances: {len(instances)}")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")

    # Show first few instances
    print("\nFirst 3 instances:")
    for inst in instances[:3]:
        print(f"  {inst.instance_id}: {inst.n_workers}w x {inst.n_shifts}s, "
              f"{inst.n_vars} vars, seed={inst.seed}")


if __name__ == "__main__":
    instances = generate_all_instances()
    save_instances(instances)
    print_instance_summary(instances)