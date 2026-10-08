"""
ShiftProof — Classical Solvers

Exact enumeration and greedy heuristic for workforce scheduling.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from src.model import Instance, check_constraints, assignment_cost, decode_bitstring, get_all_bitstrings


@dataclass(frozen=True)
class SolverResult:
    """Result from a classical solver."""
    feasible: bool
    optimal_cost: Optional[float]
    optimal_assignment: Optional[dict]
    instance_id: str
    solver_name: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


def solve_exact(instance: Instance) -> SolverResult:
    """
    Exhaustive enumeration to find the true optimal feasible assignment.

    Returns the minimum cost feasible assignment, or reports infeasible.
    Only practical for tiny instances (n_vars <= ~20).
    """
    n_vars = instance.n_vars
    if n_vars > 20:
        raise ValueError(f"Exact solver not practical for n_vars={n_vars} > 20")

    best_cost = float('inf')
    best_assignment = None
    feasible_found = False

    for bits in get_all_bitstrings(n_vars):
        assignment = decode_bitstring(instance, bits)
        feasible, _ = check_constraints(instance, assignment)

        if feasible:
            feasible_found = True
            cost = assignment_cost(instance, assignment)
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment.copy()

    if feasible_found:
        return SolverResult(
            feasible=True,
            optimal_cost=best_cost,
            optimal_assignment=best_assignment,
            instance_id=instance.instance_id,
            solver_name="exact",
            metadata={"n_vars": n_vars, "states_checked": 2 ** n_vars}
        )
    else:
        return SolverResult(
            feasible=False,
            optimal_cost=None,
            optimal_assignment=None,
            instance_id=instance.instance_id,
            solver_name="exact",
            metadata={"n_vars": n_vars, "states_checked": 2 ** n_vars, "reason": "no feasible assignment found"}
        )


def solve_greedy(instance: Instance) -> SolverResult:
    """
    Greedy heuristic: choose most constrained shift first, assign cheapest available eligible worker.

    Algorithm:
    1. Sort shifts by number of eligible workers (ascending) - most constrained first
    2. For each shift in order:
       - Among eligible workers not yet assigned, pick the one with minimum cost
       - If no available eligible worker, the algorithm fails
    3. Return the assignment if all shifts covered, else failure

    This is NOT optimal - it's a baseline for comparison.
    """
    n_workers = instance.n_workers
    n_shifts = instance.n_shifts

    # Track which workers are already assigned
    worker_assigned = [False] * n_workers
    assignment = {}

    # Sort shifts by number of eligible workers (most constrained first)
    shift_order = sorted(
        range(n_shifts),
        key=lambda s: sum(1 for w in range(n_workers) if instance.is_eligible(w, s))
    )

    for s in shift_order:
        # Find cheapest available eligible worker for this shift
        best_worker = None
        best_cost = float('inf')

        for w in range(n_workers):
            if instance.is_eligible(w, s) and not worker_assigned[w]:
                cost = instance.get_cost(w, s)
                if cost < best_cost:
                    best_cost = cost
                    best_worker = w

        if best_worker is None:
            # No available eligible worker for this shift
            return SolverResult(
                feasible=False,
                optimal_cost=None,
                optimal_assignment=None,
                instance_id=instance.instance_id,
                solver_name="greedy",
                metadata={
                    "reason": f"No available eligible worker for shift {s}",
                    "shift_order": shift_order,
                    "partial_assignment": assignment.copy()
                }
            )

        # Assign worker to shift
        assignment[(best_worker, s)] = 1
        worker_assigned[best_worker] = True

    # All shifts assigned - verify feasibility
    feasible, details = check_constraints(instance, assignment)
    if feasible:
        cost = assignment_cost(instance, assignment)
        return SolverResult(
            feasible=True,
            optimal_cost=cost,
            optimal_assignment=assignment,
            instance_id=instance.instance_id,
            solver_name="greedy",
            metadata={"shift_order": shift_order}
        )
    else:
        # Should not happen if algorithm is correct, but verify anyway
        return SolverResult(
            feasible=False,
            optimal_cost=None,
            optimal_assignment=None,
            instance_id=instance.instance_id,
            solver_name="greedy",
            metadata={"reason": "Algorithm produced infeasible assignment", "details": details}
        )


def solve_both(instance: Instance) -> tuple[SolverResult, SolverResult]:
    """Run both exact and greedy solvers."""
    exact_result = solve_exact(instance)
    greedy_result = solve_greedy(instance)
    return exact_result, greedy_result