"""
ShiftProof — Mathematical Model

Instance representation, constraints, encoding/decoding, and cost calculation.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Instance:
    """
    Workforce scheduling instance.

    Workers w ∈ {0, ..., n_workers-1}
    Shifts  s ∈ {0, ..., n_shifts-1}

    Binary variable x[w,s] = 1 means worker w assigned to shift s.
    Ineligible pairs are removed from the variable set (variable reduction).
    """
    n_workers: int
    n_shifts: int
    cost_matrix: np.ndarray  # shape (n_workers, n_shifts), non-negative
    eligibility: np.ndarray  # shape (n_workers, n_shifts), bool
    instance_id: str
    seed: int

    # Derived: variable-to-qubit mapping (only eligible pairs)
    var_to_qubit: dict = field(init=False, repr=False)
    qubit_to_var: dict = field(init=False, repr=False)
    n_vars: int = field(init=False, repr=False)

    def __post_init__(self):
        # Build variable mapping (only eligible pairs)
        var_to_qubit = {}
        qubit_to_var = {}
        q = 0
        for w in range(self.n_workers):
            for s in range(self.n_shifts):
                if self.eligibility[w, s]:
                    var_to_qubit[(w, s)] = q
                    qubit_to_var[q] = (w, s)
                    q += 1
        object.__setattr__(self, 'var_to_qubit', var_to_qubit)
        object.__setattr__(self, 'qubit_to_var', qubit_to_var)
        object.__setattr__(self, 'n_vars', q)

        # Validation
        if self.cost_matrix.shape != (self.n_workers, self.n_shifts):
            raise ValueError(f"cost_matrix shape mismatch: {self.cost_matrix.shape} vs ({self.n_workers}, {self.n_shifts})")
        if self.eligibility.shape != (self.n_workers, self.n_shifts):
            raise ValueError(f"eligibility shape mismatch: {self.eligibility.shape} vs ({self.n_workers}, {self.n_shifts})")
        if np.any(self.cost_matrix < 0):
            raise ValueError("cost_matrix must be non-negative")
        if self.n_vars == 0:
            raise ValueError("No eligible worker-shift pairs (infeasible instance)")

        # Check each shift has at least one eligible worker
        for s in range(self.n_shifts):
            if not np.any(self.eligibility[:, s]):
                raise ValueError(f"Shift {s} has no eligible workers (infeasible instance)")

    def get_cost(self, w: int, s: int) -> float:
        """Get cost for worker-shift pair."""
        return float(self.cost_matrix[w, s])

    def is_eligible(self, w: int, s: int) -> bool:
        """Check if worker-shift pair is eligible."""
        return bool(self.eligibility[w, s])

    def get_qubit(self, w: int, s: int) -> Optional[int]:
        """Get qubit index for worker-shift pair, None if ineligible."""
        return self.var_to_qubit.get((w, s))

    def get_var(self, qubit: int) -> Optional[tuple]:
        """Get (worker, shift) for qubit index."""
        return self.qubit_to_var.get(qubit)


def check_constraints(instance: Instance, assignment: dict) -> tuple[bool, dict]:
    """
    Check all constraints on an assignment.

    Args:
        instance: The scheduling instance
        assignment: Dict mapping (w, s) -> 0 or 1 (only eligible pairs need be present)

    Returns:
        (feasible, constraint_results)
        constraint_results: dict with keys 'shift_coverage', 'worker_at_most_one', 'eligibility'
    """
    n_workers = instance.n_workers
    n_shifts = instance.n_shifts

    # Constraint 1: Each shift receives exactly one eligible worker
    shift_coverage = {}
    for s in range(n_shifts):
        assigned = sum(assignment.get((w, s), 0) for w in range(n_workers) if instance.is_eligible(w, s))
        shift_coverage[s] = (assigned == 1)

    # Constraint 2: Each worker receives at most one shift
    worker_at_most_one = {}
    for w in range(n_workers):
        assigned = sum(assignment.get((w, s), 0) for s in range(n_shifts) if instance.is_eligible(w, s))
        worker_at_most_one[w] = (assigned <= 1)

    # Constraint 3: Ineligible pairs never selected (by construction in encoding,
    # but check explicitly for any assignment dict passed in)
    eligibility_ok = {}
    for (w, s), val in assignment.items():
        if val == 1:
            eligibility_ok[(w, s)] = instance.is_eligible(w, s)
        else:
            eligibility_ok[(w, s)] = True  # 0 is always ok

    all_ok = all(shift_coverage.values()) and all(worker_at_most_one.values()) and all(eligibility_ok.values())

    return all_ok, {
        'shift_coverage': shift_coverage,
        'worker_at_most_one': worker_at_most_one,
        'eligibility': eligibility_ok
    }


def assignment_cost(instance: Instance, assignment: dict) -> float:
    """
    Calculate assignment cost C(x) = Σ c[w,s] * x[w,s].

    Only sums over eligible pairs with x[w,s] = 1.
    """
    cost = 0.0
    for (w, s), val in assignment.items():
        if val == 1:
            cost += instance.get_cost(w, s)
    return cost


def decode_bitstring(instance: Instance, bitstring: str) -> dict:
    """
    Decode a bitstring into an assignment dict.

    Bitstring order: qubit 0 = leftmost or rightmost? 
    We use Qiskit convention: bitstring[0] = qubit 0 (leftmost = MSB in Qiskit's default).
    But we'll be explicit and test this.
    """
    # Qiskit's get_counts() returns bitstrings with qubit 0 as RIGHTMOST (LSB)
    # So bitstring[-1] corresponds to qubit 0
    # We'll handle both by checking length
    n_vars = instance.n_vars
    if len(bitstring) != n_vars:
        raise ValueError(f"Bitstring length {len(bitstring)} != n_vars {n_vars}")

    assignment = {}
    for q in range(n_vars):
        # Qiskit: bitstring[-(q+1)] is qubit q
        bit = int(bitstring[-(q + 1)])
        if bit == 1:
            var = instance.get_var(q)
            if var is not None:
                assignment[var] = 1
            else:
                # Should not happen if mapping is correct
                pass
    return assignment


def encode_assignment(instance: Instance, assignment: dict) -> str:
    """
    Encode an assignment dict into a bitstring (qubit 0 = rightmost/LSB).
    """
    n_vars = instance.n_vars
    bits = ['0'] * n_vars
    for (w, s), val in assignment.items():
        if val == 1:
            q = instance.get_qubit(w, s)
            if q is not None:
                bits[-(q + 1)] = '1'
    return ''.join(bits)


def get_all_bitstrings(n_vars: int) -> list[str]:
    """Generate all 2^n bitstrings for exhaustive checking."""
    return [format(i, f'0{n_vars}b') for i in range(2 ** n_vars)]


def bitstring_to_assignment(instance: Instance, bitstring: str) -> dict:
    """Alias for decode_bitstring for clarity."""
    return decode_bitstring(instance, bitstring)


def assignment_to_bitstring(instance: Instance, assignment: dict) -> str:
    """Alias for encode_assignment for clarity."""
    return encode_assignment(instance, assignment)