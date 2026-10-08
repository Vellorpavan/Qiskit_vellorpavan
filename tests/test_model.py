"""
Tests for the mathematical model (Phase 1).
"""

import numpy as np
import pytest
from src.model import (
    Instance,
    check_constraints,
    assignment_cost,
    decode_bitstring,
    encode_assignment,
    get_all_bitstrings,
    bitstring_to_assignment,
    assignment_to_bitstring,
)


def test_instance_creation():
    """Test basic instance creation with variable mapping."""
    cost = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])
    elig = np.array([[True, True, False],
                     [True, False, True],
                     [False, True, True]])

    inst = Instance(
        n_workers=3,
        n_shifts=3,
        cost_matrix=cost,
        eligibility=elig,
        instance_id="test_001",
        seed=42
    )

    assert inst.n_workers == 3
    assert inst.n_shifts == 3
    assert inst.n_vars == 6  # 3x3 - 3 ineligible = 6

    # Check mapping
    assert inst.get_qubit(0, 0) is not None
    assert inst.get_qubit(0, 1) is not None
    assert inst.get_qubit(0, 2) is None  # ineligible
    assert inst.get_qubit(1, 0) is not None
    assert inst.get_qubit(1, 1) is None  # ineligible
    assert inst.get_qubit(1, 2) is not None
    assert inst.get_qubit(2, 0) is None  # ineligible
    assert inst.get_qubit(2, 1) is not None
    assert inst.get_qubit(2, 2) is not None


def test_instance_ineligible_shift():
    """Test that instance with no eligible worker for a shift fails."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, False],
                     [False, False]])  # Shift 1 has no eligible workers

    with pytest.raises(ValueError, match="no eligible workers"):
        Instance(
            n_workers=2,
            n_shifts=2,
            cost_matrix=cost,
            eligibility=elig,
            instance_id="test_bad",
            seed=42
        )


def test_instance_no_eligible_pairs():
    """Test that instance with no eligible pairs fails."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[False, False],
                     [False, False]])

    with pytest.raises(ValueError, match="No eligible worker-shift pairs"):
        Instance(
            n_workers=2,
            n_shifts=2,
            cost_matrix=cost,
            eligibility=elig,
            instance_id="test_bad",
            seed=42
        )


def test_check_constraints_feasible():
    """Test constraint checking on a feasible assignment."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)

    # Feasible: worker 0 -> shift 0, worker 1 -> shift 1
    assignment = {(0, 0): 1, (1, 1): 1}
    feasible, details = check_constraints(inst, assignment)
    assert feasible is True
    assert all(details['shift_coverage'].values())
    assert all(details['worker_at_most_one'].values())


def test_check_constraints_shift_uncovered():
    """Test constraint checking when a shift has no worker."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)

    # Infeasible: shift 1 uncovered
    assignment = {(0, 0): 1}
    feasible, details = check_constraints(inst, assignment)
    assert feasible is False
    assert details['shift_coverage'][0] is True
    assert details['shift_coverage'][1] is False


def test_check_constraints_worker_two_shifts():
    """Test constraint checking when a worker gets two shifts."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)

    # Infeasible: worker 0 gets both shifts
    assignment = {(0, 0): 1, (0, 1): 1}
    feasible, details = check_constraints(inst, assignment)
    assert feasible is False
    assert details['worker_at_most_one'][0] is False
    assert details['worker_at_most_one'][1] is True


def test_check_constraints_ineligible_selected():
    """Test constraint checking when ineligible pair is selected."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, False],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)

    # Infeasible: worker 0 assigned to shift 1 (ineligible)
    assignment = {(0, 0): 1, (0, 1): 1, (1, 1): 1}
    feasible, details = check_constraints(inst, assignment)
    assert feasible is False
    assert details['eligibility'][(0, 1)] is False


def test_assignment_cost():
    """Test assignment cost calculation."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)

    assignment = {(0, 0): 1, (1, 1): 1}
    cost_val = assignment_cost(inst, assignment)
    assert cost_val == 1.0 + 4.0 == 5.0

    # Zero assignment
    assignment_empty = {}
    assert assignment_cost(inst, assignment_empty) == 0.0


def test_encode_decode_roundtrip():
    """Test encoding and decoding round-trip."""
    cost = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])
    elig = np.array([[True, True, False],
                     [True, False, True],
                     [False, True, True]])

    inst = Instance(3, 3, cost, elig, "test", 42)

    # Test all possible bitstrings
    for bits in get_all_bitstrings(inst.n_vars):
        assignment = decode_bitstring(inst, bits)
        encoded = encode_assignment(inst, assignment)
        assert encoded == bits, f"Round-trip failed: {bits} -> {encoded}"


def test_bitstring_ordering():
    """Test bitstring ordering matches Qiskit convention (qubit 0 = rightmost)."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)
    assert inst.n_vars == 4  # 2x2 all eligible = 4 variables

    # Assignment: qubit 0 = 1, others 0
    # Should give bitstring "0001" (qubit 3,2,1 left, qubit 0 right)
    assignment = {inst.get_var(0): 1}  # qubit 0 = 1
    bits = encode_assignment(inst, assignment)
    assert bits == "0001", f"Expected '0001', got '{bits}'"

    # Assignment: qubit 1 = 1
    assignment = {inst.get_var(1): 1}  # qubit 1 = 1
    bits = encode_assignment(inst, assignment)
    assert bits == "0010", f"Expected '0010', got '{bits}'"

    # Assignment: qubit 2 = 1
    assignment = {inst.get_var(2): 1}  # qubit 2 = 1
    bits = encode_assignment(inst, assignment)
    assert bits == "0100", f"Expected '0100', got '{bits}'"

    # Assignment: qubit 3 = 1
    assignment = {inst.get_var(3): 1}  # qubit 3 = 1
    bits = encode_assignment(inst, assignment)
    assert bits == "1000", f"Expected '1000', got '{bits}'"

    # All 1
    assignment = {inst.get_var(q): 1 for q in range(4)}
    bits = encode_assignment(inst, assignment)
    assert bits == "1111", f"Expected '1111', got '{bits}'"


def test_decode_known_basis():
    """Test decoding known basis states."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test", 42)
    assert inst.n_vars == 4

    # |0000> -> empty assignment
    assignment = decode_bitstring(inst, "0000")
    assert assignment == {}

    # |0001> (qubit 0 = 1) -> first variable
    assignment = decode_bitstring(inst, "0001")
    assert len(assignment) == 1
    assert list(assignment.keys())[0] == inst.get_var(0)

    # |0010> (qubit 1 = 1) -> second variable
    assignment = decode_bitstring(inst, "0010")
    assert len(assignment) == 1
    assert list(assignment.keys())[0] == inst.get_var(1)

    # |1111> -> all variables
    assignment = decode_bitstring(inst, "1111")
    assert len(assignment) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])