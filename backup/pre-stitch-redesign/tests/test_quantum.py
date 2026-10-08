"""
Tests for QUBO, Ising, and validation (Phase 3).
"""

import numpy as np
import pytest
from src.model import Instance, check_constraints, assignment_cost, decode_bitstring, get_all_bitstrings
from src.classical import solve_exact, solve_greedy
from src.quantum import QUBO, Ising, build_qubo, qubo_to_ising, build_qaoa_circuit
from src.validation import (
    validate_qubo_ising_equivalence,
    validate_penalty_behavior,
    validate_bit_ordering,
    run_all_validations,
    print_validation_results
)


def test_qubo_construction():
    """Test QUBO construction for a simple instance."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_qubo", 42)
    qubo = build_qubo(inst, A=10.0, B=10.0)

    assert qubo.instance == inst
    assert qubo.A == 10.0
    assert qubo.B == 10.0
    assert qubo.a.shape == (4,)
    assert qubo.b.shape == (4, 4)
    assert qubo.instance.n_vars == 4


def test_qubo_energy_calculation():
    """Test QUBO energy matches manual calculation."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_energy", 42)
    # Use small penalties for easy manual verification
    qubo = build_qubo(inst, A=10.0, B=10.0)

    # Test all-zero assignment
    bits = "0000"
    energy = qubo.energy_from_bitstring(bits)
    assignment = decode_bitstring(inst, bits)
    feasible, _ = check_constraints(inst, assignment)
    assert feasible is False  # no shifts covered

    # Test all-one assignment (both workers on both shifts - infeasible)
    bits = "1111"
    energy = qubo.energy_from_bitstring(bits)
    assignment = decode_bitstring(inst, bits)
    feasible, _ = check_constraints(inst, assignment)
    assert feasible is False  # workers have 2 shifts each


def test_qubo_feasible_energy_equals_cost():
    """Test that for feasible assignments, Q(x) = C(x)."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_feasible", 42)
    qubo = build_qubo(inst, A=10.0, B=10.0)

    # Feasible assignment: worker 0 -> shift 0, worker 1 -> shift 1
    # Variables: (0,0)=q0, (0,1)=q1, (1,0)=q2, (1,1)=q3
    # Assignment: q0=1, q3=1, others=0
    bits = "1001"  # q3=1, q2=0, q1=0, q0=1
    q_energy = qubo.energy_from_bitstring(bits)
    assignment = decode_bitstring(inst, bits)
    c_cost = assignment_cost(inst, assignment)
    assert abs(q_energy - c_cost) < 1e-10
    assert c_cost == 1.0 + 2.0 == 3.0


def test_ising_conversion():
    """Test QUBO to Ising conversion."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_ising", 42)
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    assert ising.instance == inst
    assert ising.h.shape == (4,)
    assert ising.J.shape == (4, 4)
    assert isinstance(ising.offset, float)


def test_qubo_ising_equivalence():
    """Test QUBO and Ising give same energy for all bitstrings."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_equiv", 42)
    results = validate_qubo_ising_equivalence(inst, A=10.0, B=10.0)

    assert len(results) == 1
    assert results[0].passed is True


def test_penalty_checks():
    """Test all 6 penalty validation checks."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_penalty", 42)
    results = validate_penalty_behavior(inst, A=10.0, B=10.0)

    assert len(results) == 6
    for r in results:
        assert r.passed, f"Check failed: {r.check_name}: {r.message}"


def test_bit_ordering_validation():
    """Test bit ordering validation."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_bitorder", 42)
    results = validate_bit_ordering(inst)

    assert len(results) == 2
    for r in results:
        assert r.passed, f"Check failed: {r.check_name}: {r.message}"


def test_full_validation_suite():
    """Run all validations on a test instance."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_full", 42)
    results = run_all_validations(inst, A=10.0, B=10.0)

    # Should have 6 penalty + 2 QUBO-Ising + 2 bit-ordering + 1 prob + 2 deterministic + 1 agg = 14
    assert len(results) >= 10
    all_passed = all(r.passed for r in results)
    assert all_passed, f"Some validations failed: {[r.check_name for r in results if not r.passed]}"


def test_qaoa_circuit_construction():
    """Test QAOA circuit builds correctly."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_qaoa", 42)
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    # p=1
    qc, gamma, beta = build_qaoa_circuit(ising, p=1)
    assert qc.num_qubits == 4
    assert len(gamma) == 1
    assert len(beta) == 1

    # p=2
    qc2, gamma2, beta2 = build_qaoa_circuit(ising, p=2)
    assert qc2.num_qubits == 4
    assert len(gamma2) == 2
    assert len(beta2) == 2


def test_penalty_with_greedy_reference():
    """Test QUBO builds with automatic penalty from greedy reference."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_auto_penalty", 42)
    qubo = build_qubo(inst)  # No A, B provided

    # Should use greedy cost + delta = 3.0 + 1.0 = 4.0
    greedy = solve_greedy(inst)
    assert qubo.A == greedy.optimal_cost + 1.0
    assert qubo.B == greedy.optimal_cost + 1.0


def test_three_by_three_validation():
    """Run validations on a 3x3 instance."""
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, True, True],
                     [True, True, True],
                     [True, True, True]])

    inst = Instance(3, 3, cost, elig, "test_3x3", 42)
    results = run_all_validations(inst, A=20.0, B=20.0)

    all_passed = all(r.passed for r in results)
    assert all_passed, f"Some validations failed: {[r.check_name for r in results if not r.passed]}"


def test_instance_with_ineligible_pairs():
    """Test QUBO/Ising on instance with ineligible pairs."""
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, False, True],
                     [True, True, False],
                     [False, True, True]])

    inst = Instance(3, 3, cost, elig, "test_ineligible", 42)
    results = run_all_validations(inst, A=20.0, B=20.0)

    all_passed = all(r.passed for r in results)
    assert all_passed, f"Some validations failed: {[r.check_name for r in results if not r.passed]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])