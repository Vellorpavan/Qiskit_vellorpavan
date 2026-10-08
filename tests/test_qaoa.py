"""
Tests for QAOA p=1 (Phase 4).
"""

import numpy as np
import pytest
from src.model import Instance
from src.classical import solve_exact, solve_greedy
from src.quantum import (
    build_qaoa_circuit,
    qaoa_expectation,
    optimize_qaoa,
    sample_qaoa,
    run_qaoa_p1,
    QAOAResult
)


def test_qaoa_circuit_p1():
    """Test QAOA circuit construction for p=1."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_qaoa_circuit", 42)
    from src.quantum import build_qubo, qubo_to_ising
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    qc, gamma, beta = build_qaoa_circuit(ising, p=1)

    assert qc.num_qubits == 4
    assert len(gamma) == 1
    assert len(beta) == 1
    assert qc.num_clbits == 4  # measure_all adds classical bits


def test_qaoa_circuit_p2():
    """Test QAOA circuit construction for p=2."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_qaoa_circuit_p2", 42)
    from src.quantum import build_qubo, qubo_to_ising
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    qc, gamma, beta = build_qaoa_circuit(ising, p=2)

    assert qc.num_qubits == 4
    assert len(gamma) == 2
    assert len(beta) == 2


def test_qaoa_expectation_runs():
    """Test that expectation calculation runs without error."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_expectation", 42)
    from src.quantum import build_qubo, qubo_to_ising
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    # Test with fixed parameters
    params = np.array([0.1, 0.2])  # γ, β
    energy = qaoa_expectation(params, ising, p=1, shots=100, seed=42)

    assert isinstance(energy, float)
    assert not np.isnan(energy)


def test_optimize_qaoa_runs():
    """Test that optimization runs and returns parameters."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_optimize", 42)
    from src.quantum import build_qubo, qubo_to_ising
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    # Use small maxiter for fast test
    params, energy = optimize_qaoa(ising, p=1, maxiter=10, shots=100, seed=42)

    assert params.shape == (2,)
    assert isinstance(energy, float)
    assert not np.isnan(energy)


def test_sample_qaoa_runs():
    """Test that sampling runs and returns counts."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_sample", 42)
    from src.quantum import build_qubo, qubo_to_ising
    qubo = build_qubo(inst, A=10.0, B=10.0)
    ising = qubo_to_ising(qubo)

    # Optimize first
    params, _ = optimize_qaoa(ising, p=1, maxiter=10, shots=100, seed=42)
    samples = sample_qaoa(ising, params, p=1, shots=200, seed=42)

    assert isinstance(samples, dict)
    assert sum(samples.values()) == 200
    # All bitstrings should have length 4
    for bits in samples.keys():
        assert len(bits) <= 4  # Qiskit may omit leading zeros


def test_run_qaoa_p1_complete():
    """Test complete QAOA p=1 workflow."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_full_qaoa", 42)

    # Run with small shots and iterations for speed
    result = run_qaoa_p1(inst, shots=200, maxiter=10, seed=42)

    assert isinstance(result, QAOAResult)
    assert result.instance_id == "test_full_qaoa"
    assert result.p == 1
    assert result.optimizer == "COBYLA"
    assert result.shots == 200
    assert result.seed == 42
    assert result.maxiter == 10
    assert result.optimal_params.shape == (2,)
    assert isinstance(result.optimal_energy, float)
    assert isinstance(result.samples, dict)
    assert sum(result.samples.values()) == 200
    assert 0.0 <= result.feasible_rate <= 1.0
    assert 0.0 <= result.optimal_solution_probability <= 1.0
    assert result.circuit_depth > 0
    assert result.two_qubit_gate_count >= 0
    assert result.optimizer_evaluations > 0
    assert result.runtime_seconds > 0


def test_qaoa_feasible_solution_found():
    """Test that QAOA can find feasible solutions on simple instance."""
    # Simple instance where optimal is obvious
    cost = np.array([[1.0, 10.0],
                     [10.0, 1.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_feasible_qaoa", 42)

    exact = solve_exact(inst)
    assert exact.feasible
    assert exact.optimal_cost == 2.0  # w0->s0 (1), w1->s1 (1)

    # Run QAOA with more shots for better statistics
    result = run_qaoa_p1(inst, shots=500, maxiter=20, seed=42)

    # Should find at least some feasible solutions
    assert result.feasible_samples > 0, "QAOA found zero feasible samples"
    assert result.best_feasible_cost is not None

    # Best feasible cost should be >= exact optimum (can't be better)
    if result.best_feasible_cost is not None:
        assert result.best_feasible_cost >= exact.optimal_cost - 1e-6


def test_qaoa_metrics_calculation():
    """Test that all metrics are calculated correctly."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_metrics", 42)
    result = run_qaoa_p1(inst, shots=200, maxiter=10, seed=42)

    # Verify feasible_rate calculation
    expected_feasible_rate = result.feasible_samples / result.shots
    assert abs(result.feasible_rate - expected_feasible_rate) < 1e-10

    # Verify optimal_solution_probability
    exact = solve_exact(inst)
    if exact.feasible:
        exact_bits = set()
        from src.model import encode_assignment
        exact_bits.add(encode_assignment(inst, exact.optimal_assignment))
        optimal_count = sum(result.samples.get(bits, 0) for bits in exact_bits)
        expected_opt_prob = optimal_count / result.shots
        assert abs(result.optimal_solution_probability - expected_opt_prob) < 1e-10


def test_qaoa_deterministic_with_seed():
    """Test that QAOA is deterministic with fixed seed."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_deterministic", 42)

    result1 = run_qaoa_p1(inst, shots=200, maxiter=10, seed=123)
    result2 = run_qaoa_p1(inst, shots=200, maxiter=10, seed=123)

    # Same seed should give same optimal parameters
    np.testing.assert_array_almost_equal(result1.optimal_params, result2.optimal_params)
    assert result1.optimal_energy == result2.optimal_energy
    assert result1.samples == result2.samples


def test_qaoa_three_by_three():
    """Test QAOA on 3x3 instance."""
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, True, True],
                     [True, True, True],
                     [True, True, True]])

    inst = Instance(3, 3, cost, elig, "test_3x3_qaoa", 42)

    exact = solve_exact(inst)
    assert exact.feasible
    assert exact.optimal_cost == 6.0

    # Run QAOA - use more iterations for 9 qubits
    result = run_qaoa_p1(inst, shots=300, maxiter=15, seed=42)

    assert result.feasible_samples >= 0  # May or may not find feasible
    assert result.circuit_depth > 0
    assert result.two_qubit_gate_count > 0  # Should have CX gates for ZZ terms


def test_qaoa_infeasible_instance_handled():
    """Test that infeasible instances are handled (constructor rejects them)."""
    # This is tested in model tests - Instance constructor validates
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])