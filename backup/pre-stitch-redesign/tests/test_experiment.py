"""
Tests for Experiment Engine and Dataset (Phase 6, 7).
"""

import pytest
import numpy as np
from src.dataset import (
    generate_easy_instance,
    generate_constraint_heavy_instance,
    generate_cost_conflict_instance,
    generate_all_instances,
    save_instances,
    load_instances,
    InstanceRecord
)
from src.experiment import run_single_experiment, ExperimentMetrics


def test_generate_easy_instance():
    """Test easy instance generation."""
    inst = generate_easy_instance(42, "test_easy")
    assert inst.n_workers == 3
    assert inst.n_shifts == 3
    assert inst.n_vars > 0
    assert inst.instance_id == "test_easy"
    assert inst.seed == 42


def test_generate_constraint_heavy_instance():
    """Test constraint-heavy instance generation."""
    inst = generate_constraint_heavy_instance(42, "test_constrained")
    assert inst.n_workers == 3
    assert inst.n_shifts == 3
    assert inst.n_vars > 0
    # Should have fewer eligible pairs than easy
    assert inst.n_vars <= 9


def test_generate_cost_conflict_instance():
    """Test cost-conflict instance generation."""
    inst = generate_cost_conflict_instance(42, "test_conflict")
    assert inst.n_workers == 3
    assert inst.n_shifts == 3
    assert inst.n_vars > 0


def test_generate_all_instances():
    """Test generating all 100 instances."""
    instances = generate_all_instances()
    assert len(instances) == 100

    # Count categories
    easy = sum(1 for i in instances if i.instance_id.startswith("easy"))
    constrained = sum(1 for i in instances if i.instance_id.startswith("constraint_heavy"))
    conflict = sum(1 for i in instances if i.instance_id.startswith("cost_conflict"))

    assert easy == 40
    assert constrained == 30
    assert conflict == 30

    # All instances should be valid
    for inst in instances:
        assert inst.n_vars > 0
        assert inst.n_workers == 3
        assert inst.n_shifts == 3


def test_save_load_instances():
    """Test saving and loading instances."""
    instances = generate_all_instances()
    save_instances(instances, "results/test_instances.json")
    loaded = load_instances("results/test_instances.json")

    assert len(loaded) == 100
    for orig, loaded_inst in zip(instances, loaded):
        assert orig.instance_id == loaded_inst.instance_id
        assert orig.seed == loaded_inst.seed
        np.testing.assert_array_equal(orig.cost_matrix, loaded_inst.cost_matrix)
        np.testing.assert_array_equal(orig.eligibility, loaded_inst.eligibility)


def test_run_single_experiment():
    """Test running a single experiment."""
    inst = generate_easy_instance(42, "test_exp")
    metrics = run_single_experiment(inst, shots=100, maxiter=5, seed=42)

    assert isinstance(metrics, ExperimentMetrics)
    assert metrics.instance_id == "test_exp"
    assert metrics.n_vars == inst.n_vars
    assert metrics.qaoa_shots == 100
    assert metrics.qaoa_p == 1
    assert metrics.qaoa_optimizer == "COBYLA"
    assert metrics.qaoa_maxiter == 5
    assert metrics.qaoa_seed == 42
    assert 0.0 <= metrics.qaoa_feasible_rate <= 1.0
    assert 0.0 <= metrics.qaoa_optimal_solution_probability <= 1.0


def test_experiment_metrics_gaps():
    """Test that gap metrics are computed correctly."""
    inst = generate_easy_instance(42, "test_gaps")
    metrics = run_single_experiment(inst, shots=200, maxiter=10, seed=42)

    if metrics.exact_optimal_cost is not None and metrics.qaoa_best_feasible_cost is not None:
        expected_abs_gap = metrics.qaoa_best_feasible_cost - metrics.exact_optimal_cost
        assert abs(metrics.absolute_gap - expected_abs_gap) < 1e-10

        if metrics.exact_optimal_cost > 0:
            expected_rel_gap = expected_abs_gap / metrics.exact_optimal_cost
            assert abs(metrics.relative_gap - expected_rel_gap) < 1e-10


def test_deterministic_experiment():
    """Test that experiment is deterministic with same seed."""
    inst = generate_easy_instance(42, "test_det")

    m1 = run_single_experiment(inst, shots=200, maxiter=10, seed=123)
    m2 = run_single_experiment(inst, shots=200, maxiter=10, seed=123)

    assert m1.qaoa_optimal_params == m2.qaoa_optimal_params
    assert m1.qaoa_optimal_energy == m2.qaoa_optimal_energy
    assert m1.qaoa_feasible_rate == m2.qaoa_feasible_rate
    assert m1.qaoa_feasible_samples == m2.qaoa_feasible_samples


def test_deterministic_experiment():
    """Test that experiment is deterministic with same seed."""
    inst = generate_easy_instance(42, "test_det")

    m1 = run_single_experiment(inst, shots=200, maxiter=10, seed=123)
    m2 = run_single_experiment(inst, shots=200, maxiter=10, seed=123)

    assert m1.qaoa_optimal_params == m2.qaoa_optimal_params
    assert m1.qaoa_optimal_energy == m2.qaoa_optimal_energy
    assert m1.qaoa_feasible_rate == m2.qaoa_feasible_rate
    assert m1.qaoa_feasible_samples == m2.qaoa_feasible_samples


def test_denominator_counts():
    """Test that denominator counts match expected values from the 100-instance benchmark."""
    from src.experiment import load_metrics_csv
    metrics = load_metrics_csv('results/metrics.csv')
    
    # Total instances
    assert len(metrics) == 100
    
    # QAOA feasible
    qaoa_feasible = [m for m in metrics if m.qaoa_feasible]
    assert len(qaoa_feasible) == 97
    
    # QAOA zero-feasible failures
    qaoa_failed = [m for m in metrics if not m.qaoa_feasible]
    assert len(qaoa_failed) == 3
    
    # Greedy feasible
    greedy_feasible = [m for m in metrics if m.greedy_feasible]
    assert len(greedy_feasible) == 95
    
    # Greedy failures
    greedy_failed = [m for m in metrics if m.greedy_failed]
    assert len(greedy_failed) == 5
    
    # Mutually feasible
    mutually_feasible = [m for m in metrics if m.qaoa_feasible and m.greedy_feasible]
    assert len(mutually_feasible) == 92
    
    # Exact feasible (all 100)
    exact_feasible = [m for m in metrics if m.exact_feasible]
    assert len(exact_feasible) == 100


def test_optimal_denominators():
    """Test that optimal counts use correct method-specific denominators."""
    from src.experiment import load_metrics_csv
    metrics = load_metrics_csv('results/metrics.csv')
    
    qaoa_feasible = [m for m in metrics if m.qaoa_feasible]
    greedy_feasible = [m for m in metrics if m.greedy_feasible]
    
    # QAOA optimal: among QAOA-feasible
    qaoa_optimal = sum(1 for m in qaoa_feasible if m.qaoa_optimal_solution_probability > 0)
    assert qaoa_optimal == 80
    assert qaoa_optimal / len(qaoa_feasible) == pytest.approx(0.825, rel=1e-2)
    
    # Greedy optimal: among greedy-feasible
    greedy_optimal = sum(1 for m in greedy_feasible if m.greedy_cost is not None and m.exact_optimal_cost is not None and abs(m.greedy_cost - m.exact_optimal_cost) < 1e-6)
    assert greedy_optimal == 52
    assert greedy_optimal / len(greedy_feasible) == pytest.approx(0.547, rel=1e-2)


def test_direct_comparison_denominator():
    """Test that direct QAOA vs Greedy comparison uses mutually feasible subset."""
    from src.experiment import load_metrics_csv
    metrics = load_metrics_csv('results/metrics.csv')
    
    mutually_feasible = [m for m in metrics if m.qaoa_feasible and m.greedy_feasible]
    assert len(mutually_feasible) == 92
    
    qaoa_better = sum(1 for m in mutually_feasible if m.qaoa_best_feasible_cost is not None and m.greedy_cost is not None and m.qaoa_best_feasible_cost < m.greedy_cost - 1e-6)
    greedy_better = sum(1 for m in mutually_feasible if m.qaoa_best_feasible_cost is not None and m.greedy_cost is not None and m.greedy_cost < m.qaoa_best_feasible_cost - 1e-6)
    same = sum(1 for m in mutually_feasible if m.qaoa_best_feasible_cost is not None and m.greedy_cost is not None and abs(m.qaoa_best_feasible_cost - m.greedy_cost) < 1e-6)
    
    assert qaoa_better == 38
    assert greedy_better == 7
    assert same == 47
    assert qaoa_better + greedy_better + same == len(mutually_feasible)


def test_category_generated_counts():
    """Test that category generated counts remain 40/30/30."""
    from src.experiment import load_metrics_csv
    metrics = load_metrics_csv('results/metrics.csv')
    
    easy = [m for m in metrics if m.category == 'easy']
    constraint_heavy = [m for m in metrics if m.category == 'constraint_heavy']
    cost_conflict = [m for m in metrics if m.category == 'cost_conflict']
    
    assert len(easy) == 40
    assert len(constraint_heavy) == 30
    assert len(cost_conflict) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])