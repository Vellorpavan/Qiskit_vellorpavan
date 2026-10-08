"""
Tests for Exhaustive Validation Suite (Phase 5).
"""

import pytest
from src.validation_runner import create_test_instances, run_validation_suite, run_validation_with_penalty_sweep


def test_create_test_instances():
    """Test that test instances are created correctly."""
    instances = create_test_instances()
    assert len(instances) == 7

    # Check each instance is valid
    for inst in instances:
        assert inst.n_vars > 0
        assert inst.n_workers > 0
        assert inst.n_shifts > 0


def test_validation_suite_all_pass():
    """Test that all validation checks pass on all test instances."""
    instances = create_test_instances()
    result = run_validation_suite(instances)

    assert result.all_passed, f"Validation failed: {result.total_passed}/{result.total_checks} passed"
    assert result.total_checks > 0


def test_penalty_sweep():
    """Test penalty multiplier sweep."""
    instances = create_test_instances()
    sweep_results = run_validation_with_penalty_sweep(instances)

    # All instances should have results for all multipliers
    for inst_id, inst_results in sweep_results.items():
        assert len(inst_results) == 5  # 5 multipliers
        for lam, res in inst_results.items():
            # At λ=1.0, should always pass (safe penalty)
            if lam == 1.0:
                assert res["all_passed"], f"Safe penalty failed for {inst_id}: {res['checks']}"


def test_individual_instance_validations():
    """Test each instance individually for detailed debugging."""
    instances = create_test_instances()

    for inst in instances:
        result = run_validation_suite([inst])
        assert result.all_passed, f"Instance {inst.instance_id} failed validation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])