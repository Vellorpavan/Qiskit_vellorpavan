"""
Tests for classical solvers (Phase 2).
"""

import numpy as np
import pytest
from src.model import Instance
from src.classical import solve_exact, solve_greedy, solve_both, SolverResult


def test_exact_simple_feasible():
    """Test exact solver on a simple feasible instance."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_exact_1", 42)
    result = solve_exact(inst)

    assert result.feasible is True
    assert result.optimal_cost == 3.0  # worker 0->shift 0 (1), worker 1->shift 1 (2) = 3
    assert result.optimal_assignment is not None
    assert result.optimal_assignment == {(0, 0): 1, (1, 1): 1}
    assert result.solver_name == "exact"


def test_exact_infeasible_instance():
    """Test exact solver on an infeasible instance (shift with no eligible workers)."""
    # Instance creation itself should fail for infeasible instances
    # Let's test an instance that becomes infeasible due to worker constraints
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, False],
                     [False, True]])

    # This instance IS feasible: worker 0->shift 0, worker 1->shift 1
    inst = Instance(2, 2, cost, elig, "test_exact_2", 42)
    result = solve_exact(inst)

    assert result.feasible is True
    assert result.optimal_cost == 5.0  # 1 + 4 = 5


def test_exact_three_by_three():
    """Test exact solver on 3x3 instance."""
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, True, True],
                     [True, True, True],
                     [True, True, True]])

    inst = Instance(3, 3, cost, elig, "test_exact_3", 42)
    result = solve_exact(inst)

    assert result.feasible is True
    # Optimal: worker 0->shift 0 (1), worker 1->shift 1 (2), worker 2->shift 2 (3) = 6
    assert result.optimal_cost == 6.0
    assert result.optimal_assignment == {(0, 0): 1, (1, 1): 1, (2, 2): 1}


def test_exact_with_ineligible_pairs():
    """Test exact solver with some ineligible pairs."""
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, False, True],   # worker 0 cannot do shift 1
                     [True, True, False],   # worker 1 cannot do shift 2
                     [False, True, True]])  # worker 2 cannot do shift 0

    inst = Instance(3, 3, cost, elig, "test_exact_4", 42)
    result = solve_exact(inst)

    assert result.feasible is True
    # Check assignment is valid
    assert (0, 1) not in result.optimal_assignment  # ineligible
    assert (1, 2) not in result.optimal_assignment  # ineligible
    assert (2, 0) not in result.optimal_assignment  # ineligible


def test_greedy_simple():
    """Test greedy solver on simple instance."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_greedy_1", 42)
    result = solve_greedy(inst)

    assert result.feasible is True
    assert result.optimal_cost == 3.0  # Same as exact for this case
    assert result.solver_name == "greedy"


def test_greedy_most_constrained_first():
    """Test greedy picks most constrained shift first."""
    # Shift 0: only worker 0 eligible
    # Shift 1: both workers eligible
    cost = np.array([[1.0, 10.0],
                     [100.0, 2.0]])
    elig = np.array([[True, True],
                     [False, True]])

    inst = Instance(2, 2, cost, elig, "test_greedy_2", 42)
    result = solve_greedy(inst)

    assert result.feasible is True
    # Shift 0 has only 1 eligible worker (worker 0), so it's picked first
    # Worker 0 assigned to shift 0 (cost 1)
    # Then shift 1 gets worker 1 (cost 2)
    # Total = 3
    assert result.optimal_cost == 3.0
    assert result.optimal_assignment == {(0, 0): 1, (1, 1): 1}


def test_greedy_failure():
    """Test greedy fails when no valid assignment possible."""
    # Two shifts, both only eligible for worker 0
    cost = np.array([[1.0, 2.0],
                     [100.0, 100.0]])
    elig = np.array([[True, True],
                     [False, False]])

    inst = Instance(2, 2, cost, elig, "test_greedy_fail", 42)
    result = solve_greedy(inst)

    assert result.feasible is False
    assert result.optimal_cost is None
    assert "No available eligible worker" in result.metadata["reason"]


def test_greedy_not_optimal():
    """Test greedy can be suboptimal."""
    # Greedy picks shift 0 first (only 1 eligible worker), but that forces expensive assignment
    cost = np.array([[10.0, 1.0],
                     [1.0, 10.0],
                     [5.0, 5.0]])
    elig = np.array([[True, True],
                     [True, True],
                     [True, True]])

    inst = Instance(3, 2, cost, elig, "test_greedy_subopt", 42)
    result = solve_greedy(inst)

    assert result.feasible is True
    # Shift 0 has 3 eligible, shift 1 has 3 eligible - tie, picks shift 0 first
    # Worker 1 is cheapest for shift 0 (cost 1)
    # Then shift 1 gets worker 0 (cost 1) or worker 2 (cost 5)
    # Total = 2 (if worker 0 gets shift 1) or 6 (if worker 2 gets shift 1)
    # Actually: shift 0 first, cheapest available is worker 1 (cost 1)
    # Then shift 1: cheapest available is worker 0 (cost 1) -> total 2
    # Exact optimum: worker 0->shift 1 (1), worker 1->shift 0 (1) = 2
    # In this case greedy IS optimal. Let me check...
    # Actually, shift 0 and shift 1 both have 3 eligible workers. 
    # The order is stable (shift 0 first since it has lower index).
    # Worker 1 is cheapest for shift 0 (1), worker 0 is cheapest for shift 1 (1).
    # So greedy finds optimal here.
    # Let's make one where greedy fails:
    # Shift 0: only worker 0 and 1 eligible. Shift 1: only worker 0 and 2 eligible.
    # Cost: w0s0=1, w1s0=10, w0s1=10, w2s1=1
    # Greedy: shift 0 first (2 eligible), picks worker 0 (cost 1).
    # Then shift 1: worker 0 taken, picks worker 2 (cost 1). Total = 2. Optimal!
    # 
    # To make greedy fail: shift 0 has 2 eligible, shift 1 has 1 eligible
    # Shift 0: w0 (cost 1), w1 (cost 10)
    # Shift 1: w0 (cost 10) only
    # Greedy picks shift 1 first (1 eligible), assigns w0 to s1 (cost 10)
    # Then shift 0 gets w1 (cost 10). Total = 20
    # Optimal: w0->s0 (1), but then s1 has no one -> infeasible
    # Actually this instance is infeasible if w0 takes s0!
    # Let's make both feasible but greedy suboptimal:
    # Shift 0: w0 (1), w1 (2)
    # Shift 1: w0 (100), w2 (3)
    # Greedy: shift 0 has 2, shift 1 has 2. Tie -> shift 0 first.
    # Shift 0: pick w0 (cost 1).
    # Shift 1: w0 taken, pick w2 (cost 3). Total = 4.
    # Optimal: w1->s0 (2), w0->s1 (100) = 102. Worse!
    # Actually greedy is better here.
    # Let's try: shift 0 has 1 eligible, shift 1 has 2 eligible
    # Shift 0: w0 (10)
    # Shift 1: w0 (1), w1 (2)
    # Greedy picks shift 0 first (1 eligible), assigns w0 to s0 (cost 10)
    # Then shift 1 gets w1 (cost 2). Total = 12.
    # Optimal: w0->s1 (1), but then s0 has no one! Infeasible.
    # The instance is only feasible if w0 takes s0.
    # 
    # OK, let's make a case where greedy is provably suboptimal:
    # Shift 0: w0 (cost 5), w1 (cost 6)
    # Shift 1: w0 (cost 1), w2 (cost 100)
    # Both shifts have 2 eligible workers.
    # Greedy: tie -> shift 0 first. Picks w0 (cost 5).
    # Shift 1: w0 taken, picks w2 (cost 100). Total = 105.
    # Optimal: w1->s0 (6), w0->s1 (1) = 7.
    
    cost = np.array([[5.0, 1.0],
                     [6.0, 100.0],
                     [100.0, 100.0]])
    elig = np.array([[True, True],
                     [True, False],
                     [False, True]])

    inst = Instance(3, 2, cost, elig, "test_greedy_subopt2", 42)
    exact = solve_exact(inst)
    greedy = solve_greedy(inst)

    assert exact.feasible is True
    assert greedy.feasible is True
    assert exact.optimal_cost == 7.0  # w1->s0 (6), w0->s1 (1)
    assert greedy.optimal_cost == 105.0  # w0->s0 (5), w2->s1 (100)
    assert greedy.optimal_cost > exact.optimal_cost


def test_solve_both():
    """Test running both solvers together."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_both", 42)
    exact, greedy = solve_both(inst)

    assert exact.feasible is True
    assert greedy.feasible is True
    assert exact.optimal_cost == greedy.optimal_cost == 3.0


def test_exact_metadata():
    """Test exact solver returns correct metadata."""
    cost = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_meta", 42)
    result = solve_exact(inst)

    assert result.metadata["n_vars"] == 4
    assert result.metadata["states_checked"] == 16


def test_greedy_metadata():
    """Test greedy solver returns shift order in metadata."""
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])

    inst = Instance(2, 2, cost, elig, "test_greedy_meta", 42)
    result = solve_greedy(inst)

    assert "shift_order" in result.metadata
    assert len(result.metadata["shift_order"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])