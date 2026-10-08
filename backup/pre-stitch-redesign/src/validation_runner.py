"""
ShiftProof — Validation Suite Runner (Phase 5)

Runs exhaustive validation on test instances.
This is separate from the experiment engine (src/experiment.py).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.model import Instance
from src.validation import run_all_validations, print_validation_results, ValidationResult


@dataclass(frozen=True)
class ValidationSuiteResult:
    """Result of running validation suite on multiple instances."""
    instance_results: dict  # instance_id -> list[ValidationResult]
    all_passed: bool
    total_checks: int
    total_passed: int


def create_test_instances() -> list[Instance]:
    """Create a set of test instances for exhaustive validation."""
    instances = []

    # Instance 1: Simple 2x2, all eligible
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, True],
                     [True, True]])
    instances.append(Instance(2, 2, cost, elig, "val_2x2_simple", 1))

    # Instance 2: 2x2 with ineligible pairs
    cost = np.array([[1.0, 5.0],
                     [3.0, 2.0]])
    elig = np.array([[True, False],
                     [True, True]])
    instances.append(Instance(2, 2, cost, elig, "val_2x2_ineligible", 2))

    # Instance 3: 3x3 all eligible
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, True, True],
                     [True, True, True],
                     [True, True, True]])
    instances.append(Instance(3, 3, cost, elig, "val_3x3_all", 3))

    # Instance 4: 3x3 with ineligible pairs (constraint-heavy)
    cost = np.array([[1.0, 5.0, 9.0],
                     [4.0, 2.0, 8.0],
                     [7.0, 6.0, 3.0]])
    elig = np.array([[True, False, True],
                     [True, True, False],
                     [False, True, True]])
    instances.append(Instance(3, 3, cost, elig, "val_3x3_constrained", 4))

    # Instance 5: Cost-conflict (multiple shifts want same cheap worker)
    cost = np.array([[1.0, 1.0, 10.0],
                     [10.0, 10.0, 1.0],
                     [5.0, 5.0, 5.0]])
    elig = np.array([[True, True, True],
                     [True, True, True],
                     [True, True, True]])
    instances.append(Instance(3, 3, cost, elig, "val_3x3_cost_conflict", 5))

    # Instance 6: 3 workers, 2 shifts (more workers than shifts)
    cost = np.array([[1.0, 5.0],
                     [2.0, 4.0],
                     [3.0, 6.0]])
    elig = np.array([[True, True],
                     [True, True],
                     [True, True]])
    instances.append(Instance(3, 2, cost, elig, "val_3w2s", 6))

    # Instance 7: 2 workers, 2 shifts with one worker doing both (ineligible for one)
    cost = np.array([[1.0, 100.0],
                     [100.0, 2.0]])
    elig = np.array([[True, False],
                     [False, True]])
    instances.append(Instance(2, 2, cost, elig, "val_2w2s_disjoint", 7))

    return instances


def run_validation_suite(instances: list[Instance] = None, A: float = None, B: float = None, delta: float = 1.0) -> ValidationSuiteResult:
    """
    Run exhaustive validation on multiple instances.

    For each instance, runs all validation checks:
    - QUBO-Ising equivalence
    - Penalty behavior (6 checks)
    - Bit ordering
    - Deterministic behavior
    """
    if instances is None:
        instances = create_test_instances()

    instance_results = {}
    all_passed = True
    total_checks = 0
    total_passed = 0

    for inst in instances:
        print(f"\nValidating instance: {inst.instance_id} (n_vars={inst.n_vars})")
        results = run_all_validations(inst, A, B, delta)
        instance_results[inst.instance_id] = results

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        total_checks += total
        total_passed += passed

        if passed == total:
            print(f"  ✓ All {total} checks passed")
        else:
            print(f"  ✗ {passed}/{total} checks passed")
            all_passed = False
            for r in results:
                if not r.passed:
                    print(f"    FAIL: {r.check_name} - {r.message}")

    return ValidationSuiteResult(
        instance_results=instance_results,
        all_passed=all_passed,
        total_checks=total_checks,
        total_passed=total_passed
    )


def run_validation_with_penalty_sweep(instances: list[Instance] = None) -> dict:
    """
    Run validation with penalty multiplier sweep (λ).
    Tests λ = 0.5, 1.0, 2.0, 5.0, 10.0
    """
    if instances is None:
        instances = create_test_instances()

    multipliers = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}

    for inst in instances:
        print(f"\nPenalty sweep for {inst.instance_id}:")
        inst_results = {}

        # Get baseline penalty from greedy
        from src.classical import solve_greedy
        greedy = solve_greedy(inst)
        if not greedy.feasible:
            from src.classical import solve_exact
            exact = solve_exact(inst)
            if not exact.feasible:
                continue
            base_cost = exact.optimal_cost
        else:
            base_cost = greedy.optimal_cost

        for lam in multipliers:
            A = B = lam * (base_cost + 1.0)
            val_results = run_all_validations(inst, A, B, delta=1.0)
            penalty_checks = [r for r in val_results if r.check_name.startswith("Penalty check")]
            all_penalty_passed = all(r.passed for r in penalty_checks)

            inst_results[lam] = {
                "A": A,
                "B": B,
                "all_passed": all_penalty_passed,
                "checks": {r.check_name: r.passed for r in penalty_checks}
            }
            status = "✓" if all_penalty_passed else "✗"
            print(f"  λ={lam}: A=B={A:.2f} {status}")

        results[inst.instance_id] = inst_results

    return results


def print_suite_summary(result: ValidationSuiteResult):
    """Print summary of validation suite."""
    print("\n" + "=" * 70)
    print("EXHAUSTIVE VALIDATION SUITE SUMMARY")
    print("=" * 70)
    for inst_id, checks in result.instance_results.items():
        passed = sum(1 for r in checks if r.passed)
        total = len(checks)
        status = "✓ PASS" if passed == total else "✗ FAIL"
        print(f"  {inst_id:30s} : {passed}/{total} {status}")

    print("-" * 70)
    print(f"Total: {result.total_passed}/{result.total_checks} checks passed")
    if result.all_passed:
        print("ALL VALIDATIONS PASSED - SAFE TO PROCEED TO QAOA EXPERIMENTS")
    else:
        print("VALIDATIONS FAILED - DO NOT PROCEED")
    print("=" * 70)


if __name__ == "__main__":
    instances = create_test_instances()
    result = run_validation_suite(instances)
    print_suite_summary(result)

    # Also run penalty sweep
    print("\n" + "=" * 70)
    print("PENALTY MULTIPLIER SWEEP")
    print("=" * 70)
    sweep_results = run_validation_with_penalty_sweep(instances)