"""
ShiftProof — Experiment Engine (Phase 6)

Runs QAOA experiments on instances, collects all metrics,
compares with classical baselines, and writes results.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import json
import time
from datetime import datetime

from src.model import Instance, check_constraints, assignment_cost, decode_bitstring, encode_assignment
from src.classical import solve_exact, solve_greedy, SolverResult
from src.quantum import (
    build_qubo,
    qubo_to_ising,
    run_qaoa_p1,
    QAOAResult
)
from src.dataset import generate_all_instances, save_instances, load_instances


@dataclass(frozen=True)
class ExperimentMetrics:
    """Complete metrics for one instance."""
    # Instance info
    instance_id: str
    seed: int
    category: str
    n_workers: int
    n_shifts: int
    n_vars: int
    n_qubits: int

    # Classical exact
    exact_feasible: bool
    exact_optimal_cost: Optional[float]
    exact_optimal_assignment: Optional[dict]
    exact_runtime: float

    # Classical greedy
    greedy_feasible: bool
    greedy_cost: Optional[float]
    greedy_assignment: Optional[dict]
    greedy_runtime: float
    greedy_failed: bool

    # QAOA
    qaoa_feasible: bool
    qaoa_optimal_energy: float
    qaoa_optimal_params: list
    qaoa_shots: int
    qaoa_p: int
    qaoa_optimizer: str
    qaoa_maxiter: int
    qaoa_seed: int
    qaoa_feasible_samples: int
    qaoa_feasible_rate: float
    qaoa_optimal_solution_probability: float
    qaoa_best_feasible_cost: Optional[float]
    qaoa_mean_feasible_cost: Optional[float]
    qaoa_assignment_cost_best: Optional[float]
    qaoa_qubo_energy_best: Optional[float]
    qaoa_circuit_depth: int
    qaoa_two_qubit_gate_count: int
    qaoa_optimizer_evaluations: int
    qaoa_runtime: float

    # Derived metrics
    absolute_gap: Optional[float] = None
    relative_gap: Optional[float] = None

    # Metadata
    python_version: str = ""
    qiskit_version: str = ""
    qiskit_aer_version: str = ""
    backend: str = "AerSimulator"
    timestamp: str = ""

    def __post_init__(self):
        # Compute gaps
        if self.exact_optimal_cost is not None and self.qaoa_best_feasible_cost is not None:
            object.__setattr__(self, 'absolute_gap', self.qaoa_best_feasible_cost - self.exact_optimal_cost)
            if self.exact_optimal_cost > 0:
                object.__setattr__(self, 'relative_gap', self.absolute_gap / self.exact_optimal_cost)


def run_single_experiment(instance: Instance, shots: int = 1024, maxiter: int = 100, seed: int = 42) -> ExperimentMetrics:
    """Run complete experiment on a single instance."""
    import sys
    import qiskit
    import qiskit_aer

    start_time = time.time()

    # Exact solver
    exact_start = time.time()
    exact_result = solve_exact(instance)
    exact_runtime = time.time() - exact_start

    # Greedy solver
    greedy_start = time.time()
    greedy_result = solve_greedy(instance)
    greedy_runtime = time.time() - greedy_start

    # QAOA
    qaoa_result = run_qaoa_p1(instance, shots=shots, maxiter=maxiter, seed=seed)

    # Build metrics
    # Extract category properly from instance_id
    if instance.instance_id.startswith("easy"):
        category = "easy"
    elif instance.instance_id.startswith("constraint_heavy"):
        category = "constraint_heavy"
    elif instance.instance_id.startswith("cost_conflict"):
        category = "cost_conflict"
    else:
        category = instance.instance_id.split('_')[0]

    metrics = ExperimentMetrics(
        instance_id=instance.instance_id,
        seed=instance.seed,
        category=category,
        n_workers=instance.n_workers,
        n_shifts=instance.n_shifts,
        n_vars=instance.n_vars,
        n_qubits=instance.n_vars,

        # Exact
        exact_feasible=exact_result.feasible,
        exact_optimal_cost=exact_result.optimal_cost,
        exact_optimal_assignment=exact_result.optimal_assignment,
        exact_runtime=exact_runtime,

        # Greedy
        greedy_feasible=greedy_result.feasible,
        greedy_cost=greedy_result.optimal_cost,
        greedy_assignment=greedy_result.optimal_assignment,
        greedy_runtime=greedy_runtime,
        greedy_failed=not greedy_result.feasible,

        # QAOA
        qaoa_feasible=qaoa_result.feasible,
        qaoa_optimal_energy=qaoa_result.optimal_energy,
        qaoa_optimal_params=qaoa_result.optimal_params.tolist(),
        qaoa_shots=qaoa_result.shots,
        qaoa_p=qaoa_result.p,
        qaoa_optimizer=qaoa_result.optimizer,
        qaoa_maxiter=qaoa_result.maxiter,
        qaoa_seed=qaoa_result.seed,
        qaoa_feasible_samples=qaoa_result.feasible_samples,
        qaoa_feasible_rate=qaoa_result.feasible_rate,
        qaoa_optimal_solution_probability=qaoa_result.optimal_solution_probability,
        qaoa_best_feasible_cost=qaoa_result.best_feasible_cost,
        qaoa_mean_feasible_cost=qaoa_result.mean_feasible_cost,
        qaoa_assignment_cost_best=qaoa_result.assignment_cost_best,
        qaoa_qubo_energy_best=qaoa_result.qubo_energy_best,
        qaoa_circuit_depth=qaoa_result.circuit_depth,
        qaoa_two_qubit_gate_count=qaoa_result.two_qubit_gate_count,
        qaoa_optimizer_evaluations=qaoa_result.optimizer_evaluations,
        qaoa_runtime=qaoa_result.runtime_seconds,

        # Metadata
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        qiskit_version=qiskit.__version__,
        qiskit_aer_version=qiskit_aer.__version__,
        backend="AerSimulator",
        timestamp=datetime.now().isoformat()
    )

    return metrics


def run_experiment_suite(instances: list[Instance] = None, shots: int = 1024, maxiter: int = 100, seed: int = 42,
                         output_file: str = "results/metrics.csv") -> list[ExperimentMetrics]:
    """Run experiments on all instances and save to CSV."""
    if instances is None:
        instances = generate_all_instances()

    all_metrics = []

    print(f"\nRunning experiments on {len(instances)} instances...")
    print(f"Shots: {shots}, Maxiter: {maxiter}, Seed: {seed}")
    print("-" * 70)

    for i, inst in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {inst.instance_id} (n_vars={inst.n_vars})...", end=" ", flush=True)

        try:
            metrics = run_single_experiment(inst, shots=shots, maxiter=maxiter, seed=seed)
            all_metrics.append(metrics)
            status = "✓" if metrics.qaoa_feasible else "✗"
            print(f"{status} feasible_rate={metrics.qaoa_feasible_rate:.3f}, "
                  f"gap={metrics.absolute_gap:.3f}" if metrics.absolute_gap is not None else "✓")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            # Create failed metrics record
            import sys
            import qiskit
            import qiskit_aer
            failed_metrics = ExperimentMetrics(
                instance_id=inst.instance_id,
                seed=inst.seed,
                category=inst.instance_id.split('_')[0],
                n_workers=inst.n_workers,
                n_shifts=inst.n_shifts,
                n_vars=inst.n_vars,
                n_qubits=inst.n_vars,
                exact_feasible=False,
                exact_optimal_cost=None,
                exact_optimal_assignment=None,
                exact_runtime=0,
                greedy_feasible=False,
                greedy_cost=None,
                greedy_assignment=None,
                greedy_runtime=0,
                greedy_failed=True,
                qaoa_feasible=False,
                qaoa_optimal_energy=float('nan'),
                qaoa_optimal_params=[],
                qaoa_shots=shots,
                qaoa_p=1,
                qaoa_optimizer="COBYLA",
                qaoa_maxiter=maxiter,
                qaoa_seed=seed,
                qaoa_feasible_samples=0,
                qaoa_feasible_rate=0.0,
                qaoa_optimal_solution_probability=0.0,
                qaoa_best_feasible_cost=None,
                qaoa_mean_feasible_cost=None,
                qaoa_assignment_cost_best=None,
                qaoa_qubo_energy_best=None,
                qaoa_circuit_depth=0,
                qaoa_two_qubit_gate_count=0,
                qaoa_optimizer_evaluations=0,
                qaoa_runtime=0,
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                qiskit_version=qiskit.__version__,
                qiskit_aer_version=qiskit_aer.__version__,
                backend="AerSimulator",
                timestamp=datetime.now().isoformat()
            )
            all_metrics.append(failed_metrics)

    # Save to CSV
    save_metrics_csv(all_metrics, output_file)
    print("-" * 70)
    print(f"Saved {len(all_metrics)} results to {output_file}")

    return all_metrics


def save_metrics_csv(metrics_list: list[ExperimentMetrics], filepath: str):
    """Save metrics to CSV file."""
    import csv

    if not metrics_list:
        return

    # Convert dataclasses to dicts
    rows = [asdict(m) for m in metrics_list]

    # Flatten nested structures for CSV
    flat_rows = []
    for row in rows:
        flat = {}
        for k, v in row.items():
            if isinstance(v, dict):
                # Convert dict with tuple keys to string-keyed dict
                if v and any(isinstance(key, tuple) for key in v.keys()):
                    flat[k] = json.dumps({f"{w},{s}": val for (w, s), val in v.items()})
                else:
                    flat[k] = json.dumps(v)
            elif isinstance(v, list):
                flat[k] = json.dumps(v)
            else:
                flat[k] = v
        flat_rows.append(flat)

    fieldnames = list(flat_rows[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def load_metrics_csv(filepath: str) -> list[ExperimentMetrics]:
    """Load metrics from CSV file."""
    import csv

    metrics_list = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse JSON fields
            for k, v in row.items():
                if v.startswith('[') or v.startswith('{'):
                    row[k] = json.loads(v)
                elif v == '':
                    row[k] = None
                elif k.endswith('_float') or k.endswith('_cost') or k.endswith('_gap') or k.endswith('_rate') or k.endswith('_probability') or k.endswith('_energy') or k.endswith('_runtime'):
                    try:
                        row[k] = float(v) if v != 'nan' else float('nan')
                    except:
                        pass
                elif k.endswith('_int') or k.endswith('_count') or k.endswith('_samples') or k.endswith('_iterations') or k.endswith('_depth') or k.endswith('_gate_count') or k.endswith('_evaluations') or k.endswith('_workers') or k.endswith('_shifts') or k.endswith('_vars') or k.endswith('_qubits') or k.endswith('_shots') or k.endswith('_p') or k.endswith('_maxiter') or k.endswith('_seed'):
                    try:
                        row[k] = int(v)
                    except:
                        pass
                elif k.endswith('_bool') or k.endswith('_feasible') or k.endswith('_failed'):
                    row[k] = v.lower() == 'true'

            metrics_list.append(ExperimentMetrics(**row))

    return metrics_list


def print_aggregate_results(metrics_list: list[ExperimentMetrics]):
    """Print aggregate statistics with explicit denominators."""
    if not metrics_list:
        return

    # Define subsets
    total = len(metrics_list)
    qaoa_feasible = [m for m in metrics_list if m.qaoa_feasible]
    qaoa_failed = [m for m in metrics_list if not m.qaoa_feasible]
    greedy_feasible = [m for m in metrics_list if m.greedy_feasible]
    greedy_failed = [m for m in metrics_list if m.greedy_failed]
    exact_feasible = [m for m in metrics_list if m.exact_feasible]
    
    # Mutually feasible (both QAOA and Greedy produced feasible solutions)
    mutually_feasible = [m for m in metrics_list if m.qaoa_feasible and m.greedy_feasible]
    
    # QAOA-feasible AND exact-feasible (for optimality comparison)
    qaoa_feasible_exact = [m for m in metrics_list if m.qaoa_feasible and m.exact_feasible]
    # Greedy-feasible AND exact-feasible
    greedy_feasible_exact = [m for m in metrics_list if m.greedy_feasible and m.exact_feasible]

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"Total benchmark instances: {total}")
    print()
    print("--- Feasibility (denominator = all 100 instances) ---")
    print(f"QAOA feasible (sampled >=1 feasible): {len(qaoa_feasible)}/{total} = {len(qaoa_feasible)/total*100:.1f}%")
    print(f"QAOA zero-feasible failures: {len(qaoa_failed)}/{total} = {len(qaoa_failed)/total*100:.1f}%")
    print(f"Greedy feasible: {len(greedy_feasible)}/{total} = {len(greedy_feasible)/total*100:.1f}%")
    print(f"Greedy failures: {len(greedy_failed)}/{total} = {len(greedy_failed)/total*100:.1f}%")
    print(f"Exact feasible (ground truth): {len(exact_feasible)}/{total} = {len(exact_feasible)/total*100:.1f}%")
    print()
    print("--- Mutual feasibility (for direct QAOA vs Greedy comparison) ---")
    print(f"Both QAOA and Greedy feasible: {len(mutually_feasible)}/{total}")
    print()
    print("--- Optimality (method-specific denominators) ---")
    # QAOA optimality: among QAOA-feasible & exact-feasible
    qaoa_optimal = sum(1 for m in qaoa_feasible_exact if m.qaoa_optimal_solution_probability > 0)
    print(f"QAOA found exact optimum: {qaoa_optimal}/{len(qaoa_feasible_exact)} = {qaoa_optimal/len(qaoa_feasible_exact)*100:.1f}% of QAOA-feasible instances")
    # Greedy optimality: among greedy-feasible & exact-feasible
    greedy_optimal = sum(1 for m in greedy_feasible_exact if m.greedy_cost is not None and m.exact_optimal_cost is not None and abs(m.greedy_cost - m.exact_optimal_cost) < 1e-6)
    print(f"Greedy found exact optimum: {greedy_optimal}/{len(greedy_feasible_exact)} = {greedy_optimal/len(greedy_feasible_exact)*100:.1f}% of greedy-feasible instances")
    print()
    print("--- Direct QAOA vs Greedy cost comparison (mutually feasible subset) ---")
    if mutually_feasible:
        qaoa_better = sum(1 for m in mutually_feasible if m.qaoa_best_feasible_cost is not None and m.greedy_cost is not None and m.qaoa_best_feasible_cost < m.greedy_cost - 1e-6)
        greedy_better = sum(1 for m in mutually_feasible if m.qaoa_best_feasible_cost is not None and m.greedy_cost is not None and m.greedy_cost < m.qaoa_best_feasible_cost - 1e-6)
        same = sum(1 for m in mutually_feasible if m.qaoa_best_feasible_cost is not None and m.greedy_cost is not None and abs(m.qaoa_best_feasible_cost - m.greedy_cost) < 1e-6)
        print(f"QAOA best cost < Greedy: {qaoa_better}/{len(mutually_feasible)}")
        print(f"Greedy best cost < QAOA: {greedy_better}/{len(mutually_feasible)}")
        print(f"Same cost: {same}/{len(mutually_feasible)}")
    print()
    print("--- Per-instance QAOA metrics (on QAOA-feasible & exact-feasible subset) ---")
    if qaoa_feasible_exact:
        feasible_rates = [m.qaoa_feasible_rate for m in qaoa_feasible_exact]
        opt_probs = [m.qaoa_optimal_solution_probability for m in qaoa_feasible_exact]
        abs_gaps = [m.absolute_gap for m in qaoa_feasible_exact if m.absolute_gap is not None]
        rel_gaps = [m.relative_gap for m in qaoa_feasible_exact if m.relative_gap is not None]

        print(f"Feasibility rate: mean={np.mean(feasible_rates):.3f}, median={np.median(feasible_rates):.3f}")
        print(f"Optimal solution prob: mean={np.mean(opt_probs):.3f}, median={np.median(opt_probs):.3f}")
        if abs_gaps:
            print(f"Absolute gap: mean={np.mean(abs_gaps):.3f}, median={np.median(abs_gaps):.3f}")
        if rel_gaps:
            print(f"Relative gap: mean={np.mean(rel_gaps):.3f}, median={np.median(rel_gaps):.3f}")

        # Per-category breakdown
        for cat in ["easy", "constraint_heavy", "cost_conflict"]:
            cat_all = [m for m in metrics_list if m.category == cat]
            cat_qaoa_feas = [m for m in cat_all if m.qaoa_feasible]
            cat_greedy_feas = [m for m in cat_all if m.greedy_feasible]
            cat_qaoa_feas_exact = [m for m in cat_all if m.qaoa_feasible and m.exact_feasible]
            
            if cat_qaoa_feas_exact:
                print(f"\n  {cat}: generated={len(cat_all)}, QAOA-feasible={len(cat_qaoa_feas)}, QAOA failures={len(cat_all)-len(cat_qaoa_feas)}, greedy-feasible={len(cat_greedy_feas)}, greedy failures={len(cat_all)-len(cat_greedy_feas)}")
                print(f"    QAOA metrics (denom={len(cat_qaoa_feas_exact)} QAOA-feasible):")
                print(f"    Feasibility: mean={np.mean([m.qaoa_feasible_rate for m in cat_qaoa_feas_exact]):.3f}")
                print(f"    Opt prob: mean={np.mean([m.qaoa_optimal_solution_probability for m in cat_qaoa_feas_exact]):.3f}")
                cat_gaps = [m.absolute_gap for m in cat_qaoa_feas_exact if m.absolute_gap is not None]
                if cat_gaps:
                    print(f"    Abs gap: mean={np.mean(cat_gaps):.3f}")
    print()
    print("--- Runtime ---")
    if qaoa_feasible:
        runtimes = [m.qaoa_runtime for m in qaoa_feasible]
        print(f"Mean QAOA runtime on QAOA-feasible instances: {np.mean(runtimes):.2f}s")
    if qaoa_feasible_exact:
        runtimes_exact = [m.qaoa_runtime for m in qaoa_feasible_exact]
        print(f"Mean QAOA runtime on QAOA-feasible & exact-feasible: {np.mean(runtimes_exact):.2f}s")


if __name__ == "__main__":
    # Generate instances if not exist
    import os
    if not os.path.exists("results/instances.json"):
        instances = generate_all_instances()
        save_instances(instances)
    else:
        instances = load_instances()

    # Run experiments
    metrics = run_experiment_suite(instances, shots=1024, maxiter=100, seed=42)
    print_aggregate_results(metrics)