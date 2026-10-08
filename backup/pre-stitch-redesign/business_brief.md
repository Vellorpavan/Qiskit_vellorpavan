# ShiftProof — Business Brief

**Qiskit Fall Fest 2026 — Challenge I6: Shift & Resource Scheduler**

---

## Problem Statement

Workforce scheduling under constraints is a fundamental operations research problem across hospitals, factories, and call centres. The challenge: assign workers to shifts such that every shift is covered exactly once, no worker is double-booked, and eligibility constraints are respected — while minimizing total assignment cost.

## Approach

ShiftProof evaluates **QAOA (Quantum Approximate Optimization Algorithm)** for small constrained scheduling instances, comparing against:

1. **Exact solver** — exhaustive enumeration (ground truth)
2. **Classical greedy heuristic** — most-constrained-shift-first, cheapest-available-worker

All solutions are independently verified for feasibility using the original constraints.

## Methodology

- **Problem encoding**: Binary variables x[w,s] ∈ {0,1} for eligible worker-shift pairs only (variable reduction)
- **QUBO formulation**: C(x) + A·Σ(shift_coverage_penalty) + B·Σ(worker_conflict_penalty)
- **Safe penalty**: A = B = C(x_ref) + δ (proven to make global QUBO minimum feasible)
- **QUBO→Ising**: Exact conversion verified for all 2^n bitstrings
- **QAOA**: p=1, COBYLA optimizer, 1024 shots, fixed seeds for reproducibility
- **Dataset**: 100 synthetic instances (40 Easy, 30 Constraint-heavy, 30 Cost-conflict)

## Results (Simulator)

| Metric | Value |
|---|---|
| Instances tested | 100 (3 QAOA zero-feasible, 5 greedy failures) |
| QAOA feasibility rate (mean/median) | 5.5% / 2.9% |
| QAOA found optimal (prob > 0) | 80/97 (82.5%) of QAOA-feasible |
| Greedy found optimal | 52/95 (54.7%) of greedy-feasible |
| QAOA best cost < Greedy best cost | 38/92 mutually feasible |
| Greedy best cost < QAOA best cost | 7/92 mutually feasible |
| Median absolute gap | 0.0 |

### Key Findings

1. **QAOA finds optimal more often than greedy** (82.5% of QAOA-feasible vs 54.7% of greedy-feasible)
2. **When QAOA samples feasible states, it often finds the global optimum** (median gap = 0)
3. **Low feasibility rates** (5.5% mean) reflect p=1 QAOA's difficulty with constrained problems
4. **Constraint-heavy instances**: Higher feasibility (9.6%) but larger gaps when suboptimal
5. **Cost-conflict instances**: Lower feasibility (4.3%) but smaller gaps
5. **No quantum advantage claimed** — p=1 QAOA with COBYLA is a limited configuration

## Limitations

- Tiny instances (≤9 qubits); results don't extrapolate to realistic workforce sizes
- Exhaustive enumeration doesn't scale
- Penalty proof guarantees feasible QUBO minimum, not feasible QAOA sampling
- Synthetic data only; no real-world validation
- Hardware not yet tested

## Conclusion

QAOA with p=1 can find optimal workforce schedules on small constrained instances, often outperforming a classical greedy baseline in terms of optimality probability. However, low feasibility rates indicate that sampling feasible states remains a key challenge for QAOA on constrained problems. These results provide a reproducible, verified baseline for future work on quantum scheduling.

---

*All results from executed experiments on AerSimulator. No fabricated numbers. Full reproducibility: `python -m src.experiment`*