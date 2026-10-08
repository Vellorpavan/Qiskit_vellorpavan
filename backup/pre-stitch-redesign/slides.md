# Slide 1: Problem & Approach

## ShiftProof — Constraint-Verified Quantum Scheduling

**Qiskit Fall Fest 2026 — Challenge I6**

### The Problem
Assign workers to shifts under constraints:
- Every shift: exactly one eligible worker
- Every worker: at most one shift
- Ineligible pairs: never selected
- **Minimize total assignment cost**

### Our Approach
| Method | Description |
|---|---|
| **Exact** | Exhaustive enumeration (ground truth) |
| **Greedy** | Most-constrained shift → cheapest available worker |
| **QAOA** | QUBO → Ising → p=1 QAOA on AerSimulator |

**Key principle**: Every solution independently verified by original constraint checker

---

# Slide 2: Methodology & Validation

## Mathematical Rigor

### QUBO with Safe Penalties
```
Q(x) = C(x) + A·Σ(Σx[w,s]−1)² + B·ΣΣx[w,s]x[w,t]
A = B = C(x_ref) + δ  (δ=1)
```
**Proven**: Global QUBO minimum is feasible

### Exhaustive Verification (all 2^n bitstrings)
✅ Q(x) = H(z(x)) — QUBO/Ising equivalence  
✅ Feasible x: Q(x) = C(x)  
✅ Infeasible x: Q(x) ≥ min(A,B) > C(x_ref)  
✅ argmin Q is feasible  
✅ C(argmin Q) = exact optimum  
✅ Infeasible instances detected at construction

### Penalty Sensitivity Sweep
| λ | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|
| Passes | 1/7 | 7/7 | 7/7 | 7/7 | 7/7 |

---

# Slide 3: Results & Conclusions

## 100-Instance Benchmark (AerSimulator, 1024 shots, p=1)

| Metric | Value |
|---|---|
| QAOA feasibility rate (mean) | 5.5% |
| QAOA feasible instances | 97/100 (97.0%) |
| Greedy feasible instances | 95/100 (95.0%) |
| QAOA found optimal (prob > 0) | **82.5%** (80/97 QAOA-feasible) |
| Greedy found optimal | **54.7%** (52/95 greedy-feasible) |
| QAOA best < Greedy best | 38/92 mutually feasible |
| Greedy best < QAOA best | 7/92 mutually feasible |
| Median absolute gap | **0.0** |

## Category Breakdown (generated / QAOA-feasible / greedy-feasible)

| Category | Generated | QAOA-feasible | Greedy-feasible | QAOA Opt% | Greedy Opt% |
|---|---|---|---|---|---|
| Easy | 40 | 40 | 40 | 100% | 100% |
| Constraint-heavy | 30 | 27 | 25 | 85.2% | 76.0% |
| Cost-conflict | 30 | 30 | 30 | 76.7% | 80.0% |

## Conclusion

> **QAOA p=1 finds optimal schedules more often than greedy** (82.5% of QAOA-feasible vs 54.7% of greedy-feasible), but low feasibility rates (5.5%) show constrained sampling remains challenging. No quantum advantage claimed — this is a verified, reproducible baseline for small instances.