# ShiftProof — Constraint-Verified Quantum Scheduling

> ShiftProof is a reproducible Qiskit experiment that evaluates QAOA for small constrained workforce scheduling by comparing quantum-generated schedules with a classical heuristic and an exact reference, while independently verifying feasibility and reporting solution quality, optimality gaps, and limitations.

**Hackathon:** Qiskit Fall Fest 2026 — Challenge **I6, Shift & Resource Scheduler**

| | |
|---|---|
| **No quantum-advantage claim** | This project does not claim quantum advantage, speedup, or superiority over classical methods. |
| **Synthetic data only** | No real hospital, factory, or call-centre data is used. No cost savings are claimed. |
| **No invented results** | Any number not produced by an executed run is shown as `NOT YET RUN`. |
| **Hardware is optional** | Hardware results appear only if a real IBM Quantum job completed. |

---

## Source-of-truth hierarchy

1. **Official hackathon materials** (the organizer's PPT/brief) define the *requirements*. If this README conflicts with them, the official materials win.
2. **This README** defines *our engineering design*: model, penalties, dataset, UI, and process.
3. **Code and executed results** define *what is actually true*. Documentation never overrides measured output.

Items marked **[OUR DESIGN]** are project decisions, not hackathon requirements.

**Rules for the implementing engineer (human or AI):**

- Never describe an **[OUR DESIGN]** choice as an official requirement. Example: "9 qubits", "3 workers × 3 shifts", and "100 instances" are our choices, not I6 limits.
- Prose arguments in this README (including the penalty proof) are **hypotheses until the code verifies them**. If verified code behavior contradicts the README, the code result wins and the README must be corrected.
- The UI is never a source of truth. Data flows one way: `src/` (tested mathematics and algorithms) → experiment results → UI visualization.

**Spec status:** frozen as the build specification (v1.0). Changes after this point must come from a failed verification or a correction to an official requirement, not from new concept rewrites. Next action: Phase 0 environment audit, then implementation.

---

## Contents

1. [Scope and requirements](#1-scope-and-requirements)
2. [Architecture](#2-architecture)
3. [Mathematical model](#3-mathematical-model)
4. [QUBO and penalty selection](#4-qubo-and-penalty-selection)
5. [QUBO → Ising](#5-qubo--ising)
6. [QAOA](#6-qaoa)
7. [Classical methods](#7-classical-methods)
8. [Synthetic benchmark dataset](#8-synthetic-benchmark-dataset)
9. [Validation](#9-validation)
10. [Experiment methodology and metrics](#10-experiment-methodology-and-metrics)
11. [Results](#11-results)
12. [IBM Quantum hardware](#12-ibm-quantum-hardware-optional-bonus)
13. [User interface](#13-user-interface)
14. [Limitations](#14-limitations)
15. [Build process and status](#15-build-process-and-status)
16. [Repository layout](#16-repository-layout)
17. [Installation and reproducibility](#17-installation-and-reproducibility)
18. [Security](#18-security)
19. [Deliverables map and final checklist](#19-deliverables-map-and-final-checklist)

---

## 1. Scope and requirements

**Official problem (I6).** A hospital, factory, or call centre must assign staff under constraints. The official build is a small scheduling QUBO solved with QAOA and compared with a classical heuristic.

**Official requirements as understood by this project** (verify against the official brief before submission): Qiskit implementation; simulator first; classical baseline with fair comparison; public GitHub repo with README and requirements; one-page business brief; 2-minute demo or 3 slides; public or synthetic data only; honest results and limitations; no quantum-advantage claims; real IBM hardware earns bonus points.

**[OUR DESIGN] choices — not official limits:**

| Choice | Value |
|---|---|
| Default instance size | 3 workers × 3 shifts (≤ 9 qubits) |
| Benchmark dataset | 100 synthetic instances |
| Exact reference | Exhaustive enumeration, tiny instances only |
| Application | A UI built on the tested backend, after the backend is validated |

**Business framing.** A generalized workforce-assignment problem relevant to hospitals, factories, and call centres.

**Business question.** *How can a constrained workforce assignment be modeled and evaluated as a small optimization problem, and what does QAOA actually achieve compared with a classical heuristic and the exact optimum?*

---

## 2. Architecture

```
                         ShiftProof
                              │
                        User Interface
                              │
                 ┌────────────┴────────────┐
                 │                         │
           Dataset / Input             Experiment
                 │                         │
          100 synthetic            ┌───────┴────────┐
           instances               │                │
                 │            Classical          Quantum
                 │            ├ Exact            ├ QUBO
                 │            └ Greedy           ├ Ising
                 │                               └ QAOA
                 └────────────────┬───────────────┘
                                  ▼
                     Decode + independent validation
                                  ▼
                         Benchmark analysis
                                  ▼
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
          Per-instance results            Aggregate (100) results
                 └────────────────┬────────────────┘
                                  ▼
                       Optional IBM hardware
                                  ▼
                         Final UI report
```

**Dependency direction:** `src/` → experiment results → UI. The UI imports `src/` and reads results; it never contains optimization, QUBO, or constraint logic.

The decode-and-verify stage sits **after** every method. No schedule from any method is trusted until the original constraint checker accepts it.

---

## 3. Mathematical model

- Workers `w ∈ W`, shifts `s ∈ S`, with neutral IDs (`Worker_001`, `Shift_001`, …).
- Binary variable `x[w,s] ∈ {0,1}`: 1 means worker `w` is assigned to shift `s`.
- Inputs: non-negative cost matrix `c[w,s]`, eligibility mask `e[w,s]`, seed.

**Objective (assignment cost):**

```
C(x) = Σ_{w,s} c[w,s] · x[w,s]
```

**Constraints (always checked on the original problem, never on the QUBO):**

1. Each shift receives **exactly one** eligible worker.
2. Each worker receives **at most one** shift in the toy horizon.
3. Ineligible worker–shift pairs are never selected.

**Three quantities kept strictly separate:**

| Quantity | Meaning |
|---|---|
| Assignment cost `C(x)` | The business objective |
| QUBO energy `Q(x)` | Penalized value the quantum algorithm minimizes |
| Feasibility | Boolean from the original constraint checker |

**[OUR DESIGN] Variable reduction.** Ineligible pairs are removed from the variable set instead of penalized, so constraint 3 holds by construction and fewer qubits are needed. If a shift has no eligible worker, the instance is infeasible and is reported as such. The variable-to-qubit map is stored with every instance and run.

---

## 4. QUBO and penalty selection

```
Q(x) = C(x)
     + A · Σ_s ( Σ_w x[w,s] − 1 )²
     + B · Σ_w Σ_{s<t} x[w,s] · x[w,t]
```

For binary variables: `(Σ_i x_i − 1)² = 1 − Σ_i x_i + 2 Σ_{i<j} x_i x_j`. This identity gives the constant `q0`, linear `a_i`, and quadratic `b_ij` coefficients.

### 4.1 Provably safe penalty **[OUR DESIGN]**

Let `x_ref` be a feasible reference solution (greedy if it succeeds, otherwise the exact optimum). Set

```
A = B = C(x_ref) + δ,   δ > 0   (default δ = 1 for integer costs)
```

*Argument.* Any infeasible `x` violates at least one constraint. A shift with no worker contributes `A`; a shift with `k ≥ 2` workers contributes `A(k−1)² ≥ A`; a worker with two shifts contributes at least `B`. Since `C(x) ≥ 0`, every infeasible `x` satisfies `Q(x) ≥ min(A,B) > C(x_ref) ≥ (optimal feasible cost)`. Therefore the **global QUBO minimum is feasible**, and for that state `Q(x) = C(x)`.

### 4.1a Required exhaustive verification (the proof is not sufficient on its own)

The prose argument assumes every infeasible state receives at least one full penalty term. That depends on the exact coefficients the code builds, so the **implemented QUBO** must be verified by brute force over all `2^n` bitstrings, for every test instance and every instance in the benchmark that uses the safe penalty:

| # | Check on the implemented QUBO | Must hold |
|---|---|---|
| 1 | `x` feasible | `Q(x) = C(x)` (penalty terms are exactly 0) |
| 2 | `x` infeasible | `Q(x) ≥ min(A,B)` |
| 3 | `x` infeasible | `Q(x) > C(x_ref)` |
| 4 | `argmin Q` | is feasible |
| 5 | `C(argmin Q)` | equals the exact optimum from enumeration |
| 6 | Infeasible instance (e.g. a shift with no eligible worker) | detected and reported; no penalty claim is made |

**Fail-stop rule:** if any check fails, stop, fix the QUBO construction or correct this section, and do not run QAOA experiments until all checks pass. Passing these checks on small instances validates the construction; it does not extend to larger sizes without re-verification.

### 4.2 What the proof does *not* give

- It does **not** show QAOA will sample feasible states reliably. Feasibility is always measured.
- Large penalties relative to the real costs can flatten the cost differences QAOA must resolve, which can lower feasibility and optimality rates. This is an expected experimental finding, not a bug.

### 4.3 Penalty sensitivity study

The experiment sweeps a penalty multiplier `λ` over the safe baseline (`A = B = λ · (C(x_ref) + δ)`). For every `λ` the exhaustive check reports whether the QUBO global minimum is still feasible. Values of `λ < 1` are labeled **unproven** and kept only if the exhaustive check passes for that instance. All sweep results are reported, including bad ones.

---

## 5. QUBO → Ising

Substitute `x_i = (1 − Z_i)/2` into `Q(x) = q0 + Σ a_i x_i + Σ_{i<j} b_ij x_i x_j`:

```
J_ij   = b_ij / 4
h_i    = −a_i / 2 − (1/4) Σ_{j≠i} b_ij        (over pairs containing i)
offset = q0 + Σ_i a_i / 2 + Σ_{i<j} b_ij / 4

H = offset + Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
```

The offset is retained so Ising energies equal QUBO energies exactly. **Validation:** `Q(x)` equals `H(z(x))` for every bitstring of the test instances.

---

## 6. QAOA

Pipeline: **QUBO → Ising → QAOA → measurement → decoding → constraint verification → metrics**.

| Element | Specification |
|---|---|
| Qubit mapping | Variable `i` ↔ qubit `i`, explicit, stored per run |
| Initial state | `|+⟩^n` |
| Cost layer | `exp(−iγH)`: `RZ(2γ·h_i)` and `RZZ(2γ·J_ij)`; angle conventions verified against the installed Qiskit, not assumed |
| Mixer | `RX(2β)` on every qubit |
| Depth | `p = 1` first; `p = 2` only after the full p=1 workflow is validated |
| Optimizer | Classical optimizer (default COBYLA) on expected QUBO energy, bounded evaluation budget, fixed seeds |
| Sampling | Final parameters → measurement shots |
| Bit ordering | Qiskit ordering is **not assumed**; tests use known basis states |
| Verification | Every sampled bitstring is decoded and checked; infeasible samples stay in all statistics |

Exact package versions and APIs come from the environment audit (`results/environment.json`), not from this README.

---

## 7. Classical methods

**A. Greedy heuristic (required baseline).** Choose the most constrained shift first (fewest eligible workers), assign the cheapest still-available eligible worker, repeat. If no valid choice remains, report **failure**. Greedy is never described as optimal.

**B. Exact solver (ground truth).** Exhaustive enumeration with the original constraints. Returns feasibility, true minimum feasible cost, an optimal assignment, and handles infeasible instances explicitly. It is a reference for tiny instances, **not** a scalable production algorithm.

---

## 8. Synthetic benchmark dataset

**[OUR DESIGN]** 100 reproducible synthetic instances. Not required by the hackathon.

| Category | Count | Purpose |
|---|---|---|
| Easy | 40 | Wide eligibility, little cost conflict |
| Constraint-heavy | 30 | Sparse eligibility, tight constrained shifts |
| Cost-conflict | 30 | Several shifts compete for the same cheap worker |

Each record holds: `instance_id`, workers, shifts, cost matrix, eligibility, planted feasible assignment (where used), `random_seed`, category, and metadata. Every instance is regenerable from its seed. The actual counts written to `results/instances.json` must match this table.

Data is realistic in *structure* (availability, unavailable pairs, competing cheap options) but is **not** real data.

**Scale policy.** If running QAOA on all 100 instances is impractical, the full dataset is kept and a documented representative quantum subset is used for deeper experiments. Nothing is silently dropped.

---

## 9. Validation

All checks must pass **before** any QAOA result is interpreted. If any fails, fix the mathematical layer first.

| Check | Method |
|---|---|
| Input validation | Shapes, non-negativity, eligibility |
| Encode/decode | Round trip over all bitstrings |
| Constraint checker | Known feasible and infeasible cases |
| Assignment cost | Compared with an independent computation |
| QUBO | `Q(x)` vs definition, all bitstrings |
| Ising equivalence | `Q(x) = H(z(x))`, all bitstrings |
| Exact solver | Hand-solved and infeasible cases |
| Greedy | Success and failure cases |
| Penalty behavior | Checks 1–6 of Section 4.1a on the implemented QUBO |
| Bit ordering | Known basis-state circuits |
| Probability normalization | Probabilities sum to 1 |
| Aggregation | Metrics recomputed independently on a fixture |

There is deliberately **no test asserting that QAOA finds the optimum**.

---

## 10. Experiment methodology and metrics

**Recorded per run:** instance ID, seed, QAOA parameters, optimizer and budget, shots, Qiskit and Python versions, backend, timestamp, circuit resources, runtime.

| Metric | Definition |
|---|---|
| Feasibility rate | feasible shots / total shots |
| Optimal-solution probability | shots equal to an exact-optimal assignment / total shots |
| Best feasible cost | min `C(x)` over feasible samples |
| Mean feasible cost | mean `C(x)` over feasible samples |
| Absolute gap | best feasible cost − exact optimum |
| Relative gap | absolute gap / exact optimum, **only if optimum > 0** |
| Optimizer evaluations, shots, runtime | measured |
| Qubits, depth, two-qubit gates | measured on the transpiled circuit |
| Seed variation | spread across seeds |

If there are **zero feasible samples**, cost metrics are reported as *"No feasible quantum sample observed"*, never as 0.

**Comparison:** Exact (ground truth) vs Greedy (baseline) vs QAOA. Both views are always shown:

- **Per-instance results**, including failures
- **100-instance aggregate**: mean and median feasibility and gap, how often each method found the optimum, failure counts for QAOA and greedy, runtime statistics

---

## 11. Results

> Every value below is `NOT YET RUN` until produced by an executed experiment. Results are written to `results/metrics.csv` and copied here from that file, never typed by hand.

**11.1 Controlled demonstration instances**

| Instance | Exact optimum | Greedy | QAOA feasibility | QAOA best cost | Gap |
|---|---|---|---|---|---|
| Easy (easy_000) | 9.033 | 9.245 | 1.0% | 9.033 | 0.000 |
| Constrained (constraint_heavy_001) | 15.474 | 16.896 | 6.6% | 15.474 | 0.000 |
| Cost-conflicted (cost_conflict_009) | 7.459 | 10.082 | 17.0% | 7.459 | 0.000 |

**11.2 100-instance aggregate**

| Statistic | Value |
|---|---|
| Mean / median QAOA feasibility rate | 5.5% / 2.9% |
| Mean / median QAOA optimality gap | 0.468 / 0.000 (absolute) |
| QAOA feasible instances | 97/100 (97.0%) |
| Greedy feasible instances | 95/100 (95.0%) |
| Instances where QAOA found the optimum | 80/97 (82.5%) of QAOA-feasible |
| Instances where greedy found the optimum | 52/95 (54.7%) of greedy-feasible |
| QAOA failures / greedy failures | 3 / 5 (different instances) |
| Both QAOA and Greedy feasible | 92/100 |
| QAOA best < Greedy (mutually feasible) | 38/92 |
| Greedy best < QAOA (mutually feasible) | 7/92 |
| Same cost (mutually feasible) | 47/92 |
| Mean QAOA runtime | 0.20s per QAOA-feasible instance |

**11.3 Penalty sensitivity**

| λ (multiplier) | Instances with feasible QUBO minimum |
|---|---|
| 0.5 (unproven) | 2/7 test instances |
| 1.0 (safe) | 7/7 test instances |
| 2.0 | 7/7 test instances |
| 5.0 | 7/7 test instances |
| 10.0 | 7/7 test instances |

**11.4 Figures** (`results/figures/`): Status: NOT YET RUN (plotting not yet implemented).

**11.5 What quantum computing actually achieved**

On 100 benchmark instances (all exact-feasible):

- QAOA sampled at least one feasible state on 97/100 instances (3 zero-feasible failures)
- Greedy produced a feasible schedule on 95/100 instances (5 failures)
- Both methods feasible on 92/100 instances (direct comparison set)

- QAOA found the exact optimal assignment with non-zero probability on **80 of 97 QAOA-feasible instances (82.5%)**
- Greedy found the exact optimum on **52 of 95 greedy-feasible instances (54.7%)**
- For the 92 mutually feasible instances: QAOA best cost < Greedy on 38, Greedy best < QAOA on 7, tied on 47
- Mean QAOA feasibility rate was 5.5%, reflecting the difficulty of sampling feasible states with p=1 QAOA
- Median absolute gap was 0.0 (QAOA found optimal on majority of instances where it sampled feasible states)
- Constraint-heavy instances had higher QAOA feasibility rates (9.6% mean) but larger gaps when suboptimal
- Cost-conflict instances had lower QAOA feasibility rates (4.3% mean) but smaller gaps
- No quantum advantage is claimed; QAOA with p=1 and COBYLA is a limited configuration

---

## 12. IBM Quantum hardware (optional bonus)

**Status: NOT YET RUN.** The project is fully functional without hardware.

Only after simulator validation:

1. Pick a small validated instance; freeze simulator-trained parameters (no tuning through the hardware queue).
2. Authenticate with the API actually available in the installed environment, using environment variables or IBM's local credential store.
3. Transpile; record depth and two-qubit gate count.
4. Run on a real backend; decode and verify every sample.
5. Save only safe metadata: backend name, job ID, timestamp, shots, circuit resources, counts.
6. Compare ideal simulator vs hardware and report degradation honestly.

**IDEAL SIMULATOR** and **REAL IBM QUANTUM HARDWARE** are always labeled separately and never mixed. This section reports hardware results if, and only if, a real job completed.

---

## 13. User interface

**[OUR DESIGN]** Built only after the backend passes all tests. It imports the same functions as the notebook and tests; no optimization or constraint logic is duplicated.

**Sections:** Dashboard · Problem Setup · Dataset · Constraints · Classical Optimization · Quantum Optimization · Results · Comparison · Quantum Analysis · IBM Hardware · Experiment History · Technical Details.

**Rules**
- Status is shown as `QUEUED / RUNNING / COMPLETED / FAILED`, with real error messages.
- Anything not executed shows `NOT YET RUN`.
- Editing a problem marks earlier results as belonging to the previous instance; cached results are never silently reused.
- Metrics carry meaningful labels (for example "QAOA Feasibility Rate"), never opaque scores.
- Each solution shows ✓/✗ per constraint, bitstring, variable mapping, assignment cost, and QUBO energy as separate fields.
- A dataset explorer shows workers, shifts, cost matrix, eligibility, seed, constraints, and exact optimum.
- A "How Quantum Optimization Works" view walks: scheduling problem → binary variables → QUBO → Ising → QAOA circuit → measurement → decoding → verification → result.

The main results screen answers: what was the problem, what data, what classical found, what QAOA found, what the true optimum was, how close QAOA came, how often QAOA was feasible, and what happened on hardware.

---

## 14. Limitations

- Tiny instances (about 9 qubits by default); results do not extrapolate to realistic workforce sizes.
- Exhaustive enumeration does not scale.
- The penalty proof guarantees a feasible QUBO optimum, not feasible QAOA sampling.
- p=1 QAOA with a bounded optimizer budget is a limited configuration; results depend on seeds and initialization.
- Synthetic data only; no real-world validation and no cost-savings claims.
- Any hardware result reflects one backend, one time window, and limited shots.
- **No quantum advantage is claimed.**

---

## 15. Build process and status

Each phase follows **BUILD → RUN → TEST → VERIFY → FREEZE → EXTEND**. A phase is marked done only after its gate passes.

| Phase | Description | Status |
|---|---|---|
| 0 | Environment audit | ✓ PASS |
| 1 | Mathematical model | ✓ PASS |
| 2 | Exact + greedy | ✓ PASS |
| 3 | QUBO + Ising | ✓ PASS |
| 4 | QAOA p=1 | ✓ PASS |
| 5 | Exhaustive validation | ✓ PASS |
| 6 | Experiment engine | ✓ PASS |
| 7 | 100 synthetic instances | ✓ PASS |
| 8 | Result analysis | ✓ PASS |
| 9 | User interface | NOT YET RUN |
| 10 | IBM hardware (optional) | NOT YET RUN |
| 11 | README, brief, slides/demo | ✓ PASS (README updated) |
| 12 | Clean-environment verification | NOT YET RUN |

---

## 16. Repository layout

```
shiftproof/
├── README.md
├── requirements.txt          # minimal set chosen from the environment audit
├── .gitignore                # secrets, .env, credentials
├── business_brief.md         # one page, measured results only
├── slides.md                 # 3-slide content, measured results only
├── demo.ipynb                # calls src/, runs from a clean kernel
├── src/
│   ├── model.py              # instance, constraints, cost, encode/decode
│   ├── classical.py          # exact enumeration + greedy
│   ├── quantum.py            # QUBO, Ising, QAOA circuit, sampling
│   ├── validation.py         # independent checks, penalty verification
│   ├── dataset.py            # seeded 100-instance generator
│   ├── experiment.py         # runs, metrics, aggregation, logging
│   └── hardware.py           # optional IBM execution, no credentials
├── app/                      # UI, calls src/ only
├── tests/
│   ├── test_model.py
│   ├── test_classical.py
│   ├── test_quantum.py
│   └── test_validation.py
└── results/
    ├── environment.json
    ├── instances.json
    ├── metrics.csv
    └── figures/
```

Adjust only for a concrete engineering reason, and document it.

---

## 17. Installation and reproducibility

```bash
git clone <repo-url> && cd shiftproof
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # must pass before any experiment
python -m src.dataset                      # regenerate the 100 instances from seeds
python -m src.experiment                   # baselines + QAOA, writes results/
jupyter nbconvert --execute demo.ipynb     # clean-kernel check
```

Command names are finalized after the environment audit; update this section to match the code as built.

Every run logs instance ID, seed, QAOA parameters, optimizer, budget, shots, Qiskit and Python versions, backend, timestamp, and circuit resources. The same configuration should reproduce results within expected numerical and sampling variation.

---

## 18. Security

- No API keys, tokens, or credentials in source, notebooks, logs, results, or screenshots.
- Credentials come only from environment variables or IBM's local credential mechanism.
- `.gitignore` covers `.env`, credential files, and local secret stores.
- Scan the repo and its history for accidental secrets before publishing.

---

## 19. Deliverables map and final checklist

| Official requirement | Where |
|---|---|
| Qiskit implementation | `src/quantum.py`, `tests/test_quantum.py` |
| Industry relevance | Section 1, `business_brief.md` |
| Classical benchmark | `src/classical.py`, Sections 7 and 11 |
| Presentation and brief | `business_brief.md`, `slides.md` or demo |
| Code quality | Typed, tested, modular `src/`, deterministic seeds |
| Real IBM hardware bonus | Section 12, only if actually executed |
| Public repo, README, requirements | This repository |

**Final checklist**

- [ ] Official requirements re-checked against the organizer's brief
- [ ] Environment audit recorded
- [ ] QUBO, Ising, and bit ordering validated exhaustively
- [ ] Penalty rule verified exhaustively on the implemented QUBO (checks 1–6 in Section 4.1a); sensitivity sweep reported
- [ ] Exact and greedy validated
- [ ] QAOA p=1 validated; every sample constraint-checked
- [ ] 100 instances reproducible; category counts match Section 8
- [ ] Per-instance and aggregate results both shown; failures visible
- [ ] All results come from real execution
- [ ] UI uses backend only, with no duplicated logic
- [ ] Hardware results real, or section marked NOT YET RUN
- [ ] Tests pass; notebook runs from a clean kernel
- [ ] No credentials committed; secret scan done
- [ ] README, brief, and slides match actual results
- [ ] No quantum-advantage claim anywhere
- [ ] Limitations stated
