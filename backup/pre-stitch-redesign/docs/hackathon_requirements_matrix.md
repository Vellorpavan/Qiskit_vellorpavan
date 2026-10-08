# ShiftProof — Hackathon Requirements Traceability Matrix

> **Source of Rubric:** Working rubric supplied by project owner (Qiskit Fall Fest 2026 — Challenge I6: Shift & Resource Scheduler).  
> **Status:** Traceability and implementation evidence. No guaranteed points claimed.

---

## Traceability Summary Table

| Category | Available Points | Implementation Status | Primary Implementation Files | Test / Verification Evidence |
|---|---|---|---|---|
| **1. Qiskit Implementation** | 25 | **IMPLEMENTED** | `src/quantum.py`, `app/pages/05_qubo.py`–`08_measurement.py` | 23 unit & integration tests (`test_quantum.py`, `test_qaoa.py`) |
| **2. Industry Relevance** | 20 | **IMPLEMENTED** | `business_brief.md`, `app/pages/00_workspace.py`, `app/pages/14_data_management.py` | End-to-end dataset ingestion, validation pipeline, CSV schedule export |
| **3. Classical Benchmark** | 20 | **IMPLEMENTED** | `src/classical.py`, `results/metrics.csv`, `app/pages/04_classical.py`, `app/pages/10_benchmark.py` | 100-instance frozen benchmark, exact reference, greedy heuristic |
| **4. Presentation & Business Brief** | 20 | **IMPLEMENTED** | `business_brief.md`, `slides.md`, `demo_executed.ipynb`, `README.md` | Executed Jupyter walkthrough, pitch slide deck, executive brief |
| **5. Code Quality & Engineering** | 15 | **IMPLEMENTED** | `src/`, `tests/`, `app/`, `app/services/`, `app/database/` | 61/61 pytest passing, zero warnings/errors, safe parameterized SQL |
| **6. Real IBM Hardware Bonus** | +10 (Bonus) | **NOT YET RUN** | `app/pages/13_ibm_hardware.py` | Strict boundary page; zero fake data; real execution deferred |

---

## Detailed Requirement Breakdown

### 1. Qiskit Implementation (25 Points)
- **Requirement:** Formulation of problem as a quantum circuit, QUBO/Ising mapping, QAOA optimization, and measurement sampling on Qiskit.
- **Status:** **IMPLEMENTED**
- **Evidence:**
  - QUBO formulation with variable reduction pruning ineligible worker-shift pairs.
  - Safe penalty multiplier derivation: $A = B = C(x_{\text{ref}}) + 1.0$, mathematically guaranteeing feasible global QUBO minimum.
  - Exact Pauli-$Z$ Ising Hamiltonian mapping: $x_i = (1 - Z_i)/2$ yielding $h_i$ linear fields and $J_{ij}$ quadratic couplings.
  - Parametric QAOA $p=1$ circuit in Qiskit:
    - Superposition state $|+\rangle^n$
    - Cost layer: $RZ(2\gamma h_i)$ and $RZZ(2\gamma J_{ij})$ compiled via CNOT gates
    - Mixer layer: $RX(2\beta)$ on all qubits
  - Classical optimizer: COBYLA with maxiter=100.
  - Simulator backend: Qiskit `AerSimulator` (noiseless ideal simulation, 1024 shots, seed=42).
- **Files:**
  - `src/quantum.py`: Mathematical construction of QUBO, Ising, QAOA circuit, and simulation.
  - `app/pages/05_qubo.py`: Interactive QUBO formulation and coefficient heatmap.
  - `app/pages/06_ising.py`: Pauli-$Z$ conversion explorer and coupling matrix.
  - `app/pages/07_qaoa.py`: Transpiled depth, gate count, and $p=1$ circuit diagram.
  - `app/pages/08_measurement.py`: Sampled measurement distribution histogram.
- **Verification:**
  - Tests: `tests/test_quantum.py` (12 tests), `tests/test_qaoa.py` (11 tests). All passing.
- **Remaining Gap:** Real IBM Quantum physical hardware run (evaluated under separate bonus).

---

### 2. Industry Relevance & Application (20 Points)
- **Requirement:** Real-world applicability to workforce and shift scheduling; handling operational constraints; dynamic dataset support.
- **Status:** **IMPLEMENTED**
- **Evidence:**
  - Addresses real enterprise challenge: scheduling workers across shifts under capacity, coverage, and skill requirements.
  - Ingestion pipeline supporting real tabular files (CSV, XLSX) and structured JSON.
  - Automated column profiling and interactive schema mapping.
  - 7-point mathematical validation auditing shift coverage feasibility, worker capacity, non-negative wage costs, and duplicate entries.
  - Interactive scheduling workspace allowing users to run solvers and export verified final schedules to CSV.
  - Business brief with market analysis, cost reduction metrics, and deployment ROI model.
- **Files:**
  - `business_brief.md`: Executive ROI model, market overview, implementation roadmap.
  - `app/pages/00_workspace.py`: Operational scheduling hub.
  - `app/pages/14_data_management.py`: Dataset ingestion, column profiling, and schema mapping.
  - `app/services/dataset_service.py` & `app/services/scheduling_service.py`: Dynamic database-to-model adapter.
- **Verification:**
  - Validated on built-in synthetic hospital ward dataset (`synthetic_demo_ward_3x3`).
  - Tested against malformed, uncovered, and negative-cost datasets with clean blocking.
- **Remaining Gap:** None.

---

### 3. Classical Benchmark & Baselines (20 Points)
- **Requirement:** Fair, rigorous classical comparison against established algorithms; transparent reporting without misleading quantum claims.
- **Status:** **IMPLEMENTED**
- **Evidence:**
  - Exact exhaustive enumeration reference (`solve_exact`): guarantees ground-truth optimal assignment and cost.
  - Priority-constrained greedy baseline (`solve_greedy`): assigns most constrained shifts first with lowest-cost available worker.
  - 100-instance reproducible benchmark across 3 difficulty tiers:
    - 40 Easy
    - 30 Constraint-Heavy
    - 30 Cost-Conflict
  - Strict preservation of explicit denominators:
    - QAOA Feasible: **97 / 100**
    - Greedy Feasible: **95 / 100**
    - Exact Feasible: **100 / 100**
    - QAOA Exact Optimum: **80 / 97** (82.5% of QAOA-feasible)
    - Greedy Exact Optimum: **52 / 95** (54.7% of Greedy-feasible)
    - Mutually Feasible: **92 / 100** (QAOA lower cost: **38 / 92**, Greedy lower cost: **7 / 92**, Tie: **47 / 92**)
- **Files:**
  - `src/classical.py`: Exact and greedy solver implementations.
  - `results/metrics.csv`: Frozen benchmark results across all 100 instances.
  - `app/pages/04_classical.py`: Classical baselines explorer.
  - `app/pages/10_benchmark.py`: 7 comparative Plotly visualizations.
- **Verification:**
  - Tests: `tests/test_classical.py` (11 tests), `tests/test_experiment.py` (12 tests).
  - Frozen metric verification script: 100% match across all 9 denominators.
- **Remaining Gap:** None.

---

### 4. Presentation & Business Brief (20 Points)
- **Requirement:** Clear narrative, reproducible artifacts, business brief, and presentation slides.
- **Status:** **IMPLEMENTED**
- **Evidence:**
  - Executive Business Brief: Problem statement, executive summary, ROI analysis, market sizing ($3.8B shift planning market), and risk disclosures.
  - Slide Deck: 12-slide markdown presentation covering problem, mathematics, benchmark findings, transparent limitations, and future roadmap.
  - Executed Jupyter Notebook: Fully run end-to-end demo notebook (`demo_executed.ipynb`) reproducing all benchmark calculations.
  - Main Application Portal: Clean interactive Streamlit application.
- **Files:**
  - `business_brief.md`
  - `slides.md`
  - `demo.ipynb` & `demo_executed.ipynb`
  - `README.md`
- **Verification:** Notebook executed without errors; markdown documents formatted to Diataxis standards.
- **Remaining Gap:** None.

---

### 5. Code Quality, Engineering & Scientific Honesty (15 Points)
- **Requirement:** High code quality, modularity, test suite, security guardrails, transparent limitations.
- **Status:** **IMPLEMENTED**
- **Evidence:**
  - Comprehensive automated test suite: 61 unit and integration tests passing in <3 seconds.
  - Clean architectural separation: core mathematics (`src/`), tests (`tests/`), frozen benchmark (`results/`), application workspace (`app/`), database (`data/`).
  - Independent constraint verification layer auditing all candidate solutions directly against problem constraints.
  - Transparent limitations section: explicitly discloses toy problem scale ($\le 9$ variables), 5.5% mean sampling feasibility rate, noiseless simulation caveats, and zero claims of quantum advantage or speedup.
  - Security guardrails: 5 MB file size limit, extension whitelisting, parameterized SQL, no `eval` or `pickle`.
- **Files:**
  - `tests/*.py` (6 test suites)
  - `app/services/validation_service.py`
  - `app/pages/09_verification.py`
  - `app/pages/12_limitations.py`
- **Verification:**
  - `python3 -m pytest`: 61 passed, 0 failures.
  - `python3 -m py_compile`: 0 errors across all files.
- **Remaining Gap:** None.

---

### 6. Real IBM Quantum Hardware Bonus (+10 Points Bonus)
- **Requirement:** Executing circuits on physical IBM Quantum hardware (QPU) using Qiskit Runtime.
- **Status:** **NOT YET RUN** (Integration boundary established)
- **Evidence:**
  - Hardware boundary page (`app/pages/13_ibm_hardware.py`) clearly set to `STATUS: NOT YET RUN`.
  - Zero fake hardware data, zero mock job IDs, zero simulated noise labeled as hardware.
  - Execution protocol and JSON ingestion schema prepared for physical submission when authenticated credentials become available.
- **Files:**
  - `app/pages/13_ibm_hardware.py`
- **Verification:** Verified that boundary is strictly maintained and no fake results are shown.
- **Remaining Gap:** Submission to physical IBM Quantum hardware (planned for subsequent phase).
