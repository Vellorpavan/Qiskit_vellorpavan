# ShiftProof — Constraint-Verified Quantum Scheduling

> A reproducible Qiskit optimization platform evaluating how reliably the Quantum Approximate Optimization Algorithm (QAOA) discovers valid, cost-minimized workforce shift schedules compared with exact and heuristic classical references on both local simulators and real physical IBM Quantum hardware.

**Challenge Context:** Qiskit Fall Fest — **Challenge I6: Shift & Resource Scheduler**  
**Core Frameworks:** Qiskit 2.3.1 · Qiskit Aer 0.17.2 · Qiskit IBM Runtime 0.46.1 · Streamlit 1.55.0 · Python 3.10+  
**Repository:** [https://github.com/Vellorpavan/Qiskit_vellorpavan](https://github.com/Vellorpavan/Qiskit_vellorpavan)

---

## 1. Overview

**ShiftProof** is a quantum optimization and decision engine for workforce scheduling. It translates real-world staffing rules—worker availability, qualification constraints, capacity limits, and assignment costs—into a Quadratic Unconstrained Binary Optimization (QUBO) problem, maps it to an Ising spin Hamiltonian, and executes QAOA circuits.

ShiftProof is an **optimization and decision engine**, not a predictive machine-learning system. It evaluates whether quantum optimization can solve combinatorial workforce constraints, audits feasibility independently, and compares results against classical algorithms:

| Scientific Principle | ShiftProof Standard |
| :--- | :--- |
| **No Quantum Advantage Claims** | ShiftProof does not claim quantum advantage, asymptotic speedup, or commercial superiority over classical operations research. |
| **Zero Fabricated Results** | Every metric, probability, and measurement distribution originates from actual execution (Qiskit AerSimulator or IBM Quantum hardware). Unrun states display `NOT RUN`. |
| **Strict Dataset Isolation** | Execution results are cryptographically bound to dataset and instance SHA-256 fingerprints. Results never bleed across datasets. |
| **Dual Backend Verification** | QAOA is executed locally on `AerSimulator` (noiseless classical simulation) and verified on real physical quantum hardware (`ibm_fez`, 156-qubit QPU). |

---

## 2. Problem

Workforce scheduling requires assigning personnel to operating shifts while respecting strict operational constraints:
- **Workers ($w \in W$):** Staff members with specific skills and maximum shift capacity.
- **Shifts ($s \in S$):** Time blocks requiring coverage and minimum skill qualifications.
- **Eligibility & Availability:** Ineligible or unavailable worker-shift pairings must never be assigned.
- **Cost Matrix ($c[w, s]$):** Monetary or preference costs associated with assigning worker $w$ to shift $s$.

### Constraints Enforced
1. **Shift Coverage:** Every required shift must receive exactly one assigned worker:
   $$\sum_{w \in W} x_{w, s} = 1 \quad \forall s \in S$$
2. **Worker Capacity:** No worker may be assigned to more than one shift:
   $$\sum_{s \in S} x_{w, s} \le 1 \quad \forall w \in W$$
3. **Skill & Availability:** Binary variables $x_{w, s} = 0$ for all ineligible pairings.

---

## 3. What ShiftProof Does

1. **Universal Dataset Ingestion:** Ingests external scheduling data in CSV, Excel, or JSON format.
2. **Automated Schema Understanding:** Detects worker IDs, shift names, skills, and assignment costs automatically without hardcoded templates.
3. **Canonical Normalization:** Normalizes raw data into an internal scheduling `Instance` and computes a deterministic SHA-256 fingerprint.
4. **Classical Solvers:** Solves the problem with an exact branch-and-bound algorithm (for instances $\le 16$ variables) and a priority greedy heuristic.
5. **QUBO & Ising Mathematical Pipeline:** Maps shift constraints and cost objectives into an exact quadratic unconstrained objective with rigorous penalty multipliers.
6. **Logical QAOA Circuit Generation:** Constructs parameterized Qiskit `QuantumCircuit` instances with problem-specific cost unitary $U(C, \gamma)$ and transverse mixer $U(B, \beta)$.
7. **Simulation & Hardware Execution:**
   - **Local Simulation:** Samples measurement distributions using Qiskit `AerSimulator` (1,024 shots, COBYLA optimization).
   - **Real IBM QPU:** Connects via Qiskit IBM Runtime, transpiles to native basis gates (`cz`, `sx`, `rz`), and executes on IBM Quantum hardware (`ibm_fez`).
8. **Independent Verification:** Decodes bitstrings, verifies all operational constraints independently, calculates optimality gaps against classical baselines, and displays final schedule rosters.

---

## 4. Architecture

```
                    Workforce Dataset (CSV / XLSX / JSON)
                                      │
                                      ▼
                        Ingestion & Schema Classifier
                         (app/services/ingestion/)
                                      │
                                      ▼
                          Canonical Instance Model
                               (src/model.py)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          Classical Optimization                 QUBO Formulation
          ├ Exact Branch-and-Bound             (Penalty Multiplier A, B)
          └ Priority Greedy Heuristic                   │
            (src/classical.py)                          ▼
                    │                            Ising Hamiltonian
                    │                          (H = Σ h_i Z_i + Σ J_ij Z_i Z_j)
                    │                                   │
                    │                                   ▼
                    │                         Logical QAOA Circuit
                    │                         (Ansatz Depth p=1, COBYLA)
                    │                             (src/quantum.py)
                    │                                   │
                    │                    ┌──────────────┴──────────────┐
                    │                    ▼                             ▼
                    │            AerSimulator                 IBM Quantum Hardware
                    │          (Local Simulation)            (ibm_fez 156-Qubit QPU)
                    │                    │                             │
                    └────────────────────┼─────────────────────────────┘
                                         ▼
                             Independent Verification
                          (app/services/execution_proof_service.py)
                                         │
                                         ▼
                     Decision Dashboard & Schedule Roster
                                 (Streamlit UI)
```

---

## 5. Repository Structure

```
qiskit_project/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Production Python dependencies
├── pytest.ini                         # Test runner configuration
├── .gitignore                         # Strict exclusion for secrets, caches, and OS files
├── .env.example                       # Safe environment variable template
├── slides.md                          # Hackathon pitch slide deck
├── business_brief.md                  # Executive business summary
│
├── app/                               # Interactive Streamlit Web Application
│   ├── main.py                        # Multi-page router and navigation hub
│   ├── pages/
│   │   ├── 00_workspace.py            # Workspace: solver execution cards & status
│   │   ├── 01_data.py                 # Data ingestion, schema review & activation
│   │   ├── 02_results.py              # Results: classical schedules & audits
│   │   └── 03_quantum.py              # Quantum: QAOA Aer simulation & IBM QPU
│   ├── services/                      # Application backend services
│   │   ├── dataset_service.py         # Dataset management & activation
│   │   ├── execution_state.py         # Lifecycle state & duplicate-run prevention
│   │   ├── execution_proof_service.py # Cryptographic provenance & verification
│   │   ├── experiment_service.py      # Solver execution service
│   │   ├── ibm_quantum_service.py     # IBM Quantum Platform connection & discovery
│   │   ├── scheduling_service.py      # Instance reconstruction & fingerprinting
│   │   └── ingestion/                 # Multi-format dataset ingestion pipeline
│   ├── database/                      # SQLite storage & seed data (data/shiftproof.db)
│   └── styles/theme.css               # Streamlit custom styling
│
├── src/                               # Core Mathematical & Scientific Engine
│   ├── model.py                       # Scheduling model, constraints, bitstring decoding
│   ├── classical.py                   # Exact solver & priority greedy heuristic
│   ├── quantum.py                     # QUBO, Ising conversion, QAOA ansatz circuits
│   ├── validation.py                  # Energy equivalence checks & penalty audits
│   ├── dataset.py                     # Benchmark synthesis
│   └── experiment.py                  # Benchmark suite executor
│
├── research/                          # Research & Verification Evidence
│   ├── README.md                      # Notebook methodology & evidence guide
│   └── notebooks/
│       ├── 01_qaoa_aer_execution.ipynb          # QAOA mathematical equivalence
│       ├── 02_real_dataset_to_qaoa.ipynb        # Ingestion to QAOA pipeline
│       ├── 03_classical_vs_qaoa.ipynb           # Empirical classical vs QAOA benchmark
│       ├── 04_ibm_quantum_execution.ipynb       # Real IBM Quantum execution dashboard
│       └── 05_real_ibm_hardware_execution.ipynb # Real IBM QPU execution provenance
│
├── results/                           # Frozen Reproducible Benchmark Artifacts
│   ├── metrics.csv                    # 100-instance benchmark run metrics
│   ├── instances.json                 # 100 benchmark problem instances
│   └── environment.json               # Computational environment specification
│
├── data/                              # Sample Data & Local Storage
│   └── sample/                        # Reference workforce CSV datasets
│
└── tests/                             # Automated Test Suite (133 Tests)
    ├── test_model.py                  # Constraint verification tests
    ├── test_classical.py              # Classical solver accuracy tests
    ├── test_quantum.py                # QUBO, Ising, and circuit construction tests
    ├── test_qaoa.py                   # QAOA simulation tests
    ├── test_validation.py             # Validation engine tests
    ├── test_dataset_isolation.py      # Dataset boundary & fingerprint tests
    ├── test_dataset_lifecycle.py      # Ingestion & activation lifecycle tests
    ├── test_execution_lifecycle.py    # Execution state lifecycle tests
    ├── test_execution_proof.py        # Cryptographic proof chain tests
    ├── test_experiment.py             # Experiment persistence tests
    ├── test_ibm_connection.py         # IBM connection & discovery tests
    ├── test_universal_ingestion.py    # Multi-format ingestion parser tests
    └── test_ux_navigation.py          # Solver navigation & duplicate-run tests
```

---

## 6. Quick Start

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Vellorpavan/Qiskit_vellorpavan.git
cd Qiskit_vellorpavan

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the automated test suite
pytest -q

# 5. Launch the Streamlit application
streamlit run app/main.py
```

The web application opens at `http://localhost:8501`.

---

## 7. Environment Configuration

ShiftProof runs entirely out of the box for classical optimization and local QAOA simulation without any external credentials.

To enable **IBM Quantum Hardware** integration:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Insert your IBM Quantum API token from [quantum.ibm.com](https://quantum.ibm.com/):
   ```ini
   IBM_QUANTUM_TOKEN=your_real_ibm_quantum_token_here
   IBM_QUANTUM_INSTANCE=   # Optional: set only if using IBM Cloud CRN
   IBM_QUANTUM_CHANNEL=ibm_quantum
   ```

> **Security Guarantee:** `.env` is listed in `.gitignore` and is never committed. ShiftProof strictly accesses credentials via `os.environ` and never prints or persists tokens.

---

## 8. Using the Web UI

### Complete User Workflow

1. **Navigate to Data (`/data`):**
   - Upload any CSV or Excel scheduling file (e.g., `data/sample/workforce_schedule_3x3.csv`).
   - The ingestion engine classifies the schema, detects workers, shifts, and costs, and generates a SHA-256 fingerprint.
   - Click **Activate Dataset**.

2. **Open Workspace (`/workspace`):**
   - The workspace confirms:
     ```
     Dataset: <Selected Dataset>
     Status: DATASET READY
     Classical: NOT RUN
     QAOA: NOT RUN
     IBM Hardware: NOT RUN
     ```
   - Three distinct solver action cards are displayed:
     - **Classical Optimization:** Greedy heuristic and branch-and-bound solver.
     - **QAOA Simulation:** Qiskit AerSimulator (1,024 shots, COBYLA).
     - **IBM Quantum Hardware:** Real physical QPU execution interface.

3. **Execute Solvers:**
   - Click **▶ Run Classical Optimization**: Solves the instance, saves results, and automatically routes to `/results`.
   - Click **⚡ Run QAOA Simulation**: Builds the QUBO/Ising circuit, samples 1,024 shots on `AerSimulator`, and automatically routes to `/quantum`.

4. **Review Results:**
   - On `/results`: Inspect the final schedule roster, cost breakdown, and constraint verification audit.
   - On `/quantum`: Inspect the logical circuit width/depth, empirical measurement distribution, feasible sample count, and optimality comparison.

### Core UX State Guarantees
- **Activation is Not Execution:** Selecting or uploading a dataset validates the schema but **never** triggers solvers automatically.
- **No Duplicate Execution:** Navigating between `/workspace`, `/results`, and `/quantum` or refreshing the page displays existing completed results without submitting duplicate jobs.
- **Explicit Rerun:** Rerunning requires clicking explicit buttons: `[↻ Run Classical Again]` or `[↻ Run QAOA Again]`.
- **Solver-Specific Navigation:**
  - `Run Classical` / `View Classical Result` $\to$ `/results`
  - `Run QAOA` / `View Quantum Result` $\to$ `/quantum`
  - `View IBM Hardware Result` $\to$ `/quantum`
- **Error Retention:** Failed executions display error tracebacks on the current page and never navigate away.

---

## 9. Classical Optimization

ShiftProof implements two classical solvers in [src/classical.py](src/classical.py):

1. **Exact Branch-and-Bound Solver (`solve_exact`):**
   - Explores binary assignment space with pruning.
   - Guaranteed global minimum cost solution among all feasible schedules.
   - Safety scale boundary: evaluated for instances with $\le 16$ variables.
2. **Priority Greedy Heuristic (`solve_greedy`):**
   - Sorts candidate assignments by cost and feasibility rules.
   - Extremely fast ($< 1$ ms) reference for real-time baseline comparison.

---

## 10. QAOA Simulation

Implemented in [src/quantum.py](src/quantum.py):

### Formulation
- **Binary Decision Variable:** $x[w, s] \in \{0, 1\}$ indicating assignment of worker $w$ to shift $s$.
- **QUBO Objective:**
  $$\min_{x} \quad \sum_{(w, s)} c[w, s] x_{w, s} + A \sum_{s} \left(\sum_{w} x_{w, s} - 1\right)^2 + B \sum_{w} \max\left(0, \sum_{s} x_{w, s} - 1\right)$$
- **Penalty Multipliers:** Set rigorously to $A = 10 \cdot \max(c) + 1$ and $B = A$, mathematically guaranteeing that every infeasible assignment incurs higher energy than any feasible schedule.
- **Ising Transformation:** Substituting $x_i = \frac{1 - Z_i}{2}$ yields the spin Hamiltonian $H = \sum_i h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{offset}$.

### Quantum Circuit & Execution
- **Ansatz:** Depth $p=1$ QAOA circuit initialized in equal superposition $|+\rangle^{\otimes n}$.
- **Optimizer:** Scipy COBYLA classical optimizer optimizing variational parameters $(\gamma, \beta)$.
- **Backend:** Qiskit `AerSimulator` sampling 1,024 shots.
- **Demonstrator Scale Gate:** The web application enforces a scale boundary of $\le 9$ variables for interactive simulation.

---

## 11. IBM Quantum Hardware

ShiftProof connects to physical quantum hardware using Qiskit Runtime (`qiskit-ibm-runtime`):

```
Logical QAOA Circuit (7 qubits)
           │
           ▼
IBM QPU Transpiler (Target: ibm_fez)
  ├ Layout & routing on 156 physical qubits
  ├ Native basis synthesis: {cz, sx, rz, id}
  └ Optimization level: 1
           │
           ▼
Physical QPU Execution (1,024 shots)
           │
           ▼
Hardware Measurement Distribution (117 observed bitstrings)
           │
           ▼
Classical Post-Processing & Constraint Decoding
```

### Verified Phase 2 Hardware Benchmark Evidence

| Metric | Verified Real QPU Value |
| :--- | :--- |
| **Backend** | `ibm_fez` (156 physical qubits, Heron r2) |
| **IBM Job ID** | `db3jsqslf4us73c1f7j0` |
| **Execution Shots** | 1,024 |
| **Logical Qubits** | 7 logical variables ($3 \times 3$ instance, 2 pruned) |
| **Physical Qubits Allocated** | 7 physical qubits on `ibm_fez` |
| **Logical Circuit Depth** | 25 (22 two-qubit CX gates) |
| **Transpiled Circuit Depth** | 64 (29 native two-qubit CZ gates) |
| **Observed Basis States** | 117 states across 1,024 shots |
| **Feasible Quantum Shots** | 5 shots ($0.49\%$ empirical feasibility rate) |
| **Hardware Best Feasible Cost** | **$135.00** |
| **Exact Classical Minimum Cost** | **$135.00** |
| **Hardware Optimality Gap** | **$0.00** (0.00%) |

> **Hardware Disclosure:** Executed on an unmitigated physical quantum processor. The algorithm observed the optimal classical solution among physical measurements. This proves genuine hardware execution of the identical logical pipeline without claiming quantum speedup.

---

## 12. Research Notebooks

The research notebooks in [research/notebooks/](research/notebooks/) are complete, executed, and rendered directly on GitHub:

| Notebook | Purpose | Key Evidence | Backend |
| :--- | :--- | :--- | :--- |
| [01_qaoa_aer_execution.ipynb](research/notebooks/01_qaoa_aer_execution.ipynb) | Reproducible QAOA Execution | Mathematical energy equivalence, penalty audits, statevector & sampling validation. | `AerSimulator` |
| [02_real_dataset_to_qaoa.ipynb](research/notebooks/02_real_dataset_to_qaoa.ipynb) | External Dataset to QAOA | Automated schema understanding, canonical instance normalization, SHA-256 fingerprinting. | `AerSimulator` |
| [03_classical_vs_qaoa.ipynb](research/notebooks/03_classical_vs_qaoa.ipynb) | Classical vs. QAOA Benchmark | Exact solver vs. Greedy heuristic vs. QAOA comparison; $0.00 observed gap. | `AerSimulator` |
| [04_ibm_quantum_execution.ipynb](research/notebooks/04_ibm_quantum_execution.ipynb) | Real IBM Quantum Hardware Bridge | IBM platform authentication, transpilation to native CZ gates, decision dashboard of real hardware execution. | Physical QPU (`ibm_fez`) |
| [05_real_ibm_hardware_execution.ipynb](research/notebooks/05_real_ibm_hardware_execution.ipynb) | Real IBM QPU Provenance | Complete end-to-end execution notebook capturing QPU job provenance and measurement extraction. | Physical QPU (`ibm_fez`) |

---

## 13. Reproducibility

ShiftProof ensures strict scientific reproducibility:
1. **Deterministic Random Seeds:** Global seed `42` used across QAOA parameter initialization and sampling.
2. **Cryptographic Fingerprinting:** Every dataset is hashed via SHA-256 (`active_fingerprint`); every normalized problem instance carries an immutable `instance_id`.
3. **Execution UUID Tracking:** Every solver invocation receives a unique experiment ID (`exp_...`) and solution ID (`sol_...`).
4. **Frozen Benchmark Artifacts:**
   The 100-instance benchmark in [results/](results/) is permanently locked. Verified SHA-256 hashes:
   - `results/metrics.csv`: `f1aa4ffc4f67f43cde21fb2d4cbffd369948683cbd04b19f6afebd300c03a52d`
   - `results/instances.json`: `5d0831915b29726fa089ab5963dac878a1b564e8e27869ed183705ba771a75d0`
   - `results/environment.json`: `b1d49f4108ae68d348413d0eb340f69a1366d914230896a75de01850ce8a9582`

---

## 14. Verification and Tests

The test suite contains **133 automated unit and integration tests** passing with zero failures:

```bash
$ pytest -q
........................................................................ [ 54%]
.............................................................            [100%]
133 passed in 5.82s
```

### Test Coverage Breakdown
- **Mathematical Formulation ([tests/test_model.py](tests/test_model.py), [tests/test_quantum.py](tests/test_quantum.py)):** Verifies QUBO matrix symmetry, Ising eigenvalues, energy equivalence, and bitstring decoding.
- **Classical Optimization ([tests/test_classical.py](tests/test_classical.py)):** Verifies optimality of exact branch-and-bound and feasibility of greedy heuristics.
- **QAOA Pipeline ([tests/test_qaoa.py](tests/test_qaoa.py)):** Tests variational ansatz depth, parameter optimization, and statevector sampling.
- **Dataset Ingestion ([tests/test_universal_ingestion.py](tests/test_universal_ingestion.py)):** Verifies parsing of heterogeneous CSV, XLSX, and JSON tables with ambiguous headers.
- **Dataset Isolation ([tests/test_dataset_isolation.py](tests/test_dataset_isolation.py)):** Confirms results never cross between datasets.
- **Execution UX & Navigation ([tests/test_ux_navigation.py](tests/test_ux_navigation.py)):** Formally tests requirements TEST A through TEST K (dataset activation does not run solvers, no duplicate runs on page open, solver-specific navigation, and failure handling).
- **IBM Quantum Integration ([tests/test_ibm_connection.py](tests/test_ibm_connection.py)):** Validates IBM token parsing, backend discovery, and compatibility checking.

---

## 15. Scientific Honesty / Limitations

1. **Demonstrator Scale Limit:**
   Interactive QAOA simulation in the web application is gated to problem instances with $\le 9$ variables (qubits). Larger enterprise scheduling instances (e.g., 50+ workers) exceed classical simulation capacity and NISQ hardware co-design limits.
2. **Classical Exact Bound:**
   The exact branch-and-bound solver is bounded to $\le 16$ variables ($2^{16} = 65,536$ states). Larger instances rely on greedy heuristics or scalable classical solvers.
3. **Probabilistic Sampling:**
   QAOA is a heuristic sampling algorithm. Individual shots may yield infeasible states; classical post-processing filters infeasible samples and selects the minimum-cost valid assignment.
4. **Physical Noise on QPU:**
   On real quantum hardware (`ibm_fez`), two-qubit gate errors and decoherence reduce the observed feasibility rate ($0.49\%$ observed across 1,024 shots). Error mitigation and higher circuit depths ($p > 1$) will be required for fault-tolerant operation.
5. **No Quantum Advantage Claim:**
   ShiftProof demonstrates feasibility, provenance, and experimental reproducibility. It does not claim speedup over classical algorithms.

---

## 16. Results

### Empirical Benchmark Summary (100 Instances)

| Solver | Mean Runtime | Feasibility Rate | Optimality Gap |
| :--- | :--- | :--- | :--- |
| **Exact Solver** | 0.82 ms | 100.0% | 0.00% (Baseline) |
| **Greedy Heuristic** | 0.14 ms | 100.0% | +12.4% vs exact |
| **QAOA (AerSimulator)** | 1.84 s | 62.3% of shots | **0.00%** (Optimum observed in samples) |
| **IBM QPU (`ibm_fez`)** | Remote hardware job | 0.49% of shots | **0.00%** (Optimum observed in samples) |

---

## 17. Project Status

- [x] **Complete & Verified:**
  - Automated workforce dataset ingestion and schema classifier.
  - Mathematical QUBO and Ising formulation with proven penalty multipliers.
  - Classical Exact and Greedy solvers.
  - QAOA circuit generator and local AerSimulator execution.
  - Streamlit multi-page web application with duplicate-run prevention and solver navigation.
  - 133 automated unit and integration tests passing.
  - Real IBM Quantum QPU execution on `ibm_fez` (Job `db3jsqslf4us73c1f7j0`).
  - 5 research notebooks verified and executable.
  - Frozen benchmark reproducibility artifacts locked with SHA-256 hashes.
- [ ] **Experimental / Future Work:**
  - Higher-depth QAOA circuits ($p \ge 2$) with hardware-efficient pulse scheduling.
  - Advanced error mitigation (Zero-Noise Extrapolation, Readout Error Mitigation).
  - Warm-started QAOA initialized from classical greedy solutions.

---

## 18. Hackathon / Judge Demonstration Walkthrough

For evaluators and judges reviewing ShiftProof:

1. **Inspect Problem & Dataset:**
   - Launch Streamlit: `streamlit run app/main.py`.
   - Open `/data` and upload `data/sample/workforce_schedule_3x3.csv`.
   - Observe automatic schema detection mapping workers, shifts, and costs.
   - Click **Activate Dataset**.
2. **Examine Workspace:**
   - Open `/workspace`. Notice status is `DATASET READY` with solvers `NOT RUN`.
3. **Execute Classical Optimization:**
   - Click **▶ Run Classical Optimization**.
   - The application executes the solver and automatically routes to `/results`.
   - Review the verified schedule roster, cost breakdown, and zero constraint violations.
4. **Execute Quantum QAOA Simulation:**
   - Return to `/workspace` and click **⚡ Run QAOA Simulation**.
   - The application executes the QAOA pipeline on `AerSimulator` and automatically routes to `/quantum`.
   - Inspect logical circuit depth (25), measurement histogram, feasibility rate, and comparison with the classical optimum.
5. **Inspect IBM Quantum Hardware Evidence:**
   - On `/quantum`, view the **IBM Quantum Platform** section.
   - Review the connection to `ibm_fez` and actual execution provenance from IBM Job `db3jsqslf4us73c1f7j0`.
6. **Inspect Research Notebooks & Tests:**
   - Open [research/notebooks/04_ibm_quantum_execution.ipynb](research/notebooks/04_ibm_quantum_execution.ipynb).
   - Run the automated test suite in terminal: `pytest -q` (133 passing).
   - Verify frozen benchmark hashes: `shasum -a 256 results/metrics.csv`.

---

## 19. License & Acknowledgements

Developed for **Qiskit Fall Fest 2026** (Challenge I6: Shift & Resource Scheduler).  
Built using Qiskit, Qiskit Aer, and Qiskit IBM Runtime from IBM Quantum.
