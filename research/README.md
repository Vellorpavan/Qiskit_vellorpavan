# ShiftProof — Research Notebooks & Empirical Evidence

This directory contains the computational proof and research notebooks for **ShiftProof**. They serve as verifiable experimental records evaluating the Quantum Approximate Optimization Algorithm (QAOA) on workforce scheduling instances against exact classical references and on real physical IBM Quantum hardware.

---

## 1. Purpose & Relationship to Core Engine

Rather than maintaining separate, disconnected notebook code, every notebook in this suite imports directly from the project's tested production packages:
- `src.model`: Workforce scheduling formulation, constraint verification, and cost evaluation.
- `src.quantum`: Quadratic Unconstrained Binary Optimization (QUBO) formulation, Ising Hamiltonian conversion, and Qiskit QAOA ansatz circuit generation (`build_qaoa_circuit`).
- `src.classical`: Exact branch-and-bound solver (`solve_exact`) and priority greedy heuristic (`solve_greedy`).
- `src.validation`: QUBO/Ising statevector energy equivalence checks and constraint penalty audits.
- `app.services`: Dataset schema normalization, SHA-256 fingerprinting, and IBM Quantum Runtime connection service.

All mathematical definitions, variable encodings, and penalty values ($A = 10 \cdot \max(c) + 1$, $B = A$) in these notebooks are identical to those executed by the interactive web application.

---

## 2. Notebook Inventory & Execution Order

Execute or review the notebooks in sequential order:

| Notebook | Title | Backend | Primary Focus & Key Evidence |
| :--- | :--- | :--- | :--- |
| [01_qaoa_aer_execution.ipynb](notebooks/01_qaoa_aer_execution.ipynb) | Reproducible QAOA Execution | `AerSimulator` (Local Simulation) | QUBO/Ising mathematical equivalence verification, QAOA ansatz circuit construction ($p=1$, COBYLA), and noiseless sampling. |
| [02_real_dataset_to_qaoa.ipynb](notebooks/02_real_dataset_to_qaoa.ipynb) | External Dataset to QAOA Pipeline | `AerSimulator` (Local Simulation) | End-to-end ingestion of an external workforce CSV (`data/sample/workforce_schedule_3x3.csv`), schema detection, SHA-256 fingerprinting, and QAOA execution. |
| [03_classical_vs_qaoa.ipynb](notebooks/03_classical_vs_qaoa.ipynb) | Classical Solvers vs. QAOA Comparison | `AerSimulator` (Local Simulation) | Side-by-side empirical benchmarking: Exact solver vs. Greedy heuristic vs. QAOA ($p=1$, 1,024 shots). Evaluates feasibility rate, optimality gap ($0.00 observed gap), and runtimes. |
| [04_ibm_quantum_execution.ipynb](notebooks/04_ibm_quantum_execution.ipynb) | Real IBM Quantum Hardware Execution Bridge | Real Physical QPU (`ibm_fez`) | Authentication to IBM Quantum Platform, QPU discovery, transpilation to native basis gates (`cz`, `sx`, `rz`), and visual optimization decision dashboard of real hardware execution. |
| [05_real_ibm_hardware_execution.ipynb](notebooks/05_real_ibm_hardware_execution.ipynb) | Real IBM Quantum Hardware Execution Bridge | Real Physical QPU (`ibm_fez`) | Executed provenance notebook capturing IBM job execution, logical circuit vs. transpiled hardware circuit metrics, and measurement extraction. |

---

## 3. Verified IBM Quantum Hardware Evidence

Notebooks `04` and `05` capture and analyze an actual physical quantum job executed on the IBM Quantum Platform:

- **Target Backend**: `ibm_fez` (Heron r2 architecture, 156 physical qubits, operational)
- **IBM Job ID**: `db3jsqslf4us73c1f7j0`
- **Execution Shots**: 1,024
- **Logical Problem Size**: 7 logical qubits ($n_{\text{workers}} = 3$, $n_{\text{shifts}} = 3$, with 2 unavailable assignments pruned)
- **Physical Qubits Allocated**: 7 physical qubits on `ibm_fez`
- **Logical Circuit Depth**: 25 (22 two-qubit CX gates)
- **Transpiled Circuit Depth**: 64 (29 native two-qubit CZ gates)
- **Hardware Measurement**: 117 unique basis states observed across 1,024 shots
- **Feasible Quantum Samples**: 5 shots ($0.49\%$ feasibility rate under physical noise)
- **Selected Optimized Schedule**: `Worker_0 → Shift_0`, `Worker_1 → Shift_1`, `Worker_2 → Shift_2`
- **Hardware Optimal Cost**: **$135.00**
- **Exact Classical Optimum**: **$135.00**
- **Optimality Gap**: **$0.00** (0.00%)

### Scientific Disclosure
> Physical quantum hardware execution was performed on a noisy, un-error-corrected QPU. The algorithm observed the classical optimum among its measured bitstrings in this particular 1,024-shot sample. This demonstrates computational feasibility and hardware execution of the identical logical formulation, but does **not** claim quantum speedup or quantum advantage.

---

## 4. Credential Requirements & Security

- **Notebooks 01, 02, and 03**: Run completely locally via Qiskit `AerSimulator`. **Zero API tokens required**.
- **Notebooks 04 and 05**: Connect to IBM Quantum Platform via `qiskit-ibm-runtime`.
  - Credentials must be set via environment variables:
    ```bash
    export IBM_QUANTUM_TOKEN="your_token_here"
    export IBM_QUANTUM_CHANNEL="ibm_quantum" # or 'ibm_cloud'
    ```
  - **Zero Credential Storage**: Notebooks read strictly from `os.environ`. No tokens or private credentials are hardcoded or printed in notebook outputs.

---

## 5. Running the Notebooks Locally

To launch Jupyter and execute the notebooks:

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Launch Jupyter Lab or Notebook
jupyter lab research/notebooks/
```

To execute a notebook non-interactively from the command line:

```bash
jupyter nbconvert --to notebook --execute research/notebooks/01_qaoa_aer_execution.ipynb --output 01_qaoa_aer_execution.ipynb
```
