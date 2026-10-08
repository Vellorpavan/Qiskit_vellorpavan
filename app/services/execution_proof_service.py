"""
ShiftProof — Execution Proof Service

Generates the complete, unadulterated cryptographic and mathematical execution chain
for a QAOA run on the active dataset:
  Uploaded Dataset
  → Normalized Scheduling Instance
  → Binary Decision Variables
  → QUBO
  → Ising Hamiltonian
  → Actual Qiskit QuantumCircuit
  → QAOA execution
  → Qiskit AerSimulator
  → Measurement Counts
  → Independent Verification
  → Decoded Schedule
  → Classical/Exact Comparison

Zero fabrication. Zero fake proof data. Zero benchmark reuse for uploaded data.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2

from src.model import Instance, check_constraints, assignment_cost, decode_bitstring, encode_assignment
from src.classical import solve_exact, solve_greedy
from src.quantum import build_qubo, qubo_to_ising, build_qaoa_circuit
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint


def get_execution_proof(
    dataset_id: str,
    conn: sqlite3.Connection,
    experiment_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Construct the full Execution Proof chain for a dataset.
    Returns proof dictionary or status dictionary indicating missing/invalid state.
    """
    # 1. Dataset existence & metadata
    cur = conn.cursor()
    cur.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
    ds_row = cur.fetchone()
    if not ds_row:
        return {"status": "DATASET_NOT_FOUND", "message": f"Dataset '{dataset_id}' not found."}

    ds_meta = dict(ds_row)
    active_fp = compute_dataset_fingerprint(dataset_id, conn)

    # 2. Build Instance from database
    try:
        instance = db_to_instance(dataset_id, conn)
    except Exception as e:
        return {"status": "INSTANCE_ERROR", "message": f"Failed to reconstruct instance: {e}"}

    # Fetch worker and shift metadata
    cur.execute(
        "SELECT worker_id, worker_index, name, skill, max_shifts FROM workers WHERE dataset_id = ? ORDER BY worker_index ASC",
        (dataset_id,),
    )
    worker_rows = [dict(r) for r in cur.fetchall()]
    worker_names = [r["name"] for r in worker_rows] if worker_rows else [f"Worker_{i}" for i in range(instance.n_workers)]

    cur.execute(
        "SELECT shift_id, shift_index, shift_name, required_skill FROM shifts WHERE dataset_id = ? ORDER BY shift_index ASC",
        (dataset_id,),
    )
    shift_rows = [dict(r) for r in cur.fetchall()]
    shift_names = [r["shift_name"] for r in shift_rows] if shift_rows else [f"Shift_{i}" for i in range(instance.n_shifts)]

    # 3. Retrieve QAOA Experiment
    if experiment_id:
        cur.execute(
            """
            SELECT e.*, s.solution_id, s.feasible as sol_feasible, s.total_cost, s.runtime_seconds, s.assignment_json, s.verification_json
            FROM experiments e
            LEFT JOIN solutions s ON e.experiment_id = s.experiment_id
            WHERE e.experiment_id = ? AND e.dataset_id = ?
            """,
            (experiment_id, dataset_id),
        )
    else:
        cur.execute(
            """
            SELECT e.*, s.solution_id, s.feasible as sol_feasible, s.total_cost, s.runtime_seconds, s.assignment_json, s.verification_json
            FROM experiments e
            LEFT JOIN solutions s ON e.experiment_id = s.experiment_id
            WHERE e.dataset_id = ? AND e.solver_type = 'qaoa_p1' AND e.status = 'completed'
            ORDER BY e.created_at DESC LIMIT 1
            """,
            (dataset_id,),
        )

    exp_row = cur.fetchone()
    if not exp_row:
        return {
            "status": "NO_EXECUTION",
            "message": f"No QAOA execution exists for current dataset '{ds_meta.get('name', dataset_id)}'.",
            "dataset_id": dataset_id,
            "dataset_name": ds_meta.get("name", dataset_id),
            "dataset_fingerprint": active_fp,
        }

    exp_data = dict(exp_row)
    exp_id = exp_data["experiment_id"]
    params = json.loads(exp_data.get("parameters_json") or "{}")

    # 4. Freshness & Isolation Check
    stored_fp = params.get("dataset_fingerprint")
    if stored_fp and stored_fp != active_fp:
        return {
            "status": "STALE_EXECUTION",
            "message": "Dataset was modified since last execution. Fingerprint mismatch.",
            "stored_fingerprint": stored_fp,
            "current_fingerprint": active_fp,
            "dataset_id": dataset_id,
        }

    opt_mode = ds_meta.get("optimization_mode") or "COST_OPTIMIZATION"
    if np.all(instance.cost_matrix == 0.0):
        opt_mode = "FEASIBILITY_ONLY"

    # 5. Retrieve Raw AerSimulator Measurement Counts
    cur.execute(
        """
        SELECT bitstring, shot_count, probability, is_feasible, assignment_cost, qubo_energy
        FROM measurements
        WHERE experiment_id = ?
        ORDER BY shot_count DESC, bitstring ASC
        """,
        (exp_id,),
    )
    meas_rows = [dict(r) for r in cur.fetchall()]

    # 6. Build QUBO & Ising Truth Model
    qubo = build_qubo(instance)
    ising = qubo_to_ising(qubo)

    # Decision variables mapping
    decision_variables = []
    for q in range(instance.n_vars):
        var = instance.get_var(q)
        if var is not None:
            w_idx, s_idx = var
            w_name = worker_names[w_idx] if w_idx < len(worker_names) else f"Worker_{w_idx}"
            s_name = shift_names[s_idx] if s_idx < len(shift_names) else f"Shift_{s_idx}"
            decision_variables.append({
                "qubit": q,
                "symbol": f"x_{q}",
                "worker_index": w_idx,
                "shift_index": s_idx,
                "worker_name": w_name,
                "shift_name": s_name,
                "description": f"{w_name} assigned to {s_name}",
                "cost": float(instance.get_cost(w_idx, s_idx)),
                "eligible": instance.is_eligible(w_idx, s_idx),
            })

    # QUBO linear and quadratic terms
    qubo_linear = []
    for i in range(instance.n_vars):
        qubo_linear.append({
            "index": i,
            "variable": f"x_{i}",
            "coefficient": float(qubo.a[i]),
        })

    qubo_quadratic = []
    for i in range(instance.n_vars):
        for j in range(i + 1, instance.n_vars):
            coeff = float(qubo.b[i, j])
            if abs(coeff) > 1e-12:
                qubo_quadratic.append({
                    "var_i": f"x_{i}",
                    "var_j": f"x_{j}",
                    "coefficient": coeff,
                })

    # Ising linear and coupling terms
    ising_linear = []
    for i in range(len(ising.h)):
        ising_linear.append({
            "spin": i,
            "symbol": f"Z_{i}",
            "coefficient": float(ising.h[i]),
        })

    ising_couplings = []
    for i in range(len(ising.h)):
        for j in range(i + 1, len(ising.h)):
            coupling = float(ising.J[i, j])
            if abs(coupling) > 1e-12:
                ising_couplings.append({
                    "spin_i": f"Z_{i}",
                    "spin_j": f"Z_{j}",
                    "coefficient": coupling,
                })

    # Validate QUBO <-> Ising Energy Equivalence on tested bitstrings
    energy_discrepancies = []
    for m in meas_rows[:20]:
        bs = m["bitstring"]
        x = np.array([int(bs[-(k + 1)]) for k in range(instance.n_vars)])
        z = 1 - 2 * x
        q_energy = qubo.energy(x)
        # Compute Ising energy
        i_energy = ising.offset + np.dot(ising.h, z)
        for i in range(instance.n_vars):
            for j in range(i + 1, instance.n_vars):
                i_energy += ising.J[i, j] * z[i] * z[j]
        diff = abs(q_energy - i_energy)
        energy_discrepancies.append(diff)

    max_energy_diff = max(energy_discrepancies) if energy_discrepancies else 0.0

    # 7. Actual Qiskit QuantumCircuit Object
    optimal_params = params.get("optimal_params", [0.0, 0.0])
    p_layers = params.get("p", 1)
    param_qc, gamma_params, beta_params = build_qaoa_circuit(ising, p=p_layers)

    param_dict = {}
    if len(optimal_params) >= 2 * p_layers:
        for k in range(p_layers):
            param_dict[gamma_params[k]] = optimal_params[k]
            param_dict[beta_params[k]] = optimal_params[p_layers + k]
    else:
        for k in range(p_layers):
            param_dict[gamma_params[k]] = 0.0
            param_dict[beta_params[k]] = 0.0

    bound_qc = param_qc.assign_parameters(param_dict)

    # Real circuit telemetry
    circuit_text_diagram = str(bound_qc.draw(output="text"))
    try:
        qasm_str = qiskit.qasm2.dumps(bound_qc)
    except Exception:
        qasm_str = "// OpenQASM export not available for parameterized elements"

    ops = bound_qc.count_ops()
    num_qubits = bound_qc.num_qubits
    num_clbits = bound_qc.num_clbits
    depth = bound_qc.depth()
    total_gates = bound_qc.size()
    single_q_gates = sum(v for k, v in ops.items() if k in {"h", "rz", "rx", "x", "z", "y", "s", "t"})
    two_q_gates = sum(v for k, v in ops.items() if k in {"cx", "cz", "swap", "rzz", "cp"})
    measurement_ops = ops.get("measure", 0)

    # 8. Independent Verification of All Sampled Bitstrings
    verified_samples = []
    best_feasible_sample = None
    best_feas_cost = float("inf")

    for m in meas_rows:
        bs = m["bitstring"]
        asgn = decode_bitstring(instance, bs)
        is_feas, details = check_constraints(instance, asgn)
        cost_val = assignment_cost(instance, asgn) if is_feas else None

        # Build readable violations if infeasible
        violations = []
        if not is_feas:
            for s_idx, covered in details.get("shift_coverage", {}).items():
                if not covered:
                    s_name = shift_names[s_idx] if s_idx < len(shift_names) else f"Shift {s_idx}"
                    violations.append(f"Shift '{s_name}' uncovered")
            for w_idx, at_most_one in details.get("worker_at_most_one", {}).items():
                if not at_most_one:
                    w_name = worker_names[w_idx] if w_idx < len(worker_names) else f"Worker {w_idx}"
                    violations.append(f"Worker '{w_name}' assigned multiple shifts")
            for pair, elig in details.get("eligibility", {}).items():
                if not elig:
                    w_idx, s_idx = pair
                    w_name = worker_names[w_idx] if w_idx < len(worker_names) else f"Worker {w_idx}"
                    s_name = shift_names[s_idx] if s_idx < len(shift_names) else f"Shift {s_idx}"
                    violations.append(f"Ineligible assignment: {w_name} to {s_name}")

        readable_asgn = []
        for (w_idx, s_idx), val in asgn.items():
            if val == 1:
                w_name = worker_names[w_idx] if w_idx < len(worker_names) else f"Worker {w_idx}"
                s_name = shift_names[s_idx] if s_idx < len(shift_names) else f"Shift {s_idx}"
                readable_asgn.append(f"{w_name} → {s_name}")

        sample_info = {
            "bitstring": bs,
            "shot_count": m["shot_count"],
            "probability": m["probability"],
            "is_feasible": is_feas,
            "cost": cost_val,
            "qubo_energy": m["qubo_energy"],
            "readable_assignment": readable_asgn,
            "violations": violations,
        }
        verified_samples.append(sample_info)

        if is_feas and cost_val is not None and cost_val < best_feas_cost:
            best_feas_cost = cost_val
            best_feasible_sample = sample_info

    # 9. Classical Solver Comparison for the SAME Instance
    if instance.n_vars <= 16:
        classical_method = "Classical Exact (Integer Enumeration)"
        classical_res = solve_exact(instance)
    else:
        classical_method = "Classical Greedy Reference"
        classical_res = solve_greedy(instance)

    classical_cost = classical_res.optimal_cost if classical_res.feasible else None
    classical_asgn_readable = []
    if classical_res.feasible and classical_res.optimal_assignment:
        for (w_idx, s_idx), val in classical_res.optimal_assignment.items():
            if val == 1:
                w_name = worker_names[w_idx] if w_idx < len(worker_names) else f"Worker {w_idx}"
                s_name = shift_names[s_idx] if s_idx < len(shift_names) else f"Shift {s_idx}"
                classical_asgn_readable.append(f"{w_name} → {s_name}")

    # Compute Gap
    qaoa_observed_cost = best_feasible_sample["cost"] if best_feasible_sample else None
    abs_gap = None
    rel_gap = None
    observed_exact_optimum = False

    if classical_cost is not None and qaoa_observed_cost is not None:
        abs_gap = max(0.0, qaoa_observed_cost - classical_cost)
        if abs(classical_cost) > 1e-9:
            rel_gap = (abs_gap / classical_cost) * 100.0
        else:
            rel_gap = 0.0
        if abs_gap <= 1e-4:
            observed_exact_optimum = True

    # 10. Reproducibility snippet
    shots_val = params.get("shots", 1024)
    maxiter_val = params.get("maxiter", 100)
    seed_val = params.get("seed", 42)
    p_val = params.get("p", 1)

    repro_code = f"""# ShiftProof — Reproduce Execution {exp_id}
import sqlite3
from app.services.scheduling_service import db_to_instance
from src.quantum import run_qaoa_p1

conn = sqlite3.connect("data/shiftproof.db")
instance = db_to_instance("{dataset_id}", conn)

# Run identical QAOA execution on Qiskit AerSimulator:
result = run_qaoa_p1(
    instance=instance,
    shots={shots_val},
    maxiter={maxiter_val},
    seed={seed_val},
)

print(f"Feasible samples: {{result.feasible_samples}} / {{result.shots}}")
print(f"Best feasible cost: {{result.best_feasible_cost}}")
print(f"Optimal parameters: gamma={{result.optimal_params[0]:.4f}}, beta={{result.optimal_params[1]:.4f}}")
"""

    return {
        "status": "VALID",
        "dataset_identity": {
            "dataset_name": ds_meta.get("name", dataset_id),
            "dataset_id": dataset_id,
            "dataset_fingerprint": active_fp,
            "instance_fingerprint": instance.instance_id,
            "experiment_id": exp_id,
            "created_at": exp_data.get("created_at"),
            "optimization_mode": opt_mode,
        },
        "instance_data": {
            "n_workers": instance.n_workers,
            "n_shifts": instance.n_shifts,
            "n_vars": instance.n_vars,
            "worker_names": worker_names,
            "shift_names": shift_names,
            "workers": worker_rows,
            "shifts": shift_rows,
            "eligible_pairs": [list(pair) for pair in instance.var_to_qubit.keys()],
            "cost_matrix": instance.cost_matrix.tolist() if instance.cost_matrix is not None else [],
        },
        "decision_variables": decision_variables,
        "qubo_data": {
            "n_vars": instance.n_vars,
            "A": float(qubo.A),
            "B": float(qubo.B),
            "delta": 1.0,
            "x_ref_cost": float(qubo.x_ref_cost),
            "q0": float(qubo.q0),
            "linear_coefficients": qubo_linear,
            "quadratic_coefficients": qubo_quadratic,
            "optimization_mode": opt_mode,
        },
        "ising_data": {
            "offset": float(ising.offset),
            "h_coefficients": ising_linear,
            "J_couplings": ising_couplings,
            "max_energy_discrepancy": float(max_energy_diff),
            "qubo_ising_verified": bool(max_energy_diff < 1e-9),
        },
        "circuit_data": {
            "circuit": bound_qc,
            "num_qubits": num_qubits,
            "num_clbits": num_clbits,
            "p": p_layers,
            "depth": depth,
            "total_gates": total_gates,
            "single_qubit_gates": single_q_gates,
            "two_qubit_gates": two_q_gates,
            "measurement_operations": measurement_ops,
            "ops_breakdown": dict(ops),
            "text_diagram": circuit_text_diagram,
            "qasm": qasm_str,
        },
        "backend_data": {
            "backend": "Qiskit AerSimulator",
            "execution_type": "QAOA Simulation",
            "hardware_status": "IBM Quantum Hardware: Not connected",
            "authenticity_note": (
                "Qiskit AerSimulator is a high-performance classical simulator of quantum circuits. "
                "Execution took place on the local host machine, not physical quantum hardware."
            ),
        },
        "parameters_data": {
            "shots": shots_val,
            "p": p_val,
            "optimizer": params.get("optimizer", "COBYLA"),
            "maxiter": maxiter_val,
            "seed": seed_val,
            "runtime_seconds": float(exp_data.get("runtime_seconds") or 0.0),
            "qubit_count": instance.n_vars,
            "optimal_params": [float(p) for p in optimal_params],
            "optimal_energy": params.get("optimal_energy"),
            "optimizer_evaluations": params.get("optimizer_evaluations"),
            "circuit_depth": depth,
            "two_qubit_gate_count": two_q_gates,
            "feasible_rate": params.get("feasible_rate"),
            "optimal_solution_probability": params.get("optimal_solution_probability"),
        },
        "measurement_counts": meas_rows,
        "independent_verification": verified_samples,
        "best_feasible_sample": best_feasible_sample,
        "classical_comparison": {
            "classical_method": classical_method,
            "classical_cost": classical_cost,
            "classical_assignment": classical_asgn_readable,
            "classical_feasible": classical_res.feasible,
            "qaoa_cost": qaoa_observed_cost,
            "qaoa_assignment": best_feasible_sample["readable_assignment"] if best_feasible_sample else [],
            "absolute_gap": abs_gap,
            "relative_gap": rel_gap,
            "observed_classical_optimum": observed_exact_optimum,
        },
        "reproducibility": {
            "code_snippet": repro_code,
            "shots": shots_val,
            "seed": seed_val,
            "p": p_val,
        },
    }
