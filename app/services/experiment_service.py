"""
ShiftProof — Experiment Service & Workspace Execution

Executes classical (exact, greedy) and quantum (QAOA p=1) optimizations
for datasets in the interactive application workspace.
Persists all results, schedules, and measurement distributions to SQLite (data/shiftproof.db).
NEVER touches or modifies results/.
"""

from __future__ import annotations

import sys
import time
import json
import uuid
from pathlib import Path
from typing import Optional
import sqlite3

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
from src.model import Instance, check_constraints, assignment_cost, decode_bitstring
from src.classical import solve_exact, solve_greedy, SolverResult
from src.quantum import run_qaoa_p1, QAOAResult, build_qubo
from app.database.db import get_db_connection
from app.services.scheduling_service import compute_dataset_fingerprint

logger = logging.getLogger("shiftproof.experiment")


def _fmt_assignment(asgn):
    if not asgn:
        return "{}"
    return json.dumps({f"{w},{s}": int(v) for (w, s), v in asgn.items()})


def _fmt_details(details):
    if not details:
        return "{}"
    cleaned = {}
    for k, v in details.items():
        if isinstance(v, dict):
            cleaned[k] = {str(subk): subv for subk, subv in v.items()}
        else:
            cleaned[k] = v
    return json.dumps(cleaned, default=str)


def run_classical_workspace(
    instance: Instance,
    dataset_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Execute classical optimization (Greedy + Exact if <=16 vars).
    Persists solution to SQLite experiments & solutions tables.
    """
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        experiment_id = f"exp_classical_{uuid.uuid4().hex[:8]}"

        # 1. Greedy solver (always runs)
        t0 = time.time()
        greedy_res = solve_greedy(instance)
        greedy_time = time.time() - t0

        # 2. Exact solver (only runs if <= 16 vars - application safety limit)
        exact_res = None
        exact_time = 0.0
        if instance.n_vars <= 16:
            t0 = time.time()
            try:
                exact_res = solve_exact(instance)
                exact_time = time.time() - t0
            except Exception:
                exact_res = None

        # Compute deterministic dataset fingerprint
        fingerprint = compute_dataset_fingerprint(dataset_id, conn)

        # Persist experiment
        with conn:
            conn.execute(
                """
                INSERT INTO experiments (experiment_id, dataset_id, solver_type, status, parameters_json)
                VALUES (?, ?, 'classical', 'completed', ?)
                """,
                (
                    experiment_id,
                    dataset_id,
                    json.dumps({
                        "dataset_fingerprint": fingerprint,
                        "instance_fingerprint": instance.instance_id,
                        "exact_run": bool(exact_res is not None),
                        "solver": "classical",
                        "worker_count": instance.n_workers,
                        "shift_count": instance.n_shifts,
                        "variable_count": instance.n_vars,
                    }),
                ),
            )

            # Insert greedy solution
            greedy_sol_id = f"sol_greedy_{uuid.uuid4().hex[:8]}"
            greedy_feas, greedy_details = check_constraints(instance, greedy_res.optimal_assignment or {})
            conn.execute(
                """
                INSERT INTO solutions (solution_id, experiment_id, dataset_id, feasible, total_cost, runtime_seconds, assignment_json, verification_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    greedy_sol_id,
                    experiment_id,
                    dataset_id,
                    greedy_feas,
                    greedy_res.optimal_cost,
                    greedy_time,
                    _fmt_assignment(greedy_res.optimal_assignment),
                    _fmt_details(greedy_details),
                ),
            )

            # Insert exact solution if run
            exact_sol_id = None
            if exact_res and exact_res.feasible:
                exact_sol_id = f"sol_exact_{uuid.uuid4().hex[:8]}"
                exact_feas, exact_details = check_constraints(instance, exact_res.optimal_assignment or {})
                conn.execute(
                    """
                    INSERT INTO solutions (solution_id, experiment_id, dataset_id, feasible, total_cost, runtime_seconds, assignment_json, verification_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        exact_sol_id,
                        experiment_id,
                        dataset_id,
                        exact_feas,
                        exact_res.optimal_cost,
                        exact_time,
                        _fmt_assignment(exact_res.optimal_assignment),
                        _fmt_details(exact_details),
                    ),
                )

        logger.info(
            f"[DATASET] dataset_id: {dataset_id}, fingerprint: {fingerprint} | "
            f"[INSTANCE] worker_count: {instance.n_workers}, shift_count: {instance.n_shifts}, variable_count: {instance.n_vars} | "
            f"[EXPERIMENT] experiment_id: {experiment_id}, dataset_id: {dataset_id}, solver: classical | "
            f"[RESULT] greedy: {greedy_res.optimal_cost}, exact: {exact_res.optimal_cost if exact_res else 'N/A'}"
        )

        return {
            "experiment_id": experiment_id,
            "greedy_result": greedy_res,
            "greedy_runtime": greedy_time,
            "exact_result": exact_res,
            "exact_runtime": exact_time,
        }
    finally:
        if close_after:
            conn.close()


def run_qaoa_workspace(
    instance: Instance,
    dataset_id: str,
    shots: int = 1024,
    maxiter: int = 100,
    seed: int = 42,
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """
    Execute QAOA optimization on AerSimulator and persist results.
    Enforces the validated limit: instance.n_vars <= 9.
    """
    if instance.n_vars > 9:
        raise ValueError(
            f"QAOA Locked: Dataset requires {instance.n_vars} variables/qubits. "
            f"Exceeds the currently validated QAOA demonstrator scale (<= 9 variables)."
        )

    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        experiment_id = f"exp_qaoa_{uuid.uuid4().hex[:8]}"

        # Execute QAOA in memory
        qaoa_res: QAOAResult = run_qaoa_p1(
            instance,
            shots=shots,
            maxiter=maxiter,
            seed=seed,
        )

        qubo = build_qubo(instance)

        # Process per-bitstring samples
        total_shots = sum(qaoa_res.samples.values())
        best_bitstring = None
        best_assignment = None
        best_cost = float("inf")

        measurement_rows = []
        for bitstring, count in qaoa_res.samples.items():
            if len(bitstring) < instance.n_vars:
                bitstring = "0" * (instance.n_vars - len(bitstring)) + bitstring

            asgn = decode_bitstring(instance, bitstring)
            feas, _ = check_constraints(instance, asgn)
            cost = assignment_cost(instance, asgn) if feas else None
            energy = qubo.energy_from_bitstring(bitstring)
            prob = count / total_shots

            if feas and cost is not None and cost < best_cost:
                best_cost = cost
                best_bitstring = bitstring
                best_assignment = asgn

            meas_id = f"meas_{uuid.uuid4().hex[:8]}"
            measurement_rows.append((
                meas_id,
                experiment_id,
                bitstring,
                count,
                prob,
                feas,
                cost,
                energy,
            ))

        # Format assignment string
        def _fmt_assignment(asgn):
            if not asgn:
                return "{}"
            return json.dumps({f"{w},{s}": int(v) for (w, s), v in asgn.items()})

        fingerprint = compute_dataset_fingerprint(dataset_id, conn)

        # Persist experiment, solution, and measurements
        with conn:
            conn.execute(
                """
                INSERT INTO experiments (experiment_id, dataset_id, solver_type, status, parameters_json)
                VALUES (?, ?, 'qaoa_p1', 'completed', ?)
                """,
                (
                    experiment_id,
                    dataset_id,
                    json.dumps({
                        "dataset_fingerprint": fingerprint,
                        "instance_fingerprint": instance.instance_id,
                        "worker_count": instance.n_workers,
                        "shift_count": instance.n_shifts,
                        "variable_count": instance.n_vars,
                        "solver": "qaoa_p1",
                        "shots": shots,
                        "maxiter": maxiter,
                        "seed": seed,
                        "p": 1,
                        "optimizer": "COBYLA",
                        "backend": "Qiskit AerSimulator",
                        "circuit_depth": qaoa_res.circuit_depth,
                        "two_qubit_gate_count": qaoa_res.two_qubit_gate_count,
                        "feasible_rate": qaoa_res.feasible_rate,
                        "optimal_solution_probability": qaoa_res.optimal_solution_probability,
                        "optimal_params": [float(p) for p in qaoa_res.optimal_params],
                        "optimal_energy": float(qaoa_res.optimal_energy),
                        "optimizer_evaluations": int(qaoa_res.optimizer_evaluations),
                    }),
                ),
            )

            solution_id = f"sol_qaoa_{uuid.uuid4().hex[:8]}"
            best_feas, best_details = check_constraints(instance, best_assignment or {})
            conn.execute(
                """
                INSERT INTO solutions (solution_id, experiment_id, dataset_id, feasible, total_cost, runtime_seconds, assignment_json, verification_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    solution_id,
                    experiment_id,
                    dataset_id,
                    best_feas,
                    best_cost if (best_feas and best_cost != float("inf")) else None,
                    qaoa_res.runtime_seconds,
                    _fmt_assignment(best_assignment),
                    _fmt_details(best_details),
                ),
            )

            conn.executemany(
                """
                INSERT INTO measurements (measurement_id, experiment_id, bitstring, shot_count, probability, is_feasible, assignment_cost, qubo_energy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                measurement_rows,
            )

        logger.info(
            f"[DATASET] dataset_id: {dataset_id}, fingerprint: {fingerprint} | "
            f"[INSTANCE] worker_count: {instance.n_workers}, shift_count: {instance.n_shifts}, variable_count: {instance.n_vars} | "
            f"[EXPERIMENT] experiment_id: {experiment_id}, dataset_id: {dataset_id}, solver: qaoa_p1 | "
            f"[RESULT] best_cost: {best_cost}"
        )

        from src.quantum import qubo_to_ising, build_qaoa_circuit
        ising = qubo_to_ising(qubo)
        qc, gamma_params, beta_params = build_qaoa_circuit(ising, p=1)
        param_dict = {gamma_params[0]: qaoa_res.optimal_params[0], beta_params[0]: qaoa_res.optimal_params[1]}
        bound_qc = qc.assign_parameters(param_dict)

        return {
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "dataset_fingerprint": fingerprint,
            "instance_fingerprint": instance.instance_id,
            "backend": "Qiskit AerSimulator",
            "qubit_count": instance.n_vars,
            "shots": shots,
            "feasible": best_feas,
            "best_cost": best_cost if (best_feas and best_cost != float("inf")) else None,
            "best_feasible_cost": best_cost if (best_feas and best_cost != float("inf")) else None,
            "feasible_rate": qaoa_res.feasible_rate,
            "optimal_solution_probability": qaoa_res.optimal_solution_probability,
            "circuit_depth": qaoa_res.circuit_depth,
            "two_qubit_gate_count": qaoa_res.two_qubit_gate_count,
            "runtime_seconds": qaoa_res.runtime_seconds,
            "best_assignment": best_assignment,
            "qaoa_result": qaoa_res,
            "measurement_count": len(measurement_rows),
            "circuit": bound_qc,
            "parameterized_circuit": qc,
            "optimal_params": [float(p) for p in qaoa_res.optimal_params],
            "qubo": qubo,
            "ising": ising,
        }

    finally:
        if close_after:
            conn.close()


def list_dataset_experiments(dataset_id: str, conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """Retrieve all past experiment runs and solutions for a dataset."""
    close_after = False
    if conn is None:
        conn = get_db_connection()
        close_after = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.experiment_id, e.dataset_id, e.solver_type, e.status, e.parameters_json, e.created_at,
                   s.solution_id, s.feasible, s.total_cost, s.runtime_seconds, s.assignment_json
            FROM experiments e
            JOIN solutions s ON e.experiment_id = s.experiment_id
            WHERE e.dataset_id = ?
            ORDER BY e.created_at DESC, s.rowid ASC
            """,
            (dataset_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close_after:
            conn.close()
