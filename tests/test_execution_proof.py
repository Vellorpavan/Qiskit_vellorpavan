"""
ShiftProof — Execution Proof Verification Test Suite

Verifies Requirements (A through K):
A. Uploaded dataset creates its own Instance
B. QUBO comes from that Instance
C. QAOA circuit qubit count matches QUBO variable count
D. Displayed circuit is the actual generated QuantumCircuit
E. Measurement counts come from the execution
F. Every displayed bitstring is independently verified
G. Dataset switching invalidates old execution state
H. Dataset A and Dataset B produce different fingerprints and different QUBO coefficients
I. No classical solution is copied into QAOA results
J. No fake hardware result can be displayed (honest AerSimulator status)
K. Feasibility-only datasets do not receive fabricated monetary values
"""

import sqlite3
import pytest
import numpy as np
from qiskit import QuantumCircuit

from app.database.schema import init_db
from app.services.ingestion.pipeline import inspect_uploaded_file, finalize_and_save_dataset
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint
from app.services.experiment_service import run_qaoa_workspace
from app.services.execution_proof_service import get_execution_proof
from src.classical import solve_exact
from src.quantum import build_qubo


@pytest.fixture
def proof_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def test_execution_proof_end_to_end_chain(proof_db):
    """
    Verifies Requirements A, B, C, D, E, F, I, J:
    - Uploaded dataset creates its own Instance (A)
    - QUBO comes from that Instance (B)
    - Circuit qubit count matches QUBO variable count (C)
    - Displayed circuit is actual QuantumCircuit (D)
    - Measurement counts come from execution (E)
    - Every bitstring is independently verified (F)
    - No classical solution is copied into QAOA results (I)
    - Honest AerSimulator status, no fake hardware (J)
    """
    csv_bytes = b"Worker,Shift,Hourly_Rate\nNurse_A,Morning,50.0\nNurse_B,Morning,75.0\n"
    insp = inspect_uploaded_file(csv_bytes, "hospital_roster.csv")
    ds_id, norm = finalize_and_save_dataset(proof_db, insp, dataset_name="Hospital Unit")

    # A. Uploaded dataset creates its own Instance
    instance = db_to_instance(ds_id, proof_db)
    assert instance.n_workers == 2
    assert instance.n_shifts == 1
    assert instance.n_vars == 2

    # B. QUBO comes from that Instance
    qubo = build_qubo(instance)
    assert qubo.instance.n_vars == 2
    assert qubo.a.shape == (2,)

    # Run QAOA execution
    q_out = run_qaoa_workspace(instance, ds_id, shots=128, maxiter=5, seed=42, conn=proof_db)
    exp_id = q_out["experiment_id"]

    # Generate Execution Proof
    proof = get_execution_proof(ds_id, proof_db, experiment_id=exp_id)
    assert proof["status"] == "VALID"

    # Identity check
    ident = proof["dataset_identity"]
    assert ident["dataset_id"] == ds_id
    assert ident["experiment_id"] == exp_id
    assert ident["optimization_mode"] == "COST_OPTIMIZATION"
    assert ident["dataset_fingerprint"] == compute_dataset_fingerprint(ds_id, proof_db)

    # C. QAOA circuit qubit count matches QUBO variable count
    circ_info = proof["circuit_data"]
    assert circ_info["num_qubits"] == instance.n_vars
    assert circ_info["num_qubits"] == proof["qubo_data"]["n_vars"]
    assert len(proof["decision_variables"]) == instance.n_vars

    # D. Displayed circuit is the actual generated QuantumCircuit
    qc = circ_info["circuit"]
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 2
    assert circ_info["depth"] > 0
    assert circ_info["total_gates"] > 0
    assert len(circ_info["text_diagram"]) > 0
    assert "OPENQASM" in circ_info["qasm"]

    # E. Measurement counts come from the execution
    meas = proof["measurement_counts"]
    assert len(meas) > 0
    total_shots = sum(m["shot_count"] for m in meas)
    assert total_shots == 128
    for m in meas:
        assert 0.0 <= m["probability"] <= 1.0

    # F. Every displayed bitstring is independently verified
    verif = proof["independent_verification"]
    assert len(verif) == len(meas)
    for v in verif:
        assert isinstance(v["is_feasible"], bool)
        assert isinstance(v["violations"], list)
        if not v["is_feasible"]:
            assert len(v["violations"]) > 0  # Violated constraints explained

    # I. No classical solution is copied into QAOA results
    comp = proof["classical_comparison"]
    c_res = solve_exact(instance)
    assert comp["classical_cost"] == c_res.optimal_cost
    # If QAOA found a feasible sample, its cost was computed from its own sampled bitstring
    if proof["best_feasible_sample"]:
        assert proof["best_feasible_sample"]["cost"] is not None
        # QAOA cost is either equal or higher than the exact minimum, never fabricated
        assert proof["best_feasible_sample"]["cost"] >= c_res.optimal_cost - 1e-6

    # J. Honest backend status (No fake hardware result)
    backend_info = proof["backend_data"]
    assert backend_info["backend"] == "Qiskit AerSimulator"
    assert backend_info["execution_type"] == "QAOA Simulation"
    assert backend_info["hardware_status"] == "IBM Quantum Hardware: Not connected"
    assert "not physical quantum hardware" in backend_info["authenticity_note"]


def test_dataset_switching_isolation(proof_db):
    """
    Verifies Requirement G & H:
    - Dataset A and Dataset B produce different fingerprints and different QUBOs (H)
    - Switching datasets invalidates old execution state (G)
    """
    # Create Dataset A
    csv_a = b"Worker,Shift,Cost\nAlpha,Morning,15.0\nBeta,Morning,45.0\n"
    insp_a = inspect_uploaded_file(csv_a, "dataset_a.csv")
    ds_id_a, _ = finalize_and_save_dataset(proof_db, insp_a, dataset_name="Dataset A")

    # Create Dataset B
    csv_b = b"Worker,Shift,Cost\nAlpha,Morning,80.0\nBeta,Morning,10.0\n"
    insp_b = inspect_uploaded_file(csv_b, "dataset_b.csv")
    ds_id_b, _ = finalize_and_save_dataset(proof_db, insp_b, dataset_name="Dataset B")

    # Fingerprints differ
    fp_a = compute_dataset_fingerprint(ds_id_a, proof_db)
    fp_b = compute_dataset_fingerprint(ds_id_b, proof_db)
    assert fp_a != fp_b

    # Instances differ
    inst_a = db_to_instance(ds_id_a, proof_db)
    inst_b = db_to_instance(ds_id_b, proof_db)
    assert not np.array_equal(inst_a.cost_matrix, inst_b.cost_matrix)

    # QUBOs differ (Requirement H)
    qubo_a = build_qubo(inst_a)
    qubo_b = build_qubo(inst_b)
    assert not np.array_equal(qubo_a.a, qubo_b.a)

    # Execute QAOA ONLY on Dataset A
    run_qaoa_workspace(inst_a, ds_id_a, shots=64, maxiter=3, seed=42, conn=proof_db)

    # Dataset A has a valid execution proof
    proof_a = get_execution_proof(ds_id_a, proof_db)
    assert proof_a["status"] == "VALID"
    assert proof_a["dataset_identity"]["dataset_id"] == ds_id_a
    assert proof_a["dataset_identity"]["dataset_fingerprint"] == fp_a

    # Dataset B has NOT been run yet -> Must show NO_EXECUTION (Requirement G)
    proof_b = get_execution_proof(ds_id_b, proof_db)
    assert proof_b["status"] == "NO_EXECUTION"
    assert "No QAOA execution exists" in proof_b["message"]
    # Dataset A's execution must NEVER bleed or appear for Dataset B
    assert "circuit_data" not in proof_b
    assert "measurement_counts" not in proof_b


def test_feasibility_only_dataset_execution_proof(proof_db):
    """
    Verifies Requirement K:
    - Feasibility-only dataset creates real Instance and real QUBO
    - No fabricated monetary values
    - Optimization mode is FEASIBILITY_ONLY
    - QAOA execution works and proof exposes feasibility without fake dollars
    """
    # Dataset with NO cost column
    csv_bytes = b"Worker,Shift\nTech_1,Night\nTech_2,Night\n"
    insp = inspect_uploaded_file(csv_bytes, "plant_shifts.csv")
    assert insp.classification.optimization_mode == "FEASIBILITY_ONLY"

    ds_id, norm = finalize_and_save_dataset(proof_db, insp, dataset_name="Plant Shifts")
    assert norm.optimization_mode == "FEASIBILITY_ONLY"

    # Instance has zero internal mathematical feasibility cost
    inst = db_to_instance(ds_id, proof_db)
    assert np.all(inst.cost_matrix == 0.0)

    # Run QAOA
    run_qaoa_workspace(inst, ds_id, shots=64, maxiter=3, seed=42, conn=proof_db)

    proof = get_execution_proof(ds_id, proof_db)
    assert proof["status"] == "VALID"
    assert proof["dataset_identity"]["optimization_mode"] == "FEASIBILITY_ONLY"
    assert proof["qubo_data"]["optimization_mode"] == "FEASIBILITY_ONLY"

    # No fake dollar amounts
    if proof["best_feasible_sample"]:
        assert proof["best_feasible_sample"]["cost"] == 0.0

    # Decision variables have 0.0 unit cost
    for var in proof["decision_variables"]:
        assert var["cost"] == 0.0

    # Classical comparison shows feasibility
    comp = proof["classical_comparison"]
    assert comp["classical_feasible"] is True
    assert comp["classical_cost"] == 0.0
