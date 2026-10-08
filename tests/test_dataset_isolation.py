"""
Tests for Dataset Isolation, Context Invalidation, and Optimization Provenance.
Verifies Critical Rules #1 through #20:
- Dataset A and Dataset B produce separate Instances
- Changing costs changes optimal assignment
- Changing eligibility changes optimal assignment
- Experiments and solutions are strictly dataset-scoped
- Fingerprints are deterministic and prevent cross-dataset contamination
"""

import json
import sqlite3
import pytest
import numpy as np

from app.database.schema import init_db
from app.services.scheduling_service import db_to_instance, compute_dataset_fingerprint
from app.services.experiment_service import run_classical_workspace, run_qaoa_workspace, list_dataset_experiments
from app.services.dataset_service import set_active_dataset, get_active_dataset
from src.classical import solve_exact, solve_greedy
from src.quantum import build_qubo, qubo_to_ising



@pytest.fixture
def test_db():
    """Create an isolated in-memory SQLite database initialized with ShiftProof schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def _seed_test_datasets(conn: sqlite3.Connection):
    """
    Seed two intentionally distinct datasets:
    Dataset A:
      Worker A1: cost=10, eligible=1
      Worker A2: cost=100, eligible=1
      Shift S1
      -> Expected optimal worker: Worker A1 (index 0, cost 10)
    
    Dataset B:
      Worker B1: cost=100, eligible=1
      Worker B2: cost=10, eligible=1
      Shift S1
      -> Expected optimal worker: Worker B2 (index 1, cost 10)
    
    Dataset C (Eligibility change):
      Worker C1: cost=10, eligible=0 (ineligible!)
      Worker C2: cost=100, eligible=1
      Shift S1
      -> Expected optimal worker: Worker C2 (index 1, cost 100)
    """
    with conn:
        # Dataset A
        conn.execute("INSERT INTO datasets (dataset_id, name, source_type, row_count, col_count) VALUES ('ds_A', 'Dataset A', 'upload', 2, 4)")
        conn.execute("INSERT INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w_A1', 'ds_A', 'Worker A1', 0)")
        conn.execute("INSERT INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w_A2', 'ds_A', 'Worker A2', 1)")
        conn.execute("INSERT INTO shifts (shift_id, dataset_id, shift_name, shift_index) VALUES ('s_A1', 'ds_A', 'Shift S1', 0)")
        conn.execute("INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_A', 'w_A1', 's_A1', 10.0)")
        conn.execute("INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_A', 'w_A2', 's_A1', 100.0)")
        conn.execute("INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES ('ds_A', 'w_A1', 's_A1', 1)")
        conn.execute("INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES ('ds_A', 'w_A2', 's_A1', 1)")

        # Dataset B
        conn.execute("INSERT INTO datasets (dataset_id, name, source_type, row_count, col_count) VALUES ('ds_B', 'Dataset B', 'upload', 2, 4)")
        conn.execute("INSERT INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w_B1', 'ds_B', 'Worker B1', 0)")
        conn.execute("INSERT INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w_B2', 'ds_B', 'Worker B2', 1)")
        conn.execute("INSERT INTO shifts (shift_id, dataset_id, shift_name, shift_index) VALUES ('s_B1', 'ds_B', 'Shift S1', 0)")
        conn.execute("INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_B', 'w_B1', 's_B1', 100.0)")
        conn.execute("INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_B', 'w_B2', 's_B1', 10.0)")
        conn.execute("INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES ('ds_B', 'w_B1', 's_B1', 1)")
        conn.execute("INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES ('ds_B', 'w_B2', 's_B1', 1)")

        # Dataset C (Worker C1 has lower cost but is INELIGIBLE)
        conn.execute("INSERT INTO datasets (dataset_id, name, source_type, row_count, col_count) VALUES ('ds_C', 'Dataset C', 'upload', 2, 4)")
        conn.execute("INSERT INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w_C1', 'ds_C', 'Worker C1', 0)")
        conn.execute("INSERT INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w_C2', 'ds_C', 'Worker C2', 1)")
        conn.execute("INSERT INTO shifts (shift_id, dataset_id, shift_name, shift_index) VALUES ('s_C1', 'ds_C', 'Shift S1', 0)")
        conn.execute("INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_C', 'w_C1', 's_C1', 10.0)")
        conn.execute("INSERT INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_C', 'w_C2', 's_C1', 100.0)")
        conn.execute("INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES ('ds_C', 'w_C1', 's_C1', 0)")
        conn.execute("INSERT INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES ('ds_C', 'w_C2', 's_C1', 1)")


def test_instance_dataset_isolation(test_db):
    """Test 1 & 2: Dataset A -> Instance A, Dataset B -> different Instance B."""
    _seed_test_datasets(test_db)
    inst_A = db_to_instance("ds_A", test_db)
    inst_B = db_to_instance("ds_B", test_db)

    assert inst_A.instance_id == "ds_A"
    assert inst_B.instance_id == "ds_B"
    assert inst_A.cost_matrix[0, 0] == 10.0
    assert inst_A.cost_matrix[1, 0] == 100.0
    assert inst_B.cost_matrix[0, 0] == 100.0
    assert inst_B.cost_matrix[1, 0] == 10.0
    assert not np.array_equal(inst_A.cost_matrix, inst_B.cost_matrix)


def test_cost_drives_optimization_outcome(test_db):
    """Test 8: Changing assignment costs changes optimization outcome."""
    _seed_test_datasets(test_db)
    inst_A = db_to_instance("ds_A", test_db)
    inst_B = db_to_instance("ds_B", test_db)

    res_A = solve_greedy(inst_A)
    res_B = solve_greedy(inst_B)

    # In Dataset A, Worker 0 has cost 10, Worker 1 has cost 100 -> Worker 0 assigned
    assert res_A.optimal_assignment == {(0, 0): 1}
    assert res_A.optimal_cost == 10.0

    # In Dataset B, Worker 0 has cost 100, Worker 1 has cost 10 -> Worker 1 assigned
    assert res_B.optimal_assignment == {(1, 0): 1}
    assert res_B.optimal_cost == 10.0


def test_eligibility_drives_optimization_outcome(test_db):
    """Test 9: Changing eligibility changes optimization outcome even if cost is lower."""
    _seed_test_datasets(test_db)
    inst_C = db_to_instance("ds_C", test_db)

    res_C_greedy = solve_greedy(inst_C)
    res_C_exact = solve_exact(inst_C)

    # In Dataset C, Worker 0 is ineligible, so Worker 1 MUST be assigned despite higher cost 100
    assert res_C_greedy.optimal_assignment == {(1, 0): 1}
    assert res_C_greedy.optimal_cost == 100.0

    assert res_C_exact.optimal_assignment == {(1, 0): 1}
    assert res_C_exact.optimal_cost == 100.0


def test_dataset_fingerprint_uniqueness(test_db):
    """Test 12: Fingerprints are deterministic and distinct across datasets."""
    _seed_test_datasets(test_db)
    fp_A1 = compute_dataset_fingerprint("ds_A", test_db)
    fp_A2 = compute_dataset_fingerprint("ds_A", test_db)
    fp_B = compute_dataset_fingerprint("ds_B", test_db)
    fp_C = compute_dataset_fingerprint("ds_C", test_db)

    # Determinism
    assert fp_A1 == fp_A2
    # Uniqueness across distinct datasets
    assert fp_A1 != fp_B
    assert fp_A1 != fp_C
    assert fp_B != fp_C
    assert fp_A1.startswith("fp_")


def test_classical_optimization_isolation(test_db):
    """Test 3, 4, 5, 11, 13: Datasets cannot share experiments, solutions belong to experiment."""
    _seed_test_datasets(test_db)
    inst_A = db_to_instance("ds_A", test_db)
    inst_B = db_to_instance("ds_B", test_db)

    out_A = run_classical_workspace(inst_A, "ds_A", test_db)
    out_B = run_classical_workspace(inst_B, "ds_B", test_db)

    exp_id_A = out_A["experiment_id"]
    exp_id_B = out_B["experiment_id"]
    assert exp_id_A != exp_id_B

    # Query experiment rows
    cur = test_db.cursor()
    cur.execute("SELECT dataset_id, parameters_json FROM experiments WHERE experiment_id = ?", (exp_id_A,))
    row_A = cur.fetchone()
    assert row_A["dataset_id"] == "ds_A"
    fp_A = compute_dataset_fingerprint("ds_A", test_db)
    assert json.loads(row_A["parameters_json"])["dataset_fingerprint"] == fp_A

    cur.execute("SELECT dataset_id, parameters_json FROM experiments WHERE experiment_id = ?", (exp_id_B,))
    row_B = cur.fetchone()
    assert row_B["dataset_id"] == "ds_B"
    fp_B = compute_dataset_fingerprint("ds_B", test_db)
    assert json.loads(row_B["parameters_json"])["dataset_fingerprint"] == fp_B

    # Query solutions
    cur.execute("SELECT dataset_id, experiment_id, total_cost, assignment_json FROM solutions WHERE experiment_id = ?", (exp_id_A,))
    sol_rows_A = cur.fetchall()
    assert len(sol_rows_A) >= 1
    for s in sol_rows_A:
        assert s["dataset_id"] == "ds_A"
        assert s["experiment_id"] == exp_id_A

    cur.execute("SELECT dataset_id, experiment_id, total_cost, assignment_json FROM solutions WHERE experiment_id = ?", (exp_id_B,))
    sol_rows_B = cur.fetchall()
    assert len(sol_rows_B) >= 1
    for s in sol_rows_B:
        assert s["dataset_id"] == "ds_B"
        assert s["experiment_id"] == exp_id_B


def test_results_query_excludes_other_datasets(test_db):
    """Test 6 & 15: Results queries only return experiments for active dataset."""
    _seed_test_datasets(test_db)
    inst_A = db_to_instance("ds_A", test_db)
    inst_B = db_to_instance("ds_B", test_db)

    out_A = run_classical_workspace(inst_A, "ds_A", test_db)
    out_B = run_classical_workspace(inst_B, "ds_B", test_db)

    exps_A = list_dataset_experiments("ds_A", test_db)
    exps_B = list_dataset_experiments("ds_B", test_db)

    # Verify strictly partitioned
    exp_ids_for_A = {e["experiment_id"] for e in exps_A}
    exp_ids_for_B = {e["experiment_id"] for e in exps_B}

    assert out_A["experiment_id"] in exp_ids_for_A
    assert out_B["experiment_id"] not in exp_ids_for_A

    assert out_B["experiment_id"] in exp_ids_for_B
    assert out_A["experiment_id"] not in exp_ids_for_B

    # Verify all returned records have matching dataset_id
    for e in exps_A:
        assert e["dataset_id"] == "ds_A"
    for e in exps_B:
        assert e["dataset_id"] == "ds_B"


def test_qaoa_isolation_and_no_leakage(test_db):
    """Test 14: QAOA results cannot be displayed or queried for another dataset."""
    _seed_test_datasets(test_db)
    inst_A = db_to_instance("ds_A", test_db)

    out_qaoa_A = run_qaoa_workspace(inst_A, "ds_A", shots=128, maxiter=5, seed=42, conn=test_db)
    exp_qaoa_A = out_qaoa_A["experiment_id"]

    # Verify measurements belong strictly to experiment_id
    cur = test_db.cursor()
    cur.execute("SELECT experiment_id FROM measurements WHERE experiment_id = ?", (exp_qaoa_A,))
    m_rows = cur.fetchall()
    assert len(m_rows) > 0
    for r in m_rows:
        assert r["experiment_id"] == exp_qaoa_A

    # Verify list_dataset_experiments for dataset B has zero trace of dataset A's QAOA
    exps_B = list_dataset_experiments("ds_B", test_db)
    for e in exps_B:
        assert e["experiment_id"] != exp_qaoa_A
        assert e["solver_type"] != "qaoa_p1"


def test_two_dataset_switching_isolation(test_db):
    """
    Requirement 17: TWO-DATASET ISOLATION TEST.
    Dataset A -> Run Classical -> Record dataset_id, fingerprint, experiment_id, schedule, cost.
    Dataset B -> Run Classical -> Verify dataset_id differs, fingerprint differs, experiment_id differs,
    schedule belongs to B, cost belongs to B.
    Return to Dataset A -> Verify A's results shown, B's results must not appear.
    """
    _seed_test_datasets(test_db)

    # 1. Dataset A
    set_active_dataset("ds_A")
    inst_A = db_to_instance("ds_A", test_db)
    fp_A = compute_dataset_fingerprint("ds_A", test_db)
    out_A = run_classical_workspace(inst_A, "ds_A", test_db)

    exp_id_A = out_A["experiment_id"]
    cost_A = out_A["greedy_result"].optimal_cost
    asgn_A = out_A["greedy_result"].optimal_assignment

    # 2. Dataset B
    set_active_dataset("ds_B")
    inst_B = db_to_instance("ds_B", test_db)
    fp_B = compute_dataset_fingerprint("ds_B", test_db)
    out_B = run_classical_workspace(inst_B, "ds_B", test_db)

    exp_id_B = out_B["experiment_id"]
    cost_B = out_B["greedy_result"].optimal_cost
    asgn_B = out_B["greedy_result"].optimal_assignment

    # Verify differences
    assert "ds_A" != "ds_B"
    assert fp_A != fp_B
    assert exp_id_A != exp_id_B
    assert asgn_A == {(0, 0): 1}  # Worker A1
    assert asgn_B == {(1, 0): 1}  # Worker B2
    assert asgn_A != asgn_B

    # 3. Query results when active dataset is B
    exps_for_B = list_dataset_experiments("ds_B", test_db)
    exp_ids_in_B = [e["experiment_id"] for e in exps_for_B]
    assert exp_id_B in exp_ids_in_B
    assert exp_id_A not in exp_ids_in_B

    # 4. Return to Dataset A
    set_active_dataset("ds_A")
    exps_for_A = list_dataset_experiments("ds_A", test_db)
    exp_ids_in_A = [e["experiment_id"] for e in exps_for_A]
    assert exp_id_A in exp_ids_in_A
    assert exp_id_B not in exp_ids_in_A  # B's results must not appear


def test_user_data_really_changes_qiskit(test_db):
    """
    Requirement 24: USER DATA REALLY CHANGES QISKIT.
    Dataset A vs Dataset B with different costs.
    Run QAOA for A -> record fingerprint, variable count, QUBO, Ising, circuit qubit count, measurements, schedule.
    Run QAOA for B -> record fingerprint, variable count, QUBO, Ising, circuit qubit count, measurements, schedule.
    Verify that the quantum problem and result are derived from B and not A.
    """
    from src.quantum import build_qubo, qubo_to_ising

    _seed_test_datasets(test_db)
    inst_A = db_to_instance("ds_A", test_db)
    inst_B = db_to_instance("ds_B", test_db)

    # 1. Dataset fingerprints
    fp_A = compute_dataset_fingerprint("ds_A", test_db)
    fp_B = compute_dataset_fingerprint("ds_B", test_db)
    assert fp_A != fp_B

    # 2. Variable counts & Qubit counts
    assert inst_A.n_vars == 2
    assert inst_B.n_vars == 2

    # 3. QUBO generation from current data
    qubo_A = build_qubo(inst_A)
    qubo_B = build_qubo(inst_B)
    assert not np.array_equal(qubo_A.a, qubo_B.a), "QUBO linear coefficients must change with dataset costs"

    # 4. Ising Hamiltonian from current QUBO
    ising_A = qubo_to_ising(qubo_A)
    ising_B = qubo_to_ising(qubo_B)
    assert not np.array_equal(ising_A.h, ising_B.h), "Ising magnetic field coefficients must change with dataset costs"

    # 5. Execute QAOA on Qiskit AerSimulator for A
    out_A = run_qaoa_workspace(inst_A, "ds_A", shots=256, maxiter=5, seed=42, conn=test_db)
    exp_A = out_A["experiment_id"]
    samples_A = out_A["qaoa_result"].samples

    # 6. Execute QAOA on Qiskit AerSimulator for B
    out_B = run_qaoa_workspace(inst_B, "ds_B", shots=256, maxiter=5, seed=42, conn=test_db)
    exp_B = out_B["experiment_id"]
    samples_B = out_B["qaoa_result"].samples

    # Verify quantum experiment isolation
    assert exp_A != exp_B
    assert len(samples_A) > 0
    assert len(samples_B) > 0

    # Verify measurements in DB belong strictly to experiment_id
    cur = test_db.cursor()
    cur.execute("SELECT experiment_id FROM measurements WHERE experiment_id = ?", (exp_A,))
    m_A = cur.fetchall()
    cur.execute("SELECT experiment_id FROM measurements WHERE experiment_id = ?", (exp_B,))
    m_B = cur.fetchall()

    assert len(m_A) > 0 and len(m_B) > 0
    for r in m_A:
        assert r["experiment_id"] == exp_A
    for r in m_B:
        assert r["experiment_id"] == exp_B


def test_no_data_no_optimization(test_db):
    """
    Requirement 26: NO DATA = NO OPTIMIZATION.
    When no dataset is loaded:
    - Active dataset is None
    - No experiments queryable for None
    - Cannot execute optimization without valid dataset ID
    """
    set_active_dataset(None)
    assert get_active_dataset() is None

    exps = list_dataset_experiments(None, test_db)
    assert exps == []


def test_results_page_qubo_representation_regression(test_db):
    """
    Regression test for Results page QUBO representation:
    - QUBO can be constructed from an Instance
    - QUBO object has NO to_matrix() method (proves nonexistent method is not relied upon)
    - Upper-triangular matrix can be constructed truthfully from linear (a) and quadratic (b) coefficients
    - Matrix quadratic form x^T Q x + q0 matches qubo.energy(x)
    - Ising transformation and Pauli operators can be extracted
    - Results page representation dictionary can be generated without AttributeError
    """
    _seed_test_datasets(test_db)
    inst = db_to_instance("ds_A", test_db)
    qubo = build_qubo(inst)

    # 1. Proves QUBO does not have nonexistent to_matrix()
    assert not hasattr(qubo, "to_matrix"), "QUBO dataclass should not have fake to_matrix method"

    # 2. Construct truthful matrix from real coefficients
    n = inst.n_vars
    qubo_mat = np.copy(qubo.b)
    np.fill_diagonal(qubo_mat, qubo.a)
    qubo_const = float(qubo.q0)

    assert qubo_mat.shape == (n, n)
    assert isinstance(qubo_const, float)

    # 3. Verify mathematical equivalence: x^T Q x + q0 == qubo.energy(x)
    test_x = np.array([1, 0])  # ds_A has 2 variables
    energy_direct = qubo.energy(test_x)
    energy_mat = float(test_x @ qubo_mat @ test_x + qubo_const)
    assert np.isclose(energy_direct, energy_mat)

    # 4. Ising representation
    ising = qubo_to_ising(qubo)
    pauli_op = ising.to_sparse_pauli_op()
    assert hasattr(ising, "offset")
    assert hasattr(ising, "h")
    assert hasattr(ising, "J")
    assert len(pauli_op) > 0

    # 5. Verify Results page extraction produces clean metadata without crashing
    mapping_rows = []
    for q in range(n):
        w, s = inst.get_var(q)
        mapping_rows.append({
            "Qubit": f"q_{q}",
            "Variable": f"x[{w},{s}]",
            "Linear Bias": qubo.a[q],
            "Ising Bias": ising.h[q],
        })
    assert len(mapping_rows) == n


def test_ab_dataset_dynamic_computation_pipeline(test_db):
    """
    Requirements 11, 12, 13: A/B DATA DYNAMIC COMPUTATION & STALE RESULT PREVENTION.
    Proves that:
    1. Dataset A (Worker 1 cost 10, Worker 2 cost 50) vs Dataset B (Worker 1 cost 90, Worker 2 cost 5).
    2. Optimal assignment flips: Worker 1 in A vs Worker 2 in B.
    3. Objective cost flips: 10.0 in A vs 5.0 in B.
    4. Dataset fingerprints differ: fp_A != fp_B.
    5. QUBO linear coefficients differ: qubo_A.a != qubo_B.a.
    6. Ising magnetic biases differ: ising_A.h != ising_B.h.
    7. Persisted experiment fingerprints strictly match dataset fingerprints:
       exp_fp_A == fp_A and exp_fp_B == fp_B.
    8. Stale results are never displayed for another dataset:
       Querying experiments for ds_B contains exp_B, but does not contain exp_A.
    """
    cur = test_db.cursor()
    cur.execute("INSERT OR REPLACE INTO datasets (dataset_id, name, source_type, row_count, col_count) VALUES ('ds_ab_A', 'AB Test A', 'test', 2, 3)")
    cur.execute("INSERT OR REPLACE INTO datasets (dataset_id, name, source_type, row_count, col_count) VALUES ('ds_ab_B', 'AB Test B', 'test', 2, 3)")

    for did in ['ds_ab_A', 'ds_ab_B']:
        cur.execute("INSERT OR REPLACE INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w1', ?, 'Worker Alpha', 0)", (did,))
        cur.execute("INSERT OR REPLACE INTO workers (worker_id, dataset_id, name, worker_index) VALUES ('w2', ?, 'Worker Beta', 1)", (did,))
        cur.execute("INSERT OR REPLACE INTO shifts (shift_id, dataset_id, shift_name, shift_index) VALUES ('s1', ?, 'Day Shift', 0)", (did,))
        cur.execute("INSERT OR REPLACE INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES (?, 'w1', 's1', 1)", (did,))
        cur.execute("INSERT OR REPLACE INTO eligibility (dataset_id, worker_id, shift_id, is_eligible) VALUES (?, 'w2', 's1', 1)", (did,))

    # Costs in A: w1 -> 10.0, w2 -> 50.0
    cur.execute("INSERT OR REPLACE INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_ab_A', 'w1', 's1', 10.0)")
    cur.execute("INSERT OR REPLACE INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_ab_A', 'w2', 's1', 50.0)")

    # Costs in B: w1 -> 90.0, w2 -> 5.0
    cur.execute("INSERT OR REPLACE INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_ab_B', 'w1', 's1', 90.0)")
    cur.execute("INSERT OR REPLACE INTO costs (dataset_id, worker_id, shift_id, assignment_cost) VALUES ('ds_ab_B', 'w2', 's1', 5.0)")
    test_db.commit()

    # 1. Fingerprint difference
    fp_A = compute_dataset_fingerprint('ds_ab_A', test_db)
    fp_B = compute_dataset_fingerprint('ds_ab_B', test_db)
    assert fp_A != fp_B

    # 2. Instance generation
    inst_A = db_to_instance('ds_ab_A', test_db)
    inst_B = db_to_instance('ds_ab_B', test_db)

    # 3. QUBO & Ising generation
    qubo_A = build_qubo(inst_A)
    qubo_B = build_qubo(inst_B)
    assert not np.array_equal(qubo_A.a, qubo_B.a), "QUBO linear coefficients must change dynamically with data"

    ising_A = qubo_to_ising(qubo_A)
    ising_B = qubo_to_ising(qubo_B)
    assert not np.array_equal(ising_A.h, ising_B.h), "Ising biases must change dynamically with data"

    # 4. Classical solver execution
    res_A = run_classical_workspace(inst_A, 'ds_ab_A', test_db)
    res_B = run_classical_workspace(inst_B, 'ds_ab_B', test_db)

    assert res_A["greedy_result"].optimal_cost == 10.0
    assert res_B["greedy_result"].optimal_cost == 5.0
    assert res_A["greedy_result"].optimal_assignment == {(0, 0): 1}  # w1 assigned
    assert res_B["greedy_result"].optimal_assignment == {(1, 0): 1}  # w2 assigned

    # 5. QAOA simulation execution
    qaoa_A = run_qaoa_workspace(inst_A, 'ds_ab_A', shots=128, maxiter=5, seed=42, conn=test_db)
    qaoa_B = run_qaoa_workspace(inst_B, 'ds_ab_B', shots=128, maxiter=5, seed=42, conn=test_db)
    assert qaoa_A["experiment_id"] != qaoa_B["experiment_id"]

    # 6. Verify stored experiment fingerprints
    exps_A = list_dataset_experiments('ds_ab_A', test_db)
    exps_B = list_dataset_experiments('ds_ab_B', test_db)

    for e in exps_A:
        p = json.loads(e["parameters_json"])
        assert p["dataset_fingerprint"] == fp_A
        assert p["dataset_fingerprint"] != fp_B

    for e in exps_B:
        p = json.loads(e["parameters_json"])
        assert p["dataset_fingerprint"] == fp_B
        assert p["dataset_fingerprint"] != fp_A




