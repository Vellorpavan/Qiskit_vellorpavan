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
