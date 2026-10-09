"""
ShiftProof — IBM Quantum Connection & Discovery Unit Tests

Validates:
1. Safe handling when no credentials exist (reports NOT CONFIGURED without crashing).
2. Mocked successful authentication and backend discovery.
3. Mocked authentication / network failure handling.
4. Backend compatibility check (qubit capacity and operational status).
5. Zero credential exposure in returned data structures.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from app.services import ibm_quantum_service as ibm_svc


@pytest.fixture(autouse=True)
def reset_service_cache():
    """Reset module-level cache before each test."""
    ibm_svc._CACHED_SERVICE = None
    ibm_svc._CACHED_BACKENDS = None
    ibm_svc._LAST_ERROR = None
    yield
    ibm_svc._CACHED_SERVICE = None
    ibm_svc._CACHED_BACKENDS = None
    ibm_svc._LAST_ERROR = None


def test_no_credentials_reports_not_configured(monkeypatch):
    """When no token is provided, service must report NOT CONFIGURED and never crash."""
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    monkeypatch.setattr(ibm_svc, "get_token", lambda: None)

    assert not ibm_svc.is_configured()
    
    service, err = ibm_svc.connect()
    assert service is None
    assert err is not None
    assert "not configured" in err.lower()

    status = ibm_svc.get_connection_status()
    assert status["state"] == "NOT CONFIGURED"
    assert status["backend_count"] == 0
    assert status["backends"] == []


def test_mocked_successful_authentication_and_discovery(monkeypatch):
    """Verify backend discovery extracts required metadata safely."""
    monkeypatch.setattr(ibm_svc, "get_token", lambda: "mock_secret_token_abc")
    monkeypatch.setattr(ibm_svc, "is_configured", lambda: True)

    mock_backend_1 = MagicMock()
    mock_backend_1.name = "ibm_mock_brisbane"
    mock_backend_1.num_qubits = 127
    mock_backend_1.simulator = False
    mock_backend_1.status.return_value = MagicMock(operational=True, pending_jobs=2, status_msg="active")
    mock_backend_1.operation_names = ["rz", "sx", "cz", "measure"]

    mock_backend_2 = MagicMock()
    mock_backend_2.name = "ibm_mock_simulator"
    mock_backend_2.num_qubits = 32
    mock_backend_2.simulator = True
    mock_backend_2.status.return_value = MagicMock(operational=True, pending_jobs=0, status_msg="active")
    mock_backend_2.operation_names = ["cx", "u"]

    mock_service_instance = MagicMock()
    mock_service_instance.backends.return_value = [mock_backend_1, mock_backend_2]

    with patch("qiskit_ibm_runtime.QiskitRuntimeService", return_value=mock_service_instance):
        service, err = ibm_svc.connect(force_refresh=True)
        assert service is not None
        assert err is None

        backends = ibm_svc.list_backends(service=service, force_refresh=True)
        assert len(backends) == 2
        
        # Hardware backend should be sorted first
        assert backends[0]["name"] == "ibm_mock_brisbane"
        assert backends[0]["num_qubits"] == 127
        assert backends[0]["simulator"] is False
        assert backends[0]["operational"] is True
        assert backends[0]["pending_jobs"] == 2
        assert "cz" in backends[0]["supported_instructions"]

        status = ibm_svc.get_connection_status(force_refresh=True)
        assert status["state"] == "CONNECTED"
        assert status["backend_count"] == 2


def test_mocked_authentication_failure(monkeypatch):
    """Verify clean error capture when IBM API authentication fails."""
    monkeypatch.setattr(ibm_svc, "get_token", lambda: "invalid_token_123")
    monkeypatch.setattr(ibm_svc, "is_configured", lambda: True)

    with patch("qiskit_ibm_runtime.QiskitRuntimeService", side_effect=Exception("IBM Quantum Authentication Failed: 401 Unauthorized")):
        service, err = ibm_svc.connect(force_refresh=True)
        assert service is None
        assert err is not None
        assert "401" in err

        status = ibm_svc.get_connection_status(force_refresh=True)
        assert status["state"] == "CONNECTION FAILED"
        assert "401" in status["error"]
        assert status["backend_count"] == 0


def test_backend_compatibility_check():
    """Verify candidate backend capability against logical qubit counts."""
    # Scenario A: Compatible hardware
    meta_good = {
        "name": "ibm_mock_large",
        "num_qubits": 127,
        "operational": True,
        "pending_jobs": 1,
    }
    comp_good = ibm_svc.check_backend_compatibility(meta_good, logical_qubits=9)
    assert comp_good["compatible"] is True
    assert comp_good["status_text"] == "BACKEND COMPATIBLE"

    # Scenario B: Incompatible - insufficient physical qubits
    meta_small = {
        "name": "ibm_mock_tiny",
        "num_qubits": 5,
        "operational": True,
        "pending_jobs": 0,
    }
    comp_small = ibm_svc.check_backend_compatibility(meta_small, logical_qubits=9)
    assert comp_small["compatible"] is False
    assert "insufficient qubit capacity" in comp_small["status_text"].lower()

    # Scenario C: Incompatible - non-operational / maintenance
    meta_offline = {
        "name": "ibm_mock_down",
        "num_qubits": 127,
        "operational": False,
        "pending_jobs": 0,
        "status_msg": "maintenance",
    }
    comp_offline = ibm_svc.check_backend_compatibility(meta_offline, logical_qubits=9)
    assert comp_offline["compatible"] is False
    assert "non-operational" in comp_offline["status_text"].lower()


def test_token_never_exposed_in_status(monkeypatch):
    """Verify that credentials are never leaked in status dictionaries or messages."""
    fake_secret = "super_secret_token_12345"
    monkeypatch.setattr(ibm_svc, "get_token", lambda: fake_secret)
    monkeypatch.setattr(ibm_svc, "is_configured", lambda: True)

    status = ibm_svc.get_connection_status()
    # Serialize status to string and assert token is nowhere to be found
    status_str = str(status)
    assert fake_secret not in status_str


def test_ui_quantum_page_unconfigured(monkeypatch):
    """Verify Streamlit Quantum page renders NOT CONFIGURED cleanly without crashing."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ibm_svc, "get_token", lambda: None)
    monkeypatch.setattr(ibm_svc, "is_configured", lambda: False)

    at = AppTest.from_file("app/pages/03_quantum.py", default_timeout=10)
    at.run()
    assert len(at.exception) == 0


def test_ui_quantum_page_connected_mock(monkeypatch):
    """Verify Streamlit Quantum page renders CONNECTED state with backends selector cleanly."""
    from streamlit.testing.v1 import AppTest
    from app.services.dataset_service import list_datasets
    from app.database.db import get_db_connection

    mock_status = {
        "state": "CONNECTED",
        "message": "IBM Quantum Platform: CONNECTED (1 backends available)",
        "error": None,
        "channel": "ibm_cloud",
        "backend_count": 1,
        "backends": [
            {
                "name": "ibm_mock_test_backend",
                "num_qubits": 127,
                "simulator": False,
                "operational": True,
                "pending_jobs": 0,
                "status_msg": "active",
                "supported_instructions": ["cz", "rz", "sx"],
            }
        ],
    }

    monkeypatch.setattr(ibm_svc, "get_connection_status", lambda force_refresh=False: mock_status)

    at = AppTest.from_file("app/pages/03_quantum.py", default_timeout=10)
    conn = get_db_connection()
    datasets = list_datasets(conn)
    if datasets:
        at.session_state["active_dataset_id"] = datasets[0]["dataset_id"]
    at.run()
    assert len(at.exception) == 0

    metric_labels = [m.label for m in at.metric]
    metric_values = [m.value for m in at.metric]
    assert "IBM Quantum Hardware" in metric_labels
    idx = metric_labels.index("IBM Quantum Hardware")
    assert metric_values[idx] == "CONNECTED"
    assert "AerSimulator (Classical)" in metric_values


def test_audit_hardware_measurements_deterministic():
    """Verify hardware measurement audit correctly identifies feasible states, costs, and gaps."""
    from src.model import Instance
    import numpy as np

    cost_matrix = np.array([[10.0, 20.0], [15.0, 25.0]])
    eligibility = np.ones((2, 2), dtype=bool)
    inst = Instance(
        n_workers=2,
        n_shifts=2,
        cost_matrix=cost_matrix,
        eligibility=eligibility,
        instance_id="test_inst_2x2",
        seed=42,
    )
    # Bit ordering in decode_bitstring: reversed index
    # Assignment W0->S0, W1->S1 means x_{0,0}=1, x_{1,1}=1.
    # Total cost = 10 + 25 = 35.
    raw_counts = {
        "1001": 500,  # Feasible
        "0000": 300,  # Infeasible
        "1111": 200,  # Infeasible
    }
    audit = ibm_svc.audit_hardware_measurements(inst, raw_counts, exact_cost=35.0)

    assert audit["total_shots"] == 1000
    assert audit["unique_states"] == 3
    assert audit["feasible_shots"] == 500
    assert audit["feasibility_rate"] == 50.0
    assert audit["most_probable_state"] == "1001"
    assert audit["best_feasible_cost"] == 35.0
    assert audit["exact_cost"] == 35.0
    assert audit["abs_gap"] == 0.0
    assert audit["rel_gap"] == 0.0
    assert audit["optimal_probability"] == 0.5


def test_extract_hardware_counts_from_result():
    """Verify counts extraction from SamplerV2 pub result structure."""
    mock_pub = MagicMock()
    mock_pub.data.meas.get_counts.return_value = {"010": 40, "101": 60}
    mock_result = [mock_pub]

    counts = ibm_svc.extract_hardware_counts(mock_result)
    assert counts == {"010": 40, "101": 60}


def test_poll_job_completion():
    """Verify poll_job returns result when status reaches DONE."""
    mock_job = MagicMock()
    mock_job.job_id.return_value = "job_test_123"
    mock_job.status.side_effect = ["QUEUED", "RUNNING", "DONE"]
    mock_job.result.return_value = "mock_result_data"

    res = ibm_svc.poll_job(mock_job, timeout_seconds=10, poll_interval=0.01)
    assert res == "mock_result_data"


def test_poll_job_failure():
    """Verify poll_job raises RuntimeError on ERROR status."""
    mock_job = MagicMock()
    mock_job.job_id.return_value = "job_test_fail"
    mock_job.status.return_value = "ERROR"
    mock_job.error_message.return_value = "Internal QPU calibration error"

    with pytest.raises(RuntimeError) as exc_info:
        ibm_svc.poll_job(mock_job, timeout_seconds=5, poll_interval=0.01)
    assert "calibration error" in str(exc_info.value)


