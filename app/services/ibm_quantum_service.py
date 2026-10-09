"""
ShiftProof — IBM Quantum Platform Connection & Backend Discovery Service

Responsible for IBM Quantum Platform authentication, connection management,
real backend discovery, circuit transpilation, SamplerV2 execution,
job retrieval, and independent physical measurement auditing.
Never hardcodes tokens. Never exposes or logs token values.
Preserves existing classical AerSimulator workflows untouched.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Safely load environment variables from .env if present
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass


# Global in-memory cache to avoid repeated slow network handshakes
_CACHED_SERVICE: Any = None
_CACHED_BACKENDS: Optional[List[Dict[str, Any]]] = None
_LAST_ERROR: Optional[str] = None


def get_token() -> Optional[str]:
    """Return the configured IBM token from environment, or None."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if token and token.strip():
        return token.strip()
    return None


def get_instance() -> Optional[str]:
    """Return the configured IBM instance/CRN from environment, or None."""
    inst = os.environ.get("IBM_QUANTUM_INSTANCE")
    if inst and inst.strip():
        return inst.strip()
    return None


def get_channel() -> str:
    """Determine the runtime channel ('ibm_cloud' or 'ibm_quantum')."""
    channel = os.environ.get("IBM_QUANTUM_CHANNEL")
    if channel and channel.strip():
        return channel.strip()
    inst = get_instance()
    if inst and inst.startswith("crn:"):
        return "ibm_cloud"
    return "ibm_quantum"


def is_configured() -> bool:
    """
    Check whether IBM Quantum credentials are present in the environment.
    Never exposes or logs the token.
    """
    return get_token() is not None


def connect(force_refresh: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """
    Connect to IBM Quantum Platform using QiskitRuntimeService.
    
    Returns:
        (service, None) on success, or (None, error_message) on failure.
    """
    global _CACHED_SERVICE, _LAST_ERROR

    if not is_configured():
        _LAST_ERROR = "IBM_QUANTUM_TOKEN environment variable is not configured."
        return None, _LAST_ERROR

    if _CACHED_SERVICE is not None and not force_refresh:
        return _CACHED_SERVICE, None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        token = get_token()
        channel = get_channel()
        instance = get_instance()

        kwargs: Dict[str, Any] = {
            "channel": channel,
            "token": token,
        }
        if instance:
            kwargs["instance"] = instance

        service = QiskitRuntimeService(**kwargs)
        _CACHED_SERVICE = service
        _LAST_ERROR = None
        return service, None
    except Exception as exc:
        _CACHED_SERVICE = None
        _LAST_ERROR = f"{type(exc).__name__}: {str(exc)}"
        return None, _LAST_ERROR


def list_backends(
    service: Optional[Any] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Safely discover real backends available to the authenticated IBM account.
    
    Queries actual backend metadata:
    - backend name
    - number of qubits
    - simulator vs hardware flag
    - operational status
    - pending jobs
    - supported instructions
    
    Returns an empty list if not configured or connection failed.
    """
    global _CACHED_BACKENDS

    if _CACHED_BACKENDS is not None and not force_refresh:
        return _CACHED_BACKENDS

    if service is None:
        service, err = connect(force_refresh=force_refresh)
        if service is None:
            return []

    try:
        raw_backends = service.backends()
        discovered: List[Dict[str, Any]] = []

        for b in raw_backends:
            name = getattr(b, "name", "unknown")
            num_qubits = getattr(b, "num_qubits", 0)
            simulator = getattr(b, "simulator", False)

            operational = True
            pending_jobs = 0
            status_msg = "active"
            try:
                st = b.status()
                operational = getattr(st, "operational", True)
                pending_jobs = getattr(st, "pending_jobs", 0)
                status_msg = getattr(st, "status_msg", "active")
            except Exception:
                pass

            # Extract supported operations / basis gates
            instructions: List[str] = []
            if hasattr(b, "operation_names"):
                try:
                    instructions = list(b.operation_names)
                except Exception:
                    pass
            elif hasattr(b, "target") and b.target:
                try:
                    instructions = list(b.target.operation_names)
                except Exception:
                    pass

            discovered.append({
                "name": name,
                "num_qubits": num_qubits,
                "simulator": simulator,
                "operational": operational,
                "pending_jobs": pending_jobs,
                "status_msg": status_msg,
                "supported_instructions": instructions,
            })

        # Sort: hardware first, then by qubit count desc, then name
        discovered.sort(
            key=lambda x: (x["simulator"], -x["num_qubits"], x["name"])
        )
        _CACHED_BACKENDS = discovered
        return discovered
    except Exception as exc:
        global _LAST_ERROR
        _LAST_ERROR = f"Failed to list backends: {type(exc).__name__}: {str(exc)}"
        return []


def get_backend(name: str, service: Optional[Any] = None) -> Optional[Any]:
    """
    Retrieve a specific backend instance by name.
    """
    if service is None:
        service, _ = connect()
        if service is None:
            return None

    try:
        return service.backend(name)
    except Exception:
        return None


def check_backend_compatibility(
    backend_meta: Dict[str, Any],
    logical_qubits: int,
) -> Dict[str, Any]:
    """
    Verify whether the candidate IBM backend satisfies physical requirements
    for the active dataset instance (logical_qubits = instance.n_vars).
    
    Returns:
        {
            "compatible": bool,
            "status_text": str,
            "reason": str,
            "logical_qubits": int,
            "backend_qubits": int,
        }
    """
    backend_name = backend_meta.get("name", "Unknown Backend")
    backend_qubits = backend_meta.get("num_qubits", 0)
    operational = backend_meta.get("operational", True)

    if backend_qubits < logical_qubits:
        return {
            "compatible": False,
            "status_text": "BACKEND INCOMPATIBLE — insufficient qubit capacity",
            "reason": (
                f"Problem requires {logical_qubits} logical qubits, but "
                f"'{backend_name}' only has {backend_qubits} physical qubits."
            ),
            "logical_qubits": logical_qubits,
            "backend_qubits": backend_qubits,
        }

    if not operational:
        return {
            "compatible": False,
            "status_text": "BACKEND UNAVAILABLE — non-operational status",
            "reason": f"'{backend_name}' is currently offline or under maintenance ({backend_meta.get('status_msg', 'inactive')}).",
            "logical_qubits": logical_qubits,
            "backend_qubits": backend_qubits,
        }

    return {
        "compatible": True,
        "status_text": "BACKEND COMPATIBLE",
        "reason": (
            f"'{backend_name}' has {backend_qubits} qubits (>= {logical_qubits} logical required) "
            f"and is operational (pending queue: {backend_meta.get('pending_jobs', 0)})."
        ),
        "logical_qubits": logical_qubits,
        "backend_qubits": backend_qubits,
    }


def get_connection_status(force_refresh: bool = False) -> Dict[str, Any]:
    """
    High-level connection inspection for UI and diagnostic reporting.
    
    States:
    - 'NOT CONFIGURED': No IBM_QUANTUM_TOKEN environment variable.
    - 'CONNECTED': Successfully authenticated with IBM Quantum Platform.
    - 'CONNECTION FAILED': Token configured, but network/auth failed.
    """
    if not is_configured():
        return {
            "state": "NOT CONFIGURED",
            "message": "IBM Quantum Platform: NOT CONFIGURED (IBM_QUANTUM_TOKEN not set)",
            "error": None,
            "channel": None,
            "backend_count": 0,
            "backends": [],
        }

    service, err = connect(force_refresh=force_refresh)
    if service is None or err is not None:
        return {
            "state": "CONNECTION FAILED",
            "message": f"IBM Quantum Platform: CONNECTION FAILED ({err})",
            "error": err,
            "channel": get_channel(),
            "backend_count": 0,
            "backends": [],
        }

    backends = list_backends(service=service, force_refresh=force_refresh)
    return {
        "state": "CONNECTED",
        "message": f"IBM Quantum Platform: CONNECTED ({len(backends)} backends available)",
        "error": None,
        "channel": get_channel(),
        "backend_count": len(backends),
        "backends": backends,
    }


def transpile_for_hardware(
    circuit: Any,
    backend: Any,
    optimization_level: int = 3,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Transpile a logical QAOA quantum circuit to target IBM physical QPU.
    
    Returns:
        (transpiled_qc, telemetry_dict)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed,
    )
    transpiled_qc = pm.run(circuit)

    depth = transpiled_qc.depth()
    ops = dict(transpiled_qc.count_ops())
    two_q_count = ops.get("cz", 0) + ops.get("ecr", 0) + ops.get("cx", 0)

    active_physical: List[int] = []
    layout = getattr(transpiled_qc, "layout", None)
    if layout and hasattr(layout, "initial_layout"):
        v_bits = layout.initial_layout.get_virtual_bits()
        active_physical = sorted([
            phys for virt, phys in v_bits.items()
            if "ancilla" not in str(virt).lower()
        ])

    telemetry = {
        "transpiled_depth": depth,
        "transpiled_ops": ops,
        "transpiled_2q_gates": two_q_count,
        "active_physical_qubits": active_physical,
        "physical_qubit_count": len(active_physical),
        "total_qubits_on_chip": getattr(backend, "num_qubits", transpiled_qc.num_qubits),
    }
    return transpiled_qc, telemetry


def submit_hardware_sampler(
    transpiled_circuit: Any,
    backend: Any,
    shots: int = 1024,
) -> Any:
    """
    Submit a transpiled quantum circuit to an IBM backend using SamplerV2.
    
    Returns:
        job: IBM Runtime Job object
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled_circuit], shots=shots)
    return job


def poll_job(
    job: Any,
    timeout_seconds: int = 600,
    poll_interval: float = 5.0,
    status_callback: Optional[Any] = None,
) -> Any:
    """
    Poll an IBM Quantum job until completion or timeout.
    """
    import time
    start = time.time()
    last_status = None

    while True:
        status = job.status()
        status_str = str(status)
        if status_str != last_status:
            if status_callback:
                status_callback(status_str)
            last_status = status_str

        if status_str in ["DONE", "JobStatus.DONE"]:
            return job.result()
        elif status_str in ["ERROR", "CANCELLED", "JobStatus.ERROR", "JobStatus.CANCELLED"]:
            err_msg = getattr(job, "error_message", lambda: "Unknown error")()
            raise RuntimeError(f"Job {job.job_id()} failed with status {status_str}: {err_msg}")

        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Job {job.job_id()} timed out after {timeout_seconds}s (last status: {status_str}).")

        time.sleep(poll_interval)


def extract_hardware_counts(result: Any) -> Dict[str, int]:
    """
    Extract measurement counts dictionary from SamplerV2 result.
    """
    pub_res = result[0]
    data = getattr(pub_res, "data", None)
    meas = getattr(data, "meas", None) or getattr(data, "c", None)
    if meas is None:
        raise ValueError("No measurement register found in execution result.")
    return meas.get_counts()


def retrieve_job(job_id: str, service: Optional[Any] = None) -> Any:
    """
    Retrieve an existing job from IBM Quantum Platform by job ID.
    """
    if service is None:
        service, err = connect()
        if service is None:
            raise RuntimeError(f"Cannot retrieve job: {err}")
    return service.job(job_id)


def audit_hardware_measurements(
    instance: Any,
    raw_counts: Dict[str, int],
    exact_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Deterministically decode, audit, and benchmark real physical QPU measurement counts.
    """
    from src.model import decode_bitstring, check_constraints, assignment_cost

    total_shots = sum(raw_counts.values())
    audited_states: List[Dict[str, Any]] = []
    feasible_count = 0
    best_feasible_state = None
    best_feasible_cost = float("inf")
    best_feasible_assignment = None
    most_probable_state = None
    max_count = -1

    for bitstring, count in raw_counts.items():
        prob = count / total_shots if total_shots > 0 else 0.0
        if count > max_count:
            max_count = count
            most_probable_state = bitstring

        assignment = decode_bitstring(instance, bitstring)
        is_feas, viol = check_constraints(instance, assignment)
        cost = assignment_cost(instance, assignment) if is_feas else None

        if is_feas:
            feasible_count += count
            if cost is not None and cost < best_feasible_cost:
                best_feasible_cost = cost
                best_feasible_state = bitstring
                best_feasible_assignment = assignment

        audited_states.append({
            "bitstring": bitstring,
            "count": count,
            "probability": prob,
            "feasible": is_feas,
            "cost": cost,
            "violations": viol,
            "assignment": {f"W{w}->S{s}": val for (w, s), val in assignment.items() if val == 1}
        })

    audited_states.sort(key=lambda x: x["count"], reverse=True)
    feasibility_rate = (feasible_count / total_shots * 100.0) if total_shots > 0 else 0.0

    opt_prob = 0.0
    if exact_cost is not None:
        for s in audited_states:
            if s["feasible"] and s["cost"] is not None and abs(s["cost"] - exact_cost) < 1e-6:
                opt_prob += s["probability"]

    abs_gap = (best_feasible_cost - exact_cost) if (exact_cost is not None and best_feasible_cost != float("inf")) else None
    rel_gap = (abs_gap / exact_cost * 100.0) if (abs_gap is not None and exact_cost is not None and exact_cost > 0) else None

    return {
        "total_shots": total_shots,
        "unique_states": len(raw_counts),
        "feasible_shots": feasible_count,
        "feasibility_rate": feasibility_rate,
        "most_probable_state": most_probable_state,
        "most_probable_count": max_count,
        "best_feasible_state": best_feasible_state,
        "best_feasible_cost": best_feasible_cost if best_feasible_cost != float("inf") else None,
        "best_feasible_assignment": best_feasible_assignment,
        "exact_cost": exact_cost,
        "abs_gap": abs_gap,
        "rel_gap": rel_gap,
        "optimal_probability": opt_prob,
        "audited_states": audited_states,
    }


def execute_hardware_qaoa(
    instance: Any,
    backend_name: Optional[str] = None,
    shots: int = 1024,
    opt_gamma: float = 0.392699,
    opt_beta: float = 0.785398,
    optimization_level: int = 3,
    wait_for_completion: bool = True,
    timeout_seconds: int = 600,
    status_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    End-to-end QAOA execution on real physical IBM Quantum hardware.
    
    Formulates QUBO/Ising Hamiltonian, synthesizes logical QAOA circuit,
    transpiles for the selected backend, submits SamplerV2 job, optionally polls
    for physical completion, and produces deterministic audited measurement telemetry.
    """
    from src.quantum import build_qubo, qubo_to_ising, build_qaoa_circuit
    from src.classical import solve_exact

    service, err = connect()
    if not service:
        raise RuntimeError(f"IBM Quantum connection failed: {err}")

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)

    qubo = build_qubo(instance)
    ising = qubo_to_ising(qubo)
    qc, gamma_params, beta_params = build_qaoa_circuit(ising, p=1)
    bound_qc = qc.assign_parameters({gamma_params[0]: opt_gamma, beta_params[0]: opt_beta})

    transpiled_qc, telemetry = transpile_for_hardware(
        circuit=bound_qc,
        backend=backend,
        optimization_level=optimization_level,
    )

    exact_res = solve_exact(instance) if instance.n_vars <= 16 else None
    exact_cost = exact_res.optimal_cost if (exact_res and exact_res.feasible) else None

    job = submit_hardware_sampler(transpiled_qc, backend, shots=shots)
    job_id = job.job_id()

    execution_payload: Dict[str, Any] = {
        "job_id": job_id,
        "backend_name": backend.name,
        "backend_qubits": backend.num_qubits,
        "shots": shots,
        "logical_qubits": bound_qc.num_qubits,
        "logical_depth": bound_qc.depth(),
        "transpilation_telemetry": telemetry,
        "exact_cost": exact_cost,
        "status": str(job.status()),
    }

    if wait_for_completion:
        result = poll_job(
            job=job,
            timeout_seconds=timeout_seconds,
            status_callback=status_callback,
        )
        raw_counts = extract_hardware_counts(result)
        audit_data = audit_hardware_measurements(
            instance=instance,
            raw_counts=raw_counts,
            exact_cost=exact_cost,
        )
        execution_payload.update({
            "status": "DONE",
            "audit": audit_data,
            "raw_counts": raw_counts,
        })

    return execution_payload

