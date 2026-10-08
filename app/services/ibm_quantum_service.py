"""
ShiftProof — IBM Quantum Platform Connection & Backend Discovery Service

Responsible ONLY for IBM Quantum Platform authentication, connection management,
and real backend discovery.
Never hardcodes tokens. Never exposes or logs token values.
Does NOT submit any quantum jobs in Phase 1.
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
