#!/usr/bin/env python3
"""
ShiftProof — IBM Quantum Hardware Execution CLI & Audit Tool

Allows running QAOA optimization directly on real physical IBM Quantum QPUs
or retrieving and auditing existing hardware job runs.

Usage:
    python scripts/run_ibm_quantum.py --status
    python scripts/run_ibm_quantum.py --retrieve-job db3jsqslf4us73c1f7j0
    python scripts/run_ibm_quantum.py --run --dataset data/sample/workforce_schedule_3x3.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path or sys.path[0] != str(_ROOT):
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(override=False)

from app.services import ibm_quantum_service as ibm_svc
from app.services.ingestion.file_detector import detect_file_type
from app.services.ingestion.structured_parser import parse_structured_file
from app.services.ingestion.schema_understanding import understand_schema
from app.services.ingestion.scheduling_classifier import classify_dataset
from app.services.ingestion.normalizer import normalize_dataset
from app.services.scheduling_service import normalized_to_instance, compute_dataframe_fingerprint
from src.classical import solve_exact, solve_greedy


def check_status():
    print("=" * 65)
    print("IBM QUANTUM PLATFORM STATUS & BACKEND DISCOVERY")
    print("=" * 65)
    status = ibm_svc.get_connection_status(force_refresh=True)
    print(f"Connection State:    {status['state']}")
    print(f"Message:             {status['message']}")
    if status['channel']:
        print(f"Runtime Channel:     {status['channel']}")
    print(f"Discovered Backends: {status['backend_count']}")

    if status["backends"]:
        print("\nAvailable Backends:")
        for b in status["backends"]:
            b_type = "Simulator" if b["simulator"] else "Real QPU"
            op = "Operational" if b["operational"] else "Offline"
            print(f"  • {b['name']:<18} | {b['num_qubits']} qubits | {b_type:<8} | {op:<11} | Queue: {b['pending_jobs']} jobs")
    return status


def retrieve_and_audit(job_id: str, dataset_path: Path):
    print("=" * 65)
    print(f"RETRIEVING IBM QUANTUM JOB: {job_id}")
    print("=" * 65)
    
    file_bytes = dataset_path.read_bytes()
    ftype, _ = detect_file_type(dataset_path.name, file_bytes)
    parsed = parse_structured_file(file_bytes, dataset_path.name, ftype)
    cands = understand_schema(parsed.df)
    cls = classify_dataset(parsed.df, cands)
    norm = normalize_dataset(parsed.df, cls.detected_mapping, cls.is_matrix, cls.matrix_shift_columns)
    inst = normalized_to_instance(norm, f"inst_{dataset_path.stem}", seed=42)
    
    exact_res = solve_exact(inst)
    print(f"Dataset Instance:    {inst.instance_id} ({inst.n_workers}w x {inst.n_shifts}s, {inst.n_vars} vars)")
    print(f"Exact Classical Min: ${exact_res.optimal_cost:.2f}")

    service, err = ibm_svc.connect()
    if not service:
        print(f"ERROR: Cannot connect to IBM Quantum Platform: {err}")
        sys.exit(1)

    job = ibm_svc.retrieve_job(job_id, service=service)
    print(f"Job Backend:         {job.backend().name if job.backend() else 'Unknown'}")
    print(f"Job Status:          {job.status()}")
    print(f"Creation Date:       {job.creation_date}")

    result = job.result()
    raw_counts = ibm_svc.extract_hardware_counts(result)
    audit = ibm_svc.audit_hardware_measurements(inst, raw_counts, exact_cost=exact_res.optimal_cost)

    print("\n" + "=" * 65)
    print("PHYSICAL HARDWARE MEASUREMENT AUDIT")
    print("=" * 65)
    print(f"Total Measured Shots:   {audit['total_shots']}")
    print(f"Unique States:          {audit['unique_states']}")
    print(f"Feasible Shots:         {audit['feasible_shots']} ({audit['feasibility_rate']:.2f}%)")
    print(f"Most Probable State:    |{audit['most_probable_state']}> (Count: {audit['most_probable_count']})")
    print(f"Best Feasible State:    |{audit['best_feasible_state']}> (Cost: ${audit['best_feasible_cost']:.2f})")
    print(f"Exact Classical Cost:   ${audit['exact_cost']:.2f}")
    print(f"Absolute Gap:           ${audit['abs_gap']:.2f}")
    print(f"Relative Gap:           {audit['rel_gap']:.2f}%")
    print(f"Optimal Probability:    {audit['optimal_probability']*100:.2f}%")

    if audit['best_feasible_assignment']:
        print(f"Selected Assignment:    {audit['best_feasible_assignment']}")

    return audit


def run_execution(dataset_path: Path, backend_name: str | None = None, shots: int = 1024):
    print("=" * 65)
    print(f"RUNNING QAOA ON IBM QUANTUM HARDWARE")
    print("=" * 65)

    file_bytes = dataset_path.read_bytes()
    ftype, _ = detect_file_type(dataset_path.name, file_bytes)
    parsed = parse_structured_file(file_bytes, dataset_path.name, ftype)
    cands = understand_schema(parsed.df)
    cls = classify_dataset(parsed.df, cands)
    norm = normalize_dataset(parsed.df, cls.detected_mapping, cls.is_matrix, cls.matrix_shift_columns)
    inst = normalized_to_instance(norm, f"inst_{dataset_path.stem}", seed=42)

    print(f"Dataset Instance:    {inst.instance_id} ({inst.n_workers}w x {inst.n_shifts}s, {inst.n_vars} vars)")

    def cb(status_str):
        print(f"  [QPU Status Update]: {status_str}")

    res = ibm_svc.execute_hardware_qaoa(
        instance=inst,
        backend_name=backend_name,
        shots=shots,
        wait_for_completion=True,
        status_callback=cb,
    )
    print(f"\nExecution Complete! Job ID: {res['job_id']} on {res['backend_name']}")
    if "audit" in res:
        audit = res["audit"]
        print(f"Best Feasible Cost:  ${audit['best_feasible_cost']:.2f}")
        print(f"Feasibility Rate:    {audit['feasibility_rate']:.2f}%")
        print(f"Absolute Gap:        ${audit['abs_gap']:.2f}")
    return res


def main():
    parser = argparse.ArgumentParser(description="ShiftProof IBM Quantum Execution CLI")
    parser.add_argument("--status", action="store_true", help="Check IBM Quantum connection and list backends")
    parser.add_argument("--retrieve-job", type=str, default=None, help="Retrieve and audit existing IBM job ID")
    parser.add_argument("--run", action="store_true", help="Execute QAOA on IBM Quantum hardware")
    parser.add_argument("--dataset", type=str, default="data/sample/workforce_schedule_3x3.csv", help="Dataset CSV path")
    parser.add_argument("--backend", type=str, default=None, help="Specific IBM backend name (default: least busy)")
    parser.add_argument("--shots", type=int, default=1024, help="Shots count (default: 1024)")
    args = parser.parse_args()

    ds_path = _ROOT / args.dataset

    if args.retrieve_job:
        retrieve_and_audit(args.retrieve_job, ds_path)
    elif args.run:
        run_execution(ds_path, backend_name=args.backend, shots=args.shots)
    else:
        check_status()


if __name__ == "__main__":
    main()
