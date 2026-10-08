import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path or sys.path[0] != str(_ROOT):
    sys.path.insert(0, str(_ROOT))

import json
from dotenv import load_dotenv
load_dotenv(override=False)

from app.services.ingestion.file_detector import detect_file_type
from app.services.ingestion.structured_parser import parse_structured_file
from app.services.ingestion.schema_understanding import understand_schema
from app.services.ingestion.scheduling_classifier import classify_dataset
from app.services.ingestion.normalizer import normalize_dataset
from app.services.scheduling_service import normalized_to_instance, compute_dataframe_fingerprint
from src.classical import solve_exact, solve_greedy
from src.model import decode_bitstring, check_constraints, assignment_cost
from src.quantum import build_qubo, qubo_to_ising, build_qaoa_circuit
from app.services import ibm_quantum_service as ibm_svc
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys(item) for item in obj]
    return obj

def main():
    data_path = Path("data/sample/workforce_schedule_3x3.csv")
    file_bytes = data_path.read_bytes()
    file_type, file_cat = detect_file_type(data_path.name, file_bytes)
    parsed = parse_structured_file(file_bytes, data_path.name, file_type)
    df_raw = parsed.df
    cands = understand_schema(df_raw)
    cls = classify_dataset(df_raw, cands)
    norm = normalize_dataset(df_raw, cls.detected_mapping, cls.is_matrix, cls.matrix_shift_columns)
    inst = normalized_to_instance(norm, f"inst_{data_path.stem}", seed=42)
    df_canon = norm.df_normalized
    dataset_fp = compute_dataframe_fingerprint(df_canon)

    exact_res = solve_exact(inst)
    greedy_res = solve_greedy(inst)

    qubo = build_qubo(inst)
    ising = qubo_to_ising(qubo)
    qc, gamma_params, beta_params = build_qaoa_circuit(ising, p=1)
    opt_gamma = 0.392699
    opt_beta = 0.785398
    bound_qc = qc.assign_parameters({gamma_params[0]: opt_gamma, beta_params[0]: opt_beta})

    logical_depth = bound_qc.depth()
    logical_ops = dict(bound_qc.count_ops())
    logical_cx = logical_ops.get("cx", 0)

    service, _ = ibm_svc.connect()
    job_id = "db3jsqslf4us73c1f7j0"
    job = service.job(job_id)
    backend = job.backend()

    # Transpilation telemetry
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
    transpiled_qc = pm.run(bound_qc)
    transpiled_depth = transpiled_qc.depth()
    transpiled_ops = dict(transpiled_qc.count_ops())
    transpiled_2q = transpiled_ops.get("cz", 0) + transpiled_ops.get("ecr", 0) + transpiled_ops.get("cx", 0)
    
    layout = transpiled_qc.layout
    active_physical = []
    if layout and hasattr(layout, "initial_layout"):
        v_bits = layout.initial_layout.get_virtual_bits()
        active_physical = sorted([phys for virt, phys in v_bits.items() if "ancilla" not in str(virt).lower()])

    pub_res = job.result()[0]
    raw_counts = pub_res.data.meas.get_counts()
    total_shots = sum(raw_counts.values())

    audited_states = []
    feasible_count = 0
    best_feasible_state = None
    best_feasible_cost = float("inf")
    best_feasible_assignment = None
    most_probable_state = None
    max_count = -1

    for bitstring, count in raw_counts.items():
        prob = count / total_shots
        if count > max_count:
            max_count = count
            most_probable_state = bitstring

        assignment = decode_bitstring(inst, bitstring)
        is_feas, viol = check_constraints(inst, assignment)
        cost = assignment_cost(inst, assignment) if is_feas else None

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
            "violations": stringify_keys(viol),
            "assignment": {f"W{w}->S{s}": val for (w, s), val in assignment.items() if val == 1}
        })

    audited_states.sort(key=lambda x: x["count"], reverse=True)
    feasibility_rate = (feasible_count / total_shots) * 100.0

    opt_prob = 0.0
    for s in audited_states:
        if s["feasible"] and s["cost"] is not None and abs(s["cost"] - exact_res.optimal_cost) < 1e-6:
            opt_prob += s["probability"]

    abs_gap = (best_feasible_cost - exact_res.optimal_cost) if best_feasible_cost != float("inf") else None
    rel_gap = (abs_gap / exact_res.optimal_cost * 100.0) if (abs_gap is not None and exact_res.optimal_cost > 0) else None

    # Determine match message
    if abs_gap is not None and abs_gap == 0.0:
        obs_msg = "IBM hardware observed the classical optimum."
    elif best_feasible_cost != float("inf"):
        obs_msg = "IBM hardware observed a feasible solution but not the classical optimum."
    else:
        obs_msg = "No feasible IBM hardware sample observed."

    out_data = {
        "dataset_name": data_path.name,
        "dataset_fingerprint": dataset_fp,
        "instance_id": inst.instance_id,
        "n_workers": inst.n_workers,
        "n_shifts": inst.n_shifts,
        "n_vars": inst.n_vars,
        "backend_name": backend.name,
        "backend_num_qubits": backend.num_qubits,
        "job_id": job_id,
        "shots": total_shots,
        "opt_gamma": opt_gamma,
        "opt_beta": opt_beta,
        "logical_depth": logical_depth,
        "logical_cx": logical_cx,
        "logical_ops": logical_ops,
        "transpiled_depth": transpiled_depth,
        "transpiled_2q": transpiled_2q,
        "transpiled_ops": transpiled_ops,
        "active_physical_qubits": active_physical,
        "exact_cost": exact_res.optimal_cost,
        "greedy_cost": greedy_res.optimal_cost,
        "feasible_shots": feasible_count,
        "feasibility_rate": feasibility_rate,
        "most_probable_state": most_probable_state,
        "most_probable_count": max_count,
        "best_feasible_state": best_feasible_state,
        "best_feasible_cost": best_feasible_cost,
        "best_feasible_assignment": {f"W{w}->S{s}": val for (w, s), val in best_feasible_assignment.items() if val == 1} if best_feasible_assignment else None,
        "abs_gap": abs_gap,
        "rel_gap": rel_gap,
        "optimal_probability": opt_prob,
        "observed_message": obs_msg,
        "raw_counts": raw_counts,
        "top_states": audited_states[:15]
    }

    Path("scratch/ibm_qpu_execution_result.json").write_text(json.dumps(out_data, indent=2))
    print("SUCCESS: JSON saved to scratch/ibm_qpu_execution_result.json")

if __name__ == "__main__":
    main()
