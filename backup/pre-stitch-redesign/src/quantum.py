"""
ShiftProof — QUBO, Ising, and QAOA Construction

QUBO formulation with penalty terms, Ising conversion, QAOA circuit.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler

from src.model import Instance, check_constraints, assignment_cost, decode_bitstring, get_all_bitstrings, encode_assignment


@dataclass(frozen=True)
class QUBO:
    """QUBO representation: Q(x) = q0 + sum a_i x_i + sum b_ij x_i x_j"""
    q0: float
    a: np.ndarray  # linear coefficients, shape (n_vars,)
    b: np.ndarray  # quadratic coefficients, shape (n_vars, n_vars), upper triangular
    instance: Instance
    A: float  # shift coverage penalty
    B: float  # worker conflict penalty
    x_ref_cost: float  # cost of reference solution

    def energy(self, x: np.ndarray) -> float:
        """Compute QUBO energy for binary vector x."""
        linear = np.dot(self.a, x)
        quadratic = 0.0
        n = len(x)
        for i in range(n):
            for j in range(i + 1, n):
                quadratic += self.b[i, j] * x[i] * x[j]
        return self.q0 + linear + quadratic

    def energy_from_bitstring(self, bitstring: str) -> float:
        """Compute QUBO energy from bitstring."""
        x = np.array([int(bitstring[-(i + 1)]) for i in range(self.instance.n_vars)])
        return self.energy(x)


@dataclass(frozen=True)
class Ising:
    """Ising representation: H = offset + sum h_i Z_i + sum J_ij Z_i Z_j"""
    offset: float
    h: np.ndarray  # shape (n_vars,)
    J: np.ndarray  # shape (n_vars, n_vars), upper triangular
    instance: Instance

    def energy(self, z: np.ndarray) -> float:
        """Compute Ising energy for spin vector z (±1)."""
        linear = np.dot(self.h, z)
        quadratic = 0.0
        n = len(z)
        for i in range(n):
            for j in range(i + 1, n):
                quadratic += self.J[i, j] * z[i] * z[j]
        return self.offset + linear + quadratic

    def energy_from_bitstring(self, bitstring: str) -> float:
        """Compute Ising energy from bitstring (0/1 -> +1/-1)."""
        # x = (1 - z)/2  =>  z = 1 - 2x
        # bitstring '0' -> x=0 -> z=+1
        # bitstring '1' -> x=1 -> z=-1
        z = np.array([1 - 2 * int(bitstring[-(i + 1)]) for i in range(self.instance.n_vars)])
        return self.energy(z)

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Convert to Qiskit SparsePauliOp for QAOA."""
        n = self.instance.n_vars
        pauli_list = []
        coeffs = []

        # Offset term (identity)
        pauli_list.append("I" * n)
        coeffs.append(self.offset)

        # Single-qubit Z terms
        for i in range(n):
            if abs(self.h[i]) > 1e-12:
                pauli = ["I"] * n
                pauli[-(i + 1)] = "Z"  # qubit 0 is rightmost
                pauli_list.append("".join(pauli))
                coeffs.append(self.h[i])

        # Two-qubit ZZ terms
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.J[i, j]) > 1e-12:
                    pauli = ["I"] * n
                    pauli[-(i + 1)] = "Z"
                    pauli[-(j + 1)] = "Z"
                    pauli_list.append("".join(pauli))
                    coeffs.append(self.J[i, j])

        return SparsePauliOp(pauli_list, coeffs=np.array(coeffs))


def build_qubo(instance: Instance, A: float = None, B: float = None, delta: float = 1.0) -> QUBO:
    """
    Build QUBO for the scheduling problem.

    Q(x) = C(x) + A * sum_s (sum_w x[w,s] - 1)^2 + B * sum_w sum_{s<t} x[w,s] x[w,t]

    For binary variables: (sum_i x_i - 1)^2 = 1 - sum_i x_i + 2 sum_{i<j} x_i x_j

    If A, B not provided, use safe penalty: A = B = C(x_ref) + delta
    where x_ref is the greedy solution (or exact if greedy fails).
    """
    from src.classical import solve_greedy, solve_exact

    n_vars = instance.n_vars

    # Get reference solution for penalty calibration
    greedy_result = solve_greedy(instance)
    if greedy_result.feasible:
        x_ref_cost = greedy_result.optimal_cost
    else:
        exact_result = solve_exact(instance)
        if exact_result.feasible:
            x_ref_cost = exact_result.optimal_cost
        else:
            # Instance is infeasible - this shouldn't happen as Instance constructor validates
            x_ref_cost = 0.0

    if A is None:
        A = x_ref_cost + delta
    if B is None:
        B = x_ref_cost + delta

    # Initialize QUBO coefficients
    q0 = 0.0
    a = np.zeros(n_vars)
    b = np.zeros((n_vars, n_vars))

    # 1. Assignment cost C(x) = sum c[w,s] x[w,s]
    for q in range(n_vars):
        w, s = instance.get_var(q)
        a[q] += instance.get_cost(w, s)

    # 2. Shift coverage penalty: A * sum_s (sum_w x[w,s] - 1)^2
    # For each shift, collect qubits for eligible workers
    for s in range(instance.n_shifts):
        shift_qubits = []
        for w in range(instance.n_workers):
            if instance.is_eligible(w, s):
                q = instance.get_qubit(w, s)
                if q is not None:
                    shift_qubits.append(q)

        if len(shift_qubits) == 0:
            # Should not happen - Instance constructor validates
            continue

        # (sum x_i - 1)^2 = 1 - sum x_i + 2 sum_{i<j} x_i x_j
        q0 += A * 1.0
        for q in shift_qubits:
            a[q] += A * (-1.0)
        for i_idx, q_i in enumerate(shift_qubits):
            for q_j in shift_qubits[i_idx + 1:]:
                b[q_i, q_j] += A * 2.0

    # 3. Worker conflict penalty: B * sum_w sum_{s<t} x[w,s] x[w,t]
    for w in range(instance.n_workers):
        worker_qubits = []
        for s in range(instance.n_shifts):
            if instance.is_eligible(w, s):
                q = instance.get_qubit(w, s)
                if q is not None:
                    worker_qubits.append(q)

        for i_idx, q_i in enumerate(worker_qubits):
            for q_j in worker_qubits[i_idx + 1:]:
                b[q_i, q_j] += B * 1.0

    return QUBO(
        q0=q0,
        a=a,
        b=b,
        instance=instance,
        A=A,
        B=B,
        x_ref_cost=x_ref_cost
    )


def qubo_to_ising(qubo: QUBO) -> Ising:
    """
    Convert QUBO to Ising using x_i = (1 - Z_i) / 2.

    Q = q0 + sum a_i x_i + sum_{i<j} b_ij x_i x_j

    x_i = (1 - Z_i) / 2
    x_i x_j = (1 - Z_i - Z_j + Z_i Z_j) / 4

    Substituting:
    Q = q0 + sum a_i (1 - Z_i)/2 + sum b_ij (1 - Z_i - Z_j + Z_i Z_j)/4
      = q0 + sum a_i/2 - sum a_i Z_i/2 + sum b_ij/4 - sum b_ij Z_i/4 - sum b_ij Z_j/4 + sum b_ij Z_i Z_j/4

    Grouping terms:
    offset = q0 + sum a_i/2 + sum_{i<j} b_ij/4
    h_i = -a_i/2 - (1/4) sum_{j!=i} b_ij
    J_ij = b_ij/4
    """
    n_vars = qubo.instance.n_vars
    q0 = qubo.q0
    a = qubo.a
    b = qubo.b

    # Compute offset
    offset = q0 + np.sum(a) / 2.0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            offset += b[i, j] / 4.0

    # Compute h_i
    h = np.zeros(n_vars)
    for i in range(n_vars):
        h[i] = -a[i] / 2.0
        # Subtract (1/4) * sum_{j!=i} b_ij
        sum_b = 0.0
        for j in range(n_vars):
            if j != i:
                if j > i:
                    sum_b += b[i, j]
                else:
                    sum_b += b[j, i]
        h[i] -= sum_b / 4.0

    # Compute J_ij
    J = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            J[i, j] = b[i, j] / 4.0

    return Ising(
        offset=offset,
        h=h,
        J=J,
        instance=qubo.instance
    )


def build_qaoa_circuit(ising: Ising, p: int = 1) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """
    Build QAOA circuit for given Ising Hamiltonian.

    Returns (circuit, gamma_params, beta_params)
    """
    n = ising.instance.n_vars
    gamma = [Parameter(f"γ_{k}") for k in range(p)]
    beta = [Parameter(f"β_{k}") for k in range(p)]

    qc = QuantumCircuit(n)

    # Initial state: |+>^n
    qc.h(range(n))

    for layer in range(p):
        # Cost layer: exp(-i γ H)
        # H = offset + sum h_i Z_i + sum J_ij Z_i Z_j
        # Offset is global phase, ignore
        # Z term: RZ(2 γ h_i)
        # ZZ term: RZZ(2 γ J_ij)

        for i in range(n):
            if abs(ising.h[i]) > 1e-12:
                qc.rz(2 * gamma[layer] * ising.h[i], i)

        for i in range(n):
            for j in range(i + 1, n):
                if abs(ising.J[i, j]) > 1e-12:
                    # RZZ gate: apply CNOT, RZ, CNOT
                    qc.cx(i, j)
                    qc.rz(2 * gamma[layer] * ising.J[i, j], j)
                    qc.cx(i, j)

        # Mixer layer: RX(2 β) on all qubits
        for i in range(n):
            qc.rx(2 * beta[layer], i)

    qc.measure_all()
    return qc, gamma, beta


def qaoa_expectation(params: np.ndarray, ising: Ising, p: int, shots: int = 1024, seed: int = 42) -> float:
    """
    Compute QAOA expectation value for given parameters.

    params: [γ_0, ..., γ_{p-1}, β_0, ..., β_{p-1}]
    """
    qc, gamma_params, beta_params = build_qaoa_circuit(ising, p)

    # Bind parameters
    param_dict = {}
    for k in range(p):
        param_dict[gamma_params[k]] = params[k]
        param_dict[beta_params[k]] = params[p + k]

    bound_qc = qc.assign_parameters(param_dict)

    # Run on simulator
    sim = AerSimulator(seed_simulator=seed)
    job = sim.run(bound_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute expectation value
    total_energy = 0.0
    total_shots = sum(counts.values())
    n = ising.instance.n_vars

    for bitstring, count in counts.items():
        # Ensure bitstring has correct length (pad if needed)
        if len(bitstring) < n:
            bitstring = "0" * (n - len(bitstring)) + bitstring
        energy = ising.energy_from_bitstring(bitstring)
        total_energy += energy * count

    return total_energy / total_shots


def optimize_qaoa(ising: Ising, p: int = 1, optimizer: str = "COBYLA", maxiter: int = 100, shots: int = 1024, seed: int = 42) -> tuple[np.ndarray, float]:
    """
    Optimize QAOA parameters using classical optimizer.

    Returns (optimal_params, optimal_energy)
    """
    n_params = 2 * p
    # Initial guess: small random values
    np.random.seed(seed)
    init_params = np.random.uniform(0, 0.1, n_params)

    def objective(params):
        return qaoa_expectation(params, ising, p, shots, seed)

    result = minimize(objective, init_params, method=optimizer, options={"maxiter": maxiter, "disp": False})

    return result.x, result.fun


def sample_qaoa(ising: Ising, params: np.ndarray, p: int = 1, shots: int = 1024, seed: int = 42) -> dict:
    """
    Sample from QAOA circuit at optimal parameters.

    Returns counts dict: bitstring -> count
    """
    qc, gamma_params, beta_params = build_qaoa_circuit(ising, p)

    param_dict = {}
    for k in range(p):
        param_dict[gamma_params[k]] = params[k]
        param_dict[beta_params[k]] = params[p + k]

    bound_qc = qc.assign_parameters(param_dict)

    sim = AerSimulator(seed_simulator=seed)
    job = sim.run(bound_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    return counts


@dataclass(frozen=True)
class QAOAResult:
    """Result from QAOA optimization and sampling."""
    instance_id: str
    feasible: bool
    optimal_params: np.ndarray
    optimal_energy: float
    samples: dict  # bitstring -> count
    shots: int
    p: int
    optimizer: str
    maxiter: int
    seed: int
    feasible_samples: int
    feasible_rate: float
    best_feasible_cost: Optional[float]
    mean_feasible_cost: Optional[float]
    optimal_solution_probability: float
    assignment_cost_best: Optional[float]
    qubo_energy_best: Optional[float]
    circuit_depth: int
    two_qubit_gate_count: int
    optimizer_evaluations: int
    runtime_seconds: float


def run_qaoa_p1(instance: Instance, shots: int = 1024, maxiter: int = 100, seed: int = 42,
                A: float = None, B: float = None, delta: float = 1.0) -> QAOAResult:
    """
    Run complete QAOA p=1 workflow for an instance.

    1. Build QUBO with safe penalties
    2. Convert to Ising
    3. Build QAOA circuit (p=1)
    4. Optimize parameters with COBYLA
    5. Sample at optimal parameters
    6. Decode and verify all samples
    7. Compute metrics
    """
    import time
    from src.classical import solve_exact

    start_time = time.time()

    # Build QUBO and Ising
    qubo = build_qubo(instance, A, B, delta)
    ising = qubo_to_ising(qubo)

    # Build circuit
    p = 1
    qc, gamma_params, beta_params = build_qaoa_circuit(ising, p)

    # Get circuit metrics
    # Transpile to basis gates to get depth and 2q gate count
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    transpiled_qc = pm.run(qc)
    circuit_depth = transpiled_qc.depth()
    two_qubit_gate_count = transpiled_qc.count_ops().get('cx', 0) + transpiled_qc.count_ops().get('cz', 0)

    # Optimize
    np.random.seed(seed)
    init_params = np.random.uniform(0, 0.1, 2 * p)

    eval_count = [0]

    def objective(params):
        eval_count[0] += 1
        return qaoa_expectation(params, ising, p, shots, seed)

    result = minimize(objective, init_params, method="COBYLA", options={"maxiter": maxiter, "disp": False})
    optimal_params = result.x
    optimal_energy = result.fun
    optimizer_evaluations = eval_count[0]

    # Sample at optimal parameters
    samples = sample_qaoa(ising, optimal_params, p, shots, seed)

    # Decode and verify all samples
    feasible_count = 0
    feasible_costs = []
    qubo_energies = []
    optimal_solution_count = 0

    # Get exact optimum for comparison
    exact_result = solve_exact(instance)
    exact_optimum = exact_result.optimal_cost if exact_result.feasible else None
    exact_assignments = set()
    if exact_result.feasible and exact_result.optimal_assignment:
        exact_bits = encode_assignment(instance, exact_result.optimal_assignment)
        exact_assignments.add(exact_bits)

    for bitstring, count in samples.items():
        # Pad bitstring if needed
        if len(bitstring) < instance.n_vars:
            bitstring = "0" * (instance.n_vars - len(bitstring)) + bitstring

        assignment = decode_bitstring(instance, bitstring)
        feasible, _ = check_constraints(instance, assignment)
        q_energy = qubo.energy_from_bitstring(bitstring)

        qubo_energies.append(q_energy)

        if feasible:
            feasible_count += count
            cost = assignment_cost(instance, assignment)
            feasible_costs.extend([cost] * count)

            # Check if this is an exact optimal assignment
            if bitstring in exact_assignments:
                optimal_solution_count += count

    runtime = time.time() - start_time

    feasible_rate = feasible_count / shots if shots > 0 else 0.0
    optimal_solution_prob = optimal_solution_count / shots if shots > 0 else 0.0

    best_feasible_cost = min(feasible_costs) if feasible_costs else None
    mean_feasible_cost = np.mean(feasible_costs) if feasible_costs else None
    assignment_cost_best = best_feasible_cost
    qubo_energy_best = min(qubo_energies) if qubo_energies else None

    return QAOAResult(
        instance_id=instance.instance_id,
        feasible=feasible_count > 0,
        optimal_params=optimal_params,
        optimal_energy=optimal_energy,
        samples=samples,
        shots=shots,
        p=p,
        optimizer="COBYLA",
        maxiter=maxiter,
        seed=seed,
        feasible_samples=feasible_count,
        feasible_rate=feasible_rate,
        best_feasible_cost=best_feasible_cost,
        mean_feasible_cost=mean_feasible_cost,
        optimal_solution_probability=optimal_solution_prob,
        assignment_cost_best=assignment_cost_best,
        qubo_energy_best=qubo_energy_best,
        circuit_depth=circuit_depth,
        two_qubit_gate_count=two_qubit_gate_count,
        optimizer_evaluations=optimizer_evaluations,
        runtime_seconds=runtime
    )