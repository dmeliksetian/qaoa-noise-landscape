"""
Simulator and hardware execution, energy grid builder, parameter optimisation, ZNE.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import minimize

from qaoa_common import build_qaoa_circuit, bind_params, get_params


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def compute_energy(counts: dict, Q: np.ndarray) -> float:
    """
    Compute expected QUBO energy from measurement counts.

    Qiskit is little-endian: rightmost bit = qubit 0.
    """
    total = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        x = np.array([int(b) for b in reversed(bitstring)])
        energy += (count / total) * (x @ Q @ x)
    return energy


def state_energy(bitstring: str, Q: np.ndarray) -> float:
    x = np.array([int(b) for b in reversed(bitstring)])
    return float(x @ Q @ x)


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

def find_optimal_params(
    Q: np.ndarray,
    p: int = 1,
    starts: list[tuple] | None = None,
    shots: int = 4096,
    maxiter: int = 200,
) -> tuple[float, float, float]:
    """
    Find (γ*, β*) by multi-start COBYLA on a noiseless AerSimulator.

    Args:
        Q:       QUBO matrix.
        p:       Number of QAOA layers.
        starts:  List of (γ0, β0) initial points. Defaults to 4 spread-out starts.
        shots:   Shots per COBYLA evaluation.
        maxiter: Max COBYLA iterations per start.

    Returns:
        (gamma_star, beta_star, best_energy)
    """
    from qiskit_aer import AerSimulator

    if starts is None:
        starts = [(0.5, 0.5), (1.0, 0.3), (0.3, 0.8), (1.5, 0.2)]

    qc = build_qaoa_circuit(Q, p=p)
    p_by_name = get_params(qc)
    sim = AerSimulator()

    def objective(params):
        bound = bind_params(qc, gamma_vals=[params[0]], beta_vals=[params[1]])
        counts = sim.run(bound, shots=shots).result().get_counts()
        return compute_energy(counts, Q)

    best_result = None
    for g0, b0 in starts:
        r = minimize(objective, x0=[g0, b0], method='COBYLA',
                     options={'maxiter': maxiter, 'rhobeg': 0.3})
        if best_result is None or r.fun < best_result.fun:
            best_result = r

    gamma_star, beta_star = best_result.x
    return gamma_star, beta_star, best_result.fun


# ---------------------------------------------------------------------------
# Grid circuit builder
# ---------------------------------------------------------------------------

def make_grid_circuits(
    Q: np.ndarray,
    gamma_star: float,
    beta_star: float,
    p: int = 1,
    n_points: int = 7,
    grid_range: float = 0.4,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Build bound circuits for the (γ, β) parameter grid.

    Args:
        Q:           QUBO matrix.
        gamma_star:  Centre of the γ grid.
        beta_star:   Centre of the β grid.
        p:           QAOA layers.
        n_points:    Grid resolution per axis (n_points² total circuits).
        grid_range:  Half-width of the grid around (γ*, β*).

    Returns:
        (bound_circuits, param_index, gammas, betas)
    """
    qc = build_qaoa_circuit(Q, p=p)
    gammas = np.linspace(gamma_star - grid_range, gamma_star + grid_range, n_points)
    betas  = np.linspace(beta_star  - grid_range, beta_star  + grid_range, n_points)

    bound_circuits = []
    param_index    = []
    for g, b in itertools.product(gammas, betas):
        bound_circuits.append(bind_params(qc, gamma_vals=[g], beta_vals=[b]))
        param_index.append((g, b))

    return bound_circuits, param_index, gammas, betas


# ---------------------------------------------------------------------------
# Grid execution
# ---------------------------------------------------------------------------

def run_grid(
    circuits: list,
    backend_or_sim,
    shots: int = 2048,
    use_batch: bool = False,
    rep_delay: float | None = None,
) -> list[dict]:
    """
    Run all grid circuits and return a list of counts dicts.

    Args:
        circuits:       List of bound QuantumCircuits.
        backend_or_sim: AerSimulator (ideal or noisy) or IBM backend.
        shots:          Shots per circuit.
        use_batch:      True for real IBM hardware (IBM Runtime Batch).
                        False for local AerSimulator (ideal or noisy).
        rep_delay:      Seconds to wait between shots (must fall within the
                        backend's `rep_delay_range`). None uses the backend
                        default. Dominates per-shot QPU time at high shot
                        counts far more than circuit depth/gate count does —
                        ibm_miami (Nighthawk r1) defaults to 4ms, range
                        [1ms, 4ms], vs. ibm_fez's 250us; ibm_phoenix
                        (Nighthawk r2) uses active dissipative qubit reset,
                        defaults to 1us, and allows range [0, 2ms] — set
                        rep_delay=0 there to minimize real QPU time.
    """
    if use_batch:
        from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        import math

        pm = generate_preset_pass_manager(backend=backend_or_sim, optimization_level=3)
        transpiled = pm.run(circuits)

        # IBM hard limit: 10M total shots per job (error code 1520).
        # Session timeout: ~18 min observed. At ~1.1s/circuit the time limit
        # is not the binding constraint — the shot limit is.
        # Use 9M as a safe margin below the 10M cap.
        max_shots_per_job = 9_000_000
        chunk_size = max(1, max_shots_per_job // shots)
        chunks = [transpiled[i:i+chunk_size] for i in range(0, len(transpiled), chunk_size)]

        all_counts = []
        for ci, chunk in enumerate(chunks):
            print(f"  Hardware batch {ci+1}/{len(chunks)} ({len(chunk)} circuits)...")
            with Batch(backend=backend_or_sim) as batch:
                sampler = Sampler(mode=batch)
                if rep_delay is not None:
                    sampler.options.execution.rep_delay = rep_delay
                job = sampler.run(chunk, shots=shots)
            all_counts.extend([r.data.meas.get_counts() for r in job.result()])

        return all_counts
    else:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # density_matrix computes the noisy channel once and samples cheaply from
        # it -- much faster than statevector at high shot counts, but needs
        # O(4^n) memory (n=16 needs ~69GB, too big even for a 32GB GPU). Prefer
        # density_matrix whenever it comfortably fits in the target device's
        # memory; only fall back to Kraus-sampled statevector (which has no such
        # wall, but re-simulates the whole circuit per shot and was measured
        # ~10x slower than density_matrix when the latter fits) once it doesn't.
        # Empirically: GPU density_matrix 4.3s vs. GPU statevector 43.5s for one
        # n=12 circuit at 8192 shots -- statevector must not be the default just
        # because a GPU is present.
        if isinstance(backend_or_sim, AerSimulator) and circuits:
            try:
                devices = backend_or_sim.available_devices()
                use_gpu = 'GPU' in devices
                device = 'GPU' if use_gpu else 'CPU'
                # Conservative memory budget: leave headroom for intermediate
                # buffers during density-matrix ops, not just the matrix itself.
                budget_bytes = 24e9 if use_gpu else 8e9
                n = circuits[0].num_qubits
                dm_bytes = (2 ** n) * (2 ** n) * 16  # complex128 density matrix
                method = 'density_matrix' if dm_bytes < budget_bytes else 'statevector'
                backend_or_sim.set_options(method=method, device=device)
            except Exception:
                pass

        # Transpile so the noise model is applied to the correct physical gates.
        # optimization_level=1 is sufficient for local simulation.
        pm = generate_preset_pass_manager(backend=backend_or_sim, optimization_level=1)
        transpiled = pm.run(circuits)
        sampler = SamplerV2.from_backend(backend_or_sim)
        job = sampler.run(transpiled, shots=shots)
        return [r.data.meas.get_counts() for r in job.result()]


# ---------------------------------------------------------------------------
# Energy grid builder
# ---------------------------------------------------------------------------

def build_energy_grid(
    counts_list: list[dict],
    Q: np.ndarray,
    n_points: int,
    param_index: list[tuple],
) -> tuple[np.ndarray, float, str, tuple]:
    """
    Build the expected-energy landscape grid and identify the best answer.

    Best state = most probable state at the grid point with the lowest
    *expected* energy. This is consistent with COBYLA (which also minimises
    expected energy) and ensures the marker lands at the visual minimum.

    Args:
        counts_list:  One counts dict per circuit (length = n_points²).
        Q:            QUBO matrix.
        n_points:     Grid resolution per axis.
        param_index:  List of (gamma, beta) tuples matching counts_list order.

    Returns:
        (grid, best_energy, best_state, best_params)
        grid:        (n_points × n_points) expected-energy array.
        best_energy: Expected QUBO energy at the grid minimum (averaged over all shots).
        best_state:  Most probable bitstring at that grid point (the recommended portfolio).
        best_params: (gamma, beta) of the best grid point.
    """
    grid       = np.zeros((n_points, n_points))
    all_counts = []

    for idx, counts in enumerate(counts_list):
        gi, bi = idx // n_points, idx % n_points
        grid[gi, bi] = compute_energy(counts, Q)
        all_counts.append(counts)

    best_idx    = int(np.argmin(grid))
    best_counts = all_counts[best_idx]
    best_params = param_index[best_idx]
    best_state  = max(best_counts, key=best_counts.get)
    best_energy = float(grid.flat[best_idx])  # expected energy at grid minimum

    return grid, best_energy, best_state, best_params


# ---------------------------------------------------------------------------
# ZNE
# ---------------------------------------------------------------------------

def run_zne(
    circuit,
    backend_or_sim,
    Q: np.ndarray,
    scale_factors: list[int] | None = None,
    shots: int = 4096,
    use_batch: bool = False,
) -> tuple[float, list[float], list[int]]:
    """
    Zero Noise Extrapolation via circuit folding (batch-compatible).

    Folds the circuit at each scale factor, runs all at once, then
    fits a linear polynomial and extrapolates to scale=0.

    Args:
        circuit:      Bound circuit at the best (γ*, β*).
        backend_or_sim: Backend to run on.
        Q:            QUBO matrix.
        scale_factors: Odd integers to fold at (default [1, 3, 5]).
        shots:        Shots per folded circuit.
        use_batch:    True for IBM hardware.

    Returns:
        (e_zne, energies, scale_factors)
    """
    from qaoa_common import fold_circuit

    if scale_factors is None:
        scale_factors = [1, 3, 5]

    folded_circuits = [fold_circuit(circuit, s) for s in scale_factors]
    counts_list = run_grid(folded_circuits, backend_or_sim, shots=shots, use_batch=use_batch)
    energies = [compute_energy(c, Q) for c in counts_list]

    coeffs = np.polyfit(scale_factors, energies, deg=1)
    e_zne  = float(np.polyval(coeffs, 0))

    return e_zne, energies, scale_factors
