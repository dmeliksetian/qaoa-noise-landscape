#!/usr/bin/env python
"""
QAOA Noise Landscape — domain-agnostic experiment runner.

Reads a pre-built QUBO instance file (.npy or .npz) and runs the full
QAOA noise landscape experiment: noiseless simulation, calibration-noise
simulation, and (optionally) real hardware via IBM Quantum.

Instance files are prepared by domain-specific scripts (e.g.
prepare_portfolio.py) and stored in instances/.

Usage:
  # Smoke test (fast, no hardware):
  python run_experiment.py --qubo instances/6asset_lowvol.npz \\
      --mode sim --n_points 3 --shots 512

  # Full simulation run:
  python run_experiment.py --qubo instances/6asset_lowvol.npz \\
      --mode sim --n_points 13 --shots 4096

  # Hardware run:
  python run_experiment.py --qubo instances/6asset_lowvol.npz \\
      --mode full --backend ibm_fez --n_points 13 --shots 57344
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Domain-agnostic QAOA noise landscape experiment runner.')

    # ── Instance ──────────────────────────────────────────────────────────────
    p.add_argument('--qubo', required=True, metavar='FILE',
                   help='Path to instance file (.npy = raw Q matrix, '
                        '.npz = Q + optional metadata: k, e_opt, x_opt, '
                        'variable_names, random_baseline, label).')
    p.add_argument('--k', type=int, default=None,
                   help='Cardinality constraint (number of ones in feasible '
                        'bitstrings). Overrides value in .npz. Set to 0 to '
                        'disable feasibility tracking.')
    p.add_argument('--label', default=None,
                   help='Output label for results files. Overrides value in '
                        '.npz. Defaults to the instance filename stem.')
    p.add_argument('--e_opt', type=float, default=None,
                   help='Known optimal energy (skips brute-force search). '
                        'Overrides value in .npz.')
    p.add_argument('--normalize', action='store_true',
                   help='Normalise Q by max(|Q|) before circuit generation '
                        '(recommended for hardware so gate angles are O(1)).')

    # ── QAOA / grid ───────────────────────────────────────────────────────────
    p.add_argument('--p', type=int, default=1,
                   help='Number of QAOA layers (default 1).')
    p.add_argument('--n_points', type=int, default=7,
                   help='Grid resolution per axis (n_points² circuits).')
    p.add_argument('--shots', type=int, default=2048,
                   help='Shots per circuit.')
    p.add_argument('--grid_range', type=float, default=0.4,
                   help='Half-width of (γ, β) grid around (γ*, β*).')

    # ── Classical optimisation override ──────────────────────────────────────
    p.add_argument('--gamma_star', type=float, default=None,
                   help='Pin γ* (skip COBYLA). Must also set --beta_star.')
    p.add_argument('--beta_star', type=float, default=None,
                   help='Pin β* (skip COBYLA). Must also set --gamma_star.')
    p.add_argument('--opt_starts', default=None, metavar='G,B[ G,B ...]',
                   help='Space-separated γ,β starting points for COBYLA, '
                        'e.g. "0.257,1.391 0.5,1.0". Overrides default starts.')

    # ── Execution mode ────────────────────────────────────────────────────────
    p.add_argument('--mode', choices=['sim', 'full'], default='sim',
                   help='sim = local simulators only; '
                        'full = noisy sim + real hardware.')
    p.add_argument('--backend', default=None,
                   help='IBM backend name for --mode full (e.g. ibm_fez). '
                        'Defaults to least-busy device with enough qubits.')
    p.add_argument('--account', default='ibm_quantum_nersc',
                   help='Named account in qiskit-ibm.json to bill QPU time '
                        'against (default: ibm_quantum_nersc).')
    p.add_argument('--rep_delay', type=float, default=None,
                   help='Seconds between shots (must be within the backend\'s '
                        'rep_delay_range). Dominates QPU time at high shot '
                        'counts far more than circuit depth does. Default: '
                        'backend default, except on ibm_miami/Nighthawk r1 '
                        '(auto-set to its 1ms device minimum instead of the '
                        '4ms backend default). ibm_fez defaults to 250us; '
                        'ibm_phoenix/Nighthawk r2 (active dissipative qubit '
                        'reset) defaults to 1us with range [0, 2ms], so '
                        '--rep_delay 0 is valid there and cuts real QPU time '
                        'further still.')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_instance(path: str, args) -> dict:
    """
    Load a QUBO instance from a .npy or .npz file.

    Returns a dict with keys:
        Q               — QUBO matrix (n × n)
        k               — cardinality constraint (0 = no feasibility tracking)
        e_opt           — optimal energy (None if not known)
        x_opt           — optimal bitstring (None if not known)
        variable_names  — list of strings, length n
        random_baseline — C(n,k)/2^n (None if k == 0)
        label           — output label string
    """
    path = Path(path)
    meta: dict = {}

    if path.suffix == '.npy':
        Q = np.load(path)
    elif path.suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        Q = data['Q']
        for key in ('k', 'e_opt', 'x_opt', 'variable_names',
                    'random_baseline', 'label'):
            if key in data:
                val = data[key]
                # 0-d arrays → Python scalars
                meta[key] = val.item() if (isinstance(val, np.ndarray)
                                            and val.ndim == 0) else val
    else:
        raise ValueError(f'Unsupported instance file format: {path.suffix}')

    n = Q.shape[0]

    # CLI args take precedence over file metadata
    k = args.k if args.k is not None else int(meta.get('k', 0))
    e_opt = args.e_opt if args.e_opt is not None else meta.get('e_opt')
    x_opt = meta.get('x_opt')
    label = args.label or str(meta.get('label', path.stem))

    var_names = meta.get('variable_names')
    if var_names is None:
        var_names = [f'x{i}' for i in range(n)]
    else:
        var_names = list(var_names)

    random_baseline = None
    if k > 0:
        random_baseline = meta.get('random_baseline')
        if random_baseline is None:
            random_baseline = len(list(itertools.combinations(range(n), k))) / 2 ** n

    if args.normalize:
        scale = np.abs(Q).max()
        if scale > 0:
            Q = Q / scale
            if e_opt is not None:
                e_opt = e_opt / scale
            print(f'  Q normalised by {scale:.4f}')

    return dict(Q=Q, k=k, e_opt=e_opt, x_opt=x_opt,
                variable_names=var_names, random_baseline=random_baseline,
                label=label)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def feasibility_grid(counts_list: list[dict], n_points: int, k: int) -> np.ndarray:
    """Fraction of shots satisfying sum(x_i) == k at each grid point."""
    fg = np.zeros((n_points, n_points))
    for idx, counts in enumerate(counts_list):
        gi, bi = idx // n_points, idx % n_points
        total = sum(counts.values())
        feasible = sum(c for bs, c in counts.items()
                       if sum(int(b) for b in bs) == k)
        fg[gi, bi] = feasible / total
    return fg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from src import (
        brute_force_optimal,
        find_optimal_params, make_grid_circuits, run_grid, build_energy_grid,
    )

    # ── Load instance ────────────────────────────────────────────────────────
    print(f'Loading instance: {args.qubo}')
    inst = load_instance(args.qubo, args)
    Q             = inst['Q']
    k             = inst['k']
    e_opt         = inst['e_opt']
    x_opt         = inst['x_opt']
    var_names     = inst['variable_names']
    random_bl     = inst['random_baseline']
    label         = inst['label']
    n             = Q.shape[0]

    print(f'  n={n}, k={k}, label={label}')
    print(f'  variable_names: {var_names}')

    # ── Optimal energy ───────────────────────────────────────────────────────
    if e_opt is None:
        if k > 0 and n <= 24:
            print('Computing brute-force optimal ...')
            e_opt, x_opt = brute_force_optimal(Q, k=k)
        elif k == 0 and n <= 20:
            print('Computing brute-force optimal (unconstrained) ...')
            e_opt_unconstrained = min(
                np.array([int(b) for b in f'{i:0{n}b}'[::-1]]) @ Q
                @ np.array([int(b) for b in f'{i:0{n}b}'[::-1]])
                for i in range(2 ** n)
            )
            e_opt = float(e_opt_unconstrained)
        else:
            print(f'  n={n} too large for brute force — AR will not be computed.')

    if e_opt is not None:
        opt_vars = ([var_names[i] for i in range(n) if x_opt[i]]
                    if x_opt is not None else 'unknown')
        print(f'  Optimal: E={e_opt:.4f}  vars={opt_vars}')

    # ── Classical optimisation ───────────────────────────────────────────────
    if args.gamma_star is not None and args.beta_star is not None:
        gamma_star, beta_star = args.gamma_star, args.beta_star
        print(f'\nUsing pinned (γ*, β*) = ({gamma_star:.6f}, {beta_star:.6f})')
    else:
        print('\nOptimising (γ*, β*) on noiseless simulator ...')
        opt_starts = None
        if args.opt_starts:
            opt_starts = [tuple(float(v) for v in pair.split(','))
                          for pair in args.opt_starts.split()]
        gamma_star, beta_star, e_cobyla = find_optimal_params(
            Q, p=args.p, starts=opt_starts)
        print(f'  γ* = {gamma_star:.4f},  β* = {beta_star:.4f},  E = {e_cobyla:.4f}')

    # ── Build grid circuits ──────────────────────────────────────────────────
    bound_circuits, param_index, gammas, betas = make_grid_circuits(
        Q, gamma_star, beta_star,
        p=args.p, n_points=args.n_points, grid_range=args.grid_range,
    )
    print(f'\nGrid: {args.n_points}×{args.n_points} = {len(bound_circuits)} circuits')

    # ── Simulators ──────────────────────────────────────────────────────────
    from qiskit_aer import AerSimulator
    sim_ideal = AerSimulator()
    sim_noisy = None
    hw_backend = None

    grids_list, grid_labels, best_params_list = [], [], []
    all_counts: dict[str, list] = {}
    ar_values: dict[str, float] = {}

    def _run_condition(label_key: str, display: str, backend, use_batch=False):
        print(f'\nRunning {display} ...')
        counts = run_grid(bound_circuits, backend, shots=args.shots,
                          use_batch=use_batch, rep_delay=args.rep_delay)
        grid, best_e, best_s, best_p = build_energy_grid(
            counts, Q, args.n_points, param_index)
        ar = (best_e / e_opt) if e_opt is not None else None
        grids_list.append(grid)
        grid_labels.append(display)
        best_params_list.append(best_p)
        all_counts[label_key] = counts
        if ar is not None:
            ar_values[display] = ar
        best_gi, best_bi = divmod(int(np.argmin(grid)), args.n_points)
        ar_str = f'AR={ar:.4f}' if ar is not None else 'AR=n/a'
        print(f'  Best state: {best_s}  E={best_e:.4f}  {ar_str}')
        if k > 0:
            fg = feasibility_grid(counts, args.n_points, k)
            ff = fg[best_gi, best_bi]
            rb_str = f'  (random={random_bl:.4f})' if random_bl else ''
            print(f'  Feasibility at best: {ff:.4f}{rb_str}')
        return counts

    # Ideal sim (always)
    _run_condition('ideal', 'Ideal sim', sim_ideal)

    # Noisy sim (requires IBM account)
    if args.mode in ('sim', 'full'):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(name=args.account)
            backend_name = args.backend or 'least_busy'
            hw_backend = (
                service.least_busy(operational=True, simulator=False,
                                   min_num_qubits=n)
                if backend_name == 'least_busy'
                else service.backend(backend_name)
            )
            # ibm_miami (Nighthawk r1) defaults to 4ms rep_delay but supports
            # as low as 1ms; use the device minimum unless the user overrode it.
            if args.rep_delay is None and hw_backend.name == 'ibm_miami':
                args.rep_delay = 0.001
                print('  rep_delay not set — using ibm_miami device minimum '
                      '(1ms) instead of its 4ms default.')
            # Save backend calibration snapshot before submitting any jobs
            try:
                from datetime import date as _date
                props = hw_backend.properties()
                if props is not None:
                    _props_dir = Path('results')
                    _props_dir.mkdir(exist_ok=True)
                    _props_path = _props_dir / f'backend_props_{label}_{_date.today().isoformat()}.json'
                    _props_path.write_text(
                        json.dumps(props.to_dict(), indent=2, default=str))
                    print(f'  Backend properties saved to {_props_path}')
            except Exception as _exc:
                print(f'  Could not save backend properties: {_exc}')
            sim_noisy = AerSimulator.from_backend(hw_backend)
            _run_condition('noisy', 'Noisy sim', sim_noisy)
        except Exception as exc:
            print(f'  Noisy sim skipped: {exc}')

    # Hardware (full mode only)
    if args.mode == 'full' and hw_backend is not None:
        print(f'\nSubmitting hardware grid job to {hw_backend.name} ...')
        _run_condition('hardware', 'Hardware', hw_backend, use_batch=True)

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    fg_at_best = {}
    for cond_key, counts in all_counts.items():
        if k > 0:
            fg = feasibility_grid(counts, args.n_points, k)
            # Evaluate at the ideal-sim minimum for comparability across conditions
            ideal_min_idx = int(np.argmin(grids_list[0]))
            fg_at_best[cond_key] = float(fg.flat[ideal_min_idx])

    save_dict = dict(
        gammas=gammas, betas=betas,
        param_index=np.array(param_index),
        **{f'grid_{lk.lower().replace(" ", "_")}': g
           for lk, g in zip(grid_labels, grids_list)},
        **{f'counts_{ck}': np.array(v, dtype=object)
           for ck, v in all_counts.items()},
        **{f'feasible_{ck}': feasibility_grid(v, args.n_points, k)
           for ck, v in all_counts.items()} if k > 0 else {},
    )
    out_path = results_dir / f'grids_{label}.npz'
    np.savez(out_path, **save_dict)

    summary: dict = {
        'label':          label,
        'n':              n,
        'k':              k,
        'variable_names': var_names,
        'gamma_star':     gamma_star,
        'beta_star':      beta_star,
        'e_optimal':      e_opt,
        'random_baseline': random_bl,
        'approximation_ratios': ar_values,
    }
    if fg_at_best:
        summary['feasibility_at_best'] = fg_at_best

    summary_path = results_dir / f'summary_{label}.json'
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nResults saved to {out_path}')
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
