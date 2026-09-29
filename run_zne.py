#!/usr/bin/env python
"""
run_zne.py — Zero Noise Extrapolation using Qiskit built-in EstimatorV2.

Uses qiskit_ibm_runtime EstimatorV2 with zne_mitigation=True instead of
manual circuit folding. IBM handles folding and extrapolation internally,
avoiding the transpiler gate-cancellation problem with manual folding.

Key difference from SamplerV2 approach:
  - Defines QUBO cost Hamiltonian as SparsePauliOp observable
  - EstimatorV2 returns E = <psi|H|psi> directly with ZNE applied
  - No bitstring counting — cleaner energy estimate with error bars

Usage:
    ./venv/bin/python run_zne.py --backend ibm_fez --normalize
    ./venv/bin/python run_zne.py --backend ibm_fez --normalize --shots 8192
    ./venv/bin/python run_zne.py --dry_run --normalize
    ./venv/bin/python run_zne.py --backend ibm_fez --normalize --instances 6asset_lowvol_hires 8asset_tsla

Outputs:
    results/zne_results.json     — ZNE energies, AR, error bars per instance
    figures/fig_zne.pdf/.png     — AR comparison: grid vs estimator raw vs ZNE
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Instance registry ─────────────────────────────────────────────────────────
# Values taken directly from summary JSON files.
# gamma_star, beta_star, e_optimal, tickers, k must match grid runs exactly.

ALL_INSTANCES = {
    '6asset_lowvol_hires': {
        'tickers':    ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'AMZN'],
        'k':          3,
        'gamma_star': 0.3382695157240051,
        'beta_star':  0.21859318817301948,
        'e_optimal':  -16.885078379950897,
        'ar_hw':      0.9141,
    },
    '8asset_lowvol': {
        'tickers':    ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'AMZN', 'GOOGL', 'BRK-B'],
        'k':          4,
        'gamma_star': 0.15663564965190077,
        'beta_star':  0.21378191608353048,
        'e_optimal':  -46.265343264148036,
        'ar_hw':      0.9321,
    },
    '8asset_tsla': {
        'tickers':    ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'AMZN', 'TSLA', 'IBM'],
        'k':          4,
        'gamma_star': 0.12849450156052875,
        'beta_star':  0.2777036229984607,
        'e_optimal':  -56.56299575921959,
        'ar_hw':      0.9365,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='ZNE via EstimatorV2 for QAOA instances')
    p.add_argument('--backend',   default=None,
                   help='IBM backend name (required unless --dry_run)')
    p.add_argument('--shots',     type=int, default=8192,
                   help='Shots per circuit (default 8192)')
    p.add_argument('--lam',       type=float, default=0.5,
                   help='Risk aversion λ — must match grid runs (default 0.5)')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize Q — must match flag used in grid runs')
    p.add_argument('--instances', nargs='+',
                   default=list(ALL_INSTANCES.keys()),
                   choices=list(ALL_INSTANCES.keys()),
                   help='Instances to run (default: all three)')
    p.add_argument('--noise_factors', nargs='+', type=float,
                   default=[1, 3, 5],
                   help='ZNE noise amplification factors (default: 1 3 5)')
    p.add_argument('--dry_run',   action='store_true',
                   help='Inspect circuits and Hamiltonian without submitting')
    return p.parse_args()


def build_Q(tickers, k, lam, normalize):
    """Reconstruct Q matrix identical to grid run."""
    import yfinance as yf
    from sklearn.covariance import LedoitWolf
    from src import build_qubo

    prices      = yf.download(tickers, start='2020-01-01', end='2024-01-01',
                               auto_adjust=False)['Adj Close']
    log_returns = (prices / prices.shift(1)).apply(np.log).dropna()
    mu          = log_returns.mean().values * 252
    sigma       = LedoitWolf().fit(log_returns).covariance_ * 252
    return build_qubo(mu, sigma, lam=lam, k=k, normalize=normalize)


def qubo_to_pauli(Q):
    """
    Convert QUBO matrix Q to SparsePauliOp cost Hamiltonian.

    Substitution x_i = (1 - Z_i) / 2 into x^T Q x gives:
      Constant:      C  = sum_i Q[i,i]/2 + sum_{i<j} Q[i,j]/4   (identity term)
      Diagonal:     -Q[i,i]/2 * Z_i
      Off-diagonal:  Q[i,j]/4 * Z_i Z_j

    The constant term is included as an identity Pauli so that
    <psi|H_pauli|psi> = x^T Q x exactly — directly comparable to
    Sampler energies and e_optimal from brute force.
    """
    from qiskit.quantum_info import SparsePauliOp
    n, paulis = Q.shape[0], []

    # Constant term — include so Estimator energies match QUBO energies
    C = sum(Q[i, i] / 2 for i in range(n))
    C += sum(Q[i, j] / 4 for i in range(n) for j in range(i + 1, n))
    paulis.append(('I' * n, C))

    # Z terms (diagonal)
    for i in range(n):
        if abs(Q[i, i]) > 1e-10:
            label = ['I'] * n
            label[i] = 'Z'
            paulis.append((''.join(reversed(label)), -Q[i, i] / 2))

    # ZZ terms (off-diagonal)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-10:
                label = ['I'] * n
                label[i] = 'Z'
                label[j] = 'Z'
                paulis.append((''.join(reversed(label)), Q[i, j] / 4))

    return SparsePauliOp.from_list(paulis)


def run_estimator(circuit, observable, backend, shots, zne=False, noise_factors=None):
    """
    Run circuit with EstimatorV2, optionally with ZNE.
    Returns (expectation_value, std_error).
    """
    from qiskit_ibm_runtime import EstimatorV2, Batch
    from qiskit_ibm_runtime.options import EstimatorOptions
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm         = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled = pm.run(circuit)
    obs_t      = observable.apply_layout(transpiled.layout)

    options                   = EstimatorOptions()
    options.default_shots     = shots

    if zne:
        options.resilience_level                 = 2
        options.resilience.zne_mitigation        = True
        options.resilience.zne.noise_factors     = noise_factors or [1, 3, 5]
        options.resilience.zne.extrapolator      = 'linear'
    else:
        options.resilience_level = 0

    with Batch(backend=backend) as batch:
        estimator = EstimatorV2(mode=batch, options=options)
        job       = estimator.run([(transpiled, obs_t)])

    res = job.result()[0]
    return float(res.data.evs), float(res.data.stds)


def main():
    args = parse_args()

    from src import build_qaoa_circuit, bind_params

    if not args.dry_run and args.backend is None:
        raise ValueError('--backend is required unless --dry_run is set')

    hw_backend = None
    if not args.dry_run:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service    = QiskitRuntimeService(name='ibm_cloud_account')
        hw_backend = service.backend(args.backend)
        print(f'Backend:     {hw_backend.name}')
        print(f'Basis gates: {hw_backend.basis_gates}')

    results = {}

    for label in args.instances:
        inst = ALL_INSTANCES[label]
        print(f'\n=== {label} ===')
        print(f'  Tickers: {inst["tickers"]}  k={inst["k"]}')
        print(f'  γ*={inst["gamma_star"]:.4f}  β*={inst["beta_star"]:.4f}')

        try:
            Q = build_Q(inst['tickers'], inst['k'], args.lam, args.normalize)
            print(f'  Q shape={Q.shape}  range=[{Q.min():.3f}, {Q.max():.3f}]')
        except Exception as e:
            print(f'  Failed to build Q: {e}')
            continue

        # Compute e_opt_pauli from the SAME Q as the circuit.
        # H_pauli includes constant C so <H_pauli> = x^T Q x exactly.
        from src import brute_force_optimal
        e_opt_qubo, _ = brute_force_optimal(Q, inst['k'])
        n = Q.shape[0]
        C = sum(Q[i,i]/2 for i in range(n))
        C += sum(Q[i,j]/4 for i in range(n) for j in range(i+1, n))
        e_opt_pauli = e_opt_qubo - C
        delta = abs(e_opt_qubo - inst['e_optimal'])
        status = 'OK' if delta < 0.01 else f'MISMATCH delta={delta:.4f} check --normalize'
        print(f'  e_opt_qubo={e_opt_qubo:.4f}  C={C:.4f}  e_opt_pauli={e_opt_pauli:.4f}  [{status}]')

        qc    = build_qaoa_circuit(Q, p=1)
        bound = bind_params(qc, [inst['gamma_star']], [inst['beta_star']])
        bound.remove_final_measurements(inplace=True)

        H = qubo_to_pauli(Q)

        if args.dry_run:
            print(f'  Circuit depth (pre-transpile): {bound.depth()}')
            print(f'  Circuit ops:   {dict(bound.count_ops())}')
            print(f'  Hamiltonian:   {len(H)} Pauli terms')
            print(f'  Noise factors: {args.noise_factors}')
            continue

        # Run raw (no ZNE) and ZNE in sequence — two small batch jobs
        print(f'  Running raw estimator (shots={args.shots})...')
        e_raw, std_raw = run_estimator(
            bound, H, hw_backend, args.shots, zne=False)

        print(f'  Running ZNE estimator (noise_factors={args.noise_factors}, shots={args.shots})...')
        e_zne, std_zne = run_estimator(
            bound, H, hw_backend, args.shots,
            zne=True, noise_factors=args.noise_factors)

        # AR computed consistently from normalized Q throughout
        # ar_raw ~ ar_hw from Sampler if circuit and Q are consistent
        ar_raw  = e_raw / e_opt_pauli
        ar_zne  = e_zne / e_opt_pauli
        ar_gain = ar_zne - ar_raw          # ZNE improvement in AR units
        zne_pct = (e_raw - e_zne) / abs(e_raw) * 100  # % energy improvement

        print(f'  e_opt_pauli:         {e_opt_pauli:.4f}')
        print(f'  E raw  (Estimator):  {e_raw:.4f}  AR={ar_raw:.4f}  ±{std_raw:.4f}')
        print(f'  E ZNE  (Estimator):  {e_zne:.4f}  AR={ar_zne:.4f}  ±{std_zne:.4f}')
        print(f'  ZNE gain (AR units): {ar_gain:+.4f}  ({zne_pct:.1f}% energy improvement)')

        results[label] = {
            'e_raw':        e_raw,
            'e_zne':        e_zne,
            'std_raw':      std_raw,
            'std_zne':      std_zne,
            'e_opt_pauli':  e_opt_pauli,
            'ar_raw':       ar_raw,
            'ar_zne':       ar_zne,
            'ar_gain':      ar_gain,
            'zne_pct':      zne_pct,
            'noise_factors': args.noise_factors,
        }

    if args.dry_run:
        print('\nDry run complete — no hardware used.')
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    out = Path('results/zne_results.json')
    out.write_text(json.dumps(results, indent=2))
    print(f'\nSaved {out}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f'{"Instance":<25} {"AR raw":>8} {"AR ZNE":>8} {"Gain":>7} {"ZNE%":>7} {"±ZNE":>7}')
    print('-' * 68)
    for lbl, r in results.items():
        err_ar = r['std_zne'] / abs(r['e_opt_pauli'])
        print(f'{lbl:<25} {r["ar_raw"]:>8.4f} {r["ar_zne"]:>8.4f} '
              f'{r["ar_gain"]:>+7.4f} {r["zne_pct"]:>6.1f}% {err_ar:>7.4f}')

    # ── Figure ────────────────────────────────────────────────────────────────
    Path('figures').mkdir(exist_ok=True)
    n   = len(results)
    fig, ax = plt.subplots(figsize=(max(6, n * 2.5), 4))

    lbls    = list(results.keys())
    ar_grid = [r['ar_hw_grid'] for r in results.values()]
    ar_raw  = [r['ar_raw']     for r in results.values()]
    ar_zne  = [r['ar_zne']     for r in results.values()]
    err_zne = [r['std_zne'] / abs(r['e_optimal']) for r in results.values()]
    x, w    = np.arange(n), 0.25

    ax.bar(x - w, ar_grid, w, label='Hardware — Sampler grid',    color='#D85A30', alpha=0.85)
    ax.bar(x,     ar_raw,  w, label='Hardware — Estimator (raw)', color='#BA7517', alpha=0.85)
    ax.bar(x + w, ar_zne,  w, label='Hardware — Estimator + ZNE', color='#534AB7', alpha=0.85,
           yerr=err_zne, capsize=4)

    ax.axhline(y=1.0, color='black', linewidth=0.8, linestyle='--', label='Optimal (AR=1.0)')
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=15, ha='right')
    ax.set_ylabel('Approximation ratio')
    ax.set_ylim(min(ar_grid) - 0.02, 1.05)
    ax.legend(fontsize=9)
    ax.set_title('ZNE mitigation results — all instances')
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/fig_zne.{ext}', dpi=300, bbox_inches='tight')
    print('Figures saved to figures/fig_zne.pdf + .png')


if __name__ == '__main__':
    main()
