#!/usr/bin/env python
"""
run_sa_baseline.py — Classical simulated annealing baseline for all instances.

Runs SA and random search on the same QUBO instances used for QAOA,
providing the classical comparison for the paper results table.

Usage:
    ./venv/bin/python run_sa_baseline.py
    ./venv/bin/python run_sa_baseline.py --restarts 2000 --normalize

Outputs:
    results/sa_baseline.json   — AR values for SA and random search
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np
from pathlib import Path


# ── Instance registry — must match grid runs exactly ──────────────────────────

ALL_INSTANCES = {
    '6asset_lowvol_hires': {
        'tickers':   ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'AMZN'],
        'k':         3,
        'e_optimal': -16.885078379950897,
        'ar_hw':     0.9141,
    },
    '8asset_lowvol': {
        'tickers':   ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'AMZN', 'GOOGL', 'BRK-B'],
        'k':         4,
        'e_optimal': -46.265343264148036,
        'ar_hw':     0.9321,
    },
    '8asset_tsla': {
        'tickers':   ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'AMZN', 'TSLA', 'IBM'],
        'k':         4,
        'e_optimal': -56.56299575921959,
        'ar_hw':     0.9365,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='SA classical baseline')
    p.add_argument('--restarts',  type=int,   default=1000,
                   help='SA restarts per instance (default 1000)')
    p.add_argument('--lam',       type=float, default=0.5,
                   help='Risk aversion λ — must match grid runs')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize Q — must match grid runs')
    p.add_argument('--seed',      type=int,   default=42)
    return p.parse_args()


def build_Q(tickers, k, lam, normalize):
    import yfinance as yf
    from sklearn.covariance import LedoitWolf
    from src import build_qubo

    prices      = yf.download(tickers, start='2020-01-01', end='2024-01-01',
                               auto_adjust=False)['Adj Close']
    log_returns = (prices / prices.shift(1)).apply(np.log).dropna()
    mu          = log_returns.mean().values * 252
    sigma       = LedoitWolf().fit(log_returns).covariance_ * 252
    return build_qubo(mu, sigma, lam=lam, k=k, normalize=normalize)


def qubo_energy(x, Q):
    return float(x @ Q @ x)


def random_feasible(n, k, rng):
    """Sample a random feasible bitstring with exactly k ones."""
    x = np.zeros(n)
    x[rng.choice(n, size=k, replace=False)] = 1.0
    return x


def simulated_annealing(Q, k, n_restarts, rng,
                         T_start=2.0, T_end=0.01, n_steps=500):
    """
    SA over feasibility-preserving moves: swap one selected asset
    for one unselected asset (always maintains sum(x) == k).

    Returns (best_energy, best_x, n_feasible_found)
    """
    n          = Q.shape[0]
    best_e     = np.inf
    best_x     = None

    for _ in range(n_restarts):
        x = random_feasible(n, k, rng)
        e = qubo_energy(x, Q)

        for step in range(n_steps):
            T = T_start * (T_end / T_start) ** (step / n_steps)

            # Swap move: pick one selected and one unselected index
            selected   = np.where(x == 1)[0]
            unselected = np.where(x == 0)[0]
            i = rng.choice(selected)
            j = rng.choice(unselected)

            x_new    = x.copy()
            x_new[i] = 0
            x_new[j] = 1
            e_new    = qubo_energy(x_new, Q)

            delta = e_new - e
            if delta < 0 or rng.random() < np.exp(-delta / T):
                x, e = x_new, e_new

        if e < best_e:
            best_e = e
            best_x = x.copy()

    return best_e, best_x


def random_search(Q, k, n_samples, rng):
    """
    Random search: sample n_samples feasible bitstrings, return best.
    Establishes the floor — QAOA should comfortably beat this.
    """
    n      = Q.shape[0]
    best_e = np.inf
    best_x = None

    for _ in range(n_samples):
        x = random_feasible(n, k, rng)
        e = qubo_energy(x, Q)
        if e < best_e:
            best_e = e
            best_x = x.copy()

    return best_e, best_x


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    results = {}

    for label, inst in ALL_INSTANCES.items():
        print(f'\n=== {label} ===')
        n, k = len(inst['tickers']), inst['k']
        print(f'  n={n}  k={k}  tickers={inst["tickers"]}')

        try:
            Q = build_Q(inst['tickers'], k, args.lam, args.normalize)
        except Exception as e:
            print(f'  Failed to build Q: {e}')
            continue

        e_opt = inst['e_optimal']

        # ── Simulated annealing ───────────────────────────────────────────
        t0 = time.time()
        e_sa, x_sa = simulated_annealing(Q, k, args.restarts, rng)
        t_sa = time.time() - t0
        ar_sa = e_sa / e_opt
        sa_assets = [inst['tickers'][i] for i in range(n) if x_sa[i]]
        print(f'  SA:     E={e_sa:.4f}  AR={ar_sa:.4f}  '
              f'assets={sa_assets}  ({t_sa:.1f}s, {args.restarts} restarts)')

        # ── Random search ─────────────────────────────────────────────────
        t0 = time.time()
        e_rand, x_rand = random_search(Q, k, args.restarts, rng)
        t_rand = time.time() - t0
        ar_rand = e_rand / e_opt
        rand_assets = [inst['tickers'][i] for i in range(n) if x_rand[i]]
        print(f'  Random: E={e_rand:.4f}  AR={ar_rand:.4f}  '
              f'assets={rand_assets}  ({t_rand:.1f}s, {args.restarts} samples)')

        # ── Brute force (feasible for n<=8) ───────────────────────────────
        from src import brute_force_optimal
        e_bf, x_bf = brute_force_optimal(Q, k)
        ar_bf = e_bf / e_opt
        bf_assets = [inst['tickers'][i] for i in range(n) if x_bf[i]]
        print(f'  BF:     E={e_bf:.4f}  AR={ar_bf:.4f}  assets={bf_assets}')
        print(f'  HW QAOA AR (reference): {inst["ar_hw"]:.4f}')

        results[label] = {
            'e_optimal':   e_opt,
            'ar_brute_force': ar_bf,
            'ar_sa':       ar_sa,
            'ar_random':   ar_rand,
            'ar_hw_qaoa':  inst['ar_hw'],
            'sa_assets':   sa_assets,
            'bf_assets':   bf_assets,
            'sa_restarts': args.restarts,
            'sa_time_s':   round(t_sa, 2),
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    Path('results').mkdir(exist_ok=True)
    out = Path('results/sa_baseline.json')
    out.write_text(json.dumps(results, indent=2))
    print(f'\nSaved {out}')

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f'{"Instance":<25} {"BF":>7} {"SA":>7} {"QAOA hw":>9} {"Random":>8}')
    print('-' * 58)
    for lbl, r in results.items():
        print(f'{lbl:<25} {r["ar_brute_force"]:>7.4f} {r["ar_sa"]:>7.4f} '
              f'{r["ar_hw_qaoa"]:>9.4f} {r["ar_random"]:>8.4f}')

    print()
    print('Note: AR closer to 1.0 = better. BF = exact optimal.')


if __name__ == '__main__':
    main()
