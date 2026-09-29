#!/usr/bin/env python
"""
Portfolio instance preparation — finance-specific QUBO builder.

Downloads historical price data (or generates synthetic data), builds the
mean-variance QUBO matrix, solves for the optimal portfolio by brute force,
and saves a self-contained instance .npz file to instances/ for consumption
by the domain-agnostic run_experiment.py.

Usage:
  # Real market data:
  python prepare_portfolio.py \\
      --tickers AAPL MSFT JPM XOM JNJ AMZN --k 3 \\
      --label 6asset_lowvol --normalize

  # Synthetic data (reproducible, no internet required):
  python prepare_portfolio.py \\
      --synthetic --n 6 --k 3 --seed 42 --label synthetic_6_3

Then run the experiment:
  python run_experiment.py --qubo instances/6asset_lowvol.npz \\
      --mode full --backend ibm_fez --n_points 13 --shots 57344
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Build a portfolio QUBO instance file for run_experiment.py.')

    src = p.add_mutually_exclusive_group()
    src.add_argument('--tickers', nargs='+', metavar='TICKER',
                     help='Ticker symbols (real market data via yfinance).')
    src.add_argument('--synthetic', action='store_true',
                     help='Use synthetic data (no internet required).')

    p.add_argument('--n', type=int, default=6,
                   help='Number of assets for synthetic instances (ignored '
                        'when --tickers is given).')
    p.add_argument('--k', type=int, required=True,
                   help='Number of assets to select (cardinality constraint).')
    p.add_argument('--lam', type=float, default=0.5,
                   help='Risk-aversion parameter λ (default 0.5).')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for synthetic data.')
    p.add_argument('--start', default='2019-01-01',
                   help='Start date for yfinance download (default 2019-01-01).')
    p.add_argument('--end', default='2024-01-01',
                   help='End date for yfinance download (default 2024-01-01).')
    p.add_argument('--normalize', action='store_true',
                   help='Normalise Q by max(|Q|) so entries lie in [-1, 1].')
    p.add_argument('--label', default=None,
                   help='Output label (default: auto-generated from tickers/n/k).')
    p.add_argument('--out_dir', default='instances',
                   help='Output directory (default: instances/).')
    p.add_argument('--no_brute_force', action='store_true',
                   help='Skip brute-force optimal computation (for large n).')

    return p.parse_args()


def build_qubo_portfolio(mu, sigma, lam, k, normalize=False):
    from src import build_qubo
    return build_qubo(mu, sigma, lam=lam, k=k, normalize=normalize)


def main():
    args = parse_args()
    from src import build_qubo, brute_force_optimal, synthetic_instance

    # ── Load data ────────────────────────────────────────────────────────────
    if args.synthetic:
        Q, mu, sigma, tickers = synthetic_instance(
            seed=args.seed, n=args.n, k=args.k,
            lam=args.lam, normalize=args.normalize)
        print(f'Synthetic instance: n={args.n}, k={args.k}, seed={args.seed}')
    else:
        if not args.tickers:
            raise ValueError('Provide --tickers or --synthetic.')
        try:
            import yfinance as yf
            from sklearn.covariance import LedoitWolf
        except ImportError as exc:
            raise SystemExit(f'Missing dependency: {exc}. '
                             'Install yfinance and scikit-learn, or use --synthetic.') from exc

        tickers = args.tickers
        print(f'Downloading {tickers} ({args.start} – {args.end}) ...')
        prices = yf.download(tickers, start=args.start, end=args.end,
                             auto_adjust=False)['Adj Close']
        log_returns = (prices / prices.shift(1)).apply(np.log).dropna()
        mu    = log_returns.mean().values * 252
        sigma = LedoitWolf().fit(log_returns).covariance_ * 252
        Q     = build_qubo(mu, sigma, lam=args.lam, k=args.k,
                            normalize=args.normalize)
        print(f'  Annualised μ: {mu.round(4).tolist()}')

    n = Q.shape[0]

    # ── Brute-force optimal ──────────────────────────────────────────────────
    e_opt, x_opt = None, None
    if not args.no_brute_force:
        print(f'Brute-force optimal over C({n},{args.k}) = '
              f'{len(list(itertools.combinations(range(n), args.k)))} candidates ...')
        e_opt, x_opt = brute_force_optimal(Q, k=args.k)
        opt_vars = [tickers[i] for i in range(n) if x_opt[i]]
        print(f'  E_opt = {e_opt:.6f}  optimal vars = {opt_vars}')

    # ── Random baseline ──────────────────────────────────────────────────────
    n_feasible = len(list(itertools.combinations(range(n), args.k)))
    random_baseline = n_feasible / 2 ** n

    # ── Label & output path ──────────────────────────────────────────────────
    label = args.label or f'{n}var_k{args.k}_{"_".join(tickers)}'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'{label}.npz'

    # ── Save instance ────────────────────────────────────────────────────────
    save_dict: dict = dict(
        Q=Q,
        k=np.int64(args.k),
        variable_names=np.array(tickers),
        random_baseline=np.float64(random_baseline),
        label=np.str_(label),
    )
    if e_opt is not None:
        save_dict['e_opt'] = np.float64(e_opt)
        save_dict['x_opt'] = x_opt.astype(np.int8)

    np.savez(out_path, **save_dict)

    # ── Human-readable metadata sidecar ─────────────────────────────────────
    meta = {
        'label':          label,
        'n':              n,
        'k':              args.k,
        'lam':            args.lam,
        'normalized':     args.normalize,
        'variable_names': list(tickers),
        'e_opt':          float(e_opt) if e_opt is not None else None,
        'optimal_vars':   opt_vars if e_opt is not None else None,
        'random_baseline': random_baseline,
        'n_feasible':     n_feasible,
        'source':         'synthetic' if args.synthetic else 'yfinance',
        'date_range':     f'{args.start} – {args.end}' if not args.synthetic else None,
    }
    (out_dir / f'{label}.json').write_text(json.dumps(meta, indent=2))

    print(f'\nInstance saved to {out_path}')
    print(f'Run with:')
    print(f'  python run_experiment.py --qubo {out_path} '
          f'--mode sim --n_points 7 --shots 2048')


if __name__ == '__main__':
    main()
