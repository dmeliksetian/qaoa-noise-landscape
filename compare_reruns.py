#!/usr/bin/env python
"""
Compare original sim grids with post-bugfix reruns.

6asset_lowvol_hires: old 13x13  vs  new 13x13 rerun  (direct grid comparison)
8asset_lowvol:       old 7x7    vs  new 13x13 hires   (metrics only)
8asset_tsla:         old 7x7    vs  new 13x13 hires   (metrics only)
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

RESULTS = Path("results")
BACKUP  = Path("results_backup_20260426")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.corrcoef(a, b)[0, 1])


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def report_grid_comparison(label: str, old_npz: Path, new_npz: Path):
    old = np.load(old_npz, allow_pickle=True)
    new = np.load(new_npz, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"  {label}  —  direct grid comparison")
    print(f"{'='*60}")

    old_shape = old["gammas"].shape[0]
    new_shape = new["gammas"].shape[0]
    print(f"  Grid: {old_shape}x{old_shape} (old)  vs  {new_shape}x{new_shape} (new)")

    if old_shape != new_shape:
        print("  [!] Different grid resolutions — skipping element-wise diff")
        return

    for cond, key in [("Ideal sim", "grid_ideal_sim"), ("Noisy sim", "grid_noisy_sim")]:
        if key not in old or key not in new:
            print(f"  {cond}: key missing in one of the files")
            continue
        g_old = old[key]
        g_new = new[key]
        span_old = g_old.max() - g_old.min()
        span_new = g_new.max() - g_new.min()
        print(f"\n  {cond}:")
        print(f"    span  old={span_old:.5f}  new={span_new:.5f}  Δ={span_new - span_old:+.5f}")
        print(f"    max|Δ|  = {max_abs_diff(g_old, g_new):.5f}")
        print(f"    mean|Δ| = {mean_abs_diff(g_old, g_new):.5f}")
        print(f"    Pearson r = {pearson(g_old, g_new):.6f}")
        argmin_old = np.unravel_index(np.argmin(g_old), g_old.shape)
        argmin_new = np.unravel_index(np.argmin(g_new), g_new.shape)
        print(f"    argmin  old={argmin_old}  new={argmin_new}  "
              f"{'SAME' if argmin_old == argmin_new else 'SHIFTED'}")


def report_metrics_comparison(label: str,
                               old_summary: Path, new_summary: Path,
                               old_npz: Path,   new_npz: Path):
    old_s = load_summary(old_summary)
    new_s = load_summary(new_summary)
    old_d = np.load(old_npz, allow_pickle=True)
    new_d = np.load(new_npz, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"  {label}  —  metrics comparison  "
          f"({old_d['gammas'].shape[0]}x{old_d['gammas'].shape[0]} → "
          f"{new_d['gammas'].shape[0]}x{new_d['gammas'].shape[0]})")
    print(f"{'='*60}")

    print(f"\n  Optimal params:")
    print(f"    γ*  old={old_s['gamma_star']:.6f}  new={new_s['gamma_star']:.6f}")
    print(f"    β*  old={old_s['beta_star']:.6f}  new={new_s['beta_star']:.6f}")

    print(f"\n  Approximation ratios:")
    for cond in ("Ideal sim", "Noisy sim"):
        old_ar = old_s.get("approximation_ratios", {}).get(cond)
        new_ar = new_s.get("approximation_ratios", {}).get(cond)
        if old_ar is not None and new_ar is not None:
            print(f"    {cond:<12}  old={old_ar:.6f}  new={new_ar:.6f}  Δ={new_ar - old_ar:+.6f}")

    print(f"\n  Feasibility at best (γ*, β*):")
    old_feas = old_s.get("feasibility_at_best", {})
    new_feas = new_s.get("feasibility_at_best", {})
    for cond in ("ideal", "noisy"):
        old_f = old_feas.get(cond)
        new_f = new_feas.get(cond)
        if old_f is not None and new_f is not None:
            print(f"    {cond:<8}  old={old_f:.6f}  new={new_f:.6f}  Δ={new_f - old_f:+.6f}")

    for cond_key, grid_key in [("ideal", "grid_ideal_sim"), ("noisy", "grid_noisy_sim")]:
        if grid_key in old_d and grid_key in new_d:
            g_old = old_d[grid_key]
            g_new = new_d[grid_key]
            print(f"\n  Span ({cond_key}):")
            print(f"    old span={g_old.max() - g_old.min():.5f}  "
                  f"new span={g_new.max() - g_new.min():.5f}")


def main():
    print("\nPost-bugfix rerun comparison")
    print("Bug: qiskit-aer #2421 — manifests as exception (no silent corruption expected)")

    # ── 6asset: direct grid comparison ─────────────────────────────────────────
    old_6  = BACKUP  / "grids_6asset_lowvol_hires.npz"
    new_6  = RESULTS / "grids_6asset_lowvol_hires_rerun.npz"
    old_6s = BACKUP  / "summary_6asset_lowvol_hires.json"
    new_6s = RESULTS / "summary_6asset_lowvol_hires_rerun.json"

    if new_6.exists():
        report_grid_comparison("6asset_lowvol_hires", old_6, new_6)
        report_metrics_comparison("6asset_lowvol_hires", old_6s, new_6s, old_6, new_6)
    else:
        print("\n[waiting] results/grids_6asset_lowvol_hires_rerun.npz not yet available")

    # ── 8asset_lowvol: metrics comparison ──────────────────────────────────────
    old_8lv  = BACKUP  / "grids_8asset_lowvol.npz"
    new_8lv  = RESULTS / "grids_8asset_lowvol_hires.npz"
    old_8lvs = BACKUP  / "summary_8asset_lowvol.json"
    new_8lvs = RESULTS / "summary_8asset_lowvol_hires.json"

    if new_8lv.exists():
        report_metrics_comparison("8asset_lowvol", old_8lvs, new_8lvs, old_8lv, new_8lv)
    else:
        print("\n[waiting] results/grids_8asset_lowvol_hires.npz not yet available")

    # ── 8asset_tsla: metrics comparison ────────────────────────────────────────
    old_8t  = BACKUP  / "grids_8asset_tsla.npz"
    new_8t  = RESULTS / "grids_8asset_tsla_hires.npz"
    old_8ts = BACKUP  / "summary_8asset_tsla.json"
    new_8ts = RESULTS / "summary_8asset_tsla_hires.json"

    if new_8t.exists():
        report_metrics_comparison("8asset_tsla", old_8ts, new_8ts, old_8t, new_8t)
    else:
        print("\n[waiting] results/grids_8asset_tsla_hires.npz not yet available")

    print()


if __name__ == "__main__":
    main()
