# QAOA Noise Landscape

Code and results for:

> D. S. Meliksetian, *Noise in Constrained QAOA on IBM Heron Hardware:
> Landscape Compression Tracks Feasibility Loss*, arXiv:2604.19426 (v2).

Version 1 of the preprint (April 2026) proposed Landscape Span Compression
(LSC) as a device benchmark, based on a pilot at n = 6–8. Version 2 reports a
scaling study to n = 16 on `ibm_fez` (IBM Heron r2) and withdraws that claim:
for penalty-encoded constrained problems, hardware LSC grows with circuit size
and tracks the loss of feasible samples. Section V of the paper lists every
change from version 1.

---

## Repository contents

```
.
├── src/                    # Grid execution, energy computation, ZNE helpers
├── run_experiment.py       # QAOA grid experiment runner (sim / noisy sim / hardware)
├── run_zne.py              # Zero-noise extrapolation (EstimatorV2), v1 pilot only
├── run_sa_baseline.py      # Simulated-annealing and random-search baseline
├── run_figures.py          # Figures of paper version 1
├── prepare_portfolio.py    # Build portfolio QUBO instances from market data
├── compare_reruns.py       # Compare repeated runs of the same instance
├── p_value.py              # Significance tests
├── figures/                # All figures, including those of paper version 2
├── circuits/               # Logical and transpiled circuit diagrams
├── results/                # Summaries, calibration snapshots, follow-up analyses
└── paper/                  # LaTeX source and PDF of the preprint
```

Shared code (QUBO builder, circuit factory, plotting) and the problem
instances live in the companion repository
[`qaoa-common`](https://github.com/dmeliksetian/qaoa-common).

---

## Setup

Python 3.12 or later.

```bash
# Core Qiskit stack (versions used for the paper)
pip install qiskit==2.3.1 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.46.1
pip install numpy scipy matplotlib

# Shared package and instances
git clone https://github.com/dmeliksetian/qaoa-common
pip install -e qaoa-common

# Only needed to rebuild instances from market data
pip install yfinance scikit-learn
```

`qiskit-aer` GPU simulation needs cuQuantum and the CUDA runtime on
`LD_LIBRARY_PATH`; CPU simulation works without them.

### IBM Quantum credentials

Needed for calibration-based noisy simulation and hardware runs. Save your own
credentials once; nothing is read from this repository.

```bash
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='YOUR_IBM_CLOUD_API_KEY',
    instance='YOUR_IBM_CLOUD_CRN',
    name='ibm_cloud_account',
)
"
```

---

## Instances

Seven cardinality-constrained mean–variance portfolio QUBOs (k = n/2), built
from daily prices January 2019 – December 2023. The instance files are in
`qaoa-common/instances/`.

| Paper name  | n  | Instance file                          | E*        |
|-------------|----|----------------------------------------|-----------|
| 6 low-vol   | 6  | `6asset_lowvol_hires_unnorm.npz`       | −16.885   |
| 8 low-vol   | 8  | `8asset_lowvol_hires.npz`              | −46.265   |
| 8 TSLA      | 8  | `8asset_tsla_hires.npz`                | −56.563   |
| 12 low-vol  | 12 | `12asset_lowvol_hires.npz`             | −156.361  |
| 12 high-vol | 12 | `12asset_highvol_hires.npz`            | −264.648  |
| 16 low-vol  | 16 | `16asset_lowvol_hires.npz`             | −434.338  |
| 16 high-vol | 16 | `16asset_highvol_hires.npz`            | −1134.467 |

The files `6asset_lowvol_hires.npz`, `8asset_lowvol.npz` and `8asset_tsla.npz`
in `qaoa-common` are normalized variants (max |Q| = 1) and are not used in
version 2 of the paper.

---

## Reproducing the scaling study

Each instance was run on a 7×7 grid (half-width 0.4 rad) at 4,096 and 8,192
shots per point, in three conditions: noiseless simulation, calibration-based
noisy simulation, and hardware.

```bash
# Simulation
python run_experiment.py --qubo ../qaoa-common/instances/12asset_lowvol_hires.npz \
    --mode sim --n_points 7 --shots 4096

# Hardware (uses QPU time; the full scaling study used about 1,000 s on ibm_fez)
python run_experiment.py --qubo ../qaoa-common/instances/12asset_lowvol_hires.npz \
    --mode full --backend ibm_fez --n_points 7 --shots 8192
```

Each run writes `results/grids_<label>.npz` (per-point energies, counts and
feasibility for all three conditions) and `results/summary_<label>.json`.

### Results in this repository

| Files | Contents |
|-------|----------|
| `results/summary_<instance>[_4096shots\|_8192shots].json` | Reference point, E*, per-condition summaries for the scaling study |
| `results/backend_props_*.json` | Device calibration snapshot taken with each hardware run |
| `results/wp0_convergence.json` | Reference points re-optimized on the exact statevector (paper Sec. III-D) |
| `results/wp0_ops_violation_cost.json` | Best-feasible AR and feasible fraction at the ideal and hardware argmins (Sec. III-D, III-E) |
| `results/*_pittsburgh.json`, `*_nighthawk.json` | Pilot runs on `ibm_pittsburgh` (Heron r3) and `ibm_miami` (Nighthawk) |
| `results/zne_results.json`, `results/sa_baseline.json` | Version 1 pilot: ZNE and classical baseline |

In `zne_results.json`, use only `e_raw`, `e_zne`, `std_raw` and `std_zne`;
the `ar_*` fields carry a Hamiltonian constant offset.

### Data availability

The per-point grid files (`results/grids_*.npz`, including raw shot counts)
are too large for this repository and are archived at
**[Zenodo DOI — to be added]**. All numbers in the paper can be recomputed
from them.

---

## How to cite

```bibtex
@misc{meliksetian2026noise,
  author        = {Meliksetian, Dikran S.},
  title         = {Noise in Constrained {QAOA} on {IBM} Heron Hardware:
                   Landscape Compression Tracks Feasibility Loss},
  year          = {2026},
  eprint        = {2604.19426},
  archivePrefix = {arXiv},
}
```

---

## Acknowledgment

Pilot hardware access was provided by IBM Quantum through a promotion to the
IBM Quantum Open Plan. The scaling study used resources of the National Energy
Research Scientific Computing Center (NERSC) IBM Quantum Innovation Center
program, a U.S. Department of Energy Office of Science User Facility, under
NERSC Proposal QPN26-016.

## License

Code is released under the MIT License (see `LICENSE`).
