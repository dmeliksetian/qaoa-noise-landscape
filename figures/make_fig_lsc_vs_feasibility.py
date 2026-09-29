"""LSC_hw vs feasibility-range compression, 7 ibm_fez instances, 4,096 shots/pt.
Values from reports/nersc_phase1_report.tex (feasibility-range table), which were
re-derived from results/grids_*.npz."""
import matplotlib.pyplot as plt
from scipy import stats
labels = ['6 low-vol', '8 low-vol', '8 TSLA', '12 low-vol', '12 high-vol', '16 low-vol', '16 high-vol']
n      = [6, 8, 8, 12, 12, 16, 16]
feas   = [0.355, 0.346, 0.416, 0.634, 0.543, 0.805, 0.805]
lsc    = [0.248, 0.240, 0.291, 0.550, 0.660, 0.811, 0.784]
r, p = stats.pearsonr(feas, lsc)
colors = {6: '#1D9E75', 8: '#2B6CB0', 12: '#BA7517', 16: '#C0392B'}
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.plot([0.2, 0.9], [0.2, 0.9], ls='--', lw=0.8, color='0.6', zorder=0)
for f, l, k, lab in zip(feas, lsc, n, labels):
    ax.scatter(f, l, s=28, color=colors[k], zorder=2)
    dx, dy = (0.012, -0.035) if lab in ('8 low-vol', '16 high-vol') else (0.012, 0.012)
    ax.annotate(lab, (f, l), xytext=(f + dx, l + dy), fontsize=6.5)
ax.set_xlabel('Feasibility-range compression\n' r'$1-\mathrm{range}(\mathrm{FF}_{hw})/\mathrm{range}(\mathrm{FF}_{ideal})$', fontsize=8)
ax.set_ylabel(r'$\mathrm{LSC}_{hw}$', fontsize=8)
ax.set_xlim(0.2, 0.9); ax.set_ylim(0.2, 0.9)
ax.tick_params(labelsize=7)
ax.text(0.22, 0.85, f'Pearson $r={r:.3f}$, $p={p:.4f}$', fontsize=7)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/fig_lsc_vs_feasibility_compression.{ext}', dpi=300)
print(r, p)
