"""
run_figures.py — generate all paper figures to figures/ as .pdf and .png at 300 DPI.
Figures 1–6 of the first version of the paper (arXiv:2604.19426v1).
"""

import os
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

os.makedirs('figures', exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    'ideal':    '#1D9E75',
    'noisy':    '#BA7517',
    'hardware': '#D85A30',
    'zne':      '#534AB7',
    'optimal':  '#2C2C2A',
    'random':   '#888780',
}

INSTANCE_COLORS = {
    '6asset_lowvol_hires': '#185FA5',
    '8asset_lowvol':       '#0F6E56',
    '8asset_tsla':         '#993C1D',
}

INSTANCE_LABELS = {
    '6asset_lowvol_hires': '6-asset (low vol)',
    '8asset_lowvol':       '8-asset (low vol)',
    '8asset_tsla':         '8-asset (TSLA)',
}

CONDITION_ORDER = ['Ideal sim', 'Noisy sim', 'Hardware']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
d6 = np.load('results/grids_6asset_lowvol_hires.npz', allow_pickle=True)
gammas6 = d6['gammas']
betas6  = d6['betas']
g6_ideal = d6['grid_ideal_sim']
g6_noisy = d6['grid_noisy_sim']
g6_hw    = d6['grid_hardware']
f6_ideal = d6['feasible_ideal']
f6_noisy = d6['feasible_noisy']
f6_hw    = d6['feasible_hardware']

s6 = json.load(open('results/summary_6asset_lowvol_hires.json'))
gamma_star6 = s6['gamma_star']
beta_star6  = s6['beta_star']

d8lv = np.load('results/grids_8asset_lowvol.npz', allow_pickle=True)
d8ts = np.load('results/grids_8asset_tsla.npz',   allow_pickle=True)

s8lv = json.load(open('results/summary_8asset_lowvol.json'))
s8ts = json.load(open('results/summary_8asset_tsla.json'))

summaries = {
    '6asset_lowvol_hires': s6,
    '8asset_lowvol':       s8lv,
    '8asset_tsla':         s8ts,
}

grids = {
    '6asset_lowvol_hires': {
        'gammas': gammas6, 'betas': betas6,
        'ideal': g6_ideal, 'noisy': g6_noisy, 'hardware': g6_hw,
    },
    '8asset_lowvol': {
        'gammas': d8lv['gammas'], 'betas': d8lv['betas'],
        'ideal': d8lv['grid_ideal_sim'], 'noisy': d8lv['grid_noisy_sim'], 'hardware': d8lv['grid_hardware'],
    },
    '8asset_tsla': {
        'gammas': d8ts['gammas'], 'betas': d8ts['betas'],
        'ideal': d8ts['grid_ideal_sim'], 'noisy': d8ts['grid_noisy_sim'], 'hardware': d8ts['grid_hardware'],
    },
}

zne = json.load(open('results/zne_results.json'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def span(g):
    return g.max() - g.min()


def compression_pct(g_ideal, g_hw):
    return (1 - span(g_hw) / span(g_ideal)) * 100


def save_fig(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(f'figures/{stem}.{ext}')
    print(f'  saved figures/{stem}.pdf/.png')


def heatmap_extent(gammas, betas):
    """[left, right, bottom, top] for imshow."""
    dg = (gammas[-1] - gammas[0]) / (len(gammas) - 1)
    db = (betas[-1]  - betas[0])  / (len(betas)  - 1)
    return [betas[0]  - db/2, betas[-1]  + db/2,
            gammas[0] - dg/2, gammas[-1] + dg/2]


# ---------------------------------------------------------------------------
# Figure 1 — Energy landscape heatmaps (3-panel)
# ---------------------------------------------------------------------------
print('Figure 1: energy landscape heatmaps …')

panels = [
    ('(a) Noiseless simulation', g6_ideal),
    ('(b) Noise model',          g6_noisy),
    ('(c) Hardware',             g6_hw),
]
ext = heatmap_extent(gammas6, betas6)

# Shared color limits across all three panels
all_vals = np.concatenate([g6_ideal.ravel(), g6_noisy.ravel(), g6_hw.ravel()])
vmin, vmax = all_vals.min(), all_vals.max()

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True,
                          gridspec_kw={'wspace': 0.08})

ims = []
for ax, (title, grid) in zip(axes, panels):
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=ext, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    ims.append(im)

    # White star at (γ*, β*)
    ax.plot(beta_star6, gamma_star6, '*', color='white', ms=12,
            markeredgecolor='#444', markeredgewidth=0.5, zorder=5)

    # Colored X at panel minimum
    idx = np.unravel_index(grid.argmin(), grid.shape)
    ax.plot(betas6[idx[1]], gammas6[idx[0]], 'x', color='cyan',
            ms=9, mew=2, zorder=5)

    # Span annotation
    s_val = span(grid)
    s_ref = span(g6_ideal)
    pct   = (s_val / s_ref) * 100
    ax.set_title(title)
    ax.text(0.97, 0.03,
            f'LS={s_val:.2f}\n({pct:.1f}% of ideal)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#222',
                      alpha=0.85, edgecolor='white', linewidth=0.6))

    ax.set_xlabel(r'$\beta$')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

axes[0].set_ylabel(r'$\gamma$')
cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.85, pad=0.02)
cbar.set_label('Energy expectation value', fontsize=9)
save_fig(fig, 'fig1_landscape')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Feasibility heatmaps (3-panel)
# ---------------------------------------------------------------------------
print('Figure 2: feasibility heatmaps …')

fpanels = [
    ('(a) Noiseless', f6_ideal),
    ('(b) Noise model', f6_noisy),
    ('(c) Hardware',    f6_hw),
]
fvmin, fvmax = 0.0, 0.75

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True,
                          gridspec_kw={'wspace': 0.08})

fims = []
for ax, (title, fgrid) in zip(axes, fpanels):
    fim = ax.imshow(fgrid, origin='lower', aspect='auto',
                    extent=ext, cmap='YlOrRd', vmin=fvmin, vmax=fvmax)
    fims.append(fim)

    # White star
    ax.plot(beta_star6, gamma_star6, '*', color='white', ms=12,
            markeredgecolor='#444', markeredgewidth=0.5, zorder=5)

    # Dashed contour at random baseline 0.3125
    ax.contour(betas6, gammas6, fgrid, levels=[0.3125],
               colors='black', linestyles='--', linewidths=1.2)

    ax.set_title(title)
    ax.set_xlabel(r'$\beta$')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

axes[0].set_ylabel(r'$\gamma$')
cbar = fig.colorbar(fims[-1], ax=axes, shrink=0.85, pad=0.02)
cbar.set_label('Feasibility fraction', fontsize=9)
save_fig(fig, 'fig2_feasibility')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — AR bar chart (cross-instance)
# ---------------------------------------------------------------------------
print('Figure 3: approximation ratio bar chart …')

instances = ['6asset_lowvol_hires', '8asset_lowvol', '8asset_tsla']
instance_labels_short = ['6-asset\n(low vol)', '8-asset\n(low vol)', '8-asset\n(TSLA)']
cond_colors = [COLORS['ideal'], COLORS['noisy'], COLORS['hardware']]
cond_labels = ['Ideal', 'Noisy sim', 'Hardware']

x   = np.arange(len(instances))
bw  = 0.22
offsets = [-bw, 0, bw]

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(10, 4),
                                        gridspec_kw={'wspace': 0.35})

def _draw_ar_bars(ax, ylim, annotate_cost):
    for i, (cond_key, color, label) in enumerate(
            zip(CONDITION_ORDER, cond_colors, cond_labels)):
        vals = [summaries[inst]['approximation_ratios'][cond_key] for inst in instances]
        ax.bar(x + offsets[i], vals, bw * 0.9, color=color, label=label,
               edgecolor='white', linewidth=0.5)
        if annotate_cost and cond_key == 'Hardware':
            for j, (inst, hw_val) in enumerate(zip(instances, vals)):
                ideal_val = summaries[inst]['approximation_ratios']['Ideal sim']
                cost = ideal_val - hw_val
                ax.text(x[j] + offsets[i], hw_val + 0.001,
                        f'−{cost:.3f}', ha='center', va='bottom',
                        fontsize=7, color=color)
    ax.axhline(1.0, color=COLORS['optimal'], linestyle='--', linewidth=1.2,
               label='Optimal (AR=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(instance_labels_short)
    ax.set_ylim(*ylim)
    ax.set_ylabel('Approximation ratio')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1 if ylim[0] == 0 else 0.02))
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

_draw_ar_bars(ax_full, (0.0, 1.08), annotate_cost=False)
ax_full.set_title('(a) Full range')
ax_full.text(0.5, 0.55,
             'All feasible solutions\nwithin 1.71% of optimum\n— AR saturates',
             transform=ax_full.transAxes, ha='center', va='center',
             fontsize=7.5, color='#444',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                       edgecolor='#aaa', linewidth=0.7))
ax_full.legend(framealpha=0.9, loc='lower right', fontsize=7)

_draw_ar_bars(ax_zoom, (0.88, 1.025), annotate_cost=True)
ax_zoom.set_title('(b) Noise cost detail')
ax_zoom.legend(framealpha=0.9, loc='lower right', fontsize=7)

save_fig(fig, 'fig3_ar_comparison')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Feasibility degradation line plot
# ---------------------------------------------------------------------------
print('Figure 4: feasibility degradation …')

cond_keys_feas = ['ideal', 'noisy', 'hardware']
x_labels = ['Ideal\nsim', 'Noisy\nsim', 'Hardware']
x_pos = np.arange(3)

fig, ax = plt.subplots(figsize=(5.5, 4))

for inst in instances:
    s = summaries[inst]
    vals = [s['feasibility_at_best'][k] for k in cond_keys_feas]
    n_qubits = 6 if '6asset' in inst else 8
    random_bl = s['random_baseline']
    ax.plot(x_pos, vals, 'o-', color=INSTANCE_COLORS[inst],
            label=INSTANCE_LABELS[inst], linewidth=1.8, ms=6)

# Random baseline lines
ax.axhline(s6['random_baseline'], color=COLORS['random'], linestyle='--',
           linewidth=1.1, label=f"6-asset random ({s6['random_baseline']:.4f})")
ax.axhline(s8lv['random_baseline'], color=COLORS['random'], linestyle=':',
           linewidth=1.1, label=f"8-asset random ({s8lv['random_baseline']:.4f})")

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylim(0.30, 0.72)
ax.set_ylabel('Feasibility fraction')
ax.legend(framealpha=0.9, loc='upper right')
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
save_fig(fig, 'fig4_feasibility_degradation')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Span compression bar chart
# ---------------------------------------------------------------------------
print('Figure 5: span compression …')

cond_keys_grid = ['ideal', 'noisy', 'hardware']
cond_labels5   = ['Ideal', 'Noisy sim', 'Hardware']
cond_colors5   = [COLORS['ideal'], COLORS['noisy'], COLORS['hardware']]

x   = np.arange(len(instances))
bw  = 0.22
offsets = [-bw, 0, bw]

fig, ax = plt.subplots(figsize=(6.5, 4))

for i, (ck, color, label) in enumerate(zip(cond_keys_grid, cond_colors5, cond_labels5)):
    vals = [span(grids[inst][ck]) for inst in instances]
    ax.bar(x + offsets[i], vals, bw * 0.9, color=color, label=label,
           edgecolor='white', linewidth=0.5)

# Annotate hardware bars with hardware LSC, noisy bars with noisy LSC
for j, inst in enumerate(instances):
    hw_cpct    = compression_pct(grids[inst]['ideal'], grids[inst]['hardware'])
    noisy_cpct = compression_pct(grids[inst]['ideal'], grids[inst]['noisy'])
    noisy_span = span(grids[inst]['noisy'])
    ideal_span = span(grids[inst]['ideal'])
    # Hardware annotation (bold, hardware color) above ideal bar
    ax.text(x[j], ideal_span + 0.3,
            f'HW: −{hw_cpct:.1f}%', ha='center', va='bottom',
            fontsize=8, color=COLORS['hardware'], fontweight='bold')
    # Noisy annotation (smaller, noisy color) above noisy bar
    ax.text(x[j] + offsets[1], noisy_span + 0.3,
            f'−{noisy_cpct:.1f}%', ha='center', va='bottom',
            fontsize=7, color=COLORS['noisy'])

# Bracket showing unexplained gap for the first instance
inst0 = instances[0]
noisy_s = span(grids[inst0]['noisy'])
hw_s    = span(grids[inst0]['hardware'])
x0      = x[0]
ax.annotate('', xy=(x0 + offsets[2] + bw*0.5, hw_s),
            xytext=(x0 + offsets[2] + bw*0.5, noisy_s),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
ax.text(x0 + offsets[2] + bw*0.6, (noisy_s + hw_s) / 2,
        'unexplained\nby model', ha='left', va='center',
        fontsize=6.5, color='#555', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(instance_labels_short)
ax.set_ylabel('Landscape energy span')
ax.legend(framealpha=0.9)
ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
save_fig(fig, 'fig5_span_compression')
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — ZNE results
# ---------------------------------------------------------------------------
print('Figure 6: ZNE results …')

zne_labels = {
    '6asset_lowvol_hires': '6-asset',
    '8asset_lowvol':       '8-asset\n(low vol)',
    '8asset_tsla':         '8-asset\n(TSLA)',
}
zne_instances = ['6asset_lowvol_hires', '8asset_lowvol', '8asset_tsla']
zne_x = np.arange(len(zne_instances))

e_raw  = np.array([zne[k]['e_raw']   for k in zne_instances])
e_zne_ = np.array([zne[k]['e_zne']   for k in zne_instances])
s_raw  = np.array([zne[k]['std_raw'] for k in zne_instances])
s_zne_ = np.array([zne[k]['std_zne'] for k in zne_instances])

pct_improve = (e_raw - e_zne_) / np.abs(e_raw) * 100   # positive = ZNE better
std_factor  = s_zne_ / s_raw

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8),
                                 gridspec_kw={'wspace': 0.35})

# --- Left panel: e_raw vs e_zne per instance ---
bw = 0.30
ax1.bar(zne_x - bw/2, e_raw,  bw, color=COLORS['hardware'], label='Raw HW',
        yerr=s_raw,  capsize=5,
        error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': '#555'})
ax1.bar(zne_x + bw/2, e_zne_, bw, color=COLORS['zne'],      label='ZNE',
        yerr=s_zne_, capsize=5,
        error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})

# Open headroom above y=0 so directional labels sit in clear whitespace
ax1.set_ylim(bottom=None, top=0.55)

ax1.set_xticks(zne_x)
ax1.set_xticklabels([zne_labels[k] for k in zne_instances])
ax1.set_ylabel('Energy expectation value\n(more negative = better)')
ax1.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
ax1.set_title('(a) Raw vs ZNE energy')
ax1.axhline(0, color='#bbb', linewidth=0.7)  # reference line at 0

# Directional labels in the clear whitespace above y=0
for j, pct in enumerate(pct_improve):
    if pct > 0:
        ann = f'ZNE +{pct:.1f}%\nbetter'
        color = '#1a7a1a'
    else:
        ann = f'ZNE {abs(pct):.1f}%\nworse'
        color = '#c0392b'
    ax1.text(zne_x[j], 0.07, ann,
             ha='center', va='bottom', fontsize=7.5, color=color,
             fontweight='bold', linespacing=1.3)

ax1.legend(framealpha=0.9, loc='lower left', fontsize=8)

# --- Right panel: std inflation ---
ax2.bar(zne_x, std_factor, color=COLORS['zne'], edgecolor='white', linewidth=0.5)
ax2.axhline(1.0, color='#555', linestyle='--', linewidth=1.0)
ax2.set_xticks(zne_x)
ax2.set_xticklabels([zne_labels[k] for k in zne_instances])
ax2.set_ylabel(r'Std inflation ($\sigma_{\rm ZNE}\,/\,\sigma_{\rm raw}$)')
ax2.set_title('(b) Uncertainty inflation')
ax2.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)

# Annotate values
for j, sf in enumerate(std_factor):
    ax2.text(zne_x[j], sf + 0.05, f'{sf:.1f}×',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

save_fig(fig, 'fig6_zne')
plt.close(fig)

print('\nAll figures saved to figures/')
