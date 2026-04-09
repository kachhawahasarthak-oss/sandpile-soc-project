"""
Enhanced Abelian Sandpile Model Simulation
Self-Organized Criticality — IIT Delhi Complexity Science Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Global plot style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

BLUE   = '#1f77b4'
ORANGE = '#d62728'
GREEN  = '#2ca02c'
PURPLE = '#9467bd'

# ── Simulation ────────────────────────────────────────────────────────────────
def run_sandpile(N=50, threshold=4, iterations=100_000, seed=42):
    rng = np.random.default_rng(seed)
    grid = np.zeros((N, N), dtype=np.int32)
    avalanches = np.zeros(iterations, dtype=np.int32)

    for t in range(iterations):
        i, j = rng.integers(0, N, size=2)
        grid[i, j] += 1
        size = 0
        while True:
            xs, ys = np.where(grid >= threshold)
            if xs.size == 0:
                break
            for x, y in zip(xs, ys):
                if grid[x, y] >= threshold:
                    grid[x, y] -= threshold
                    size += 1
                    if x > 0:     grid[x-1, y] += 1
                    if x < N-1:   grid[x+1, y] += 1
                    if y > 0:     grid[x, y-1] += 1
                    if y < N-1:   grid[x, y+1] += 1
        avalanches[t] = size

    return grid, avalanches


# ── Helper: log-log binned PDF ────────────────────────────────────────────────
def logbin(data, n_bins=40):
    lo = np.log10(data.min())
    hi = np.log10(data.max())
    edges = np.logspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(data, bins=edges)
    widths = np.diff(edges)
    density = counts / (data.size * widths)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = density > 0
    return centers[mask], density[mask]


# ── Run ───────────────────────────────────────────────────────────────────────
print("Running sandpile simulation (N=50, 100 000 drops)…")
grid, avalanches = run_sandpile()
aval = avalanches[avalanches > 0]

print(f"  Non-zero avalanches : {len(aval):,}")
print(f"  Mean size           : {aval.mean():.1f}")
print(f"  Max size            : {aval.max():,}")
print(f"  Std dev             : {aval.std():.1f}")

# Power-law fit (MLE estimator for discrete power law)
s_min = 10            # fit above s_min to avoid finite-size effects
aval_fit = aval[aval >= s_min]
alpha_mle = 1 + len(aval_fit) / np.sum(np.log(aval_fit / (s_min - 0.5)))
print(f"  MLE exponent α      : {alpha_mle:.3f}")

# OLS fit on log-binned PDF
centers, density = logbin(aval)
mask_fit = centers >= s_min
log_c = np.log10(centers[mask_fit])
log_d = np.log10(density[mask_fit])
slope, intercept, r_val, p_val, _ = stats.linregress(log_c, log_d)
alpha_ols = -slope
print(f"  OLS exponent α      : {alpha_ols:.3f}  (R²={r_val**2:.4f})")

# Cumulative distribution
aval_sorted = np.sort(aval)
ccdf = 1 - np.arange(1, len(aval_sorted)+1) / len(aval_sorted)

# Temporal evolution (avalanche size over time)
window = 500
rolling_mean = np.convolve(avalanches, np.ones(window)/window, mode='valid')

# ── FIGURE 1: Heatmap ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
cmap = LinearSegmentedColormap.from_list('sandpile', ['#1a1a2e', '#e94560', '#f5a623', '#ffffff'])
im = ax.imshow(grid, cmap=cmap, origin='lower', vmin=0, vmax=3)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Grain count', fontsize=11)
cbar.set_ticks([0, 1, 2, 3])
ax.set_title('Final Sandpile Configuration (50×50 grid)', fontweight='bold')
ax.set_xlabel('Column index')
ax.set_ylabel('Row index')
ax.grid(False)
plt.tight_layout()
plt.savefig('/home/claude/fig1_heatmap.png')
plt.close()
print("Saved fig1_heatmap.png")

# ── FIGURE 2: Log-log PDF with fit ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(centers, density, s=20, color=BLUE, alpha=0.7, label='Empirical PDF', zorder=3)
fit_line = 10**intercept * centers[mask_fit]**slope
ax.plot(centers[mask_fit], fit_line, color=ORANGE, linewidth=2,
        linestyle='--', label=f'Power-law fit  α = {alpha_ols:.2f}  (R²={r_val**2:.3f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Avalanche size  $s$')
ax.set_ylabel('Probability density  $P(s)$')
ax.set_title('Avalanche Size Distribution (log–log scale)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, which='both', alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig('/home/claude/fig2_loglog_pdf.png')
plt.close()
print("Saved fig2_loglog_pdf.png")

# ── FIGURE 3: CCDF ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(aval_sorted, ccdf, color=PURPLE, linewidth=1.2, alpha=0.8, label='Empirical CCDF')
# Reference slope
s_ref = np.array([s_min, 5000])
ax.loglog(s_ref, (s_ref/s_min)**(-(alpha_mle-1)), '--', color=ORANGE, linewidth=2,
          label=f'Power-law reference  α−1 = {alpha_mle-1:.2f}')
ax.set_xlabel('Avalanche size  $s$')
ax.set_ylabel('$P(S \geq s)$  (complementary CDF)')
ax.set_title('Complementary Cumulative Distribution Function', fontweight='bold')
ax.legend()
ax.grid(True, which='both', alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig('/home/claude/fig3_ccdf.png')
plt.close()
print("Saved fig3_ccdf.png")

# ── FIGURE 4: Temporal evolution ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

# Panel (a): raw avalanche time series (first 5000 steps for visibility)
axes[0].plot(avalanches[:5000], color=BLUE, alpha=0.5, linewidth=0.5)
axes[0].set_ylabel('Avalanche size')
axes[0].set_title('(a) Avalanche time series (first 5 000 drops)', fontweight='bold')
axes[0].set_xlabel('Time step')

# Panel (b): rolling mean over all 100k steps
axes[1].plot(rolling_mean, color=GREEN, linewidth=0.8, alpha=0.9)
axes[1].axhline(aval.mean(), color=ORANGE, linewidth=1.5, linestyle='--',
                label=f'Global mean = {aval.mean():.1f}')
axes[1].set_ylabel(f'Rolling mean (window={window})')
axes[1].set_title('(b) Rolling-mean avalanche size — convergence to steady state', fontweight='bold')
axes[1].set_xlabel('Time step')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/claude/fig4_temporal.png')
plt.close()
print("Saved fig4_temporal.png")

# ── FIGURE 5: Grain-height histogram ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4))
vals, counts_ = np.unique(grid.flatten(), return_counts=True)
bars = ax.bar(vals, counts_ / grid.size * 100, color=[BLUE, GREEN, ORANGE, PURPLE][:len(vals)],
              edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, counts_):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v/grid.size*100:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_xlabel('Number of grains per cell')
ax.set_ylabel('Fraction of cells (%)')
ax.set_title('Final Grain-Height Distribution', fontweight='bold')
ax.set_xticks(vals)
ax.set_xticklabels([str(v) for v in vals])
plt.tight_layout()
plt.savefig('/home/claude/fig5_height_hist.png')
plt.close()
print("Saved fig5_height_hist.png")

# ── Print summary stats table ─────────────────────────────────────────────────
q = np.percentile(aval, [25, 50, 75, 90, 99])
print("\n=== SUMMARY TABLE ===")
print(f"Total drops          : 100,000")
print(f"Non-zero avalanches  : {len(aval):,}  ({len(aval)/1000:.1f}%)")
print(f"Mean size            : {aval.mean():.2f}")
print(f"Median size          : {np.median(aval):.0f}")
print(f"Std deviation        : {aval.std():.2f}")
print(f"Max size             : {aval.max():,}")
print(f"25th percentile      : {q[0]:.0f}")
print(f"75th percentile      : {q[2]:.0f}")
print(f"90th percentile      : {q[3]:.0f}")
print(f"99th percentile      : {q[4]:.0f}")
print(f"MLE exponent α       : {alpha_mle:.4f}")
print(f"OLS exponent α       : {alpha_ols:.4f}")
print(f"OLS R²               : {r_val**2:.4f}")

# Grain distribution
print("\n=== GRAIN HEIGHT DISTRIBUTION ===")
for v, c in zip(vals, counts_):
    print(f"  {v} grains: {c:,} cells ({c/grid.size*100:.1f}%)")
