"""
Simple Random Walk — Presentation Figure
FYD: Stochastic Portfolio Optimization with Regime Switching

Mathematical chain: [1] Random Walk  -->  [2] Brownian Motion  -->  [3] GBM  -->  ...
This figure covers Step 1.
"""

import matplotlib
matplotlib.use("Agg")   # save-only; swap to "TkAgg" / "Qt5Agg" for interactive window

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Parameters ────────────────────────────────────────────────────────────────
N_STEPS = 500           # discrete time steps
N_PATHS = 10            # individual paths to draw
N_DIST  = 30_000        # MC samples for final-position histogram

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0D1117"
GRID_C  = "#1F2937"
TEXT_C  = "#E5E7EB"
MUTED_C = "#6B7280"

path_colors = plt.cm.plasma(np.linspace(0.15, 0.90, N_PATHS))

# ── Figure / axes ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor=BG)
gs  = gridspec.GridSpec(
    1, 2,
    width_ratios=[3, 1],
    wspace=0.04,
    left=0.07, right=0.97,
    top=0.87,  bottom=0.12,
)
ax_w = fig.add_subplot(gs[0])   # random walk paths
ax_d = fig.add_subplot(gs[1])   # final-position distribution


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED_C, labelsize=11, length=4)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_C, linewidth=0.6)


_style(ax_w)
_style(ax_d)

# ── Left panel: paths ─────────────────────────────────────────────────────────
n_idx = np.arange(N_STEPS + 1)

for i in range(N_PATHS):
    xi   = RNG.choice([-1, 1], size=N_STEPS)
    path = np.concatenate(([0], np.cumsum(xi)))
    ax_w.plot(n_idx, path, alpha=0.75, linewidth=1.1,
              color=path_colors[i], zorder=3)

# Diffusion envelopes:  Var(X_t) = n  -->  std = sqrt(n)
sigma_env = np.sqrt(n_idx)

ax_w.fill_between(n_idx, -2 * sigma_env, 2 * sigma_env,
                  alpha=0.06, color="#FFFFFF", zorder=1,
                  label=r"$\pm 2\sigma = \pm 2\sqrt{n}$")
ax_w.fill_between(n_idx, -sigma_env, sigma_env,
                  alpha=0.12, color="#FFFFFF", zorder=2,
                  label=r"$\pm\,\sigma = \pm\sqrt{n}$")
ax_w.plot(n_idx,  sigma_env, color="#FFFFFF", lw=0.9,
          ls="--", alpha=0.35, zorder=2)
ax_w.plot(n_idx, -sigma_env, color="#FFFFFF", lw=0.9,
          ls="--", alpha=0.35, zorder=2)

ax_w.axhline(0, color=MUTED_C, linewidth=0.8, zorder=1)

ax_w.set_xlabel("Step  $n$", fontsize=13)
ax_w.set_ylabel("Position  $X_t$", fontsize=13)
ax_w.set_xlim(0, N_STEPS)

ax_w.legend(
    loc="upper left", fontsize=10, framealpha=0.25,
    labelcolor=TEXT_C, edgecolor=GRID_C, facecolor="#1A2030",
)

# Model equation box (using only matplotlib-supported mathtext)
ax_w.text(
    0.985, 0.04,
    r"$X_t = \sum_{i=1}^{n} Y_i, \quad Y_i \in \{-1,\,+1\}$",
    transform=ax_w.transAxes,
    fontsize=11, color="#9CA3AF",
    ha="right", va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="#161B22", alpha=0.90,
        edgecolor=GRID_C, linewidth=0.8,
    ),
)

# ── Right panel: final-position histogram ─────────────────────────────────────
finals = RNG.choice([-1, 1], size=(N_DIST, N_STEPS)).sum(axis=1)

ax_d.hist(
    finals, bins=55, orientation="horizontal",
    density=True, color="#7C3AED", alpha=0.55,
    edgecolor="none", zorder=2,
)

# Theoretical N(0, N_STEPS)
y_rng    = np.linspace(finals.min(), finals.max(), 500)
sigma_th = np.sqrt(N_STEPS)
pdf      = norm.pdf(y_rng, loc=0, scale=sigma_th)

ax_d.plot(pdf, y_rng,
          color="#38BDF8", linewidth=2.2, zorder=3,
          label=r"$\mathcal{N}(0,\,n)$")

# Align y-axes with the walk panel
y_lo, y_hi = ax_w.get_ylim()
ax_d.set_ylim(y_lo, y_hi)
ax_d.set_yticklabels([])
ax_d.set_xlabel("Density", fontsize=11)
ax_d.set_title(f"$X_{{500}}$\nDistribution", fontsize=12, pad=8,
               color=TEXT_C)
ax_d.xaxis.set_major_locator(MaxNLocator(3))
ax_d.legend(
    loc="upper right", fontsize=9, framealpha=0.25,
    labelcolor=TEXT_C, edgecolor=GRID_C, facecolor="#1A2030",
)

# ── Figure-level titles ───────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    "Simple Random Walk  →  Brownian Motion",
    ha="center", fontsize=18, fontweight="bold", color=TEXT_C,
)
fig.text(
    0.5, 0.910,
    r"CLT:  as $n \to \infty$,   $X_t / \sqrt{n} \to \mathcal{N}(0,1)$",
    ha="center", fontsize=12, color=MUTED_C,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "random_walk_figure.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out_path}")
