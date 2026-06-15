"""
Stochastic Population Growth — SDE Explanation Figure
FYD: Stochastic Portfolio Optimization with Regime Switching

Model: Stochastic Logistic Growth
  dN = rN(1 - N/K) dt  +  sigma*N dW_t
       [  drift       ]    [ diffusion ]

Numerical scheme: Euler-Maruyama
  N_{t+dt} = N_t + rN_t(1-N_t/K)*dt + sigma*N_t*sqrt(dt)*Z,  Z ~ N(0,1)
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Model parameters ──────────────────────────────────────────────────────────
r     = 0.6       # intrinsic growth rate  (yr^-1)
K     = 1000.0    # carrying capacity
N0    = 40.0      # initial population
sigma = 0.22      # noise intensity
T     = 14.0      # total time (years)
dt    = 0.005     # time step
n_steps = int(T / dt)
t_arr   = np.linspace(0, T, n_steps + 1)

N_PATHS = 25
N_DIST  = 8_000   # paths for final-time distribution

# ── Deterministic logistic solution (analytic) ────────────────────────────────
exp_rt = np.exp(r * t_arr)
N_det  = K * N0 * exp_rt / (K + N0 * (exp_rt - 1))

# ── Euler-Maruyama simulator ──────────────────────────────────────────────────
def euler_maruyama(n_paths, rng):
    N = np.full(n_paths, N0)
    out = np.zeros((n_paths, n_steps + 1))
    out[:, 0] = N0
    for j in range(n_steps):
        drift     = r * N * (1.0 - N / K) * dt
        diffusion = sigma * N * np.sqrt(dt) * rng.standard_normal(n_paths)
        N = np.maximum(N + drift + diffusion, 0.0)
        out[:, j + 1] = N
    return out

paths = euler_maruyama(N_PATHS, RNG)

dist_rng  = np.random.default_rng(7)
all_paths = euler_maruyama(N_DIST, dist_rng)
finals    = all_paths[:, -1]

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0D1117"
GRID_C  = "#1F2937"
TEXT_C  = "#E5E7EB"
MUTED_C = "#6B7280"
DET_C   = "#FBBF24"    # gold  — deterministic baseline

path_colors = plt.cm.plasma(np.linspace(0.12, 0.88, N_PATHS))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), facecolor=BG)
gs  = gridspec.GridSpec(
    1, 2,
    width_ratios=[3, 1],
    wspace=0.04,
    left=0.07, right=0.97,
    top=0.87,  bottom=0.12,
)
ax_p = fig.add_subplot(gs[0])
ax_d = fig.add_subplot(gs[1])


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED_C, labelsize=11, length=4)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_C, linewidth=0.6)


_style(ax_p)
_style(ax_d)

# ── Left panel: stochastic paths + deterministic baseline ─────────────────────
for i in range(N_PATHS):
    ax_p.plot(t_arr, paths[i], alpha=0.50, linewidth=0.9,
              color=path_colors[i], zorder=2)

ax_p.plot(t_arr, N_det,
          color=DET_C, linewidth=2.4, zorder=4,
          label=r"Deterministic  $dN/dt = rN(1-N/K)$")

ax_p.axhline(K, color=MUTED_C, linewidth=1.1, linestyle=":",
             zorder=1, label=f"Carrying capacity  $K = {int(K)}$")

ax_p.set_xlabel("Time  $t$  (years)", fontsize=13)
ax_p.set_ylabel("Population  $N(t)$", fontsize=13)
ax_p.set_xlim(0, T)
ax_p.set_ylim(bottom=0)

ax_p.legend(loc="center right", fontsize=10, framealpha=0.25,
            labelcolor=TEXT_C, edgecolor=GRID_C, facecolor="#1A2030")

# ── Drift / diffusion annotation arrows ───────────────────────────────────────
# Arrow pointing at the deterministic curve (drift)
t_arrow = T * 0.30
N_arrow = float(np.interp(t_arrow, t_arr, N_det))
ax_p.annotate(
    r"drift:  $rN(1-N/K)\,dt$",
    xy=(t_arrow, N_arrow),
    xytext=(t_arrow - 1.2, N_arrow - 210),
    color=DET_C, fontsize=10,
    arrowprops=dict(arrowstyle="-|>", color=DET_C, lw=1.3),
    zorder=5,
)

# Arrow pointing at a high stochastic path (diffusion)
t_diff  = T * 0.60
idx_d   = int(t_diff / dt)
high_N  = float(paths[:, idx_d].max())
ax_p.annotate(
    r"diffusion:  $\sigma N\,dW_t$",
    xy=(t_diff, high_N),
    xytext=(t_diff + 0.6, high_N + 110),
    color="#38BDF8", fontsize=10,
    arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=1.3),
    zorder=5,
)

# SDE equation box (bottom-right of left panel)
ax_p.text(
    0.985, 0.05,
    r"$dN = rN(1-N/K)\,dt \;+\; \sigma N\,dW_t$",
    transform=ax_p.transAxes,
    fontsize=11, color="#9CA3AF",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#161B22",
              alpha=0.90, edgecolor=GRID_C, linewidth=0.8),
)

# ── Right panel: N(T) distribution ────────────────────────────────────────────
ax_d.hist(finals, bins=55, orientation="horizontal",
          density=True, color="#7C3AED", alpha=0.55,
          edgecolor="none", zorder=2)

kde   = gaussian_kde(finals, bw_method=0.15)
y_rng = np.linspace(max(0.0, finals.min()), finals.max(), 500)
ax_d.plot(kde(y_rng), y_rng,
          color="#38BDF8", linewidth=2.2, zorder=3, label="KDE")

# Mark deterministic final value
N_det_T = float(N_det[-1])
ax_d.axhline(N_det_T, color=DET_C, linewidth=1.6, linestyle="--",
             zorder=4, label=f"Det.  $N(T)$")

y_lo, y_hi = ax_p.get_ylim()
ax_d.set_ylim(y_lo, y_hi)
ax_d.set_yticklabels([])
ax_d.set_xlabel("Density", fontsize=11)
ax_d.set_title(f"$N({int(T)})$\nDistribution", fontsize=12, pad=8, color=TEXT_C)
ax_d.xaxis.set_major_locator(MaxNLocator(3))
ax_d.legend(loc="upper right", fontsize=8, framealpha=0.25,
            labelcolor=TEXT_C, edgecolor=GRID_C, facecolor="#1A2030")

# ── Figure-level titles ───────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    "Stochastic Population Growth  —  What is an SDE?",
    ha="center", fontsize=18, fontweight="bold", color=TEXT_C,
)
fig.text(
    0.5, 0.910,
    r"Euler-Maruyama:  $N_{t+\Delta t} = N_t + rN_t(1-N_t/K)\Delta t"
    r" + \sigma N_t \sqrt{\Delta t}\,Z_t, \quad Z_t \sim \mathcal{N}(0,1)$",
    ha="center", fontsize=11, color=MUTED_C,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "stochastic_population_figure.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out_path}")
