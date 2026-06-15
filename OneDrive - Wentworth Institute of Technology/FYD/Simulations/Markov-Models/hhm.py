import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


# ── Gaussian HMM — Baum-Welch (EM) from scratch ──────────────────────────────
#
# Model (Section 2.1):
#   Z_t in {0, 1}          latent state (bull / bear)
#   Y_t | Z_t = j  ~  N(mu_j, sigma_j^2)
#   Transition matrix A:   A[i,j] = P(Z_t=j | Z_{t-1}=i)
#
# Algorithms implemented:
#   forward      — scaled alpha pass
#   backward     — scaled beta pass
#   baum_welch   — EM parameter estimation
#   viterbi      — most likely state sequence

def _emissions(obs, mus, sigmas):
    """B[t, j] = N(obs_t; mu_j, sigma_j)  shape (T, K)"""
    return np.column_stack([norm.pdf(obs, mus[j], sigmas[j])
                            for j in range(len(mus))])


def _forward(B, pi, A):
    """Scaled forward pass. Returns alpha (T,K) and scales (T,)."""
    T, K = B.shape
    alpha  = np.zeros((T, K))
    scales = np.zeros(T)

    alpha[0]  = pi * B[0]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]

    for t in range(1, T):
        alpha[t]  = (alpha[t - 1] @ A) * B[t]
        scales[t] = alpha[t].sum()
        alpha[t] /= scales[t]

    return alpha, scales


def _backward(B, A, scales):
    """Scaled backward pass. Returns beta (T,K)."""
    T, K = B.shape
    beta       = np.zeros((T, K))
    beta[T - 1] = 1.0 / scales[T - 1]

    for t in range(T - 2, -1, -1):
        beta[t] = (A * (B[t + 1] * beta[t + 1])).sum(axis=1) / scales[t]

    return beta


def baum_welch(obs, n_states=2, n_iter=200, tol=1e-5):
    """
    Fit a Gaussian HMM via the Baum-Welch (EM) algorithm.

    Parameters
    ----------
    obs        : 1-D array of observations
    n_states   : number of hidden states (K)
    n_iter     : max EM iterations
    tol        : convergence threshold on log-likelihood

    Returns
    -------
    pi, A, mus, sigmas : fitted parameters
    gamma              : posterior state probabilities (T, K)
    """
    T, K = len(obs), n_states

    # ── Initialise ────────────────────────────────────────────────────────────
    pi     = np.ones(K) / K
    # Near-diagonal transition matrix so states are persistent
    A      = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(A, 0.9)
    # Split sorted observations into K chunks for initial means
    idx    = np.argsort(obs)
    chunk  = T // K
    mus    = np.array([obs[idx[i * chunk:(i + 1) * chunk]].mean()
                       for i in range(K)])
    sigmas = np.full(K, obs.std())

    prev_ll = -np.inf

    for iteration in range(n_iter):
        B              = _emissions(obs, mus, sigmas)
        alpha, scales  = _forward(B, pi, A)
        beta           = _backward(B, A, scales)

        # ── E-step: gamma and xi ──────────────────────────────────────────────
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)   # (T, K)

        # xi[t, i, j] = P(Z_t=i, Z_{t+1}=j | Y)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = (alpha[t, :, None] * A
                     * (B[t + 1] * beta[t + 1])[None, :])
            xi[t] /= xi[t].sum()

        # ── M-step: update parameters ─────────────────────────────────────────
        pi = gamma[0] / gamma[0].sum()

        xi_sum = xi.sum(axis=0)                     # (K, K)
        A      = xi_sum / xi_sum.sum(axis=1, keepdims=True)

        for j in range(K):
            w       = gamma[:, j]
            mus[j]  = (w * obs).sum() / w.sum()
            sigmas[j] = np.sqrt((w * (obs - mus[j]) ** 2).sum() / w.sum())

        ll = np.sum(np.log(scales + 1e-300))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return pi, A, mus, sigmas, gamma


def viterbi(obs, pi, A, mus, sigmas):
    """
    Viterbi algorithm — returns the most likely hidden state sequence.
    Uses log-space to avoid underflow.
    """
    T, K   = len(obs), len(pi)
    log_A  = np.log(A + 1e-300)

    log_delta = np.zeros((T, K))
    psi       = np.zeros((T, K), dtype=int)

    log_delta[0] = np.log(pi + 1e-300) + np.array(
        [norm.logpdf(obs[0], mus[j], sigmas[j]) for j in range(K)])

    for t in range(1, T):
        log_emit = np.array([norm.logpdf(obs[t], mus[j], sigmas[j])
                             for j in range(K)])
        candidates     = log_delta[t - 1, :, None] + log_A   # (K, K)
        psi[t]         = candidates.argmax(axis=0)
        log_delta[t]   = candidates.max(axis=0) + log_emit

    states       = np.zeros(T, dtype=int)
    states[T - 1] = log_delta[T - 1].argmax()
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


# ── Data ──────────────────────────────────────────────────────────────────────

TICKERS    = {"KO": "Coca-Cola", "PEP": "PepsiCo"}
START_DATE = "2007-09-01"
END_DATE   = "2017-09-01"


def load_weekly_returns(ticker):
    raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    close = raw["Close"].resample("W").last().dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    pct = close.pct_change().dropna()
    close = close.loc[pct.index]
    return close, pct


def fit_stock(ticker):
    close, pct = load_weekly_returns(ticker)
    obs = pct.values
    pi, A, mus, sigmas, gamma = baum_welch(obs)
    states = viterbi(obs, pi, A, mus, sigmas)
    bull = int(np.argmax(mus))
    bear = 1 - bull
    return {
        "close": close, "pct": pct, "obs": obs, "dates": pct.index,
        "pi": pi, "A": A, "mus": mus, "sigmas": sigmas,
        "gamma": gamma, "states": states,
        "bull": bull, "bear": bear,
        "is_bull": (states == bull),
    }


print("Fitting HMMs for Coca-Cola (KO) and PepsiCo (PEP)...")
results = {t: fit_stock(t) for t in TICKERS}

# Align on shared dates
dates_ko  = results["KO"]["dates"]
dates_pep = results["PEP"]["dates"]
shared    = dates_ko.intersection(dates_pep)

for t in TICKERS:
    r = results[t]
    mask = r["dates"].isin(shared)
    r["close"]   = r["close"].loc[shared]
    r["pct"]     = r["pct"].loc[shared]
    r["obs"]     = r["pct"].values
    r["dates"]   = shared
    r["gamma"]   = r["gamma"][mask]
    r["states"]  = r["states"][mask]
    r["is_bull"] = r["is_bull"][mask]

dates = shared

# ── Print summaries ───────────────────────────────────────────────────────────
for ticker, name in TICKERS.items():
    r = results[ticker]
    bull, bear = r["bull"], r["bear"]
    A, mus, sigmas = r["A"], r["mus"], r["sigmas"]
    print("=" * 55)
    print(f"  2-State Gaussian HMM  —  {ticker} ({name})")
    print("=" * 55)
    print(f"  Bull (j={bull}):  mu = {mus[bull]*100:.2f}%   sigma = {sigmas[bull]*100:.2f}%")
    print(f"  Bear (j={bear}):  mu = {mus[bear]*100:.2f}%   sigma = {sigmas[bear]*100:.2f}%")
    print(f"\n  Transition matrix:")
    print(f"    Bull→Bull: {A[bull,bull]:.3f}   Bull→Bear: {A[bull,bear]:.3f}")
    print(f"    Bear→Bull: {A[bear,bull]:.3f}   Bear→Bear: {A[bear,bear]:.3f}\n")

# Joint state: 0=both bear, 1=KO bull only, 2=PEP bull only, 3=both bull
ko_bull  = results["KO"]["is_bull"].astype(int)
pep_bull = results["PEP"]["is_bull"].astype(int)
joint    = ko_bull * 2 + pep_bull   # 0,1,2,3

joint_colors = {
    0: ("#d73027", "Both Bear"),
    1: ("#fee08b", "KO Bull / PEP Bear"),
    2: ("#91bfdb", "KO Bear / PEP Bull"),
    3: ("#1a9850", "Both Bull"),
}

# ── Figures ───────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Figure 1: Posterior P(bull) for both stocks
fig1, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
for ax, (ticker, name) in zip(axes, TICKERS.items()):
    r = results[ticker]
    ax.plot(dates, r["gamma"][:, r["bull"]], linewidth=0.8,
            color="steelblue" if ticker == "KO" else "darkorange")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Bull | Y)")
    ax.set_title(f"Posterior Bull Probability — {ticker} ({name})")
axes[-1].set_xlabel("Date")
plt.tight_layout()

# Figure 2: Closing prices colored by each stock's own bull/bear state
fig2, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
leg = [Line2D([0], [0], color="green", label="Bull"),
       Line2D([0], [0], color="red",   label="Bear")]
for ax, (ticker, name) in zip(axes, TICKERS.items()):
    r = results[ticker]
    close_vals = r["close"].values
    for i in range(len(dates) - 1):
        c = "green" if r["is_bull"][i] else "red"
        ax.plot(dates[i:i+2], close_vals[i:i+2], color=c, linewidth=0.9)
    ax.legend(handles=leg, title="Regime")
    ax.set_ylabel("Close ($)")
    ax.set_title(f"{ticker} ({name}) — Weekly Close with Bull/Bear Regime")
axes[-1].set_xlabel("Date")
plt.tight_layout()

# Figure 3: Joint regime relationship (KO vs PEP states)
fig3, ax3 = plt.subplots(figsize=(13, 3))
for code, (color, label) in joint_colors.items():
    mask = joint == code
    if mask.any():
        ax3.fill_between(dates, 0, 1,
                         where=mask, color=color, alpha=0.85, label=label,
                         transform=ax3.get_xaxis_transform())
ax3.set_yticks([])
ax3.set_xlim(dates[0], dates[-1])
ax3.set_xlabel("Date")
ax3.set_title("Joint Bull/Bear Regime: Coca-Cola vs PepsiCo")
ax3.legend(loc="upper right", ncol=2, fontsize=9,
           handles=[Patch(color=c, label=l) for c, l in
                    [(v[0], v[1]) for v in joint_colors.values()]])
plt.tight_layout()

# Figure 4: Regime agreement rate rolling 52-week window
agreement = (ko_bull == pep_bull).astype(float)
agree_series = pd.Series(agreement, index=dates)
rolling_agree = agree_series.rolling(52, min_periods=26).mean()

fig4, ax4 = plt.subplots(figsize=(13, 3))
ax4.plot(dates, rolling_agree, color="purple", linewidth=1.0,
         label="52-week rolling agreement")
ax4.axhline(rolling_agree.mean(), color="grey", linestyle="--", linewidth=0.8,
            label=f"Overall: {rolling_agree.mean():.1%}")
ax4.set_ylim(0, 1)
ax4.set_ylabel("Fraction in same regime")
ax4.set_title("Rolling Regime Agreement — KO vs PEP")
ax4.set_xlabel("Date")
ax4.legend()
plt.tight_layout()

plt.show()

# ── Regime co-occurrence table ────────────────────────────────────────────────
total = len(dates)
print("\nJoint regime co-occurrence:")
print(f"  {'State':<25} {'Weeks':>6}  {'Share':>6}")
for code, (_, label) in joint_colors.items():
    n = (joint == code).sum()
    print(f"  {label:<25} {n:>6}  {n/total:>6.1%}")
overall = (ko_bull == pep_bull).mean()
print(f"\n  Overall agreement rate: {overall:.1%}")
