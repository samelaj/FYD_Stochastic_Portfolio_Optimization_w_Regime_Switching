"""
regime_hedger.py
────────────────
Regime-aware option pricing and delta-hedging for KO and PEP.

Pipeline:
  1. Fit Gaussian HMM on weekly returns  -> bull/bear sigma, mu, transition matrix A
  2. Compute P(bear) for the most recent observation
  3. Price ATM protective puts at bull-vol, bear-vol, and P-blended vol
  4. Compute Greeks (delta, gamma, vega) at each regime vol
  5. Simulate regime-switching GBM paths vs. constant-vol GBM
  6. Plot: regime dashboard, put cost vs P(bear), GBM comparison, Greeks comparison

Run:
    python regime_hedger.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yfinance as yf
from scipy.stats import norm

# ── Import Black-Scholes tools from sibling directory ─────────────────────────
_bs_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Black-Scholes")
)
sys.path.insert(0, _bs_path)
from black_scholes_simulator import (
    bs_call, bs_put, bs_greeks, simulate_gbm, market_inputs,
)


# ── HMM core ──────────────────────────────────────────────────────────────────
# Copied from hhm.py because hhm.py runs fitting code at module level
# (no __main__ guard), so importing it would re-run the whole script.

def _emissions(obs, mus, sigmas):
    return np.column_stack(
        [norm.pdf(obs, mus[j], sigmas[j]) for j in range(len(mus))]
    )


def _forward(B, pi, A):
    T, K   = B.shape
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
    T, K        = B.shape
    beta        = np.zeros((T, K))
    beta[T - 1] = 1.0 / scales[T - 1]
    for t in range(T - 2, -1, -1):
        beta[t] = (A * (B[t + 1] * beta[t + 1])).sum(axis=1) / scales[t]
    return beta


def baum_welch(obs, n_states=2, n_iter=200, tol=1e-5):
    T, K   = len(obs), n_states
    pi     = np.ones(K) / K
    A      = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(A, 0.9)
    idx    = np.argsort(obs)
    chunk  = T // K
    mus    = np.array([obs[idx[i * chunk:(i + 1) * chunk]].mean()
                       for i in range(K)])
    sigmas = np.full(K, obs.std())
    prev_ll = -np.inf
    for _ in range(n_iter):
        B             = _emissions(obs, mus, sigmas)
        alpha, scales = _forward(B, pi, A)
        beta          = _backward(B, A, scales)
        gamma  = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = (alpha[t, :, None] * A
                     * (B[t + 1] * beta[t + 1])[None, :])
            xi[t] /= xi[t].sum()
        pi     = gamma[0] / gamma[0].sum()
        xi_sum = xi.sum(axis=0)
        A      = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        for j in range(K):
            w         = gamma[:, j]
            mus[j]    = (w * obs).sum() / w.sum()
            sigmas[j] = np.sqrt((w * (obs - mus[j]) ** 2).sum() / w.sum())
        ll = np.sum(np.log(scales + 1e-300))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return pi, A, mus, sigmas, gamma


def viterbi(obs, pi, A, mus, sigmas):
    T, K      = len(obs), len(pi)
    log_A     = np.log(A + 1e-300)
    log_delta = np.zeros((T, K))
    psi       = np.zeros((T, K), dtype=int)
    log_delta[0] = np.log(pi + 1e-300) + np.array(
        [norm.logpdf(obs[0], mus[j], sigmas[j]) for j in range(K)]
    )
    for t in range(1, T):
        log_emit   = np.array(
            [norm.logpdf(obs[t], mus[j], sigmas[j]) for j in range(K)]
        )
        candidates   = log_delta[t - 1, :, None] + log_A
        psi[t]       = candidates.argmax(axis=0)
        log_delta[t] = candidates.max(axis=0) + log_emit
    states        = np.zeros(T, dtype=int)
    states[T - 1] = log_delta[T - 1].argmax()
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


# ── Data & HMM fitting ────────────────────────────────────────────────────────

START_DATE = "2015-01-01"
END_DATE   = pd.Timestamp.today().strftime("%Y-%m-%d")
T_HORIZON  = 0.25   # 3-month option


def fit_hmm_for_ticker(ticker):
    """Fetch weekly returns, fit Gaussian HMM, return all regime metrics."""
    raw   = yf.download(ticker, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False)
    close = raw["Close"].resample("W").last().dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    pct   = close.pct_change().dropna()
    close = close.loc[pct.index]

    obs                    = pct.values
    pi, A, mus, sigmas, gamma = baum_welch(obs)
    states                 = viterbi(obs, pi, A, mus, sigmas)
    bull                   = int(np.argmax(mus))
    bear                   = 1 - bull

    # Annualize weekly params
    sigma_bull = sigmas[bull] * np.sqrt(52)
    sigma_bear = sigmas[bear] * np.sqrt(52)
    mu_bull    = mus[bull]    * 52
    mu_bear    = mus[bear]    * 52

    # Current-week posterior
    p_bull_now = float(gamma[-1, bull])
    p_bear_now = float(gamma[-1, bear])

    # Expected weeks remaining in current regime  E[duration] = 1 / (1 - p_stay)
    state_idx = bull if states[-1] == bull else bear
    exp_weeks = 1.0 / (1.0 - float(A[state_idx, state_idx]))

    return {
        "ticker":              ticker,
        "close":               close,
        "pct":                 pct,
        "dates":               pct.index,
        "obs":                 obs,
        "pi":                  pi,
        "A":                   A,
        "mus":                 mus,       # weekly, state-indexed
        "sigmas":              sigmas,    # weekly, state-indexed
        "gamma":               gamma,
        "states":              states,
        "bull":                bull,
        "bear":                bear,
        "is_bull":             (states == bull),
        "sigma_bull":          sigma_bull,
        "sigma_bear":          sigma_bear,
        "mu_bull":             mu_bull,
        "mu_bear":             mu_bear,
        "p_bull_now":          p_bull_now,
        "p_bear_now":          p_bear_now,
        "current_state":       "bull" if states[-1] == bull else "bear",
        "exp_weeks_remaining": exp_weeks,
    }


# ── Regime-switching GBM ──────────────────────────────────────────────────────

def simulate_regime_switching_gbm(S0, mus_ann, sigmas_ann, A, pi,
                                   T, n_steps=252, n_paths=500, seed=42):
    """
    GBM where mu and sigma switch between states each step via transition matrix A.
    mus_ann / sigmas_ann must be indexed in the same state order as A and pi.
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    K      = len(pi)
    paths  = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    states = rng.choice(K, size=n_paths, p=pi)

    for t in range(n_steps):
        for j in range(K):
            mask = states == j
            if not mask.any():
                continue
            Z     = rng.standard_normal(mask.sum())
            drift = (mus_ann[j] - 0.5 * sigmas_ann[j] ** 2) * dt
            diff  = sigmas_ann[j] * np.sqrt(dt) * Z
            paths[mask, t + 1] = paths[mask, t] * np.exp(drift + diff)

        states = np.array([rng.choice(K, p=A[states[i]])
                           for i in range(n_paths)])

    return paths


# ── Hedge metrics ─────────────────────────────────────────────────────────────

def compute_hedge_summary(hmm, mkt, T=T_HORIZON):
    """Price ATM puts and compute Greeks at bull, bear, and blended vol."""
    S  = mkt["S"]
    r  = mkt["r"]
    K  = round(S)          # at-the-money strike

    sb    = hmm["sigma_bull"]
    sr    = hmm["sigma_bear"]
    pb    = hmm["p_bull_now"]
    pr    = hmm["p_bear_now"]
    s_mix = pb * sb + pr * sr   # probability-weighted vol

    put_bull  = bs_put(S, K, r, sb,           T)
    put_bear  = bs_put(S, K, r, sr,           T)
    put_blend = bs_put(S, K, r, s_mix,        T)
    put_naive = bs_put(S, K, r, mkt["sigma"], T)

    g_bull  = bs_greeks(S, K, r, sb,    T)
    g_bear  = bs_greeks(S, K, r, sr,    T)
    g_blend = bs_greeks(S, K, r, s_mix, T)

    vol_gap    = (sr - sb) * 100           # percentage points
    mispricing = g_bull["vega"] * vol_gap  # $ underpricing if using bull vol in bear

    return {
        "S":           S,
        "K":           K,
        "r":           r,
        "T":           T,
        "sigma_bull":  sb,
        "sigma_bear":  sr,
        "sigma_blend": s_mix,
        "sigma_naive": mkt["sigma"],
        "put_bull":    put_bull,
        "put_bear":    put_bear,
        "put_blend":   put_blend,
        "put_naive":   put_naive,
        "g_bull":      g_bull,
        "g_bear":      g_bear,
        "g_blend":     g_blend,
        "p_bull":      pb,
        "p_bear":      pr,
        "vol_gap":     vol_gap,
        "mispricing":  mispricing,
    }


# ── Console report ────────────────────────────────────────────────────────────

def print_hedge_report(ticker, hmm, s):
    w = 62
    print(f"\n{'=' * w}")
    print(f"  REGIME HEDGE REPORT  —  {ticker}")
    print(f"{'=' * w}")
    print(f"  Current regime            : {hmm['current_state'].upper()}")
    print(f"  P(bear) now               : {s['p_bear']:.1%}")
    print(f"  P(bull) now               : {s['p_bull']:.1%}")
    print(f"  Expected weeks remaining  : {hmm['exp_weeks_remaining']:.1f}")
    print()
    print(f"  {'Volatility Input':<32} {'Annualized':>10}")
    print(f"  {'-' * 42}")
    print(f"  {'Bull regime sigma':<32} {s['sigma_bull']:>10.2%}")
    print(f"  {'Bear regime sigma':<32} {s['sigma_bear']:>10.2%}")
    print(f"  {'Blended (P-weighted) sigma':<32} {s['sigma_blend']:>10.2%}  <- recommended")
    print(f"  {'21-day realized sigma (naive)':<32} {s['sigma_naive']:>10.2%}")
    print()
    print(f"  ATM Put  (S=${s['S']:.2f}, K={s['K']}, T={s['T']}y, r={s['r']:.2%})")
    print(f"  {'-' * 42}")
    print(f"  {'Bull vol put':<32} ${s['put_bull']:>8.3f}")
    print(f"  {'Bear vol put':<32} ${s['put_bear']:>8.3f}  (+${s['put_bear'] - s['put_bull']:.3f})")
    print(f"  {'Blended vol put':<32} ${s['put_blend']:>8.3f}  <- USE THIS")
    print(f"  {'Naive realized vol put':<32} ${s['put_naive']:>8.3f}")
    print()
    print(f"  Delta hedge ratio (put delta):")
    print(f"  {'-' * 42}")
    print(f"  {'Bull regime delta':<32} {s['g_bull']['delta_put']:>+10.4f}")
    print(f"  {'Bear regime delta':<32} {s['g_bear']['delta_put']:>+10.4f}")
    print(f"  {'Blended delta':<32} {s['g_blend']['delta_put']:>+10.4f}  <- USE THIS")
    print()
    print(f"  Regime mispricing risk:")
    print(f"  {'-' * 42}")
    print(f"  {'Vol gap (bear - bull)':<32} {s['vol_gap']:>9.1f} %pts")
    print(f"  {'Vega (per 1% vol move)':<32} ${s['g_bull']['vega']:>8.4f}")
    print(f"  {'Put underpriced by ~':<32} ${s['mispricing']:>8.2f}  if using bull vol in bear")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_regime_dashboard(results):
    """4-panel dashboard: P(bear), closing price, rolling put cost, joint state."""
    tickers    = list(results.keys())
    palette    = {"KO": "steelblue", "PEP": "darkorange"}
    fig, axes  = plt.subplots(2, 2, figsize=(15, 9))

    # Panel 1: P(bear) over time ──────────────────────────────────────────────
    ax = axes[0, 0]
    for t in tickers:
        h = results[t]["hmm"]
        ax.plot(h["dates"], 1 - h["gamma"][:, h["bull"]],
                color=palette[t], linewidth=0.9, label=t)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Bear | data)")
    ax.set_title("Posterior Bear Probability")
    ax.legend()

    # Panel 2: Closing price coloured by regime ───────────────────────────────
    ax  = axes[0, 1]
    leg = [Line2D([0], [0], color="green", label="Bull"),
           Line2D([0], [0], color="red",   label="Bear")]
    for t in tickers:
        h  = results[t]["hmm"]
        cv = h["close"].values
        dt = h["dates"]
        ls = "--" if t == "PEP" else "-"
        for i in range(len(dt) - 1):
            c = "green" if h["is_bull"][i] else "red"
            ax.plot(dt[i:i+2], cv[i:i+2],
                    color=c, linewidth=0.8, linestyle=ls, alpha=0.75)
    ax.legend(handles=leg + [
        Line2D([0], [0], color="grey", ls="-",  label="KO"),
        Line2D([0], [0], color="grey", ls="--", label="PEP"),
    ])
    ax.set_ylabel("Close ($)")
    ax.set_title("Closing Price — Bull/Bear Regime")

    # Panel 3: Rolling blended ATM put cost ───────────────────────────────────
    ax = axes[1, 0]
    for t in tickers:
        h  = results[t]["hmm"]
        s  = results[t]["summary"]
        rf = s["r"]
        sb = h["sigma_bull"]
        sr = h["sigma_bear"]
        p_bull    = h["gamma"][:, h["bull"]]
        sig_blend = p_bull * sb + (1 - p_bull) * sr
        S_vals    = h["close"].values
        put_cost  = np.array([
            bs_put(S_vals[i], round(S_vals[i]), rf, sig_blend[i], T_HORIZON)
            for i in range(len(S_vals))
        ])
        ax.plot(h["dates"], put_cost, color=palette[t], linewidth=0.9, label=t)
    ax.set_ylabel(f"ATM Put Price ($)")
    ax.set_title(f"Regime-Blended ATM Put Cost  (T={T_HORIZON}y)")
    ax.legend()

    # Panel 4: Joint KO / PEP regime ──────────────────────────────────────────
    ax    = axes[1, 1]
    h_ko  = results["KO"]["hmm"]
    h_pep = results["PEP"]["hmm"]
    shared = h_ko["dates"].intersection(h_pep["dates"])
    ko_b   = pd.Series(h_ko["is_bull"].astype(int),  index=h_ko["dates"]).loc[shared].values
    pep_b  = pd.Series(h_pep["is_bull"].astype(int), index=h_pep["dates"]).loc[shared].values
    joint  = ko_b * 2 + pep_b   # 0=both bear, 1=KO bear/PEP bull, 2=KO bull/PEP bear, 3=both bull
    joint_palette = {
        0: ("#d73027", "Both Bear"),
        1: ("#fee08b", "KO Bear / PEP Bull"),
        2: ("#91bfdb", "KO Bull / PEP Bear"),
        3: ("#1a9850", "Both Bull"),
    }
    for code, (color, label) in joint_palette.items():
        mask = joint == code
        if mask.any():
            ax.fill_between(shared, 0, 1, where=mask, color=color,
                            alpha=0.85, label=label,
                            transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(shared[0], shared[-1])
    ax.set_title("Joint KO / PEP Regime State")
    ax.legend(loc="upper right", fontsize=8, ncol=2,
              handles=[Patch(color=v[0], label=v[1])
                       for v in joint_palette.values()])

    for ax in axes.flat:
        ax.set_xlabel("Date")
    plt.suptitle("Regime Hedge Dashboard — KO vs PEP", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_put_vs_pbear(results):
    """
    For each stock: put price as a continuous function of P(bear).
    Shows where current P(bear) sits on the curve vs naive realized-vol put.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (ticker, data) in zip(axes, results.items()):
        s = data["summary"]
        p_range   = np.linspace(0, 1, 300)
        sig_range = (1 - p_range) * s["sigma_bull"] + p_range * s["sigma_bear"]
        put_curve = [bs_put(s["S"], s["K"], s["r"], sv, s["T"]) for sv in sig_range]

        ax.plot(p_range * 100, put_curve, color="purple", linewidth=1.6)
        ax.axvline(s["p_bear"] * 100, color="red", linestyle="--", linewidth=1.1,
                   label=f"Now  P(bear)={s['p_bear']:.0%}  (${s['put_blend']:.2f})")
        ax.axhline(s["put_naive"], color="grey", linestyle=":", linewidth=1.1,
                   label=f"Naive 21d-vol put  (${s['put_naive']:.2f})")
        ax.set_xlabel("P(Bear)  %")
        ax.set_ylabel("ATM Put Price ($)")
        ax.set_title(f"{ticker} — Put Cost vs Bear Probability")
        ax.legend(fontsize=8)

    plt.suptitle("Protective Put Price as a Function of Regime Probability",
                 fontsize=12)
    plt.tight_layout()
    return fig


def plot_gbm_comparison(ticker, hmm, mkt, T=T_HORIZON):
    """Standard GBM (fixed sigma) vs regime-switching GBM side by side."""
    S0  = mkt["S"]
    r   = mkt["r"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Standard GBM — naive realized vol
    paths_std = simulate_gbm(S0, r, mkt["sigma"], T, n_steps=63, n_paths=300)
    t_ax      = np.linspace(0, T, paths_std.shape[1])
    ax1.plot(t_ax, paths_std[:30].T, alpha=0.3, linewidth=0.6, color="steelblue")
    ax1.plot(t_ax, np.median(paths_std, axis=0),
             color="black", linewidth=2, label="Median")
    ax1.fill_between(t_ax,
                     np.percentile(paths_std,  5, axis=0),
                     np.percentile(paths_std, 95, axis=0),
                     alpha=0.15, color="steelblue", label="5–95th pct")
    ax1.axhline(S0, color="grey", linestyle="--", linewidth=0.7)
    ax1.set_title(f"Standard GBM  (σ={mkt['sigma']:.1%} fixed)")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock Price ($)")
    ax1.legend(fontsize=8)

    # Regime-switching GBM — sigmas array is state-indexed (same order as A, pi)
    mus_ann  = hmm["mus"] * 52
    sigs_ann = hmm["sigmas"] * np.sqrt(52)
    paths_rs = simulate_regime_switching_gbm(
        S0, mus_ann, sigs_ann, hmm["A"], hmm["pi"],
        T, n_steps=63, n_paths=300
    )
    ax2.plot(t_ax, paths_rs[:30].T, alpha=0.3, linewidth=0.6, color="darkorange")
    ax2.plot(t_ax, np.median(paths_rs, axis=0),
             color="black", linewidth=2, label="Median")
    ax2.fill_between(t_ax,
                     np.percentile(paths_rs,  5, axis=0),
                     np.percentile(paths_rs, 95, axis=0),
                     alpha=0.15, color="darkorange", label="5–95th pct")
    ax2.axhline(S0, color="grey", linestyle="--", linewidth=0.7)
    ax2.set_title(
        f"Regime-Switching GBM\n"
        f"(σ_bull={hmm['sigma_bull']:.1%},  σ_bear={hmm['sigma_bear']:.1%})"
    )
    ax2.set_xlabel("Time (years)")
    ax2.legend(fontsize=8)

    plt.suptitle(f"{ticker} — Standard vs Regime-Switching Price Paths", fontsize=12)
    plt.tight_layout()
    return fig


def plot_greeks_comparison(ticker, s):
    """Delta, Gamma, Vega across stock price range at bull vol vs bear vol."""
    S_range = np.linspace(s["S"] * 0.7, s["S"] * 1.3, 300)
    g_bull  = [bs_greeks(sv, s["K"], s["r"], s["sigma_bull"], s["T"]) for sv in S_range]
    g_bear  = [bs_greeks(sv, s["K"], s["r"], s["sigma_bear"], s["T"]) for sv in S_range]

    metrics = [
        ("delta_put",  "Put Delta",       "Hedge Ratio"),
        ("gamma",      "Gamma",           "Delta Change per $1"),
        ("vega",       "Vega (per 1% σ)", "$ per 1% vol move"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (key, label, ylabel) in zip(axes, metrics):
        ax.plot(S_range, [g[key] for g in g_bull],
                color="green", linewidth=1.6,
                label=f"Bull vol ({s['sigma_bull']:.1%})")
        ax.plot(S_range, [g[key] for g in g_bear],
                color="red", linewidth=1.6,
                label=f"Bear vol ({s['sigma_bear']:.1%})")
        ax.axvline(s["K"], linestyle="--", color="grey",  alpha=0.6, label=f"K={s['K']}")
        ax.axvline(s["S"], linestyle=":",  color="black", alpha=0.6,
                   label=f"S={s['S']:.0f}")
        ax.set_title(label)
        ax.set_xlabel("Stock Price ($)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    plt.suptitle(f"{ticker} — Greeks: Bull Vol vs Bear Vol", fontsize=12)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TICKERS = {"KO": "Coca-Cola", "PEP": "PepsiCo"}

    print("Fitting HMMs and fetching live prices...")
    results = {}

    for ticker in TICKERS:
        hmm = fit_hmm_for_ticker(ticker)
        mkt = market_inputs(ticker)
        s   = compute_hedge_summary(hmm, mkt)
        results[ticker] = {"hmm": hmm, "mkt": mkt, "summary": s}
        print_hedge_report(ticker, hmm, s)

    # Figure 1: 4-panel regime dashboard
    plot_regime_dashboard(results)

    # Figure 2: Put price as a function of P(bear) for each stock
    plot_put_vs_pbear(results)

    # Figures 3 & 4: Standard vs regime-switching GBM paths
    for ticker in TICKERS:
        plot_gbm_comparison(ticker, results[ticker]["hmm"], results[ticker]["mkt"])

    # Figures 5 & 6: Greeks at bull vol vs bear vol
    for ticker in TICKERS:
        plot_greeks_comparison(ticker, results[ticker]["summary"])

    plt.show()