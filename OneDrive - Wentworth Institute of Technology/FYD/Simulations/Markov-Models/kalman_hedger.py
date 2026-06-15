"""
kalman_hedger.py
────────────────
Regime-switching Kalman Filter for dynamic pairs hedging of KO and PEP.

Model:
  P_KO_t = alpha_t + beta_t * P_PEP_t + epsilon_t
  [alpha_t, beta_t] evolves as a random walk (state)

Regime integration:
  HMM from regime_hedger supplies P(bear) each week.
  Bull regime → slow process noise Q  (stable hedge ratio)
  Bear regime → fast process noise Q  (hedge ratio updates aggressively)

Outputs:
  - Time-varying beta_t with 95% confidence band
  - Spread z-score entry/exit signal
  - Cumulative P&L: Kalman dynamic hedge vs fixed OLS beta
  - Live dashboard: current beta, z-score, position recommendation

Run:
  python kalman_hedger.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import yfinance as yf

# ── Sibling-directory imports ─────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "Black-Scholes")))
from black_scholes_simulator import bs_put, bs_greeks, market_inputs

sys.path.insert(0, _here)
from regime_hedger import fit_hmm_for_ticker   # safe — has __main__ guard


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanPairsFilter:
    """
    Online Kalman Filter estimating a time-varying linear hedge relationship:

        P_KO_t  =  alpha_t  +  beta_t * P_PEP_t  +  epsilon_t

    State transition (random walk on parameters):
        [alpha_t, beta_t]  =  [alpha_{t-1}, beta_{t-1}]  +  w_t
        w_t ~ N(0, Q_t)

    Observation model:
        P_KO_t  =  H_t @ theta_t  +  e_t,   e_t ~ N(0, R)
        H_t  =  [1,  P_PEP_t]

    Process noise Q_t is blended between bull and bear scaling via P(bear),
    so the hedge ratio tracks faster when the HMM detects a bear regime.
    """

    def __init__(self, delta_bull=1e-4, delta_bear=5e-4, R=0.01):
        """
        delta_bull : process noise in bull regime — hedge ratio drifts slowly
        delta_bear : process noise in bear regime — hedge ratio updates faster
        R          : observation noise variance ($^2)
        """
        self.delta_bull = delta_bull
        self.delta_bear = delta_bear
        self.R          = R
        self.theta      = np.zeros(2)       # [alpha, beta]
        self.P          = np.eye(2) * 10.0  # vague initial covariance

    def _Q(self, p_bear):
        """Process noise covariance blended by bear probability."""
        p_bear = float(np.clip(p_bear, 0.0, 1.0))   # guard against NaN / out-of-range
        delta  = (1.0 - p_bear) * self.delta_bull + p_bear * self.delta_bear
        return (delta / (1.0 - delta)) * np.eye(2)

    def step(self, y, x, p_bear=0.0):
        """
        One predict-update cycle.

        Parameters
        ----------
        y      : P_KO observation (float)
        x      : P_PEP observation (float, regressor)
        p_bear : HMM bear probability for this timestep

        Returns
        -------
        alpha      : updated intercept
        beta       : updated hedge ratio
        spread     : prior spread  y - H @ theta_{t-1|t-1}  (use for trading)
        beta_std   : posterior std dev of beta estimate
        innov_var  : innovation variance S_t (for standardised signal)
        """
        H      = np.array([1.0, x])
        Q      = self._Q(p_bear)

        # ── Predict ───────────────────────────────────────────────────────────
        P_pred = self.P + Q                   # theta_pred = self.theta

        # ── Innovation (PRIOR spread — computed before the update) ────────────
        y_pred = float(H @ self.theta)
        spread = y - y_pred                   # = P_KO - (alpha + beta*P_PEP)
        S      = float(H @ P_pred @ H) + self.R

        # ── Update ────────────────────────────────────────────────────────────
        K          = P_pred @ H / S           # Kalman gain, shape (2,)
        self.theta = self.theta + K * spread
        self.P     = (np.eye(2) - np.outer(K, H)) @ P_pred

        return (
            float(self.theta[0]),             # alpha
            float(self.theta[1]),             # beta
            spread,                           # prior spread (for signals)
            float(np.sqrt(self.P[1, 1])),     # beta uncertainty
            S,                                # innovation variance
        )

    def run(self, prices_ko, prices_pep, p_bear_series):
        """
        Run the filter over the full aligned price series.

        Returns
        -------
        DataFrame with columns: alpha, beta, spread, beta_std, innov_var
        indexed by prices_ko.index
        """
        n   = len(prices_ko)
        buf = np.zeros((n, 5))

        for t in range(n):
            buf[t] = self.step(
                float(prices_ko.iloc[t]),
                float(prices_pep.iloc[t]),
                float(p_bear_series.iloc[t]),
            )

        return pd.DataFrame(
            buf,
            columns=["alpha", "beta", "spread", "beta_std", "innov_var"],
            index=prices_ko.index,
        )


# ── Data loading ──────────────────────────────────────────────────────────────

START_DATE    = "2015-01-01"
END_DATE      = pd.Timestamp.today().strftime("%Y-%m-%d")
ENTRY_Z       = 2.0    # open trade when |z| crosses above this
EXIT_Z        = 0.5    # close trade when |z| drops below this
ZSCORE_WINDOW = 30     # rolling days for z-score normalisation
COST_BPS      = 5      # transaction cost per leg in basis points (5 bps = 0.05%)
RISK_FREE_ANN = 0.045  # annualised risk-free rate (~4.5% T-bill)


def load_daily_prices(ticker):
    raw   = yf.download(ticker, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def build_daily_p_bear(hmm_ko, hmm_pep, daily_index):
    """
    Average KO and PEP weekly P(bear) then forward-fill to daily business days.

    pandas resample("W") labels weeks with Sundays, which are not trading days.
    Reindexing directly onto business days produces all-NaN. Fix: expand to the
    union of weekly + daily dates, forward-fill across that union, then select
    only the daily dates.
    """
    p_ko  = pd.Series(1 - hmm_ko["gamma"][:, hmm_ko["bull"]],   index=hmm_ko["dates"])
    p_pep = pd.Series(1 - hmm_pep["gamma"][:, hmm_pep["bull"]], index=hmm_pep["dates"])

    combined = pd.concat([p_ko, p_pep], axis=1).mean(axis=1)

    all_idx = combined.index.union(daily_index).sort_values()
    return (
        combined
        .reindex(all_idx)
        .ffill()
        .bfill()
        .reindex(daily_index)
        .fillna(0.5)   # fallback: neutral if still missing
    )


# ── Backtest ──────────────────────────────────────────────────────────────────

def _signal_from_zscore(z, entry, exit_):
    """State-machine signal: +1 long spread, -1 short spread, 0 flat."""
    sig = np.zeros(len(z))
    pos = 0
    for t in range(1, len(z)):
        v = z[t]
        if np.isnan(v):
            pos = 0
        elif pos == 0:
            if v < -entry:
                pos =  1   # spread too low → expect it to rise → long KO short PEP
            elif v > entry:
                pos = -1   # spread too high → expect it to fall → short KO long PEP
        elif pos ==  1 and v > -exit_:
            pos = 0
        elif pos == -1 and v <  exit_:
            pos = 0
        sig[t] = pos
    return sig


def _sharpe(pnl, rf_ann=RISK_FREE_ANN):
    """Annualised Sharpe with daily risk-free subtracted.
    Returns 0.0 when std is negligible (zero-trade windows etc.)."""
    rf_daily = rf_ann / 252
    excess   = pnl - rf_daily
    if pnl.std() < 1e-8:   # no meaningful trading activity
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def _add_costs(pnl_series, signal_series, prices_ko, prices_pep, beta_series,
               cost_bps=COST_BPS):
    """
    Deduct transaction costs on every signal change.
    Cost = cost_bps/10000 per leg * 2 legs * notional.
    Notional = |P_KO| + |beta| * |P_PEP|  (dollar value of both legs).
    """
    cost_per_leg = cost_bps / 10_000
    sig_change   = signal_series.diff().abs()              # 1 when position flips
    notional     = prices_ko.abs() + beta_series.abs() * prices_pep.abs()
    cost         = sig_change * notional * cost_per_leg * 2
    return (pnl_series - cost.fillna(0)).fillna(0)


def run_backtest(kf_df, prices_ko, prices_pep):
    """
    Two parallel pairs strategies:
      Kalman — z-score on the Kalman time-varying spread
      OLS    — z-score on a rolling-252d OLS spread (no look-ahead bias)

    FIX 1: Correct P&L formula.
      Bug: original code used sp_kf.diff() which mixes actual price moves
      with Kalman parameter updates (phantom gains when alpha/beta drift).
      Fix: P&L = signal[t-1] * (dKO[t] - beta[t-1] * dPEP[t])
      i.e. the actual dollar return of the position held from t-1 to t.

    FIX 2: OLS baseline uses rolling 252-day window, not full-history fit.

    FIX 3: Transaction costs deducted at each signal change.

    FIX 4: Sharpe uses excess return (minus daily Rf = 4.5%/252).
    """
    ko_aligned  = prices_ko.reindex(kf_df.index).ffill()
    pep_aligned = prices_pep.reindex(kf_df.index).ffill()

    # ── Kalman spread z-score (signal generation) ─────────────────────────────
    sp_kf  = kf_df["spread"]
    mu_kf  = sp_kf.rolling(ZSCORE_WINDOW, min_periods=10).mean()
    std_kf = sp_kf.rolling(ZSCORE_WINDOW, min_periods=10).std().replace(0, np.nan)
    z_kf   = (sp_kf - mu_kf) / std_kf

    sig_arr = _signal_from_zscore(z_kf.values, ENTRY_Z, EXIT_Z)
    sig_kf  = pd.Series(sig_arr, index=kf_df.index)
    sig_s1  = sig_kf.shift(1)                  # position held on day t

    # FIX 1 — P&L from actual price moves using yesterday's hedge ratio
    beta_held  = kf_df["beta"].shift(1)         # beta known at end of t-1
    dko        = ko_aligned.diff()
    dpep       = pep_aligned.diff()
    gross_kf   = (sig_s1 * (dko - beta_held * dpep)).fillna(0)

    # FIX 3 — subtract transaction costs on signal changes
    net_kf = _add_costs(gross_kf, sig_kf, ko_aligned, pep_aligned, beta_held)

    # ── Rolling OLS baseline (FIX 2 — no full-history look-ahead) ────────────
    roll_window = 252
    ols_betas   = []
    for i in range(len(ko_aligned)):
        start = max(0, i - roll_window + 1)
        x = pep_aligned.iloc[start:i + 1].values
        y = ko_aligned.iloc[start:i + 1].values
        if len(x) >= 30:
            b = float(np.polyfit(x, y, 1)[0])
        else:
            b = 0.3  # sensible prior before enough data
        ols_betas.append(b)
    ols_beta_series = pd.Series(ols_betas, index=kf_df.index)
    ols_beta_scalar = float(ols_beta_series.iloc[-1])   # current value for report

    sp_ols  = ko_aligned - ols_beta_series * pep_aligned
    mu_ols  = sp_ols.rolling(ZSCORE_WINDOW, min_periods=10).mean()
    std_ols = sp_ols.rolling(ZSCORE_WINDOW, min_periods=10).std().replace(0, np.nan)
    z_ols   = (sp_ols - mu_ols) / std_ols

    sig_ols_arr = _signal_from_zscore(z_ols.values, ENTRY_Z, EXIT_Z)
    sig_ols     = pd.Series(sig_ols_arr, index=kf_df.index)
    sig_ols_s1  = sig_ols.shift(1)
    beta_ols_held = ols_beta_series.shift(1)
    gross_ols   = (sig_ols_s1 * (dko - beta_ols_held * dpep)).fillna(0)
    net_ols     = _add_costs(gross_ols, sig_ols, ko_aligned, pep_aligned, beta_ols_held)

    n_trades_kf  = int((sig_kf.diff().abs() > 0).sum())
    n_trades_ols = int((sig_ols.diff().abs() > 0).sum())

    return {
        "z_kf":            z_kf,
        "sig_kf":          sig_kf,
        "pnl_kf":          net_kf,
        "gross_pnl_kf":    gross_kf,
        "cum_pnl_kf":      net_kf.cumsum(),
        "z_ols":           z_ols,
        "sig_ols":         sig_ols,
        "pnl_ols":         net_ols,
        "cum_pnl_ols":     net_ols.cumsum(),
        "ols_beta":        ols_beta_scalar,
        "ols_beta_series": ols_beta_series,
        "spread_ols":      sp_ols,
        "n_trades_kf":     n_trades_kf,
        "n_trades_ols":    n_trades_ols,
        "ko_aligned":      ko_aligned,
        "pep_aligned":     pep_aligned,
    }


# ── Console report ────────────────────────────────────────────────────────────

def _strategy_stats(pnl, n_days):
    """Return dict of key strategy statistics."""
    rf_daily = RISK_FREE_ANN / 252
    excess   = pnl - rf_daily
    sharpe   = float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0
    vol_ann  = float(pnl.std() * np.sqrt(252))
    ret_ann  = float(pnl.mean() * 252)
    cum      = pnl.cumsum()
    roll_max = cum.cummax()
    dd       = cum - roll_max
    max_dd   = float(dd.min())
    return {"sharpe": sharpe, "vol_ann": vol_ann, "ret_ann": ret_ann,
            "max_dd": max_dd, "cum_pnl": float(cum.iloc[-1])}


def print_live_report(kf_df, bt, hmm_ko, hmm_pep, mkt_ko, mkt_pep):
    current_beta     = kf_df["beta"].iloc[-1]
    current_alpha    = kf_df["alpha"].iloc[-1]
    current_beta_std = kf_df["beta_std"].iloc[-1]
    current_z        = bt["z_kf"].iloc[-1]
    current_sig      = bt["sig_kf"].iloc[-1]
    p_bear           = 0.5 * (hmm_ko["p_bear_now"] + hmm_pep["p_bear_now"])

    if current_sig == 1:
        rec = "LONG KO  /  SHORT {:.2f} shares PEP".format(current_beta)
    elif current_sig == -1:
        rec = "SHORT KO  /  LONG {:.2f} shares PEP".format(current_beta)
    else:
        rec = "FLAT  (no position)"

    n_days = len(bt["pnl_kf"])
    stats_kf  = _strategy_stats(bt["pnl_kf"],  n_days)
    stats_ols = _strategy_stats(bt["pnl_ols"], n_days)

    # Step-by-step Sharpe breakdown (gross -> net of costs -> net of Rf)
    gross_raw_sharpe = float(
        (bt["gross_pnl_kf"].mean() / bt["gross_pnl_kf"].std()) * np.sqrt(252)
    ) if bt["gross_pnl_kf"].std() > 0 else 0.0
    gross_ex_rf = float(
        ((bt["gross_pnl_kf"] - RISK_FREE_ANN/252).mean() /
          bt["gross_pnl_kf"].std()) * np.sqrt(252)
    ) if bt["gross_pnl_kf"].std() > 0 else 0.0
    years = n_days / 252

    w = 68
    print(f"\n{'=' * w}")
    print(f"  KALMAN PAIRS HEDGE REPORT  --  KO / PEP")
    print(f"{'=' * w}")
    print(f"  Avg P(bear) now            : {p_bear:.1%}")
    print(f"  Regime mode                : {'BEAR -- fast delta' if p_bear > 0.5 else 'BULL -- slow delta'}")
    print()
    print(f"  Current hedge ratio (beta) : {current_beta:.4f}  +/- {2*current_beta_std:.4f}  (95% CI)")
    print(f"  Rolling OLS beta (baseline): {bt['ols_beta']:.4f}")
    print(f"  Current alpha (intercept)  : ${current_alpha:.2f}")
    print()
    print(f"  Current z-score            : {current_z:+.2f}")
    print(f"  Entry threshold            : +/-{ENTRY_Z:.1f}")
    print(f"  Exit threshold             : +/-{EXIT_Z:.1f}")
    print()
    print(f"  Recommended position       : {rec}")
    print()
    print(f"  {'='*62}")
    print(f"  SHARPE AUDIT -- step-by-step breakdown")
    print(f"  {'='*62}")
    print(f"  [1] Gross Sharpe (correct P&L, no costs, no Rf) : {gross_raw_sharpe:>6.2f}")
    print(f"  [2] Gross Sharpe minus Rf only                  : {gross_ex_rf:>6.2f}")
    print(f"  [3] Net Sharpe  (costs {COST_BPS}bps/leg, Rf deducted) : {stats_kf['sharpe']:>6.2f}  << FINAL")
    print()
    print(f"  Backtest ({years:.1f} yrs)  |  Kalman  |  Rolling OLS")
    print(f"  {'-'*52}")
    print(f"  Ann. Return ($)            : ${stats_kf['ret_ann']:>7.2f}  |  ${stats_ols['ret_ann']:>7.2f}")
    print(f"  Ann. Volatility ($)        : ${stats_kf['vol_ann']:>7.2f}  |  ${stats_ols['vol_ann']:>7.2f}")
    print(f"  Sharpe (net, ex-Rf)        : {stats_kf['sharpe']:>8.2f}  |  {stats_ols['sharpe']:>8.2f}")
    print(f"  Max Drawdown ($)           : ${stats_kf['max_dd']:>7.2f}  |  ${stats_ols['max_dd']:>7.2f}")
    print(f"  Cumulative P&L             : ${stats_kf['cum_pnl']:>7.2f}  |  ${stats_ols['cum_pnl']:>7.2f}")
    print(f"  Total Trades               : {bt['n_trades_kf']:>8}  |  {bt['n_trades_ols']:>8}")
    print(f"  Avg Hold (days)            : {n_days/max(bt['n_trades_kf'],1):>8.1f}  |  {n_days/max(bt['n_trades_ols'],1):>8.1f}")
    print(f"  Cost assumed               : {COST_BPS} bps/leg ({COST_BPS*2} bps round-trip)")
    print(f"  Risk-free rate             : {RISK_FREE_ANN:.1%} annualised")


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_sharpe(kf_df, prices_ko, prices_pep,
                         train_days=252, test_days=63):
    """
    Rolling walk-forward: re-run the backtest on each out-of-sample (OOS) window.
    Reports per-window Sharpe and the mean/std across all windows.

    Parameters
    ----------
    train_days : length of training window (Kalman warms up on this period)
    test_days  : length of each OOS evaluation window
    """
    ko  = prices_ko.reindex(kf_df.index).ffill()
    pep = prices_pep.reindex(kf_df.index).ffill()
    n   = len(kf_df)

    oos_sharpes = []
    print(f"\n  Walk-Forward (train={train_days}d, test={test_days}d):")
    print(f"  {'Window':<22} {'OOS Start':<14} {'OOS End':<14} {'Sharpe':>7}  {'Trades':>7}")
    print(f"  {'-' * 66}")

    start = train_days
    window_num = 0
    while start + test_days <= n:
        test_slice = slice(start, start + test_days)
        kf_window  = kf_df.iloc[test_slice]
        ko_window  = ko.iloc[test_slice]
        pep_window = pep.iloc[test_slice]

        sp    = kf_window["spread"]
        mu    = sp.rolling(ZSCORE_WINDOW, min_periods=5).mean()
        std   = sp.rolling(ZSCORE_WINDOW, min_periods=5).std().replace(0, np.nan)
        z     = (sp - mu) / std
        sig   = pd.Series(_signal_from_zscore(z.values, ENTRY_Z, EXIT_Z), index=kf_window.index)
        beta_held = kf_window["beta"].shift(1)

        gross = (sig.shift(1) * (ko_window.diff() - beta_held * pep_window.diff())).fillna(0)
        net   = _add_costs(gross, sig, ko_window, pep_window, beta_held)
        sh    = _sharpe(net)
        n_tr  = int((sig.diff().abs() > 0).sum())

        oos_sharpes.append(sh)
        d0 = kf_window.index[0].strftime("%Y-%m-%d")
        d1 = kf_window.index[-1].strftime("%Y-%m-%d")
        print(f"  Window {window_num + 1:<16} {d0:<14} {d1:<14} {sh:>7.2f}  {n_tr:>7}")

        start      += test_days
        window_num += 1

    arr = np.array(oos_sharpes)
    print(f"\n  OOS Sharpe: mean={arr.mean():.2f}  std={arr.std():.2f}  "
          f"min={arr.min():.2f}  max={arr.max():.2f}")
    print(f"  Windows positive: {(arr > 0).sum()}/{len(arr)}")
    return arr


# ── Figures ───────────────────────────────────────────────────────────────────

def _shade_bear(ax, dates, p_bear_daily, threshold=0.5):
    """Shade bear-regime periods on an existing axes."""
    in_bear = False
    t_start = None
    for t, d in enumerate(dates):
        pb = p_bear_daily.reindex([d]).iloc[0] if d in p_bear_daily.index else 0.0
        if pb > threshold and not in_bear:
            in_bear  = True
            t_start  = d
        elif pb <= threshold and in_bear:
            ax.axvspan(t_start, d, color="#ffcccc", alpha=0.35, zorder=0)
            in_bear  = False
    if in_bear:
        ax.axvspan(t_start, dates[-1], color="#ffcccc", alpha=0.35, zorder=0)


def plot_beta_series(kf_df, p_bear_daily):
    """Beta_t with 95% confidence band and bear-regime shading."""
    fig, ax = plt.subplots(figsize=(13, 4))
    dates   = kf_df.index
    beta    = kf_df["beta"]
    bstd    = kf_df["beta_std"]

    _shade_bear(ax, dates, p_bear_daily)
    ax.plot(dates, beta, color="steelblue", linewidth=1.1, label="β_t (Kalman)")
    ax.fill_between(dates, beta - 2 * bstd, beta + 2 * bstd,
                    alpha=0.2, color="steelblue", label="95% CI")
    ax.axhline(kf_df["beta"].mean(), color="grey", linestyle="--",
               linewidth=0.8, label=f"Mean β = {kf_df['beta'].mean():.3f}")
    ax.set_ylabel("Hedge Ratio  β_t  (shares PEP per share KO)")
    ax.set_xlabel("Date")
    ax.set_title("Kalman-Estimated Hedge Ratio — KO / PEP  (red shading = bear regime)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_zscore_signal(bt, p_bear_daily):
    """Z-score of Kalman spread with entry/exit bands and position coloring."""
    fig, ax = plt.subplots(figsize=(13, 4))
    dates   = bt["z_kf"].index
    z       = bt["z_kf"]
    sig     = bt["sig_kf"]

    _shade_bear(ax, dates, p_bear_daily)

    # Shade position regions
    for i in range(len(dates) - 1):
        s = sig.iloc[i]
        if s == 1:
            ax.axvspan(dates[i], dates[i + 1], color="green", alpha=0.12, zorder=0)
        elif s == -1:
            ax.axvspan(dates[i], dates[i + 1], color="red",   alpha=0.12, zorder=0)

    ax.plot(dates, z, color="black", linewidth=0.8, label="Z-score")
    ax.axhline( ENTRY_Z, color="green", linestyle="--", linewidth=1.0, label=f"+{ENTRY_Z} entry")
    ax.axhline(-ENTRY_Z, color="red",   linestyle="--", linewidth=1.0, label=f"-{ENTRY_Z} entry")
    ax.axhline( EXIT_Z,  color="green", linestyle=":",  linewidth=0.7)
    ax.axhline(-EXIT_Z,  color="red",   linestyle=":",  linewidth=0.7)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("Spread Z-score")
    ax.set_xlabel("Date")
    ax.set_title("Kalman Spread Z-score — Entry/Exit Signal  (green=long KO, red=short KO)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_pnl_comparison(bt):
    """Cumulative P&L: Kalman dynamic hedge vs fixed OLS beta."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(bt["cum_pnl_kf"],  color="steelblue",   linewidth=1.2,
            label=f"Kalman  (cumulative ${bt['cum_pnl_kf'].iloc[-1]:.2f})")
    ax.plot(bt["cum_pnl_ols"], color="darkorange",  linewidth=1.2, linestyle="--",
            label=f"Fixed OLS β={bt['ols_beta']:.3f}  (cumulative ${bt['cum_pnl_ols'].iloc[-1]:.2f})")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_xlabel("Date")
    ax.set_title("Pairs Strategy Cumulative P&L — Kalman vs Fixed OLS Hedge Ratio")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_live_dashboard(kf_df, bt, hmm_ko, hmm_pep, mkt_ko, mkt_pep):
    """
    4-panel current-state snapshot:
      TL: rolling Sharpe (Kalman vs OLS)
      TR: beta_t over last 90 days zoomed
      BL: current spread distribution with z-score marker
      BR: put price comparison (regime-blended vs naive) for each stock
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # ── TL: Rolling 63-day Sharpe ─────────────────────────────────────────────
    ax = axes[0, 0]
    roll_sharpe_kf  = (bt["pnl_kf"].rolling(63).mean()
                       / bt["pnl_kf"].rolling(63).std().replace(0, np.nan)
                       * np.sqrt(252))
    roll_sharpe_ols = (bt["pnl_ols"].rolling(63).mean()
                       / bt["pnl_ols"].rolling(63).std().replace(0, np.nan)
                       * np.sqrt(252))
    ax.plot(roll_sharpe_kf,  color="steelblue",  linewidth=0.9, label="Kalman")
    ax.plot(roll_sharpe_ols, color="darkorange", linewidth=0.9, linestyle="--", label="OLS")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_title("Rolling 63-day Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)

    # ── TR: Beta — last 90 trading days ──────────────────────────────────────
    ax      = axes[0, 1]
    tail    = kf_df.tail(90)
    beta    = tail["beta"]
    bstd    = tail["beta_std"]
    ax.plot(tail.index, beta, color="steelblue", linewidth=1.3)
    ax.fill_between(tail.index, beta - 2 * bstd, beta + 2 * bstd,
                    alpha=0.25, color="steelblue")
    ax.axhline(kf_df["beta"].mean(), color="grey", linestyle="--",
               linewidth=0.8, label="Full-period mean β")
    ax.set_title("Beta — Last 90 Days (zoomed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("β_t")
    ax.legend(fontsize=8)

    # ── BL: Spread distribution with current z-score ─────────────────────────
    ax      = axes[1, 0]
    spreads = kf_df["spread"].dropna()
    ax.hist(spreads, bins=60, color="steelblue", alpha=0.6, density=True)
    current_sp = kf_df["spread"].iloc[-1]
    ax.axvline(current_sp, color="red", linewidth=1.5,
               label=f"Now  spread={current_sp:.2f}  z={bt['z_kf'].iloc[-1]:+.2f}")
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_title("Spread Distribution  (prior residual)")
    ax.set_xlabel("Spread ($)")
    ax.legend(fontsize=8)

    # ── BR: ATM put price at bull-vol vs bear-vol for KO and PEP ─────────────
    ax     = axes[1, 1]
    tickers = {"KO": (hmm_ko, mkt_ko), "PEP": (hmm_pep, mkt_pep)}
    x      = np.arange(len(tickers))
    labels = list(tickers.keys())
    put_bull_vals  = []
    put_bear_vals  = []
    put_blend_vals = []

    for ticker, (hmm, mkt) in tickers.items():
        S  = mkt["S"]
        K  = round(S)
        r  = mkt["r"]
        T  = 0.25
        sb = hmm["sigma_bull"]
        sr = hmm["sigma_bear"]
        pb = hmm["p_bull_now"]
        pr = hmm["p_bear_now"]
        put_bull_vals.append(bs_put(S, K, r, sb, T))
        put_bear_vals.append(bs_put(S, K, r, sr, T))
        put_blend_vals.append(bs_put(S, K, r, pb * sb + pr * sr, T))

    width = 0.25
    ax.bar(x - width, put_bull_vals,  width, label="Bull vol put",  color="green",  alpha=0.7)
    ax.bar(x,         put_blend_vals, width, label="Blended put",   color="steelblue", alpha=0.7)
    ax.bar(x + width, put_bear_vals,  width, label="Bear vol put",  color="red",    alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ATM Put Price ($)")
    ax.set_title("Protective Put Cost by Regime Vol  (T=0.25y)")
    ax.legend(fontsize=8)

    plt.suptitle("Live Hedge Dashboard — KO / PEP", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching prices and fitting HMMs...")

    # ── 1. Fit weekly HMMs for regime detection ───────────────────────────────
    hmm_ko  = fit_hmm_for_ticker("KO")
    hmm_pep = fit_hmm_for_ticker("PEP")

    # ── 2. Load daily prices, align to shared trading days ───────────────────
    prices_ko  = load_daily_prices("KO")
    prices_pep = load_daily_prices("PEP")
    shared     = prices_ko.index.intersection(prices_pep.index)
    prices_ko  = prices_ko.loc[shared]
    prices_pep = prices_pep.loc[shared]

    # ── 3. Forward-fill weekly P(bear) to daily ───────────────────────────────
    p_bear_daily = build_daily_p_bear(hmm_ko, hmm_pep, shared)

    # ── 4. Run Kalman Filter ──────────────────────────────────────────────────
    print("Running regime-switching Kalman Filter...")
    kf     = KalmanPairsFilter(delta_bull=1e-4, delta_bear=5e-4, R=0.01)
    kf_df  = kf.run(prices_ko, prices_pep, p_bear_daily)

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    bt = run_backtest(kf_df, prices_ko, prices_pep)

    # ── 6. Fetch live market data for option pricing ──────────────────────────
    print("Fetching live option inputs...")
    mkt_ko  = market_inputs("KO")
    mkt_pep = market_inputs("PEP")

    # ── 7. Console report ─────────────────────────────────────────────────────
    print_live_report(kf_df, bt, hmm_ko, hmm_pep, mkt_ko, mkt_pep)

    # ── 8. Walk-forward validation ────────────────────────────────────────────
    print("\nRunning walk-forward validation...")
    walk_forward_sharpe(kf_df, prices_ko, prices_pep)

    # ── 9. Figures ────────────────────────────────────────────────────────────
    plot_beta_series(kf_df, p_bear_daily)
    plot_zscore_signal(bt, p_bear_daily)
    plot_pnl_comparison(bt)
    plot_live_dashboard(kf_df, bt, hmm_ko, hmm_pep, mkt_ko, mkt_pep)

    plt.show()