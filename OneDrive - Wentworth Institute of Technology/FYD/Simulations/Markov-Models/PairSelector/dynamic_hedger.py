"""
dynamic_hedger.py
─────────────────
Orchestrates: cointegration screening → Kalman hedging → BS vol filter → P&L per fold.

Interface to existing modules (confirmed from reading source files):
  KalmanPairsFilter(delta_bull, delta_bear, R)
      .run(prices_y, prices_x, p_bear_series) -> DataFrame[alpha, beta, spread, beta_std, innov_var]

  bs_call(S, K, r, sigma, T)  -- from black_scholes_simulator.py
  bs_put(S, K, r, sigma, T)   -- from black_scholes_simulator.py

P&L formula (never use spread differences):
  gross[t] = signal[t-1] * (dY[t] - beta[t-1] * dX[t])   correct
  gross[t] = signal[t-1] * (spread[t] - spread[t-1])       WRONG — mixes price moves with parameter drift

Module 4b (HMM regime gate) is wired in at the bottom — set USE_HMM_GATE = True to enable.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Sibling imports ───────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _parent)
sys.path.insert(0, _here)

from kalman_hedger import KalmanPairsFilter
from cointegration_screener import screen_pairs, print_screening_report
from bs_vol_filter import (
    realized_vol, iv_rank, vol_entry_gate,
    spread_vol_estimate, vol_scaled_notional, bs_spread_diagnostic,
)

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_PAIRS        = 5       # top N cointegrated pairs per fold
ENTRY_Z          = 2.0     # z-score to open position
EXIT_Z           = 0.5     # z-score to close position
STOP_Z           = 3.5     # stop-loss z-score (exit if spread blows through)
COST_BPS         = 10      # round-trip transaction cost (5 bps per leg × 2)
IV_RANK_GATE     = 0.80    # block entries if vol rank > this
ROLLING_Z_WINDOW = 20      # days for z-score rolling mean/std (must be << TEST_DAYS=63)
USE_HMM_GATE     = False   # set True after Module 4b is wired in


# ── Kalman adapter ────────────────────────────────────────────────────────────

def run_kalman(prices_y: pd.Series,
               prices_x: pd.Series,
               p_bear: pd.Series = None) -> pd.DataFrame:
    """
    Thin adapter: instantiate KalmanPairsFilter and run it.
    p_bear defaults to neutral (0.0 = bull mode = slow process noise = stable beta).
    Returns DataFrame with columns [alpha, beta, spread, beta_std, innov_var].
    """
    if p_bear is None:
        p_bear = pd.Series(0.0, index=prices_y.index)
    kf = KalmanPairsFilter(delta_bull=1e-4, delta_bear=5e-4, R=0.01)
    return kf.run(prices_y, prices_x, p_bear)


def run_kalman_warmup(train_y: pd.Series,
                      train_x: pd.Series,
                      test_y: pd.Series,
                      test_x: pd.Series,
                      p_bear: pd.Series = None) -> pd.DataFrame:
    """
    Warm up the Kalman filter on training data, then continue on test data.
    Returns only the test-window slice of the filter output.

    Why this matters: KalmanPairsFilter initialises with theta=[0,0], P=10*I.
    On the first step, the predicted KO price is 0, so the spread = KO_price ~$40.
    This initialization spike corrupts the rolling z-score if the filter only
    runs on 63 days of test data (60-day window leaves only 3 valid bars).
    Running on train+test means the filter has converged before we trade.
    """
    all_y = pd.concat([train_y, test_y])
    all_x = pd.concat([train_x, test_x])
    if p_bear is None:
        p_bear = pd.Series(0.0, index=all_y.index)
    else:
        p_bear = p_bear.reindex(all_y.index).ffill().fillna(0.0)

    kf     = KalmanPairsFilter(delta_bull=1e-4, delta_bear=5e-4, R=0.01)
    kf_all = kf.run(all_y, all_x, p_bear)
    return kf_all.loc[test_y.index]


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(spread: pd.Series, vol_gate: pd.Series,
                     z_mean: float = None, z_std: float = None) -> pd.Series:
    """
    Z-score signal with entry / exit / stop-loss logic.

    If z_mean and z_std are supplied (training-period statistics), the z-score
    is fixed: z = (spread - z_mean) / z_std. This is the correct mode when the
    spread is the Kalman residual from log-price regression — it measures how far
    the current spread is from the TRAINING-PERIOD equilibrium, which is the
    long-run mean-reversion target.

    If z_mean/z_std are None, falls back to a 20-day rolling z-score (kept for
    legacy compatibility, but avoid: rolling z on a non-stationary spread causes
    immediate stop-outs).

    State machine (evaluated in priority order):
      |z| > STOP_Z           → exit immediately (stop loss)
      |z| < EXIT_Z           → exit (mean-reversion complete)
      z > ENTRY_Z & gate_ok  → short spread (z too high, expect reversion down)
      z < -ENTRY_Z & gate_ok → long spread  (z too low,  expect reversion up)
    """
    if z_mean is not None and z_std is not None and z_std > 1e-8:
        z = (spread - z_mean) / z_std
    else:
        mu  = spread.rolling(ROLLING_Z_WINDOW, min_periods=max(10, ROLLING_Z_WINDOW // 3)).mean()
        std = spread.rolling(ROLLING_Z_WINDOW, min_periods=max(10, ROLLING_Z_WINDOW // 3)).std()
        std = std.replace(0, np.nan)
        z   = (spread - mu) / std

    signal   = pd.Series(0.0, index=spread.index)
    position = 0

    for i in range(len(z)):
        zi       = z.iloc[i]
        gate_ok  = bool(vol_gate.iloc[i]) if i < len(vol_gate) else True

        if np.isnan(zi):
            signal.iloc[i] = 0
            continue

        if abs(zi) > STOP_Z:
            position = 0
        elif abs(zi) < EXIT_Z:
            position = 0
        elif position == 0:
            if zi > ENTRY_Z and gate_ok:
                position = -1   # spread too high → short: short Y, long X
            elif zi < -ENTRY_Z and gate_ok:
                position =  1   # spread too low  → long:  long Y, short X

        signal.iloc[i] = position

    return signal


# ── P&L computation ───────────────────────────────────────────────────────────

def compute_pnl(signal: pd.Series,
                dY: pd.Series,
                dX: pd.Series,
                beta_held: pd.Series,
                prices_y: pd.Series,
                prices_x: pd.Series,
                notional: float,
                rf_daily: pd.Series,
                log_price_mode: bool = False) -> pd.Series:
    """
    CORRECT P&L formula: signal[t-1] * (dY[t] - beta[t-1] * dX[t])

    log_price_mode=True (default when Kalman runs on log prices):
      dY, dX are log returns (dimensionless). raw P&L is already a return.
      dollar_pnl = raw_pnl_return * notional  — no spread_notional division.

    log_price_mode=False (legacy: Kalman on raw prices):
      dY, dX are dollar price changes. Normalize by dollar cost of position.
      dollar_pnl = (raw_dollar_pnl / spread_notional) * notional

    Parameters
    ----------
    signal         : position signal (0, +1, -1) at each close
    dY             : daily change of Y (log return if log_price_mode, else price change)
    dX             : daily change of X (log return if log_price_mode, else price change)
    beta_held      : Kalman beta at each t (shift(1) applied internally)
    prices_y       : dollar price of Y (for notional calculation in non-log mode)
    prices_x       : dollar price of X (for notional calculation in non-log mode)
    notional       : target dollar position size
    rf_daily       : daily risk-free rate (fraction, e.g. 0.045/252)
    log_price_mode : True when dY/dX are log returns (dimensionless)
    """
    sig_s1  = signal.shift(1)
    beta_s1 = beta_held.shift(1)

    raw_pnl = sig_s1 * (dY - beta_s1 * dX)

    if log_price_mode:
        # dY and dX are log returns → raw_pnl is already a fraction return
        dollar_pnl = raw_pnl.fillna(0) * notional
    else:
        # dY and dX are dollar price changes → normalize by position cost
        spread_notional = (prices_y.abs() + beta_s1.abs() * prices_x.abs()).replace(0, np.nan)
        dollar_pnl      = (raw_pnl / spread_notional).fillna(0) * notional

    # Transaction costs: COST_BPS / 10000 of notional at each signal change
    trade_bars = (signal != signal.shift(1)).astype(float)
    cost       = trade_bars * (COST_BPS / 10_000) * notional

    # Daily risk-free opportunity cost (only when position is held)
    pos_held = signal.shift(1).abs()
    rf       = rf_daily.reindex(dY.index).ffill().fillna(0) * notional * pos_held

    return (dollar_pnl - cost - rf).fillna(0)


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.std() < 1e-8 or len(r) < 5:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(252))


# ── HMM Regime Gate (Module 4b) ───────────────────────────────────────────────

def _get_hmm_gate(train_returns: pd.DataFrame, test_index: pd.Index) -> pd.Series:
    """
    Fit a 2-state HMM on training market returns, then run the forward algorithm
    on the full (train+test) returns to get causal P(bear) at each test date.

    Returns boolean Series: True = risk-on (allow entries), False = risk-off (block).
    Uses SPY if available, otherwise equal-weighted market return.

    Requires: regime_hedger.baum_welch, _emissions, _forward
    """
    try:
        sys.path.insert(0, _parent)
        from regime_hedger import baum_welch, _emissions, _forward

        # Use SPY if in universe, else equal-weight
        if "SPY" in train_returns.columns:
            obs = train_returns["SPY"].dropna().values
        else:
            obs = train_returns.mean(axis=1).dropna().values

        if len(obs) < 52:
            return pd.Series(True, index=test_index)

        pi, A, mus, sigmas, _ = baum_welch(obs)
        bull = int(np.argmax(mus))

        # Forward-only pass for causal P(bear)
        B      = _emissions(obs, mus, sigmas)
        alpha, _ = _forward(B, pi, A)
        p_bear = alpha[:, 1 - bull]

        # Map weekly-ish gamma to daily test index (forward-fill)
        p_bear_s  = pd.Series(p_bear, index=train_returns.index[-len(obs):])
        full_idx  = p_bear_s.index.union(test_index).sort_values()
        p_bear_d  = p_bear_s.reindex(full_idx).ffill().bfill()
        p_bear_test = p_bear_d.reindex(test_index).fillna(0.5)

        return p_bear_test < 0.5   # True = risk-on (bull regime)

    except Exception:
        return pd.Series(True, index=test_index)


# ── Fold runner ───────────────────────────────────────────────────────────────

def run_fold(train_data: dict,
             test_data: dict,
             rf_daily: pd.Series,
             verbose: bool = False) -> dict:
    """
    Full pipeline for one walk-forward fold.

    Parameters
    ----------
    train_data : dict with keys 'prices' and 'log_prices' (training window)
    test_data  : dict with keys 'prices' and 'log_prices' (OOS test window)
    rf_daily   : daily risk-free rate series (full history, reindexed internally)
    verbose    : print per-pair results

    Returns
    -------
    dict:
      pnl          : pd.Series of aggregate daily dollar P&L
      pairs        : list of per-pair result dicts
      n_pairs_used : int
    """
    # Step 1: screen pairs on training data only
    ranked    = screen_pairs(train_data["log_prices"])
    top_pairs = ranked[:MAX_PAIRS]

    if not top_pairs:
        empty = pd.Series(0.0, index=test_data["prices"].index)
        return {"pnl": empty, "pairs": [], "n_pairs_used": 0}

    if verbose:
        print(f"    Screened: {len(ranked)} cointegrated pairs, using top {len(top_pairs)}")
        print_screening_report(top_pairs, top_n=len(top_pairs))

    # HMM regime gate (Module 4b)
    hmm_gate = None
    if USE_HMM_GATE:
        train_rets = train_data["log_prices"].diff().dropna()
        hmm_gate   = _get_hmm_gate(train_rets, test_data["prices"].index)

    fold_pnl     = pd.Series(0.0, index=test_data["prices"].index)
    pair_results = []

    for info in top_pairs:
        t1, t2 = info["pair"]

        if t1 not in test_data["prices"].columns or t2 not in test_data["prices"].columns:
            continue

        # Use LOG PRICES for Kalman — consistent with cointegration screening.
        # Kalman on raw prices produces a non-stationary spread when prices trend;
        # the rolling z-score then immediately hits STOP_Z → avg hold of ~2 days.
        # Log prices produce a stationary Kalman residual if the pair is cointegrated.
        Y_train_log = train_data["log_prices"][t1]
        X_train_log = train_data["log_prices"][t2]
        Y_test_log  = test_data["log_prices"][t1]
        X_test_log  = test_data["log_prices"][t2]

        # Dollar prices kept for vol estimation and notional calculation only
        Y_train_px  = train_data["prices"][t1]
        X_train_px  = train_data["prices"][t2]
        Y_test_px   = test_data["prices"][t1]
        X_test_px   = test_data["prices"][t2]

        # Step 2a: OLS cointegrating spread — used for SIGNAL GENERATION.
        # The Kalman spread is minimized by the filter (it's the observation residual),
        # so it never reaches ±2σ to trigger entry. The OLS spread with FIXED training
        # parameters is the stationary mean-reverting series we want to trade.
        # screener already stored ols_alpha and ols_beta in the info dict.
        alpha_ols = info.get("ols_alpha", 0.0)
        beta_ols  = info["ols_beta"]
        ols_train = Y_train_log - beta_ols * X_train_log - alpha_ols
        z_mean    = float(ols_train.mean())  # ≈ 0 (alpha absorbs intercept)
        z_std     = float(ols_train.std())   # spread volatility from training

        ols_test  = Y_test_log - beta_ols * X_test_log - alpha_ols

        # Step 2b: warm-up Kalman on train+test log prices → time-varying beta for hedging.
        # Kalman beta is used in the P&L formula (better hedge than fixed OLS beta
        # when the cointegrating ratio drifts over time).
        kf_df  = run_kalman_warmup(Y_train_log, X_train_log, Y_test_log, X_test_log)
        beta_k = kf_df["beta"].ffill()

        # Step 3: vol filter — entry gate (uses actual prices, not log prices)
        vol_gate = vol_entry_gate(Y_test_px, X_test_px, IV_RANK_GATE)
        if hmm_gate is not None:
            vol_gate = vol_gate & hmm_gate.reindex(vol_gate.index).fillna(True)

        # Step 4: vol-scaled notional (use training window vol for stability)
        rv_y = realized_vol(Y_train_px)
        rv_x = realized_vol(X_train_px)
        sv   = spread_vol_estimate(
            float(rv_y.dropna().iloc[-1]) if len(rv_y.dropna()) else 0.20,
            float(rv_x.dropna().iloc[-1]) if len(rv_x.dropna()) else 0.20,
            float(beta_k.dropna().iloc[-1]) if len(beta_k.dropna()) else 1.0,
        )
        notional = vol_scaled_notional(sv)

        # Step 5: generate signal on OLS spread using training-period statistics
        signal = generate_signals(ols_test, vol_gate, z_mean=z_mean, z_std=z_std)

        # Step 6: P&L — Kalman beta for hedging, log-return mode
        # signal * (r_Y - beta_kalman * r_X) gives portfolio log-return fraction
        # dollar_pnl = fraction * notional (no spread_notional division needed)
        pnl = compute_pnl(
            signal, Y_test_log.diff(), X_test_log.diff(), beta_k,
            Y_test_px, X_test_px, notional, rf_daily,
            log_price_mode=True,
        )

        n_trades    = int((signal != signal.shift(1)).sum())
        pair_sharpe = _sharpe(pnl)

        # BS diagnostic on current OLS spread state
        spread_now  = float(ols_test.dropna().iloc[-1]) if len(ols_test.dropna()) else 0.0
        bs_diag     = bs_spread_diagnostic(spread_now, sv, info["half_life"])

        fold_pnl = fold_pnl.add(pnl, fill_value=0)

        pair_results.append({
            "pair":        (t1, t2),
            "pvalue":      info["pvalue"],
            "half_life":   info["half_life"],
            "notional":    notional,
            "spread_vol":  sv,
            "n_trades":    n_trades,
            "pair_sharpe": pair_sharpe,
            "bs_call":     bs_diag["bs_call"],
            "bs_put":      bs_diag["bs_put"],
        })

        if verbose:
            avg_hold = test_data["prices"].shape[0] / max(n_trades, 1)
            print(f"    {t1}/{t2:<10} Sharpe={pair_sharpe:>5.2f}  "
                  f"Trades={n_trades:>3}  Hold={avg_hold:.1f}d  "
                  f"Notional=${notional:,.0f}  BS_call={bs_diag['bs_call']:.3f}")

    return {
        "pnl":          fold_pnl,
        "pairs":        pair_results,
        "n_pairs_used": len(pair_results),
    }


if __name__ == "__main__":
    import yfinance as yf

    print("Testing dynamic_hedger on KO/PEP/XOM/CVX (mini universe)...")
    tickers = ["KO", "PEP", "XOM", "CVX"]
    raw     = yf.download(tickers, start="2018-01-01", end="2024-01-01",
                          auto_adjust=True, progress=False)["Close"].ffill().dropna()
    log_px  = np.log(raw)

    n = len(raw)
    split = int(n * 0.7)
    train = {"prices": raw.iloc[:split],   "log_prices": log_px.iloc[:split]}
    test  = {"prices": raw.iloc[split:],   "log_prices": log_px.iloc[split:]}
    rf    = pd.Series(0.045 / 252, index=raw.index)

    result = run_fold(train, test, rf, verbose=True)
    print(f"\nFold aggregate Sharpe: {_sharpe(result['pnl']):.2f}")
    print(f"Fold cumulative P&L:   ${result['pnl'].sum():.2f}")
