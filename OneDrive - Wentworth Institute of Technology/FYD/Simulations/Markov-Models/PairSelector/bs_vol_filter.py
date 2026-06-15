"""
bs_vol_filter.py
────────────────
Wraps black_scholes_simulator.py to provide:
  1. Realized vol per ticker (rolling window)
  2. IV rank — blocks entries when vol is spiking (>80th percentile)
  3. Vol-scaled position sizing (target 2% annualized vol per pair)
  4. BS spread diagnostic (treat spread as underlying, zero as strike)

CONFIRMED function names from reading black_scholes_simulator.py:
  bs_call(S, K, r, sigma, T)   -- European call
  bs_put(S, K, r, sigma, T)    -- European put via put-call parity
  bs_greeks(S, K, r, sigma, T) -- delta, gamma, vega, theta, rho
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ── Import from existing Black-Scholes simulator ──────────────────────────────
_bs_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Black-Scholes")
)
sys.path.insert(0, _bs_path)
from black_scholes_simulator import bs_call, bs_put, bs_greeks   # exact names confirmed


# ── Realized vol ──────────────────────────────────────────────────────────────

def realized_vol(prices: pd.Series, window: int = 30) -> pd.Series:
    """Annualized realized vol from log returns, rolling `window` days."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


# ── IV rank ───────────────────────────────────────────────────────────────────

def iv_rank(vol_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Percentile rank of current realized vol vs trailing `window` days.
    0 = lowest vol in window, 1 = highest.
    Gate entries when iv_rank > 0.80 (vol is spiking).
    """
    lo = vol_series.rolling(window, min_periods=60).min()
    hi = vol_series.rolling(window, min_periods=60).max()
    return ((vol_series - lo) / (hi - lo + 1e-10)).clip(0.0, 1.0)


def vol_entry_gate(prices_y: pd.Series,
                   prices_x: pd.Series,
                   iv_rank_threshold: float = 0.80) -> pd.Series:
    """
    Returns boolean Series: True = ok to enter, False = vol spike blocks entry.
    Gate fires when EITHER leg's vol rank exceeds the threshold.
    """
    vr_y = iv_rank(realized_vol(prices_y))
    vr_x = iv_rank(realized_vol(prices_x))
    combined = vr_y.reindex(prices_y.index).ffill().fillna(0.5)
    other    = vr_x.reindex(prices_y.index).ffill().fillna(0.5)
    return (combined < iv_rank_threshold) & (other < iv_rank_threshold)


# ── Vol-scaled position sizing ────────────────────────────────────────────────

def spread_vol_estimate(vol_y: float, vol_x: float, beta: float) -> float:
    """
    Approximate annualized vol of the spread (Y - beta*X) assuming independence.
    Actual correlation will be higher (KO/PEP ~0.7), so this is conservative.
    """
    return float(np.sqrt(vol_y**2 + (beta * vol_x)**2))


def vol_scaled_notional(spread_vol: float,
                         target_vol: float   = 0.02,
                         base_notional: float = 10_000,
                         min_notional: float  = 5_000,
                         max_notional: float  = 20_000) -> float:
    """
    Scale position so each pair contributes ~2% annualized P&L volatility.
    Capped between $5k and $20k.
    target_vol / spread_vol  = fraction of base_notional to deploy.
    """
    if spread_vol <= 1e-6:
        return base_notional
    raw = (target_vol / spread_vol) * base_notional
    return float(np.clip(raw, min_notional, max_notional))


# ── BS spread diagnostic ──────────────────────────────────────────────────────

def bs_spread_diagnostic(spread_level: float,
                          spread_vol: float,
                          half_life_days: float,
                          rf: float = 0.045) -> dict:
    """
    Treat the spread as an underlying asset with:
      S = |spread_level| (current spread magnitude)
      K = 0.01           (near-zero strike — reversion to zero)
      T = half_life / 252 (expected time to reversion)
      sigma = spread_vol  (annualized spread vol)

    A high call price means the spread is wide and has time value —
    there is optionality in the mean-reversion. A low call price means
    either the spread is already near zero or vol is low.

    Uses bs_call and bs_put from black_scholes_simulator.py.
    """
    S     = max(abs(spread_level), 0.001)
    K     = 0.01
    T     = max(half_life_days / 252.0, 1.0 / 252.0)
    sigma = max(spread_vol, 0.01)

    call_val = bs_call(S, K, rf, sigma, T)
    put_val  = bs_put(S, K, rf, sigma, T)

    return {
        "bs_call":         call_val,
        "bs_put":          put_val,
        "S":               S,
        "K":               K,
        "T_years":         T,
        "sigma":           sigma,
        "spread_level":    spread_level,
        "half_life_days":  half_life_days,
    }


if __name__ == "__main__":
    import yfinance as yf

    print("Testing bs_vol_filter on KO / PEP...")
    raw = yf.download(["KO", "PEP"], start="2020-01-01", auto_adjust=True,
                      progress=False)["Close"]
    ko  = raw["KO"].dropna()
    pep = raw["PEP"].dropna()

    vol_ko  = realized_vol(ko)
    vol_pep = realized_vol(pep)
    ivr_ko  = iv_rank(vol_ko)
    ivr_pep = iv_rank(vol_pep)
    gate    = vol_entry_gate(ko, pep)

    print(f"  KO  current vol   : {vol_ko.dropna().iloc[-1]:.1%}")
    print(f"  PEP current vol   : {vol_pep.dropna().iloc[-1]:.1%}")
    print(f"  KO  IV rank       : {ivr_ko.dropna().iloc[-1]:.2f}")
    print(f"  PEP IV rank       : {ivr_pep.dropna().iloc[-1]:.2f}")
    print(f"  Entry gate today  : {'OPEN' if gate.iloc[-1] else 'BLOCKED'}")

    beta = 0.35
    sv   = spread_vol_estimate(vol_ko.iloc[-1], vol_pep.iloc[-1], beta)
    n    = vol_scaled_notional(sv)
    print(f"\n  Spread vol (ann.) : {sv:.1%}")
    print(f"  Notional target   : ${n:,.0f}")

    import numpy as np
    spread_now = float(ko.iloc[-1] - beta * pep.iloc[-1])
    diag = bs_spread_diagnostic(spread_now, sv, half_life_days=15.0)
    print(f"\n  BS Spread Diagnostic:")
    for k, v in diag.items():
        print(f"    {k:<18}: {v:.4f}")
