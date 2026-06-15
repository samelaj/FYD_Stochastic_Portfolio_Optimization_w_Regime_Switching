"""
cointegration_screener.py
─────────────────────────
Screen pairs in the universe for cointegration on a rolling training window.

Filters applied in order (first failure exits early):
  1. Same sector only        — no economic rationale for cross-sector cointegration
  2. Return correlation >= 0.40 — rejects structurally diverging intra-sector pairs
  3. EG p-value < 0.15       — sufficient power on 504-day windows (EG needs ~500 obs)
  4. phi < -0.003            — rejects EG false positives (phi~0 = correlated drift)
  5. Half-life 5-500 days    — tradeable mean reversion speed

No Johansen test: debug showed 0 pairs reach it; it's dead weight on this window length.
Correlation uses 60-day rolling LOG RETURNS (not log prices) — price-level correlation is
spuriously high for any two trending stocks in the same sector.

MUST be called ONLY on training-window data. Never pass test data here.
"""

import warnings
import numpy as np
import pandas as pd
from itertools import combinations

from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")


# ── Sector map ────────────────────────────────────────────────────────────────
# Only pairs sharing a sector are economically motivated to mean-revert.
# Cross-sector pairs that pass EG are statistical coincidences.

SECTOR_MAP = {
    "KO":   "staples",  "PEP":  "staples",  "MCD":  "staples",
    "YUM":  "staples",  "SBUX": "staples",
    "XOM":  "energy",   "CVX":  "energy",   "COP":  "energy",
    "SLB":  "energy",   "BKR":  "energy",
    "JPM":  "fin",      "BAC":  "fin",      "WFC":  "fin",
    "GS":   "fin",      "MS":   "fin",
    "JNJ":  "health",   "PFE":  "health",   "MRK":  "health",
    "ABT":  "health",   "BMY":  "health",
    "NEE":  "util",     "DUK":  "util",     "SO":   "util",
    "D":    "util",     "AEP":  "util",
}


# ── Half-life ─────────────────────────────────────────────────────────────────

def half_life(spread: pd.Series) -> tuple:
    """
    Ornstein-Uhlenbeck half-life via AR(1) on the spread.
    Fits: delta_spread_t = phi * spread_{t-1} + epsilon_t

    Returns (hl, phi):
      hl  = half-life in days (np.inf if phi >= 0)
      phi = AR(1) coefficient (negative = mean-reverting)
    """
    s = spread.dropna()
    if len(s) < 20:
        return np.inf, 0.0
    lag   = s.shift(1).dropna()
    delta = s.diff().dropna()
    idx   = delta.index.intersection(lag.index)
    if len(idx) < 20:
        return np.inf, 0.0

    phi = float(OLS(delta.loc[idx].values, lag.loc[idx].values).fit().params[0])
    if phi >= 0:
        return np.inf, phi
    return float(-np.log(2) / np.log(1 + phi)), phi


# ── Main screener ─────────────────────────────────────────────────────────────

def screen_pairs(log_prices: pd.DataFrame,
                 pvalue_threshold: float  = 0.15,
                 min_half_life: float     = 5.0,
                 max_half_life: float     = 500.0,
                 min_phi_magnitude: float = 0.003,
                 min_ret_corr: float      = 0.40,
                 min_obs: int             = 200) -> list:
    """
    Economically-filtered cointegration screen.

    Parameters
    ----------
    log_prices       : training-window log prices (days x tickers)
    pvalue_threshold : EG p-value cutoff (default 0.15)
    min_half_life    : minimum OU half-life in days (default 5)
    max_half_life    : maximum OU half-life in days (default 500)
    min_phi_magnitude: minimum |AR(1) phi| to reject phi~0 false positives (default 0.003)
    min_ret_corr     : minimum mean 60-day rolling LOG RETURN correlation (default 0.40)
    min_obs          : minimum shared trading days (default 200)

    Returns
    -------
    List of dicts sorted by EG p-value ascending.
    """
    results = []
    tickers = log_prices.columns.tolist()

    for t1, t2 in combinations(tickers, 2):

        # ── Filter 1: same sector ─────────────────────────────────────────────
        if SECTOR_MAP.get(t1) != SECTOR_MAP.get(t2):
            continue

        aligned = log_prices[[t1, t2]].dropna()
        if len(aligned) < min_obs:
            continue

        # ── Filter 2: 60-day rolling LOG RETURN correlation ───────────────────
        # Use returns (not price levels) — price-level correlation is spuriously
        # high for any two trending stocks even without cointegration.
        ret1 = aligned[t1].diff()
        ret2 = aligned[t2].diff()
        rolling_corr = ret1.rolling(60, min_periods=30).corr(ret2).dropna()
        if rolling_corr.empty or rolling_corr.mean() < min_ret_corr:
            continue

        # ── Filter 3: Engle-Granger p-value ──────────────────────────────────
        _, pvalue, _ = coint(aligned[t1].values, aligned[t2].values)
        if pvalue > pvalue_threshold:
            continue

        # ── Build OLS residuals (same series EG tests for stationarity) ──────
        # CRITICAL: subtract BOTH slope and intercept so spread has mean≈0.
        # Without subtracting alpha, AR(1) without constant is biased to phi≈0.
        x         = aligned[t2].values
        y         = aligned[t1].values
        ols_fit   = OLS(y, add_constant(x)).fit()
        alpha_ols = float(ols_fit.params[0])
        beta_ols  = float(ols_fit.params[1])
        spread    = pd.Series(y - beta_ols * x - alpha_ols, index=aligned.index)

        # ── Filter 4: phi magnitude — reject correlated drift (phi ~ 0) ──────
        hl, phi = half_life(spread)
        if abs(phi) < min_phi_magnitude:
            continue

        # ── Filter 5: half-life range ─────────────────────────────────────────
        if not (min_half_life <= hl <= max_half_life):
            continue

        results.append({
            "pair":          (t1, t2),
            "sector":        SECTOR_MAP.get(t1, "unknown"),
            "pvalue":        float(pvalue),
            "coint_score":   float(_coint_score(aligned[t1].values, aligned[t2].values)),
            "half_life":     float(hl),
            "phi":           float(phi),
            "mean_ret_corr": float(rolling_corr.mean()),
            "spread_std":    float(spread.std()),
            "ols_beta":      float(beta_ols),
            "ols_alpha":     float(alpha_ols),
        })

    results.sort(key=lambda r: r["pvalue"])
    return results


def _coint_score(y: np.ndarray, x: np.ndarray) -> float:
    try:
        score, _, _ = coint(y, x)
        return float(score)
    except Exception:
        return float("nan")


def print_screening_report(results: list, top_n: int = 10) -> None:
    if not results:
        print("  No pairs passed screening.")
        return
    print(f"\n  {'Pair':<16} {'Sector':<9} {'p-val':>7}  {'HL':>6}  {'RetCorr':>8}  {'phi':>8}  {'Beta':>8}")
    print(f"  {'-' * 72}")
    for r in results[:top_n]:
        t1, t2 = r["pair"]
        print(f"  {t1}/{t2:<12} {r['sector']:<9} {r['pvalue']:>7.4f}  "
              f"{r['half_life']:>5.0f}d  {r['mean_ret_corr']:>8.2f}  "
              f"{r['phi']:>8.4f}  {r['ols_beta']:>8.4f}")
    if len(results) > top_n:
        print(f"  ... and {len(results) - top_n} more")


if __name__ == "__main__":
    import yfinance as yf
    from pair_universe import UNIVERSE

    print(f"Testing screener on {len(UNIVERSE)}-ticker universe, 2015-2023...")
    raw    = yf.download(UNIVERSE, start="2015-01-01", end="2023-01-01",
                         auto_adjust=True, progress=False)["Close"]
    log_px = np.log(raw.ffill().dropna(axis=1, thresh=int(len(raw) * 0.95))
                       .ffill().bfill().dropna())

    # Test on a 504-day training slice (mirrors main.py fold structure)
    train_lp = log_px.iloc[:504]
    results  = screen_pairs(train_lp)
    print(f"\nFound {len(results)} pairs on first 504-day window (2015-2017):")
    print_screening_report(results)

    # Also test on a more recent slice
    train_lp2 = log_px.iloc[-504:]
    results2  = screen_pairs(train_lp2)
    print(f"\nFound {len(results2)} pairs on last 504-day window (2021-2023):")
    print_screening_report(results2)
