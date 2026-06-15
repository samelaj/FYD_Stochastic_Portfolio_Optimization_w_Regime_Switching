"""
pair_universe.py
────────────────
Extends the existing data pipeline to pull the full 30-ticker universe.

intial_data_pipeline.py has no __main__ guard — top-level code downloads
14 tickers at import time. We import only fetch_prices() using importlib
to avoid triggering that download, then build fetch_universe() on top of it.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Note on intial_data_pipeline.py ──────────────────────────────────────────
# That file has no __main__ guard: lines 47-58 call fetch_prices() and
# build_features() at import time, which triggers a yfinance download and
# then crashes when the empty-DataFrame stub is passed to build_features()
# (KeyError: 'SPY'). Safe importlib patching is not possible without
# rewriting the upstream file.
#
# fetch_universe() below follows the identical fetch_prices() pattern from
# that file (same yfinance call, same 95% threshold, same ffill/bfill),
# extended for the 30-ticker universe and with ^IRX for risk-free rate.


# ── Universe ──────────────────────────────────────────────────────────────────

UNIVERSE = [
    "KO",  "PEP", "MCD", "YUM",  "SBUX",   # Consumer Staples / Restaurants
    "XOM", "CVX", "COP", "SLB",  "BKR",    # Energy
    "JPM", "BAC", "WFC", "GS",   "MS",     # Financials
    "JNJ", "PFE", "MRK", "ABT",  "BMY",    # Healthcare
    "NEE", "DUK", "SO",  "D",    "AEP",    # Utilities
]


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_universe(tickers: list, start: str, end: str) -> dict:
    """
    Download adjusted close prices for the full universe plus ^IRX (risk-free rate).

    Returns
    -------
    dict with keys:
      prices      : pd.DataFrame (date x ticker, adjusted close)
      log_prices  : pd.DataFrame (natural log of prices)
      returns     : pd.DataFrame (daily log returns)
      rf_daily    : pd.Series   (daily risk-free rate = ^IRX / 100 / 252)
    """
    print(f"  Downloading {len(tickers)} tickers + ^IRX from {start} to {end}...")
    raw = yf.download(
        tickers + ["^IRX"],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    # Risk-free rate
    rf_series = close["^IRX"].ffill() / 100 / 252 if "^IRX" in close.columns \
                else pd.Series(0.045 / 252, index=close.index)
    prices = close.drop(columns=["^IRX"], errors="ignore")

    # Drop tickers with >5% missing data then forward-fill gaps
    missing_pct = prices.isnull().mean()
    prices = prices.loc[:, missing_pct < 0.05].ffill().bfill()

    tickers_kept = prices.columns.tolist()
    dropped = [t for t in tickers if t not in tickers_kept]
    if dropped:
        print(f"  Dropped (>5% missing): {dropped}")
    print(f"  Universe after quality filter: {len(tickers_kept)} tickers")

    log_prices = np.log(prices.clip(lower=1e-6))
    returns    = log_prices.diff()

    return {
        "prices":     prices,
        "log_prices": log_prices,
        "returns":    returns,
        "rf_daily":   rf_series,
        "tickers":    tickers_kept,
    }


if __name__ == "__main__":
    data = fetch_universe(UNIVERSE, "2015-01-01", "2024-12-31")
    print(f"\nPrices shape : {data['prices'].shape}")
    print(f"Date range   : {data['prices'].index[0].date()} to {data['prices'].index[-1].date()}")
    print(f"Tickers      : {data['tickers']}")
    print(data["prices"].tail(3))
