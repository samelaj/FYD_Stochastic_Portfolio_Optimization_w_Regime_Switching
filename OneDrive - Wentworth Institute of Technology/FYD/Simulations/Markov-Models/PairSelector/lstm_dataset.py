"""
lstm_dataset.py
───────────────
Builds labeled training sequences for the two-head LSTM.

Dataset A (reversion): currently cointegrated pairs → did spread half-revert in 5 days?
Dataset B (emergence): same-sector non-cointegrated pairs → does pair become cointegrated within 63 days?

Feature vector (12 features per bar):
  z_score, z_momentum, spread_vol_20, beta, beta_change,
  iv_rank_leg1, iv_rank_leg2, corr_60, half_life,
  regime, days_since_cross, pvalue_rolling

Interfaces confirmed from reading existing files:
  screen_pairs(log_prices)         → list of dicts with ols_alpha, ols_beta, half_life
  realized_vol(prices, window=30)  → needs RAW prices (not log)
  iv_rank(vol_series)              → percentile rank 0-1
  regime_hedger.baum_welch(obs)    → (pi, A, mus, sigmas, gamma) — safe to import (has __main__ guard)
  hhm.py has top-level execution code at import — do NOT import it directly
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")

_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.abspath(os.path.join(_here, ".."))
sys.path.insert(0, _here)
sys.path.insert(0, _parent)

from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from cointegration_screener import SECTOR_MAP, screen_pairs
from bs_vol_filter import realized_vol, iv_rank

# Import HMM from regime_hedger (has __main__ guard — safe to import).
# Do NOT import hhm.py directly — it has top-level download + plot code.
from regime_hedger import baum_welch, _emissions, _forward

SEQUENCE_LEN = 20    # 20 bars of history per training example (4 trading weeks)
N_FEATURES   = 12


# ── Label A: reversion ────────────────────────────────────────────────────────

def label_reversion(z_score: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Binary: did the spread move at least 50% toward zero within `horizon` bars?

    Only labels bars where |z| >= 1.5 (signal is active).
    A z of 2.1 that reaches 1.0 within 5 days = 1 (50% toward zero).
    """
    z    = z_score.values
    n    = len(z)
    out  = np.zeros(n, dtype=np.float32)
    for i in range(n - horizon):
        current = z[i]
        if abs(current) < 1.5:
            continue
        target = current * 0.5
        future = z[i + 1 : i + horizon + 1]
        if current > 0 and np.any(future <= target):
            out[i] = 1.0
        elif current < 0 and np.any(future >= target):
            out[i] = 1.0
    return pd.Series(out, index=z_score.index)


# ── Label B: emergence ────────────────────────────────────────────────────────

def label_emergence(log_prices: pd.DataFrame, t1: str, t2: str,
                    horizon: int = 63, step: int = 5) -> pd.Series:
    """
    Binary: does this pair pass EG cointegration (p < 0.15) in any 63-day
    window starting within `horizon` bars of today?

    Uses step=5 to sub-sample the forward window and speed up computation.
    Each outer iteration still runs only ONE EG test on the forward window.
    """
    series = pd.concat([log_prices[t1], log_prices[t2]], axis=1).dropna()
    n      = len(series)
    out    = np.zeros(n, dtype=np.float32)

    for i in range(0, n - horizon - 60, step):
        future = series.iloc[i + 1 : i + horizon + 1]
        if len(future) < 60:
            continue
        try:
            _, pvalue, _ = coint(future[t1].values, future[t2].values)
            label = 1.0 if pvalue < 0.15 else 0.0
        except Exception:
            label = 0.0
        # Fill all bars in this step with the same label (forward-fill between steps)
        out[i : min(i + step, n)] = label

    return pd.Series(out, index=series.index)


# ── HMM regime feature ────────────────────────────────────────────────────────

def compute_regime_series(log_prices: pd.DataFrame) -> pd.Series:
    """
    Fit a 2-state Gaussian HMM on the equal-weighted portfolio log-return.
    Returns a daily Series: 0 = risk-on (bull), 1 = risk-off (bear).

    Uses regime_hedger.baum_welch (safe to import — has __main__ guard).
    hhm.py is NOT used directly because it runs data downloads at import time.
    """
    eq_returns = log_prices.diff().dropna().mean(axis=1).values
    if len(eq_returns) < 50:
        return pd.Series(0, index=log_prices.index)

    try:
        pi, A, mus, sigmas, gamma = baum_welch(eq_returns)
        bull      = int(np.argmax(mus))
        p_bull    = gamma[:, bull]
        # regime = 0 (risk-on) when p_bull > 0.5, else 1 (risk-off)
        regime    = (p_bull <= 0.5).astype(float)
        # Align to original log_prices index (gamma is 1 obs shorter due to diff)
        idx       = log_prices.index[1:]   # diff() drops first row
        regime_s  = pd.Series(regime, index=idx)
        return regime_s.reindex(log_prices.index).ffill().fillna(0)
    except Exception:
        return pd.Series(0.0, index=log_prices.index)


# ── Rolling half-life (vectorized) ───────────────────────────────────────────

def _rolling_half_life(spread: pd.Series, window: int = 60) -> pd.Series:
    """
    Fast vectorized rolling AR(1) coefficient using rolling cov/var.
    phi = Cov(delta_spread, lag_spread) / Var(lag_spread)
    half_life = -log(2) / log(1 + phi)  [phi < 0 required for mean reversion]

    Avoids O(n²) nested loop from the original spec's rolling_hl.
    """
    lag   = spread.shift(1)
    ds    = spread.diff()
    # rolling OLS: slope = cov(ds, lag) / var(lag)
    phi   = ds.rolling(window, min_periods=30).cov(lag) / \
            lag.rolling(window, min_periods=30).var()
    # Clamp phi to meaningful range (negative = mean-reverting)
    phi   = phi.clip(-0.99, -0.001)
    hl    = -np.log(2) / np.log(1 + phi)
    return hl.clip(1, 500)


# ── Rolling EG p-value ────────────────────────────────────────────────────────

def _rolling_eg_pvalue(log_prices: pd.DataFrame, t1: str, t2: str,
                        window: int = 60, step: int = 5) -> pd.Series:
    """Compute rolling 60-day EG p-value with step-subsampling for speed."""
    n   = len(log_prices)
    idx = log_prices.index
    pv  = pd.Series(np.nan, index=idx)

    for i in range(window, n, step):
        w = log_prices[[t1, t2]].iloc[i - window : i].dropna()
        if len(w) < 40:
            continue
        try:
            _, p, _ = coint(w[t1].values, w[t2].values)
            pv.iloc[i - step : i] = p
        except Exception:
            pass

    return pv.ffill().fillna(0.5)


# ── Days since last zero-crossing ─────────────────────────────────────────────

def _days_since_cross(z: pd.Series) -> pd.Series:
    """
    Count bars since z-score last crossed zero.
    Returns a Series capped at 60 and normalized to [0, 1].
    """
    cross       = ((z > 0) != (z.shift(1) > 0)).astype(int)
    cross_idx   = cross[cross == 1].index
    days        = pd.Series(60, index=z.index, dtype=float)

    if len(cross_idx) == 0:
        return days / 60

    for i, dt in enumerate(z.index):
        past = cross_idx[cross_idx <= dt]
        if len(past) == 0:
            days.loc[dt] = 60
        else:
            days.loc[dt] = (z.index.get_loc(dt) - z.index.get_loc(past[-1]))

    return (days.clip(0, 60) / 60)


# ── Full feature matrix for one pair ─────────────────────────────────────────

def build_feature_matrix(log_prices: pd.DataFrame,
                          t1: str, t2: str,
                          kalman_beta: pd.Series,
                          regime_series: pd.Series,
                          ols_alpha: float = 0.0,
                          ols_beta: float = 1.0) -> pd.DataFrame:
    """
    Compute all 12 LSTM features for a pair over the full price history.

    Parameters
    ----------
    log_prices    : full log-price DataFrame
    t1, t2        : ticker names
    kalman_beta   : time-varying Kalman hedge ratio Series (same index as log_prices)
    regime_series : HMM regime (0=risk-on, 1=risk-off) for each date
    ols_alpha     : from screen_pairs() result — intercept of cointegrating vector
    ols_beta      : from screen_pairs() result — slope of cointegrating vector

    Returns DataFrame (n_dates, N_FEATURES) ready for sequence slicing.
    """
    # OLS spread (fixed training parameters — stationary by construction)
    spread      = log_prices[t1] - ols_beta * log_prices[t2] - ols_alpha
    spread_mean = spread.mean()
    spread_std  = spread.std() + 1e-8

    # Z-score from training-period equilibrium
    z           = (spread - spread_mean) / spread_std

    # Spread vol (annualized, 20-day)
    spread_vol  = spread.diff().rolling(20, min_periods=10).std() * np.sqrt(252)

    # 60-day rolling return correlation (log returns)
    ret1        = log_prices[t1].diff()
    ret2        = log_prices[t2].diff()
    corr_60     = ret1.rolling(60, min_periods=30).corr(ret2)

    # Rolling half-life from the OLS spread
    hl_raw      = _rolling_half_life(spread)

    # IV rank (realized vol on RAW prices — bs_vol_filter expects prices not log)
    prices_t1   = np.exp(log_prices[t1])
    prices_t2   = np.exp(log_prices[t2])
    vol_t1      = realized_vol(prices_t1)
    vol_t2      = realized_vol(prices_t2)
    ivr_t1      = iv_rank(vol_t1)
    ivr_t2      = iv_rank(vol_t2)

    # Rolling EG p-value (cointegration strength over time)
    pv_roll     = _rolling_eg_pvalue(log_prices, t1, t2)

    # Days since last zero crossing
    dsc         = _days_since_cross(z)

    # Regime aligned to log_prices index
    regime      = regime_series.reindex(log_prices.index).ffill().fillna(0)

    feat = pd.DataFrame({
        'z_score':          z,
        'z_momentum':       z - z.shift(5),
        'spread_vol_20':    spread_vol.clip(0, 2),
        'beta':             kalman_beta.reindex(log_prices.index).ffill().fillna(ols_beta),
        'beta_change':      (kalman_beta - kalman_beta.shift(10))
                            .reindex(log_prices.index).ffill().fillna(0),
        'iv_rank_leg1':     ivr_t1.fillna(0.5),
        'iv_rank_leg2':     ivr_t2.fillna(0.5),
        'corr_60':          corr_60.fillna(0),
        'half_life':        hl_raw.clip(1, 500) / 500,     # normalize [0,1]
        'regime':           regime,
        'days_since_cross': dsc,
        'pvalue_rolling':   pv_roll.clip(0, 1),
    }, index=log_prices.index)

    return feat.ffill().fillna(0).astype(np.float32)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_datasets(log_prices: pd.DataFrame,
                   screened_pairs: list,
                   all_same_sector_pairs: list,
                   kalman_betas: dict,
                   regime_series: pd.Series,
                   seq_len: int   = SEQUENCE_LEN,
                   horizon_rev: int  = 5,
                   horizon_emer: int = 63) -> tuple:
    """
    Build sequence arrays and labels for both LSTM heads.

    Parameters
    ----------
    log_prices            : full log-price history (dates x tickers)
    screened_pairs        : list of dicts from screen_pairs() — currently cointegrated
    all_same_sector_pairs : list of (t1,t2) tuples for all same-sector pairs
    kalman_betas          : dict {(t1,t2): pd.Series of Kalman beta values}
    regime_series         : daily regime (0/1) Series

    Returns
    -------
    sequences : np.ndarray (N, seq_len, N_FEATURES)
    labels    : np.ndarray (N, 2) — [reversion_label, emergence_label]
    """
    sequences, labels = [], []
    coint_set         = {p['pair'] for p in screened_pairs}

    # ── Dataset A: reversion labels on cointegrated pairs ─────────────────────
    print(f"  Building Dataset A: {len(screened_pairs)} cointegrated pairs...")
    for info in screened_pairs:
        t1, t2 = info['pair']
        alpha   = info.get('ols_alpha', 0.0)
        beta    = info['ols_beta']
        k_beta  = kalman_betas.get((t1, t2),
                                   pd.Series(beta, index=log_prices.index))

        feat     = build_feature_matrix(log_prices, t1, t2, k_beta,
                                        regime_series, alpha, beta)
        z        = feat['z_score']
        rev_lbl  = label_reversion(z, horizon=horizon_rev)
        em_lbl   = label_emergence(log_prices, t1, t2, horizon=horizon_emer)

        feat_arr = feat.values
        rev_arr  = rev_lbl.values
        em_arr   = em_lbl.reindex(log_prices.index).ffill().fillna(0).values

        for i in range(seq_len, len(feat_arr) - max(horizon_rev, horizon_emer)):
            seq = feat_arr[i - seq_len : i]
            if np.isnan(seq).any() or np.isinf(seq).any():
                continue
            sequences.append(seq)
            labels.append([rev_arr[i], em_arr[i]])

    # ── Dataset B: emergence labels on non-cointegrated pairs ─────────────────
    non_coint = [p for p in all_same_sector_pairs if p not in coint_set]
    print(f"  Building Dataset B: {len(non_coint)} non-cointegrated pairs...")
    for (t1, t2) in non_coint:
        k_beta = kalman_betas.get((t1, t2),
                                  pd.Series(1.0, index=log_prices.index))
        feat    = build_feature_matrix(log_prices, t1, t2, k_beta,
                                       regime_series, 0.0, 1.0)
        em_lbl  = label_emergence(log_prices, t1, t2, horizon=horizon_emer)

        feat_arr = feat.values
        em_arr   = em_lbl.reindex(log_prices.index).ffill().fillna(0).values

        for i in range(seq_len, len(feat_arr) - horizon_emer):
            seq = feat_arr[i - seq_len : i]
            if np.isnan(seq).any() or np.isinf(seq).any():
                continue
            sequences.append(seq)
            labels.append([0.0, em_arr[i]])   # reversion label = 0 (not active)

    if not sequences:
        raise ValueError("No valid sequences built — check data length and pair list")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.float32)
    print(f"\n  Dataset built: {len(X)} sequences  shape={X.shape}")
    print(f"  Reversion positives: {y[:,0].mean():.1%}")
    print(f"  Emergence positives: {y[:,1].mean():.1%}")
    return X, y


# ── PyTorch Dataset wrapper ───────────────────────────────────────────────────

def make_pair_dataset(sequences: np.ndarray, labels: np.ndarray):
    """
    Returns a torch Dataset. Import delayed so module can be imported even
    before PyTorch is installed (e.g., for testing build_feature_matrix alone).
    """
    import torch
    from torch.utils.data import Dataset

    class PairDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return PairDataset(sequences, labels)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import yfinance as yf
    from dynamic_hedger import run_kalman

    UNIVERSE = list(SECTOR_MAP.keys())
    print(f"Downloading 2 years of data for {len(UNIVERSE)} tickers...")
    raw   = yf.download(UNIVERSE, start='2022-01-01', end='2024-06-30',
                        auto_adjust=True, progress=False)['Close'].ffill().dropna()
    lp    = np.log(raw)

    print("Computing regime series...")
    regime = compute_regime_series(lp)
    print(f"  Regime: {regime.value_counts().to_dict()}")

    print("\nRunning cointegration screen...")
    pairs = screen_pairs(lp)
    print(f"  Found {len(pairs)} cointegrated pairs")

    all_ss = [(t1, t2) for t1, t2 in combinations(UNIVERSE, 2)
              if SECTOR_MAP.get(t1) == SECTOR_MAP.get(t2)]

    # Stub Kalman betas
    k_betas = {p['pair']: pd.Series(p['ols_beta'], index=lp.index) for p in pairs}

    X, y = build_datasets(lp, pairs, all_ss, k_betas, regime)
    print(f"\nFinal dataset: sequences={X.shape}  labels={y.shape}")
    print(f"Label check: reversion={y[:,0].mean():.1%}  emergence={y[:,1].mean():.1%}")
    assert X.shape[1:] == (SEQUENCE_LEN, N_FEATURES), f"Unexpected shape: {X.shape}"
    print("Shape assertion passed.")
