# debug_screener.py
import sys, os

import yfinance as yf
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

SECTOR_MAP = {
    'KO': 'staples', 'PEP': 'staples', 'MCD': 'staples', 'YUM': 'staples', 'SBUX': 'staples',
    'XOM': 'energy',  'CVX': 'energy',  'COP': 'energy',  'SLB': 'energy',  'BKR': 'energy',
    'JPM': 'fin',     'BAC': 'fin',     'WFC': 'fin',     'GS': 'fin',      'MS': 'fin',
    'JNJ': 'health',  'PFE': 'health',  'MRK': 'health',  'ABT': 'health',  'BMY': 'health',
    'NEE': 'util',    'DUK': 'util',    'SO': 'util',     'D': 'util',      'AEP': 'util',
}

UNIVERSE = list(SECTOR_MAP.keys())

# Window selector: old=2012-2014, mid=2017-2019, recent=2022-2024, live=2024-today
window_label = sys.argv[1] if len(sys.argv) > 1 else 'old'
if window_label == 'old':
    start, end = '2012-01-01', '2014-01-01'
elif window_label == 'mid':
    start, end = '2017-01-01', '2019-01-01'
elif window_label == 'live':
    start = '2024-01-01'
    end   = pd.Timestamp.today().strftime('%Y-%m-%d')   # always up to today
else:
    start, end = '2022-01-01', '2024-01-01'

print(f"Window: {start} to {end}")
raw = yf.download(UNIVERSE, start=start, end=end,
                  auto_adjust=True, progress=False)['Close'].ffill().dropna()
log_prices = np.log(raw)

print(f"Data shape: {log_prices.shape}")
print(f"\n{'Pair':<14} {'Sector':>8} {'Corr':>6} {'EG_p':>8} {'HL':>8} {'phi':>8} {'Status':>10}")
print("-" * 72)

passed_sector = passed_corr = passed_eg = passed_phi = passed_hl = total_pass = 0

for t1, t2 in combinations(UNIVERSE, 2):
    # Filter 1: Same sector
    if SECTOR_MAP.get(t1) != SECTOR_MAP.get(t2):
        continue
    passed_sector += 1

    aligned = pd.concat([log_prices[t1], log_prices[t2]], axis=1).dropna()
    if len(aligned) < 200:
        continue

    # Filter 2: 60-day rolling LOG RETURN correlation
    ret1 = aligned[t1].diff()
    ret2 = aligned[t2].diff()
    corr = ret1.rolling(60, min_periods=30).corr(ret2).dropna().mean()
    if corr < 0.40:
        print(f"{t1}/{t2:<8} {SECTOR_MAP[t1]:>8} {corr:>6.2f}  FAIL corr")
        continue
    passed_corr += 1

    # Filter 3: EG p-value
    try:
        _, pvalue, _ = coint(aligned[t1], aligned[t2])
    except Exception as e:
        print(f"{t1}/{t2:<8} EG error: {e}")
        continue
    if pvalue > 0.15:
        print(f"{t1}/{t2:<8} {SECTOR_MAP[t1]:>8} {corr:>6.2f} {pvalue:>8.3f}  FAIL EG")
        continue
    passed_eg += 1

    # Build OLS residuals — MUST subtract both slope and intercept so spread is mean-zero.
    # Without subtracting alpha, AR(1)-no-constant is biased to phi≈0 (EG false negatives).
    x       = aligned[t2].values
    y       = aligned[t1].values
    fit     = OLS(y, add_constant(x)).fit()
    alpha   = float(fit.params[0])
    beta    = float(fit.params[1])
    spread  = pd.Series(y - beta * x - alpha, index=aligned.index)

    # Filter 4: phi magnitude
    lag  = spread.shift(1).dropna()
    ds   = spread.diff().dropna()
    idx  = ds.index.intersection(lag.index)
    phi  = OLS(ds.loc[idx].values, lag.loc[idx].values).fit().params[0]
    if abs(phi) < 0.003:
        print(f"{t1}/{t2:<8} {SECTOR_MAP[t1]:>8} {corr:>6.2f} {pvalue:>8.3f} {'':>8} {phi:>8.4f}  FAIL phi")
        continue
    passed_phi += 1

    # Filter 5: Half-life
    hl = -np.log(2) / np.log(1 + phi) if phi < 0 else float('inf')
    if not (5 <= hl <= 500):
        print(f"{t1}/{t2:<8} {SECTOR_MAP[t1]:>8} {corr:>6.2f} {pvalue:>8.3f} {hl:>8.1f} {phi:>8.4f}  FAIL HL")
        continue
    passed_hl += 1

    print(f"{t1}/{t2:<8} {SECTOR_MAP[t1]:>8} {corr:>6.2f} {pvalue:>8.3f} {hl:>8.1f} {phi:>8.4f}  PASS")
    total_pass += 1

print(f"\n--- Filter funnel ---")
print(f"Same-sector pairs:        {passed_sector}")
print(f"After ret_corr >= 0.40:   {passed_corr}")
print(f"After EG p < 0.15:        {passed_eg}")
print(f"After |phi| >= 0.003:     {passed_phi}")
print(f"After 5 <= HL <= 500:     {passed_hl}")
print(f"TOTAL PASS:               {total_pass}")
