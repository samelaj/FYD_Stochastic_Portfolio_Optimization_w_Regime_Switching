# current_pairs.py
# Run this standalone: python current_pairs.py
# Outputs: current_pairs_recommendation.png
# Purpose: show non-technical readers exactly which pairs to trade and why

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from cointegration_screener import screen_pairs, SECTOR_MAP

END_DATE   = date.today().strftime('%Y-%m-%d')
START_DATE = (date.today() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years back
UNIVERSE   = list(SECTOR_MAP.keys())

# --- Fetch data ---
print(f"Fetching {len(UNIVERSE)} tickers from {START_DATE} to {END_DATE}...")
raw = yf.download(UNIVERSE, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)['Close'].ffill().dropna()
log_prices = np.log(raw)

# --- Screen for pairs ---
print("Screening for cointegrated pairs...")
ranked = screen_pairs(log_prices, pvalue_threshold=0.15)
top = ranked[:5]   # top 5 pairs

if not top:
    print("No cointegrated pairs found in current window.")
    sys.exit(0)

print(f"\n{'='*55}")
print(f"  CURRENT PAIRS TO TRADE  ({END_DATE})")
print(f"{'='*55}")
for i, p in enumerate(top, 1):
    t1, t2 = p['pair']
    print(f"  {i}. {t1} / {t2}  |  p={p['pvalue']:.3f}  |  half-life={p['half_life']:.0f}d")
print(f"{'='*55}\n")

# --- Build figure ---
n = len(top)
fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
fig.suptitle(f'Pairs to Trade -- {END_DATE}', fontsize=16, fontweight='bold', y=1.01)

if n == 1:
    axes = [axes]

for i, p in enumerate(top):
    t1, t2 = p['pair']
    ax_price  = axes[i][0]
    ax_spread = axes[i][1]

    # Normalize prices to 100 for easy comparison
    p1    = np.exp(log_prices[t1]).dropna()
    p2    = np.exp(log_prices[t2]).dropna()
    norm1 = p1 / p1.iloc[0] * 100
    norm2 = p2 / p2.iloc[0] * 100

    ax_price.plot(norm1.index, norm1.values, label=t1, linewidth=1.5, color='#2563EB')
    ax_price.plot(norm2.index, norm2.values, label=t2, linewidth=1.5, color='#DC2626')
    ax_price.set_title(f'{t1} vs {t2}  (sector: {p["sector"]})', fontweight='bold')
    ax_price.set_ylabel('Price (rebased to 100)')
    ax_price.legend()
    ax_price.grid(True, alpha=0.3)

    # Spread z-score using OLS cointegrating vector (same as trading system)
    alpha_ols = p.get('ols_alpha', 0.0)
    beta_ols  = p['ols_beta']
    spread    = log_prices[t1] - beta_ols * log_prices[t2] - alpha_ols
    spread_mean = spread.mean()
    spread_std  = spread.std()
    z = (spread - spread_mean) / spread_std
    z_plot = z.dropna()

    ax_spread.plot(z_plot.index, z_plot.values, color='gray', linewidth=0.8, alpha=0.5)
    ax_spread.axhline( 2.0, color='red',   linestyle='--', linewidth=1, label='Sell signal (+2 std)')
    ax_spread.axhline(-2.0, color='green', linestyle='--', linewidth=1, label='Buy signal (-2 std)')
    ax_spread.axhline( 0,   color='black', linestyle='-',  linewidth=0.5, alpha=0.3)

    # Shade signal zones
    ax_spread.fill_between(z_plot.index,  2.0, z_plot.values,
                           where=z_plot.values >  2.0, alpha=0.2, color='red',   label='Short spread')
    ax_spread.fill_between(z_plot.index, -2.0, z_plot.values,
                           where=z_plot.values < -2.0, alpha=0.2, color='green', label='Long spread')

    # Current signal callout
    current_z = float(z_plot.iloc[-1])
    if abs(current_z) < 2.0:
        signal_text = 'HOLD (no signal)'
        color_text  = 'gray'
    elif current_z < -2.0:
        signal_text = f'BUY {t1} / SELL {t2}'
        color_text  = 'green'
    else:
        signal_text = f'SELL {t1} / BUY {t2}'
        color_text  = 'red'

    ax_spread.set_title(f'Spread z-score  |  Now: {current_z:.2f} std  >>  {signal_text}',
                        fontweight='bold', color=color_text)
    ax_spread.set_ylabel('Z-score (standard deviations)')
    ax_spread.set_ylim(-4, 4)
    ax_spread.legend(fontsize=8)
    ax_spread.grid(True, alpha=0.3)

    # Annotation
    ax_spread.annotate(
        f'Half-life: {p["half_life"]:.0f} days\np-value: {p["pvalue"]:.3f}',
        xy=(0.02, 0.05), xycoords='axes fraction',
        fontsize=9, color='gray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    )

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'current_pairs_recommendation.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {out}")
