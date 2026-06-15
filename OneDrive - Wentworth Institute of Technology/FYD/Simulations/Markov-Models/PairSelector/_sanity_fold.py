import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pair_universe import fetch_universe, UNIVERSE
from dynamic_hedger import run_fold, _sharpe

data   = fetch_universe(UNIVERSE, '2010-01-01', '2014-06-30')
prices = data['prices']
log_px = data['log_prices']
rf     = data['rf_daily']
dates  = prices.index

train_idx = dates[:504]
test_idx  = dates[504:567]
train = {'prices': prices.loc[train_idx], 'log_prices': log_px.loc[train_idx]}
test  = {'prices': prices.loc[test_idx],  'log_prices': log_px.loc[test_idx]}

result   = run_fold(train, test, rf, verbose=True)
sh       = _sharpe(result['pnl'])
n_trades = sum(r['n_trades'] for r in result['pairs'])
avg_hold = len(test_idx) / max(n_trades, 1) if n_trades > 0 else 0
print(f"\nFold Sharpe: {sh:.2f}  Pairs: {result['n_pairs_used']}  n_trades: {n_trades}  Avg hold: {avg_hold:.1f}d")
print(f"Cum P&L: ${result['pnl'].sum():.2f}")
