"""
main.py
───────
Entry point for the Dynamic Pairs Trading System.

Run:
  cd FYD/Simulations/Markov-Models/PairSelector
  python main.py

Walk-forward structure:
  Training window : rolling 252 trading days (1 year)
  Test window     : next 63 trading days (1 quarter)
  Step size       : 63 days (non-overlapping OOS)
  Total folds     : ~(total_days - 252) / 63

Pairs flow per fold:
  1. Screen all 435 combinations for cointegration (on training data ONLY)
  2. Keep top MAX_PAIRS by p-value with half-life 5-40 days
  3. Run Kalman filter on each OOS pair → dynamic hedge ratio
  4. Apply BS vol gate (block entries if IV rank > 0.80)
  5. Generate z-score signals → compute P&L (correct formula)
  6. Aggregate across pairs → fold P&L
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # use interactive backend; switch to "Agg" for headless
import matplotlib.pyplot as plt
from datetime import date

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from pair_universe   import fetch_universe, UNIVERSE
from dynamic_hedger  import run_fold, _sharpe
from performance_report import generate_report, plot_report

# ── Configuration ─────────────────────────────────────────────────────────────

START      = "2010-01-01"
END        = date.today().strftime("%Y-%m-%d")   # always pulls through today
TRAIN_DAYS = 504    # 2 year training window — EG test needs 500+ obs for statistical power
TEST_DAYS  = 63     # 1 quarter OOS test window
VERBOSE    = False  # set True to print per-pair detail each fold


def main():
    print("=" * 60)
    print("  Dynamic Pairs Trading System")
    print("=" * 60)

    # ── Step 1: Download universe ─────────────────────────────────────────────
    print("\n[1/3] Fetching universe data...")
    data       = fetch_universe(UNIVERSE, START, END)
    prices     = data["prices"]
    log_prices = data["log_prices"]
    rf_daily   = data["rf_daily"]
    dates      = prices.index

    n_tickers = len(data["tickers"])
    n_days    = len(dates)
    n_pairs   = n_tickers * (n_tickers - 1) // 2
    n_folds   = max(0, (n_days - TRAIN_DAYS) // TEST_DAYS)

    print(f"  Tickers   : {n_tickers}")
    print(f"  Pairs     : {n_pairs}")
    print(f"  Days      : {n_days}  ({dates[0].date()} to {dates[-1].date()})")
    print(f"  Folds     : {n_folds}  (train={TRAIN_DAYS}d, test={TEST_DAYS}d)")

    # ── Step 2: Walk-forward loop ─────────────────────────────────────────────
    print(f"\n[2/3] Running {n_folds} walk-forward folds...")

    fold_results = []
    all_pnl      = []
    i            = TRAIN_DAYS
    fold_num     = 0

    while i + TEST_DAYS <= n_days:
        train_idx = dates[i - TRAIN_DAYS : i]
        test_idx  = dates[i : i + TEST_DAYS]

        train_data = {
            "prices":     prices.loc[train_idx],
            "log_prices": log_prices.loc[train_idx],
        }
        test_data = {
            "prices":     prices.loc[test_idx],
            "log_prices": log_prices.loc[test_idx],
        }

        fold_num += 1
        status = (
            f"  Fold {fold_num:>3}/{n_folds} | "
            f"train {train_idx[0].date()} - {train_idx[-1].date()} | "
            f"test  {test_idx[0].date()} - {test_idx[-1].date()}"
        )

        result = run_fold(train_data, test_data, rf_daily, verbose=VERBOSE)
        fold_pnl_sharpe = _sharpe(result["pnl"])

        print(f"{status} | Sharpe={fold_pnl_sharpe:>5.2f}  Pairs={result['n_pairs_used']}")

        fold_results.append(result)
        all_pnl.append(result["pnl"])
        i += TEST_DAYS

    if not fold_results:
        print("No folds completed. Check START/END dates and TRAIN_DAYS.")
        return

    all_pnl_series = pd.concat(all_pnl).sort_index()

    # ── Step 3: Report ────────────────────────────────────────────────────────
    print(f"\n[3/3] Generating report...")
    generate_report(fold_results, all_pnl_series)

    fig = plot_report(fold_results, all_pnl_series)
    plt.show()


if __name__ == "__main__":
    main()
