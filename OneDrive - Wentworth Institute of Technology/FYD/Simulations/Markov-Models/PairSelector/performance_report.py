"""
performance_report.py
─────────────────────
Aggregate walk-forward reporting for the dynamic pairs system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def _sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.std() < 1e-8 or len(r) < 5:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(252))


def _max_drawdown(pnl: pd.Series) -> float:
    cum = pnl.cumsum()
    return float((cum - cum.cummax()).min())


def generate_report(fold_results: list, all_pnl: pd.Series) -> None:
    """
    Print a comprehensive walk-forward performance report.

    Parameters
    ----------
    fold_results : list of dicts from dynamic_hedger.run_fold()
                   each has keys: pnl, pairs, n_pairs_used
    all_pnl      : concatenated daily P&L series across all folds
    """
    oos_sharpes    = [_sharpe(f["pnl"]) for f in fold_results]
    positive_folds = sum(s > 0 for s in oos_sharpes)
    ann_return     = float(all_pnl.mean() * 252)
    ann_vol        = float(all_pnl.std() * np.sqrt(252))
    sharpe         = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd         = _max_drawdown(all_pnl)

    pair_counts  = Counter()
    total_trades = 0
    pair_sharpes: dict = {}
    for f in fold_results:
        for p in f["pairs"]:
            key = f"{p['pair'][0]}/{p['pair'][1]}"
            pair_counts[key] += 1
            total_trades     += p["n_trades"]
            if key not in pair_sharpes:
                pair_sharpes[key] = []
            pair_sharpes[key].append(p["pair_sharpe"])

    avg_hold = (len(all_pnl) / max(total_trades, 1))

    w = 58
    print(f"\n{'=' * w}")
    print(f"  DYNAMIC PAIRS SYSTEM -- WALK-FORWARD REPORT")
    print(f"{'=' * w}")
    print(f"\n  OUT-OF-SAMPLE AGGREGATE")
    print(f"  {'Annualized Return ($):':<30} ${ann_return:>10,.2f}")
    print(f"  {'Annualized Volatility ($):':<30} ${ann_vol:>10,.2f}")
    print(f"  {'Sharpe Ratio:':<30} {sharpe:>11.2f}")
    print(f"  {'Max Drawdown ($):':<30} ${max_dd:>10,.2f}")
    print(f"  {'Total Trades:':<30} {total_trades:>11}")
    print(f"  {'Avg Hold Period (days):':<30} {avg_hold:>11.1f}")

    print(f"\n  WALK-FORWARD STABILITY  ({len(fold_results)} folds)")
    print(f"  {'OOS Sharpe mean:':<30} {np.mean(oos_sharpes):>11.2f}")
    print(f"  {'OOS Sharpe std:':<30} {np.std(oos_sharpes):>11.2f}")
    print(f"  {'Positive folds:':<30} {positive_folds}/{len(fold_results)} "
          f"({100*positive_folds/len(fold_results):.0f}%)")
    print(f"  {'Best fold Sharpe:':<30} {max(oos_sharpes):>11.2f}")
    print(f"  {'Worst fold Sharpe:':<30} {min(oos_sharpes):>11.2f}")

    print(f"\n  TARGET vs ACTUAL")
    _flag = lambda val, lo, hi: "OK" if lo <= val <= hi else "FLAG"
    print(f"  OOS Sharpe [target 0.7-1.4]:   {np.mean(oos_sharpes):.2f}  {_flag(np.mean(oos_sharpes),0.7,1.4)}")
    print(f"  % Positive [target >55%]:       {100*positive_folds/len(fold_results):.0f}%  {_flag(positive_folds/len(fold_results),0.55,1.0)}")
    print(f"  Avg hold [target 5-30d]:        {avg_hold:.1f}d  {_flag(avg_hold,5,30)}")
    print(f"  Max drawdown [target >-20%]:    ${max_dd:,.2f}  {'OK' if max_dd > -20 else 'FLAG'}")

    print(f"\n  TOP 10 MOST SELECTED PAIRS")
    print(f"  {'Pair':<18} {'Folds':>6}  {'Avg Sharpe':>11}")
    print(f"  {'-' * 38}")
    for pair, count in pair_counts.most_common(10):
        avg_sh = np.mean(pair_sharpes[pair]) if pair in pair_sharpes else 0.0
        print(f"  {pair:<18} {count:>6}  {avg_sh:>11.2f}")

    print(f"\n{'=' * w}\n")


def plot_report(fold_results: list, all_pnl: pd.Series) -> None:
    """4-panel performance chart."""
    oos_sharpes = [_sharpe(f["pnl"]) for f in fold_results]
    cum_pnl     = all_pnl.cumsum()
    drawdown    = cum_pnl - cum_pnl.cummax()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # TL: Cumulative P&L
    ax = axes[0, 0]
    ax.plot(cum_pnl.index, cum_pnl.values, color="steelblue", linewidth=1.2)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_title("Cumulative P&L (OOS, all pairs)")
    ax.set_ylabel("Cumulative $ P&L")
    ax.set_xlabel("Date")

    # TR: Per-fold OOS Sharpe
    ax = axes[0, 1]
    colors = ["green" if s > 0 else "red" for s in oos_sharpes]
    ax.bar(range(len(oos_sharpes)), oos_sharpes, color=colors, alpha=0.7)
    ax.axhline(0,   color="black", linewidth=0.6)
    ax.axhline(0.7, color="green", linewidth=0.8, linestyle="--", label="Target 0.7")
    ax.set_title("OOS Sharpe per Fold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Fold #")
    ax.legend(fontsize=8)

    # BL: Drawdown
    ax = axes[1, 0]
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="red", alpha=0.4)
    ax.set_title("Drawdown from Peak ($)")
    ax.set_ylabel("Drawdown ($)")
    ax.set_xlabel("Date")

    # BR: Daily P&L distribution
    ax = axes[1, 1]
    pnl_clean = all_pnl.replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(pnl_clean, bins=80, color="steelblue", alpha=0.6, density=True)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(pnl_clean.mean(), color="green", linewidth=1.0,
               linestyle="--", label=f"Mean ${pnl_clean.mean():.2f}")
    ax.set_title("Daily P&L Distribution")
    ax.set_xlabel("Daily P&L ($)")
    ax.legend(fontsize=8)

    plt.suptitle("Dynamic Pairs System -- Walk-Forward Performance", fontsize=12)
    plt.tight_layout()
    return fig
