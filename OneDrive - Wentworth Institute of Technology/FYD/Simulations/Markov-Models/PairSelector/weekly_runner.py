"""
weekly_runner.py
────────────────
Run every Monday morning: python weekly_runner.py
Outputs: weekly_recommendation.png + plain-English trade printed to terminal.

Pipeline:
  1. Download 2+ years of market data through today
  2. Compute regime series (HMM from regime_hedger — NOT hhm.py which has top-level code)
  3. Compute Kalman betas for all same-sector pairs
  4. Load trained LSTM and score all pairs
  5. Print trade recommendation + generate figure

$100 sizing: $50 per leg, dollar-neutral, fractional shares.
Exit: |z| < 0.5  OR  5 trading days  OR  |z| > 3.5 (stop-loss).
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
from datetime import date, timedelta

warnings.filterwarnings("ignore")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, '..'))

from cointegration_screener import screen_pairs, SECTOR_MAP
from bs_vol_filter          import realized_vol, iv_rank
from lstm_predictor         import load_model, score_all_pairs
from lstm_dataset           import compute_regime_series
from dynamic_hedger         import run_kalman

BUDGET        = 100.00
MIN_Z_ENTRY   = 1.8       # enter when |z| >= 1.8 (slightly below 2.0)
MIN_P_REV     = 0.45      # P(reversion) threshold
MIN_P_EMER    = 0.55      # P(emergence) threshold to flag on watch list
UNIVERSE      = list(SECTOR_MAP.keys())
TRAIN_DAYS    = 504
END_DATE      = date.today().strftime('%Y-%m-%d')
START_DATE    = (date.today() - timedelta(days=TRAIN_DAYS + 90)).strftime('%Y-%m-%d')


def _build_kalman_betas(log_prices: pd.DataFrame) -> dict:
    """
    Run the Kalman filter on each same-sector pair and collect time-varying betas.
    Uses regime_hedger's Kalman (via dynamic_hedger.run_kalman) on log prices.

    Returns dict {(t1,t2): pd.Series of beta values}.
    """
    kalman_betas = {}
    universe     = log_prices.columns.tolist()

    for t1, t2 in combinations(universe, 2):
        if SECTOR_MAP.get(t1) != SECTOR_MAP.get(t2):
            continue
        try:
            kf = run_kalman(log_prices[t1], log_prices[t2])
            kalman_betas[(t1, t2)] = kf['beta'].ffill()
        except Exception:
            kalman_betas[(t1, t2)] = pd.Series(1.0, index=log_prices.index)

    return kalman_betas


def _get_trade_sizing(t1: str, t2: str, beta: float,
                       prices: pd.DataFrame, budget: float = 100.0) -> tuple:
    """
    Size $budget across two legs, dollar-neutral.
    Returns (shares_t1, shares_t2, cost_t1, cost_t2).
    Fractional shares required for small budgets.
    """
    p1     = float(prices[t1].iloc[-1])
    p2     = float(prices[t2].iloc[-1])
    half   = budget / 2.0
    sh_t1  = round(half / p1, 4)
    sh_t2  = round((half * abs(beta)) / p2, 4)
    return sh_t1, sh_t2, round(half, 2), round(half, 2)


def run_weekly():
    print(f"\n{'='*60}")
    print(f"  WEEKLY PAIRS RECOMMENDATION  --  {END_DATE}")
    print(f"{'='*60}\n")

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    import yfinance as yf
    print("Fetching market data...")
    raw = yf.download(UNIVERSE, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)['Close'].ffill().dropna()
    prices     = raw.copy()
    log_prices = np.log(prices)

    # ── Step 2: Regime series via regime_hedger (NOT hhm.py) ─────────────────
    print("Computing HMM regime series...")
    regime = compute_regime_series(log_prices)
    bull_pct = (regime == 0).mean()
    print(f"  Current regime: {'RISK-ON (bull)' if regime.iloc[-1] == 0 else 'RISK-OFF (bear)'}"
          f"  (bull {bull_pct:.0%} of history)")

    # ── Step 3: Kalman betas ──────────────────────────────────────────────────
    print("Computing Kalman hedge ratios for all same-sector pairs...")
    kalman_betas = _build_kalman_betas(log_prices)
    print(f"  Computed betas for {len(kalman_betas)} pairs")

    # ── Step 4: Load model and score ──────────────────────────────────────────
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run: python lstm_trainer.py")
        return

    print("Scoring all pairs with LSTM...")
    scores = score_all_pairs(log_prices, kalman_betas, regime, model)

    if scores.empty:
        print("No pairs scored (check screener / model).")
        return

    # ── Step 5: Find best trade ───────────────────────────────────────────────
    tradeable = scores[
        scores['is_cointegrated'] &
        (scores['z_score_now'].abs() >= MIN_Z_ENTRY) &
        (scores['p_reversion'] >= MIN_P_REV)
    ]

    emerging = scores[
        ~scores['is_cointegrated'] &
        (scores['p_emergence'] >= MIN_P_EMER)
    ].head(3)

    # ── Step 6: Print recommendation ─────────────────────────────────────────
    print(f"\n{'='*60}")
    if tradeable.empty:
        print("  NO TRADE THIS WEEK")
        print("  No pairs meet all entry criteria right now.")
        print(f"  (Need: cointegrated + |z| >= {MIN_Z_ENTRY} + "
              f"P(reversion) >= {MIN_P_REV:.0%})")
    else:
        best      = tradeable.iloc[0]
        t1, t2    = best['t1'], best['t2']
        z         = best['z_score_now']
        beta_val  = float(kalman_betas.get((t1, t2),
                          pd.Series(best['ols_beta'])).iloc[-1])

        sh1, sh2, c1, c2 = _get_trade_sizing(t1, t2, beta_val, prices, BUDGET)

        long_leg  = t1 if z < 0 else t2
        short_leg = t2 if z < 0 else t1
        long_sh   = sh1 if z < 0 else sh2
        short_sh  = sh2 if z < 0 else sh1

        print(f"  TRADE THIS WEEK")
        print(f"  {'─'*50}")
        print(f"  Pair:         {t1} / {t2}  ({best['sector'].upper()})")
        print(f"  Signal:       z-score = {z:+.2f} std")
        print(f"  Action:       BUY  {long_sh:.4f} shares of {long_leg}  (${c1:.2f})")
        print(f"                SELL {short_sh:.4f} shares of {short_leg} (${c2:.2f})")
        print(f"  Total cost:   ${BUDGET:.2f}")
        print(f"  Exit when:    |z| < 0.5  OR  5 trading days  OR  |z| > 3.5")
        print(f"  P(reversion): {best['p_reversion']:.1%}")
        print(f"  Half-life:    {best['half_life']:.0f} days")
        print(f"  {'─'*50}")

    if not emerging.empty:
        print(f"\n  WATCH LIST  (not yet tradeable — watch for entry)")
        print(f"  {'─'*50}")
        for _, row in emerging.iterrows():
            print(f"  {row['pair']:<14}  P(cointegrates): {row['p_emergence']:.1%}"
                  f"  sector: {row['sector']}")
    print(f"{'='*60}\n")

    # ── Step 7: Generate figure ───────────────────────────────────────────────
    _generate_figure(scores, tradeable, emerging, prices, log_prices)


def _generate_figure(scores, tradeable, emerging, prices, log_prices):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Weekly Pairs Recommendation  --  {END_DATE}',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_rank   = fig.add_subplot(gs[0, :])   # full top row
    ax_spread = fig.add_subplot(gs[1, 0])
    ax_watch  = fig.add_subplot(gs[1, 1])

    # ── Panel 1: LSTM composite score bar chart (top 10) ─────────────────────
    top10  = scores.head(10)
    colors = ['#16a34a' if r else '#2563EB'
              for r in top10['is_cointegrated']]
    ax_rank.barh(top10['pair'][::-1], top10['composite_score'][::-1],
                 color=colors[::-1], alpha=0.85)
    ax_rank.axvline(0.5, color='red', linestyle='--', linewidth=1)
    ax_rank.set_xlabel('LSTM Composite Score  (0 = no signal,  1 = strong signal)')
    ax_rank.set_title('All Pairs Ranked by LSTM Score', fontweight='bold')
    ax_rank.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    ax_rank.legend(
        handles=[Patch(facecolor='#16a34a', label='Cointegrated (tradeable)'),
                 Patch(facecolor='#2563EB', label='Emerging (watch list)')],
        fontsize=9, loc='lower right',
    )

    # ── Panel 2: spread z-score for best trade ────────────────────────────────
    if not tradeable.empty:
        best = tradeable.iloc[0]
        t1, t2 = best['t1'], best['t2']
        alpha  = best.get('ols_alpha', 0.0) if 'ols_alpha' in best else 0.0
        beta   = best['ols_beta']
        spread = log_prices[t1] - beta * log_prices[t2] - alpha
        z      = (spread - spread.mean()) / (spread.std() + 1e-8)
        z_plot = z.dropna().iloc[-63:]

        ax_spread.plot(z_plot.index, z_plot.values, color='gray', linewidth=1)
        ax_spread.axhline( 2.0, color='red',   linestyle='--', linewidth=1)
        ax_spread.axhline(-2.0, color='green', linestyle='--', linewidth=1)
        ax_spread.axhline( 0.0, color='black', linestyle='-',  linewidth=0.5, alpha=0.4)
        ax_spread.fill_between(z_plot.index, 2.0, z_plot,
                               where=z_plot > 2.0,  color='red',   alpha=0.2)
        ax_spread.fill_between(z_plot.index, -2.0, z_plot,
                               where=z_plot < -2.0, color='green', alpha=0.2)

        current_z = float(z_plot.iloc[-1])
        direction = (f"BUY {t1} / SELL {t2}" if current_z < 0
                     else f"SELL {t1} / BUY {t2}")
        ax_spread.set_title(f'THIS WEEK: {t1}/{t2}\n{direction}  (z={current_z:+.2f})',
                            fontweight='bold',
                            color='green' if current_z < 0 else 'red')
        ax_spread.set_ylabel('Spread z-score')
        ax_spread.set_ylim(-4, 4)
        ax_spread.grid(True, alpha=0.3)
        ax_spread.tick_params(axis='x', rotation=30)
    else:
        ax_spread.text(0.5, 0.5, 'NO TRADE\nTHIS WEEK',
                       ha='center', va='center', fontsize=16, color='gray',
                       transform=ax_spread.transAxes)
        ax_spread.set_title("This Week's Trade", fontweight='bold')
        ax_spread.axis('off')

    # ── Panel 3: watch list ───────────────────────────────────────────────────
    if not emerging.empty:
        ax_watch.barh(emerging['pair'].tolist()[::-1],
                      emerging['p_emergence'].tolist()[::-1],
                      color='#7c3aed', alpha=0.75)
        ax_watch.axvline(MIN_P_EMER, color='gray', linestyle='--', linewidth=1,
                         label=f'Threshold ({MIN_P_EMER:.0%})')
        ax_watch.set_xlim(0, 1)
        ax_watch.set_xlabel('P(becomes tradeable within 3 months)')
        ax_watch.set_title('Emerging Pairs -- Watch List', fontweight='bold')
        ax_watch.legend(fontsize=9)
        ax_watch.grid(True, axis='x', alpha=0.3)
    else:
        ax_watch.text(0.5, 0.5, 'No emerging\npairs flagged',
                      ha='center', va='center', fontsize=13, color='gray',
                      transform=ax_watch.transAxes)
        ax_watch.set_title('Emerging Pairs -- Watch List', fontweight='bold')
        ax_watch.axis('off')

    out = os.path.join(_here, 'weekly_recommendation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out}")


if __name__ == '__main__':
    run_weekly()
