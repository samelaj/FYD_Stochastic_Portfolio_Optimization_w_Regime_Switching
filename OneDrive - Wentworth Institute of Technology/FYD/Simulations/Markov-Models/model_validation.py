"""
model_validation.py
───────────────────
Four robustness tests for the KO/PEP predictive model.

Test 1 — Ablation
  Remove z_score from features. If AUC collapses to ~0.50, the entire system
  is just a dressed-up z-score mean-reversion rule, not a multi-feature model.
  Also tests z_score alone to quantify its standalone contribution.

Test 2 — Cross-pair generalisation
  Run the identical pipeline on XOM/CVX, JPM/BAC, MSFT/GOOGL.
  If every pair gets AUC > 0.70, the model is a generic autocorrelation
  filter — not pair-specific — which changes how results should be interpreted.

Test 3 — Transaction costs
  Deduct 0.05% per leg (0.10% round-trip) every time the signal flips.
  Shows what Sharpe looks like in a realistic live-trading environment.

Test 4 — Block bootstrap AUC confidence interval
  Sample 200-day blocks with replacement (500 resamples) to build a 95% CI
  on AUC that accounts for temporal autocorrelation in the test observations.
  A standard CI assumes i.i.d. rows — that is wrong for a 30-day rolling z-score.

Run:
  python model_validation.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yfinance as yf

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# ── Sibling imports ───────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from walk_forward import (
    build_features, build_sequences_flat, train_logistic,
    simulate_pnl, load_vix, build_walk_forward_p_bear,
    FEATURE_COLS, TRAIN_FRAC, HORIZON, START_DATE, END_DATE, ENTRY_PROB,
)
from kalman_hedger import KalmanPairsFilter, load_daily_prices


# ── Shared pipeline ───────────────────────────────────────────────────────────

def run_pair_pipeline(ticker1, ticker2, vix, label=None):
    """
    Full walk-forward pipeline for any ticker pair.
    Returns dict with metrics, probabilities, and data for further analysis.
    """
    name = label or f"{ticker1}/{ticker2}"
    print(f"  [{name}] Loading prices...")

    p1 = load_daily_prices(ticker1)
    p2 = load_daily_prices(ticker2)
    shared = p1.index.intersection(p2.index)
    p1 = p1.loc[shared]
    p2 = p2.loc[shared]

    train_cutoff = shared[int(len(shared) * TRAIN_FRAC)]

    print(f"  [{name}] Fitting walk-forward HMMs (cutoff {train_cutoff.date()})...")
    pb1 = build_walk_forward_p_bear(ticker1, shared, train_cutoff)
    pb2 = build_walk_forward_p_bear(ticker2, shared, train_cutoff)
    p_bear = ((pb1 + pb2) / 2).fillna(0.5)

    kf    = KalmanPairsFilter(delta_bull=1e-4, delta_bear=5e-4, R=0.01)
    kf_df = kf.run(p1, p2, p_bear)

    df = build_features(p1, p2, kf_df, p_bear, vix)

    split     = int(len(df) * TRAIN_FRAC)
    df_train  = df.iloc[:split]
    df_test   = df.iloc[split:]

    X_tr = df_train[FEATURE_COLS].values
    y_tr = df_train["target"].values
    X_te = df_test[FEATURE_COLS].values
    y_te = df_test["target"].values

    model, scaler, prob, metrics = train_logistic(X_tr, y_tr, X_te, y_te)

    pnl, cum_pnl, sharpe = simulate_pnl(
        pd.Series(prob, index=df_test.index), df_test["spread"]
    )

    return {
        "name":     name,
        "ticker1":  ticker1,
        "ticker2":  ticker2,
        "metrics":  metrics,
        "auc":      metrics["auc"],
        "acc":      metrics["acc"],
        "sharpe":   sharpe,
        "prob":     prob,
        "y_test":   y_te,
        "df_test":  df_test,
        "df_train": df_train,
        "model":    model,
        "scaler":   scaler,
        "prices1":  p1,
        "prices2":  p2,
        "kf_df":    kf_df,
    }


# ── Test 1: Ablation ──────────────────────────────────────────────────────────

def test1_ablation(df_train, df_test):
    """
    Train logistic regression under three feature sets:
      A) All features
      B) All features EXCEPT z_score
      C) z_score ONLY
    Compare AUC to isolate z_score's contribution.
    """
    results = {}
    sets = {
        "All features":          FEATURE_COLS,
        "Without z_score":       [f for f in FEATURE_COLS if f != "z_score"],
        "z_score only":          ["z_score"],
    }
    for label, cols in sets.items():
        X_tr = df_train[cols].values
        y_tr = df_train["target"].values
        X_te = df_test[cols].values
        y_te = df_test["target"].values

        scaler = StandardScaler()
        Xtr_s  = scaler.fit_transform(X_tr)
        Xte_s  = scaler.transform(X_te)

        m = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        m.fit(Xtr_s, y_tr)
        prob = m.predict_proba(Xte_s)[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(y_te, prob)
        acc = accuracy_score(y_te, pred)

        pnl, cum_pnl, sharpe = simulate_pnl(
            pd.Series(prob, index=df_test.index), df_test["spread"]
        )
        results[label] = {
            "auc": auc, "acc": acc, "sharpe": sharpe,
            "prob": prob, "y_test": y_te,
            "pnl": pnl, "cum_pnl": cum_pnl,
        }
        print(f"    {label:<28}  AUC={auc:.3f}  Acc={acc:.1%}  Sharpe={sharpe:.2f}")

    return results


# ── Test 2: Cross-pair ────────────────────────────────────────────────────────

PAIRS = [
    ("KO",   "PEP",  "KO/PEP  (original)"),
    ("XOM",  "CVX",  "XOM/CVX (energy)"),
    ("JPM",  "BAC",  "JPM/BAC (banks)"),
    ("MSFT", "GOOGL","MSFT/GOOGL (tech)"),
]


def test2_cross_pairs(vix):
    """Run the full pipeline on each pair and collect results."""
    results = {}
    for t1, t2, label in PAIRS:
        try:
            r = run_pair_pipeline(t1, t2, vix, label)
            results[label] = r
            print(f"    {label:<28}  AUC={r['auc']:.3f}  Sharpe={r['sharpe']:.2f}")
        except Exception as e:
            print(f"    {label:<28}  FAILED: {e}")
    return results


# ── Test 3: Transaction costs ─────────────────────────────────────────────────

def simulate_pnl_with_costs(prob_series, df_test, prices1, prices2, kf_df,
                              threshold=ENTRY_PROB, cost_per_leg=0.0005):
    """
    P&L net of transaction costs.
    Charges 2 × cost_per_leg × (P_t1 + |beta_t| × P_t2) on every signal flip.

    cost_per_leg=0.0005 → 0.05% per leg → 0.10% round-trip.
    This approximates bid-ask spread + market impact for liquid large-caps.
    """
    idx = prob_series.index
    sig = np.where(prob_series > threshold,          1,
          np.where(prob_series < (1 - threshold),   -1, 0))
    sig_s = pd.Series(sig, index=idx).shift(1).fillna(0)

    spread   = df_test["spread"].reindex(idx)
    sp_diff  = spread.diff()
    gross_pnl = (sig_s * sp_diff).fillna(0)

    # Align prices and beta to signal index
    p1    = prices1.reindex(idx).ffill()
    p2    = prices2.reindex(idx).ffill()
    beta  = kf_df["beta"].reindex(idx).ffill().abs()

    # Notional = both legs combined
    notional = p1 + beta * p2

    # Cost fires when signal changes
    sig_diff = sig_s.diff().abs()
    cost     = sig_diff * notional * cost_per_leg * 2

    net_pnl  = gross_pnl - cost.fillna(0)
    n_trades = int((sig_s.diff().abs() > 0).sum())

    return {
        "gross_pnl":  gross_pnl,
        "net_pnl":    net_pnl,
        "cum_gross":  gross_pnl.cumsum(),
        "cum_net":    net_pnl.cumsum(),
        "sharpe_gross": (gross_pnl.mean() / gross_pnl.std() * np.sqrt(252))
                         if gross_pnl.std() > 0 else 0,
        "sharpe_net":   (net_pnl.mean()   / net_pnl.std()   * np.sqrt(252))
                         if net_pnl.std()  > 0 else 0,
        "n_trades":   n_trades,
        "cost_total": float(cost.sum()),
        "avg_hold_days": len(idx) / max(n_trades, 1),
    }


def test3_costs(lr_prob, df_test, prices_ko, prices_pep, kf_df):
    """Run cost analysis for LR model at several cost assumptions."""
    results = {}
    scenarios = {
        "No cost (0%)":        0.0,
        "0.05% per leg":       0.0005,
        "0.10% per leg":       0.001,
        "0.20% per leg":       0.002,
    }
    prob_s = pd.Series(lr_prob, index=df_test.index)
    for label, cost in scenarios.items():
        r = simulate_pnl_with_costs(prob_s, df_test, prices_ko, prices_pep,
                                     kf_df, cost_per_leg=cost)
        results[label] = r
        print(f"    {label:<22}  Sharpe={r['sharpe_net']:>6.2f}  "
              f"Trades={r['n_trades']:>4}  "
              f"Avg hold={r['avg_hold_days']:.1f}d  "
              f"Total cost=${r['cost_total']:.2f}")
    return results


# ── Test 4: Block bootstrap ───────────────────────────────────────────────────

def block_bootstrap_auc(y_true, y_prob, block_size=200, n_bootstrap=500, seed=42):
    """
    Stationary block bootstrap for AUC confidence interval.

    Why blocks, not i.i.d. resampling:
    The z_score feature is computed from a 30-day rolling window, so consecutive
    observations are highly autocorrelated. i.i.d. bootstrap underestimates variance.
    Block bootstrap preserves the temporal autocorrelation structure.

    Returns bootstrap distribution of AUC values.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    auc_samples = []

    for _ in range(n_bootstrap):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, max(1, n - block_size)))
            indices.extend(range(start, min(start + block_size, n)))
        idx_b = np.array(indices[:n])
        y_b   = y_true[idx_b]
        p_b   = y_prob[idx_b]

        # Skip degenerate bootstrap samples (all one class)
        if 0 < y_b.sum() < len(y_b):
            auc_samples.append(roc_auc_score(y_b, p_b))

    return np.array(auc_samples)


def test4_bootstrap(y_test, lr_prob, block_size=200, n_bootstrap=500):
    """Bootstrap AUC for LR model and naive block-shuffle baseline."""
    print(f"    Running {n_bootstrap} bootstrap resamples (block_size={block_size})...")

    auc_lr     = block_bootstrap_auc(y_test, lr_prob, block_size, n_bootstrap)
    # Null distribution: shuffle probabilities within each block → destroys signal
    rng        = np.random.default_rng(99)
    prob_null  = lr_prob.copy()
    rng.shuffle(prob_null)
    auc_null   = block_bootstrap_auc(y_test, prob_null, block_size, n_bootstrap)

    ci_lo, ci_hi = np.percentile(auc_lr, [2.5, 97.5])
    observed     = roc_auc_score(y_test, lr_prob)
    p_value      = float((auc_null >= observed).mean())

    print(f"    Observed AUC     : {observed:.3f}")
    print(f"    95% CI (block)   : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    Bootstrap mean   : {auc_lr.mean():.3f}  ±{auc_lr.std():.3f}")
    print(f"    Null mean        : {auc_null.mean():.3f}")
    print(f"    p-value vs null  : {p_value:.3f}  "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} at 5%)")

    return {
        "auc_dist":  auc_lr,
        "null_dist": auc_null,
        "observed":  observed,
        "ci_lo":     ci_lo,
        "ci_hi":     ci_hi,
        "p_value":   p_value,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_test1(ablation_results):
    """Bar chart: AUC and Sharpe for each feature set."""
    labels  = list(ablation_results.keys())
    aucs    = [ablation_results[l]["auc"]    for l in labels]
    sharpes = [ablation_results[l]["sharpe"] for l in labels]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    bars = ax1.bar(x, aucs, color=["steelblue", "salmon", "lightgreen"], width=0.5)
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Random (0.50)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("AUC"); ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Test 1 — AUC by Feature Set")
    ax1.legend(fontsize=8)
    for bar, v in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, sharpes, color=["steelblue", "salmon", "lightgreen"], width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Sharpe Ratio"); ax2.axhline(0, color="grey", linewidth=0.6)
    ax2.set_title("Test 1 — Sharpe by Feature Set")
    for bar, v in zip(bars2, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.suptitle("Ablation Study — Contribution of z_score Feature", fontsize=12)
    plt.tight_layout()
    return fig


def plot_test2(cross_results):
    """AUC and Sharpe bar chart for each pair."""
    labels  = list(cross_results.keys())
    aucs    = [cross_results[l]["auc"]    for l in labels]
    sharpes = [cross_results[l]["sharpe"] for l in labels]
    x = np.arange(len(labels))

    colors = ["steelblue", "darkorange", "green", "purple"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    bars = ax1.bar(x, aucs, color=colors[:len(labels)], width=0.5)
    ax1.axhline(0.5,  color="grey",  linestyle="--", linewidth=0.8, label="Random")
    ax1.axhline(0.58, color="black", linestyle=":",  linewidth=0.8, label="Typical edge (0.58)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8, rotation=10)
    ax1.set_ylabel("AUC"); ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Test 2 — AUC by Pair")
    ax1.legend(fontsize=8)
    for bar, v in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, sharpes, color=colors[:len(labels)], width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8, rotation=10)
    ax2.set_ylabel("Sharpe Ratio"); ax2.axhline(0, color="grey", linewidth=0.6)
    ax2.set_title("Test 2 — Sharpe by Pair")
    for bar, v in zip(bars2, sharpes):
        ypos = v + 0.05 if v >= 0 else v - 0.2
        ax2.text(bar.get_x() + bar.get_width()/2, ypos, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.suptitle("Cross-Pair Generalisation — Same Pipeline on 4 Pairs", fontsize=12)
    plt.tight_layout()
    return fig


def plot_test3(cost_results):
    """Cumulative P&L and Sharpe under each cost scenario."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    palette = ["steelblue", "darkorange", "red", "darkred"]
    labels  = list(cost_results.keys())

    for label, color in zip(labels, palette):
        r = cost_results[label]
        ax1.plot(r["cum_net"].index, r["cum_net"].values,
                 label=f"{label}  (Sharpe={r['sharpe_net']:.2f})",
                 color=color, linewidth=1.2)

    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Cumulative Net P&L ($)")
    ax1.set_xlabel("Date")
    ax1.set_title("Cumulative P&L Under Transaction Cost Scenarios")
    ax1.legend(fontsize=8)

    sharpes = [cost_results[l]["sharpe_net"] for l in labels]
    ax2.bar(range(len(labels)), sharpes, color=palette)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=10)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.axhline(1, color="grey",  linestyle="--", linewidth=0.8, label="Sharpe=1 benchmark")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Sharpe vs Transaction Cost Assumption")
    ax2.legend(fontsize=8)
    for i, v in enumerate(sharpes):
        ax2.text(i, max(v, 0) + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    plt.suptitle("Test 3 — Realistic Transaction Costs", fontsize=12)
    plt.tight_layout()
    return fig


def plot_test4(bootstrap_result):
    """Bootstrap AUC distribution vs null, with CI and p-value."""
    fig, ax = plt.subplots(figsize=(9, 5))
    r = bootstrap_result

    ax.hist(r["null_dist"], bins=40, alpha=0.5, color="grey",
            density=True, label="Null distribution (shuffled probs)")
    ax.hist(r["auc_dist"],  bins=40, alpha=0.6, color="steelblue",
            density=True, label="Bootstrap AUC distribution")

    ax.axvline(r["observed"], color="red",   linewidth=2.0,
               label=f"Observed AUC = {r['observed']:.3f}")
    ax.axvline(r["ci_lo"],    color="steelblue", linewidth=1.2, linestyle="--",
               label=f"95% CI  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    ax.axvline(r["ci_hi"],    color="steelblue", linewidth=1.2, linestyle="--")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":",
               label="Random (0.50)")

    ax.set_xlabel("AUC")
    ax.set_ylabel("Density")
    sig_str = "SIGNIFICANT" if r["p_value"] < 0.05 else "NOT significant"
    ax.set_title(
        f"Test 4 — Block Bootstrap AUC  (block=200d, n=500)\n"
        f"p-value vs null = {r['p_value']:.3f}  →  {sig_str} at 5%"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── Console summary ───────────────────────────────────────────────────────────

def print_validation_summary(abl, cross, cost, boot):
    w = 68
    print(f"\n{'=' * w}")
    print(f"  MODEL VALIDATION SUMMARY")
    print(f"{'=' * w}")

    # Test 1
    print(f"\n  TEST 1 — Ablation (z_score contribution)")
    print(f"  {'Feature set':<28} {'AUC':>6}  {'Sharpe':>7}")
    print(f"  {'-' * 44}")
    for label, r in abl.items():
        flag = " << BASELINE" if "All" in label else ""
        print(f"  {label:<28} {r['auc']:>6.3f}  {r['sharpe']:>7.2f}{flag}")
    auc_drop = abl["All features"]["auc"] - abl["Without z_score"]["auc"]
    print(f"\n  AUC drop without z_score: {auc_drop:.3f}")
    if auc_drop > 0.15:
        print(f"  >> z_score accounts for most of the signal. Other features add little.")
    elif auc_drop > 0.05:
        print(f"  >> z_score is important but other features add meaningful signal.")
    else:
        print(f"  >> Other features carry the signal. z_score is not the primary driver.")

    # Test 2
    print(f"\n  TEST 2 - Cross-pair (same pipeline, 4 pairs)")
    print(f"  {'Pair':<28} {'AUC':>6}  {'Sharpe':>7}")
    print(f"  {'-' * 44}")
    for label, r in cross.items():
        print(f"  {label:<28} {r['auc']:>6.3f}  {r['sharpe']:>7.2f}")
    aucs = [r["auc"] for r in cross.values()]
    if all(a > 0.65 for a in aucs):
        print(f"\n  >> All pairs AUC > 0.65. Model is a GENERIC mean-reversion filter.")
        print(f"     Not pair-specific. Results should be interpreted as such.")
    elif aucs[0] > max(aucs[1:]) + 0.05:
        print(f"\n  >> KO/PEP materially outperforms other pairs.")
        print(f"     Model is capturing pair-specific cointegration.")
    else:
        print(f"\n  >> Mixed results. Investigate per-pair regime dynamics.")

    # Test 3
    print(f"\n  TEST 3 - Transaction costs")
    print(f"  {'Scenario':<22} {'Sharpe':>7}  {'Trades':>7}  {'Avg hold':>9}  {'Total cost':>11}")
    print(f"  {'-' * 60}")
    for label, r in cost.items():
        print(f"  {label:<22} {r['sharpe_net']:>7.2f}  {r['n_trades']:>7}  "
              f"{r['avg_hold_days']:>7.1f}d  ${r['cost_total']:>10.2f}")
    realistic = list(cost.items())[1][1]
    if realistic["sharpe_net"] > 1.0:
        print(f"\n  >> Strategy survives realistic costs. Sharpe > 1.0 at 0.05% per leg.")
    else:
        print(f"\n  >> Strategy does NOT survive realistic costs. Not tradeable as-is.")

    # Test 4
    r = boot
    print(f"\n  TEST 4 - Block bootstrap AUC (200-day blocks, 500 resamples)")
    print(f"  Observed AUC  : {r['observed']:.3f}")
    print(f"  95% CI        : [{r['ci_lo']:.3f},  {r['ci_hi']:.3f}]")
    print(f"  p-value       : {r['p_value']:.3f}  "
          f"({'SIGNIFICANT' if r['p_value'] < 0.05 else 'NOT significant'})")
    if r["ci_lo"] > 0.5:
        print(f"  >> The LOWER bound of the CI is above 0.50.")
        print(f"     Signal is statistically robust to autocorrelation in features.")
    else:
        print(f"  >> CI lower bound crosses 0.50. Signal may not be statistically")
        print(f"     distinguishable from noise after correcting for autocorrelation.")

    print(f"\n{'=' * w}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  Model Validation Suite - KO / PEP")
    print("=" * 60)

    # ── Base pipeline (KO/PEP) ────────────────────────────────────────────────
    print("\n[0/4] Running base KO/PEP pipeline...")
    vix    = load_vix()
    base   = run_pair_pipeline("KO", "PEP", vix, "KO/PEP (original)")
    df_train   = base["df_train"]
    df_test    = base["df_test"]
    lr_prob    = base["prob"]
    y_test     = base["y_test"]
    prices_ko  = base["prices1"]
    prices_pep = base["prices2"]
    kf_df      = base["kf_df"]
    print(f"  Base AUC={base['auc']:.3f}  Sharpe={base['sharpe']:.2f}")

    # ── Test 1 ────────────────────────────────────────────────────────────────
    print("\n[1/4] Ablation study...")
    abl = test1_ablation(df_train, df_test)

    # ── Test 2 ────────────────────────────────────────────────────────────────
    print("\n[2/4] Cross-pair generalisation (this takes ~3 min)...")
    cross = test2_cross_pairs(vix)

    # ── Test 3 ────────────────────────────────────────────────────────────────
    print("\n[3/4] Transaction cost analysis...")
    cost = test3_costs(lr_prob, df_test, prices_ko, prices_pep, kf_df)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    print("\n[4/4] Block bootstrap AUC confidence interval...")
    boot = test4_bootstrap(y_test, lr_prob, block_size=200, n_bootstrap=500)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_validation_summary(abl, cross, cost, boot)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures...")
    plot_test1(abl)
    plot_test2(cross)
    plot_test3(cost)
    plot_test4(boot)
    plt.show()
