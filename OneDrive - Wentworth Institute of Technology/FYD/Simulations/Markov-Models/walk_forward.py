"""
walk_forward.py
───────────────
Walk-forward predictive modeling for the KO / PEP pairs spread.

Feature generation (no look-ahead):
  • Kalman Filter runs online — only uses data up to time t
  • HMM forward algorithm only (not smoothed Baum-Welch gamma) — P(bear | Y_1:t)
  • 10 features: z-score, beta_t, beta_std, p_bear, spread_vol,
                 ko/pep 5-day returns, log-VIX, KO-PEP correlation, spread momentum

Target:
  y_t = 1  if spread_{t+5} > spread_t  (spread rises → long KO / short PEP wins)
  y_t = 0  otherwise

Models (time-ordered 70 / 30 split — NO shuffling):
  1. Logistic Regression      — fast baseline, interpretable coefficients
  2. MLP Neural Network       — non-linear patterns, sklearn (no TensorFlow needed)
     Input: last 10 days of all 10 features flattened → 100-dim vector
     Architecture: 100 → 64 → 32 → 1

Run:
  python walk_forward.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

warnings.filterwarnings("ignore")

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    precision_score, recall_score,
)

# ── Sibling imports ───────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from kalman_hedger import KalmanPairsFilter, load_daily_prices, build_daily_p_bear
from regime_hedger import fit_hmm_for_ticker, _emissions, _forward, baum_welch


# ── HMM walk-forward regime probabilities ─────────────────────────────────────

def hmm_filtered_p_bear(obs, pi, A, mus, sigmas, bull_state):
    """
    Forward algorithm only — P(Z_t = bear | Y_1:t).
    Uses only data up to t. No look-ahead.
    """
    B          = _emissions(obs, mus, sigmas)
    alpha, _   = _forward(B, pi, A)
    return alpha[:, 1 - bull_state]             # shape (T,)


def build_walk_forward_p_bear(ticker, daily_index, train_cutoff_date):
    """
    Fit HMM on weekly training data only, then run the forward-only pass
    on the full weekly series (train + test) with training-fitted parameters.

    This ensures:
      • HMM parameters (mu, sigma, A) see NO test-period data
      • P(bear) at each date still uses only data up to that date (forward pass)

    Returns daily P(bear) series aligned to daily_index.
    """
    # Download full weekly series
    raw   = yf.download(ticker, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False)
    close = raw["Close"].resample("W").last().dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    pct   = close.pct_change().dropna()

    # Split at training cutoff
    train_pct = pct.loc[:train_cutoff_date]
    full_pct  = pct                             # forward pass runs on all weeks

    if len(train_pct) < 52:
        raise ValueError(f"Only {len(train_pct)} training weeks — need at least 52.")

    # Fit Baum-Welch on training data only
    pi_fit, A_fit, mus_fit, sigmas_fit, _ = baum_welch(train_pct.values)
    bull = int(np.argmax(mus_fit))

    # Run forward pass on full series using training-fitted parameters
    p_bear_weekly = hmm_filtered_p_bear(
        full_pct.values, pi_fit, A_fit, mus_fit, sigmas_fit, bull
    )
    p_bear_series = pd.Series(p_bear_weekly, index=full_pct.index)

    # Forward-fill weekly → daily
    all_idx = p_bear_series.index.union(daily_index).sort_values()
    return (
        p_bear_series
        .reindex(all_idx).ffill().bfill()
        .reindex(daily_index).fillna(0.5)
    )


# ── Data helpers ──────────────────────────────────────────────────────────────

START_DATE  = "2015-01-01"
END_DATE    = pd.Timestamp.today().strftime("%Y-%m-%d")
HORIZON     = 5     # days ahead for target
SEQ_LEN     = 20    # LSTM lookback window (trading days)
TRAIN_FRAC  = 0.70  # fraction of data used for training
ENTRY_PROB  = 0.55  # signal threshold for trading simulation


def load_vix(start=START_DATE, end=END_DATE):
    raw   = yf.download("^VIX", start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


# ── Feature engineering ───────────────────────────────────────────────────────

FEATURE_COLS = [
    "z_score",      # Kalman spread z-score
    "beta",         # Kalman hedge ratio
    "beta_std",     # Uncertainty in beta (higher in bear regimes)
    "p_bear",       # HMM filtered bear probability (forward-only)
    "spread_vol",   # 21-day rolling spread volatility
    "ko_ret_5d",    # KO 5-day return
    "pep_ret_5d",   # PEP 5-day return
    "log_vix",      # Log-VIX (macro fear)
    "ko_pep_corr",  # 21-day rolling KO-PEP return correlation
    "spread_mom",   # 5-day spread momentum
]


def build_features(prices_ko, prices_pep, kf_df, p_bear_daily, vix):
    """
    Construct feature matrix and target vector.
    Every feature at row t uses ONLY data available at or before time t.
    Target: 1 if spread rises HORIZON days ahead.
    """
    idx = kf_df.index

    # Align all inputs
    ko  = prices_ko.reindex(idx).ffill()
    pep = prices_pep.reindex(idx).ffill()
    pb  = p_bear_daily.reindex(idx).ffill().fillna(0.5)
    vx  = vix.reindex(idx).ffill().bfill().fillna(20.0)

    spread = kf_df["spread"]

    # ── z-score of Kalman spread ──────────────────────────────────────────────
    mu_sp  = spread.rolling(30, min_periods=10).mean()
    std_sp = spread.rolling(30, min_periods=10).std().replace(0, np.nan)
    z      = (spread - mu_sp) / std_sp

    # ── other features ────────────────────────────────────────────────────────
    sp_vol    = spread.rolling(21, min_periods=10).std()
    ko_ret5   = ko.pct_change(5)
    pep_ret5  = pep.pct_change(5)
    ko_d      = ko.pct_change()
    pep_d     = pep.pct_change()
    corr_21   = ko_d.rolling(21, min_periods=10).corr(pep_d)
    sp_mom    = spread.diff(5)

    df = pd.DataFrame({
        "z_score":    z,
        "beta":       kf_df["beta"],
        "beta_std":   kf_df["beta_std"],
        "p_bear":     pb,
        "spread_vol": sp_vol,
        "ko_ret_5d":  ko_ret5,
        "pep_ret_5d": pep_ret5,
        "log_vix":    np.log(vx.clip(lower=1e-6)),
        "ko_pep_corr": corr_21,
        "spread_mom": sp_mom,
        "spread":     spread,                       # kept for P&L, not a model feature
    }, index=idx)

    # Target: spread rises HORIZON days ahead
    df["target"] = (spread.shift(-HORIZON) > spread).astype(int)

    # Drop NaN rows and last HORIZON rows (no valid target)
    df = df.dropna().iloc[:-HORIZON]
    return df


# ── Sequence builder for MLP ──────────────────────────────────────────────────

def build_sequences_flat(X_arr, y_arr, seq_len=SEQ_LEN):
    """
    Build flattened look-back windows for the MLP.
    Each row = last seq_len days of all features concatenated → (seq_len * n_features,)
    Gives the MLP access to temporal patterns without requiring an LSTM.

    X_arr : shape (T, F)
    y_arr : shape (T,)
    Returns X_flat (T-L, L*F), y_flat (T-L,)
    """
    T, F  = X_arr.shape
    X_flat = np.lib.stride_tricks.sliding_window_view(
        X_arr, window_shape=seq_len, axis=0
    ).reshape(T - seq_len + 1, seq_len * F)
    y_flat = y_arr[seq_len - 1:]
    return X_flat, y_flat


# ── Model training ────────────────────────────────────────────────────────────

def train_logistic(X_train, y_train, X_test, y_test):
    """Logistic regression with standard scaling. Returns model, scaler, metrics."""
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xte    = scaler.transform(X_test)

    model  = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    model.fit(Xtr, y_train)

    prob   = model.predict_proba(Xte)[:, 1]
    pred   = (prob >= 0.5).astype(int)
    metrics = _eval_metrics("Logistic Regression", y_test, prob, pred)
    return model, scaler, prob, metrics


def train_mlp(X_train_flat, y_train_flat, X_test_flat, y_test_flat):
    """
    MLP neural network on flattened look-back windows.
    Input: (SEQ_LEN * n_features,) = 10 days × 10 features = 100-dim vector
    Architecture: 100 → 64 → 32 → 1 (sigmoid output via predict_proba)
    Uses sklearn MLPClassifier — no TensorFlow required.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(X_train_flat, y_train_flat)
    print(f"  MLP trained for {model.n_iter_} iterations")

    prob    = model.predict_proba(X_test_flat)[:, 1]
    pred    = (prob >= 0.5).astype(int)
    metrics = _eval_metrics("MLP Neural Net", y_test_flat, prob, pred)
    return model, prob, metrics


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _eval_metrics(name, y_true, y_prob, y_pred):
    auc  = roc_auc_score(y_true, y_prob)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    return {"name": name, "auc": auc, "acc": acc, "prec": prec, "rec": rec}


def simulate_pnl(prob_series, spread_series, threshold=ENTRY_PROB):
    """
    Convert model probability into a trading signal and compute daily P&L.
    prob_series  : P(spread rises) indexed by date
    spread_series: actual spread values, same index
    """
    sig = np.where(prob_series > threshold,          1,   # long spread
          np.where(prob_series < (1 - threshold),   -1,   # short spread
                                                     0))  # flat
    sig_s    = pd.Series(sig, index=prob_series.index).shift(1)
    sp_diff  = spread_series.diff()
    pnl      = (sig_s * sp_diff).fillna(0)
    sharpe   = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0
    return pnl, pnl.cumsum(), sharpe


def kalman_baseline_pnl(df_test):
    """Reproduce the simple z-score signal from kalman_hedger as baseline."""
    z      = df_test["z_score"]
    prob_b = ((-z + 2) / 4).clip(0, 1)   # map z in [-2, 2] → prob in [0, 1]
    return simulate_pnl(prob_b, df_test["spread"])


def print_results(metrics_list, pnl_results):
    print(f"\n{'=' * 68}")
    print(f"  WALK-FORWARD RESULTS  —  KO / PEP Spread Prediction")
    print(f"{'=' * 68}")
    print(f"  {'Model':<26} {'AUC':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'Sharpe':>8}")
    print(f"  {'-' * 62}")
    for m in metrics_list:
        sharpe = pnl_results[m["name"]]["sharpe"]
        print(f"  {m['name']:<26} {m['auc']:>6.3f} {m['acc']:>6.1%} "
              f"{m['prec']:>6.1%} {m['rec']:>6.1%} {sharpe:>8.2f}")
    print()


def predict_now(lr_model, lr_scaler, mlp_model,
                feature_row, seq_flat=None):
    """Current-date prediction using the latest feature snapshot."""
    x_lr = lr_scaler.transform(feature_row.reshape(1, -1))
    p_lr = float(lr_model.predict_proba(x_lr)[0, 1])

    p_mlp = None
    if mlp_model is not None and seq_flat is not None:
        p_mlp = float(mlp_model.predict_proba(seq_flat.reshape(1, -1))[0, 1])

    print(f"\n  {'─' * 50}")
    print(f"  CURRENT PREDICTION (next {HORIZON} trading days)")
    print(f"  {'─' * 50}")
    print(f"  Logistic Regression  P(spread rises) = {p_lr:.1%}")
    if p_mlp is not None:
        print(f"  MLP Neural Net       P(spread rises) = {p_mlp:.1%}")
    p_use = p_mlp if p_mlp is not None else p_lr

    if p_use > ENTRY_PROB:
        print(f"\n  Signal  →  LONG KO  /  SHORT PEP")
        print(f"  Rationale: spread expected to rise — KO underpriced vs PEP")
    elif p_use < (1 - ENTRY_PROB):
        print(f"\n  Signal  →  SHORT KO  /  LONG PEP")
        print(f"  Rationale: spread expected to fall — KO overpriced vs PEP")
    else:
        print(f"\n  Signal  →  FLAT  (probability too close to 50%)")


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_feature_analysis(df_train):
    """4-panel: feature distributions by target class + correlation heatmap."""
    top_features = ["z_score", "p_bear", "beta_std", "spread_vol",
                    "log_vix", "spread_mom"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, feat in zip(axes.flat, top_features):
        d0 = df_train.loc[df_train["target"] == 0, feat].dropna()
        d1 = df_train.loc[df_train["target"] == 1, feat].dropna()
        ax.hist(d0, bins=40, alpha=0.55, color="red",   density=True, label="Spread falls")
        ax.hist(d1, bins=40, alpha=0.55, color="green", density=True, label="Spread rises")
        ax.set_title(feat)
        ax.legend(fontsize=7)
    plt.suptitle("Feature Distributions by Target Class  (Training Set)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df_train):
    """Heatmap of feature-feature and feature-target correlations."""
    cols = FEATURE_COLS + ["target"]
    corr = df_train[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_roc_curves(metrics_list, roc_data):
    """ROC curves for all models on the test set."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["steelblue", "darkorange", "green"]
    for (m, (fpr, tpr)), color in zip(zip(metrics_list, roc_data), colors):
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{m['name']}  (AUC = {m['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_pnl_comparison(pnl_results):
    """Cumulative P&L for all models vs Kalman z-score baseline."""
    fig, ax = plt.subplots(figsize=(13, 4))
    palette = {"Kalman Baseline": "grey",
               "Logistic Regression": "steelblue",
               "MLP Neural Net": "darkorange"}
    for name, res in pnl_results.items():
        cum = res["cum_pnl"]
        ls  = "--" if name == "Kalman Baseline" else "-"
        ax.plot(cum.index, cum.values, label=f"{name}  (Sharpe={res['sharpe']:.2f})",
                linewidth=1.3, linestyle=ls, color=palette.get(name, "black"))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_xlabel("Date")
    ax.set_title("Out-of-Sample Cumulative P&L Comparison")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_mlp_loss_curve(mlp_model):
    """MLP training loss curve (sklearn tracks loss_curve_ attribute)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mlp_model.loss_curve_, color="steelblue", linewidth=1.3, label="Training loss")
    if hasattr(mlp_model, "validation_scores_") and mlp_model.validation_scores_:
        # validation_scores_ is accuracy not loss, plot on twin axis
        ax2 = ax.twinx()
        ax2.plot(mlp_model.validation_scores_, color="darkorange",
                 linewidth=1.3, linestyle="--", label="Val accuracy")
        ax2.set_ylabel("Validation Accuracy", color="darkorange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss", color="steelblue")
    ax.set_title("MLP Neural Net Training Curve")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_feature_importance(lr_model):
    """Logistic regression coefficient magnitudes (standardised features)."""
    coefs = pd.Series(np.abs(lr_model.coef_[0]), index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["steelblue" if c > coefs.median() else "lightsteelblue"
               for c in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.set_xlabel("|Coefficient|  (standardised features)")
    ax.set_title("Logistic Regression Feature Importance")
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Walk-Forward Predictive Model  —  KO / PEP")
    print("=" * 60)

    # ── 1. Load prices, fit HMMs on training data only, run Kalman ───────────
    print("\n[1/6] Loading prices and fitting walk-forward HMMs...")
    prices_ko  = load_daily_prices("KO")
    prices_pep = load_daily_prices("PEP")
    shared     = prices_ko.index.intersection(prices_pep.index)
    prices_ko  = prices_ko.loc[shared]
    prices_pep = prices_pep.loc[shared]

    # Approximate the train cutoff date from raw price dates.
    # The feature matrix drops some rows (rolling burnin) but the date is close.
    train_cutoff = shared[int(len(shared) * TRAIN_FRAC)]
    print(f"      HMM training cutoff: {train_cutoff.date()}  (no test data seen)")

    # Fit HMM on training window only; forward-pass on full series
    # → p_bear is genuinely out-of-sample for the test period
    p_bear_ko  = build_walk_forward_p_bear("KO",  shared, train_cutoff)
    p_bear_pep = build_walk_forward_p_bear("PEP", shared, train_cutoff)
    p_bear_daily = ((p_bear_ko + p_bear_pep) / 2).fillna(0.5)

    # Kalman Filter is already online — no changes needed here
    kf    = KalmanPairsFilter(delta_bull=1e-4, delta_bear=5e-4, R=0.01)
    kf_df = kf.run(prices_ko, prices_pep, p_bear_daily)

    # ── 2. Build features ─────────────────────────────────────────────────────
    print("[2/6] Building feature matrix...")
    vix = load_vix()
    df  = build_features(prices_ko, prices_pep, kf_df, p_bear_daily, vix)
    print(f"      Total samples: {len(df)}  |  Features: {len(FEATURE_COLS)}")
    print(f"      Target balance: {df['target'].mean():.1%} positive")

    # ── 3. Train / test split (time-ordered — NO shuffling) ───────────────────
    print("[3/6] Splitting data (time-ordered 70/30)...")
    split_idx    = int(len(df) * TRAIN_FRAC)
    df_train     = df.iloc[:split_idx]
    df_test      = df.iloc[split_idx:]

    X_train      = df_train[FEATURE_COLS].values
    y_train      = df_train["target"].values
    X_test       = df_test[FEATURE_COLS].values
    y_test       = df_test["target"].values

    print(f"      Train: {df_train.index[0].date()} → {df_train.index[-1].date()}"
          f"  ({len(df_train)} days)")
    print(f"      Test:  {df_test.index[0].date()} → {df_test.index[-1].date()}"
          f"  ({len(df_test)} days)")

    # ── 4. Logistic Regression ────────────────────────────────────────────────
    print("[4/6] Training Logistic Regression...")
    lr_model, lr_scaler, lr_prob, lr_metrics = train_logistic(
        X_train, y_train, X_test, y_test
    )
    print(f"      AUC = {lr_metrics['auc']:.3f}  |  Accuracy = {lr_metrics['acc']:.1%}")

    # ── 5. MLP Neural Network ─────────────────────────────────────────────────
    print("[5/6] Training MLP Neural Network...")
    mlp_scaler = StandardScaler()
    X_tr_sc    = mlp_scaler.fit_transform(X_train)
    X_te_sc    = mlp_scaler.transform(X_test)

    # Build flattened 10-day look-back sequences (10 days × 10 features = 100 inputs)
    X_tr_flat, y_tr_flat = build_sequences_flat(X_tr_sc, y_train, seq_len=SEQ_LEN)
    X_te_flat, y_te_flat = build_sequences_flat(X_te_sc, y_test,  seq_len=SEQ_LEN)

    mlp_model, mlp_prob_raw, mlp_metrics = train_mlp(
        X_tr_flat, y_tr_flat, X_te_flat, y_te_flat
    )
    # Align MLP prob to df_test index (MLP loses first SEQ_LEN-1 rows)
    mlp_prob = pd.Series(mlp_prob_raw, index=df_test.index[SEQ_LEN - 1:])
    print(f"      AUC = {mlp_metrics['auc']:.3f}  |  Accuracy = {mlp_metrics['acc']:.1%}")

    # ── 6. P&L simulation ─────────────────────────────────────────────────────
    print("[6/6] Simulating P&L on test set...")
    pnl_results = {}

    base_pnl, base_cum, base_sharpe = kalman_baseline_pnl(df_test)
    pnl_results["Kalman Baseline"] = {
        "pnl": base_pnl, "cum_pnl": base_cum, "sharpe": base_sharpe
    }

    lr_prob_s = pd.Series(lr_prob, index=df_test.index)
    pnl_lr, cum_lr, sharpe_lr = simulate_pnl(lr_prob_s, df_test["spread"])
    pnl_results["Logistic Regression"] = {
        "pnl": pnl_lr, "cum_pnl": cum_lr, "sharpe": sharpe_lr
    }

    pnl_mlp, cum_mlp, sharpe_mlp = simulate_pnl(mlp_prob, df_test["spread"])
    pnl_results["MLP Neural Net"] = {
        "pnl": pnl_mlp, "cum_pnl": cum_mlp, "sharpe": sharpe_mlp
    }

    # ── Results table ─────────────────────────────────────────────────────────
    all_metrics = [lr_metrics, mlp_metrics, {
        "name": "Kalman Baseline",
        "auc":  0.5,
        "acc":  float((df_test["z_score"] < 0).mean()),
        "prec": 0.0, "rec": 0.0,
    }]
    print_results(all_metrics, pnl_results)

    # ── Current prediction ────────────────────────────────────────────────────
    latest_features = df[FEATURE_COLS].iloc[-1].values
    X_all_sc   = mlp_scaler.transform(df[FEATURE_COLS].values)
    seq_flat   = X_all_sc[-SEQ_LEN:].flatten() if len(X_all_sc) >= SEQ_LEN else None

    predict_now(lr_model, lr_scaler, mlp_model, latest_features, seq_flat)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_prob)
    y_te_mlp             = y_test[SEQ_LEN - 1:]
    fpr_mlp, tpr_mlp, _ = roc_curve(y_te_mlp, mlp_prob_raw)

    plot_feature_analysis(df_train)
    plot_correlation_heatmap(df_train)
    plot_roc_curves([lr_metrics, mlp_metrics], [(fpr_lr, tpr_lr), (fpr_mlp, tpr_mlp)])
    plot_pnl_comparison(pnl_results)
    plot_feature_importance(lr_model)
    plot_mlp_loss_curve(mlp_model)

    plt.show()