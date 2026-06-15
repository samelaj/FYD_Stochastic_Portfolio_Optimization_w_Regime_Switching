# CLAUDE.md — LSTM Pair Scoring System
# Extends: FYD/Simulations/Markov-Models/PairSelector/
# Hardware: CPU only (laptop) — keep model small and fast
# Goal: weekly $100 hedge recommendation using LSTM to rank pairs

## What This Builds

A two-output LSTM that runs every week and answers two questions:
1. REVERSION HEAD  — "Which currently cointegrated pairs will mean-revert this week?"
2. EMERGENCE HEAD  — "Which pairs are about to become cointegrated?"

Combined output: a single weekly recommendation — one pair to trade, which leg to
go long/short, and how to size $100 across the two legs.

---

## New Files to Create

```
PairSelector/
├── lstm_dataset.py       ← builds labeled training dataset from walk-forward history
├── lstm_model.py         ← two-head LSTM architecture (PyTorch, CPU-optimized)
├── lstm_trainer.py       ← trains and saves the model
├── lstm_predictor.py     ← loads model, scores current pairs
├── weekly_runner.py      ← full weekly pipeline end-to-end
└── models/
    └── .gitkeep          ← saved model goes here after training
```

Do NOT modify any existing files. Import from them only.

---

## Step 0 — Read existing files first (mandatory)

Before writing any new code, read and summarize:
1. `cointegration_screener.py` — SECTOR_MAP, screen_pairs() interface, what columns
   log_prices DataFrame has
2. `dynamic_hedger.py` — how features like z-score and spread are currently computed
3. `bs_vol_filter.py` — realized_vol() and iv_rank() interfaces
4. `hhm.py` — what regime states are output, how to call it

These are your feature sources. Know their interfaces before building the dataset.

---

## Module 1 — lstm_dataset.py

### Purpose
Build two labeled datasets from historical price data:
- Dataset A (reversion): sequences for currently cointegrated pairs
- Dataset B (emergence): sequences for pairs approaching cointegration

### Feature Vector (computed per bar, per pair)

```python
# 12 features per timestep
FEATURES = [
    'z_score',           # current spread z-score (rolling 20-day)
    'z_momentum',        # z_score[t] - z_score[t-5]
    'spread_vol_20',     # 20-day realized vol of spread
    'beta',              # current Kalman hedge ratio
    'beta_change',       # beta[t] - beta[t-10] (drift in hedge ratio)
    'iv_rank_leg1',      # IV rank of long leg (from bs_vol_filter)
    'iv_rank_leg2',      # IV rank of short leg
    'corr_60',           # 60-day rolling return correlation between legs
    'half_life',         # current half-life estimate (normalized)
    'regime',            # HMM regime state: 0=risk-on, 1=risk-off
    'days_since_cross',  # days since z-score last crossed zero
    'pvalue_rolling',    # rolling 60-day EG p-value (cointegration strength)
]

SEQUENCE_LEN = 20   # 20 bars of history per training example (4 trading weeks)
```

### Label A — Reversion (for currently cointegrated pairs)

```python
def label_reversion(z_score: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Binary: did the spread move at least 50% toward zero within `horizon` bars?
    Positive = 1, No reversion = 0

    Using 50% reversion (not full close) to generate enough positive examples.
    A z-score of 2.1 that moves to 1.0 within 5 days = 1 (50% of the way to 0).
    """
    labels = pd.Series(0, index=z_score.index)
    for i in range(len(z_score) - horizon):
        current_z = z_score.iloc[i]
        if abs(current_z) < 1.5:          # only label bars where signal is active
            continue
        target = current_z * 0.5          # 50% reversion target
        future_slice = z_score.iloc[i+1 : i+horizon+1]
        # Moved toward zero by at least 50%
        if current_z > 0 and (future_slice <= target).any():
            labels.iloc[i] = 1
        elif current_z < 0 and (future_slice >= target).any():
            labels.iloc[i] = 1
    return labels
```

### Label B — Emergence (for pairs NOT yet cointegrated)

```python
def label_emergence(log_prices: pd.DataFrame, t1: str, t2: str,
                    horizon: int = 63) -> pd.Series:
    """
    Binary: does this pair pass the cointegration screen (EG p < 0.15)
    within the next `horizon` bars (63 days = 1 quarter)?
    Label each bar today as 1 if the pair becomes cointegrated within 63 days.
    """
    from statsmodels.tsa.stattools import coint
    series = pd.concat([log_prices[t1], log_prices[t2]], axis=1).dropna()
    labels = pd.Series(0, index=series.index)
    for i in range(len(series) - horizon - 60):
        future_window = series.iloc[i+1 : i+horizon+1]
        if len(future_window) < 60:
            continue
        try:
            _, pvalue, _ = coint(future_window[t1], future_window[t2])
            if pvalue < 0.15:
                labels.iloc[i] = 1
        except Exception:
            continue
    return labels
```

### Dataset Builder

```python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """
    Produces (sequence, label) pairs for LSTM training.
    sequence shape: (SEQUENCE_LEN, n_features)
    label shape:    (2,)  — [reversion_label, emergence_label]
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        # sequences: (N, SEQUENCE_LEN, n_features)
        # labels:    (N, 2)
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_feature_matrix(log_prices, t1, t2, kalman_beta,
                          vol_fn, ivrank_fn, regime_series) -> pd.DataFrame:
    """
    Compute all 12 features for a pair over the full price history.
    Returns DataFrame indexed by date with one column per feature.
    """
    spread    = log_prices[t1] - kalman_beta * log_prices[t2]
    z         = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + 1e-8)
    spread_vol = spread.diff().rolling(20).std() * np.sqrt(252)
    corr_60   = log_prices[t1].diff().rolling(60).corr(log_prices[t2].diff())

    # Half-life (rolling AR1)
    def rolling_hl(spread, window=60):
        hls = pd.Series(np.nan, index=spread.index)
        for i in range(window, len(spread)):
            s = spread.iloc[i-window:i]
            sl = s.shift(1).dropna()
            ds = s.diff().dropna()
            aligned = pd.concat([ds, sl], axis=1).dropna()
            if len(aligned) < 20:
                continue
            try:
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools import add_constant
                phi = OLS(aligned.iloc[:,0], add_constant(aligned.iloc[:,1])).fit().params.iloc[1]
                if phi < 0:
                    hls.iloc[i] = min(-np.log(2) / np.log(1 + phi), 500)
            except Exception:
                pass
        return hls

    hl = rolling_hl(spread)

    # Days since last zero crossing
    zero_cross = ((z > 0) != (z.shift(1) > 0)).astype(int)
    days_since = zero_cross[::-1].expanding().apply(
        lambda x: next((i for i, v in enumerate(x) if v == 1), len(x)), raw=False
    )[::-1]

    # Rolling EG p-value
    from statsmodels.tsa.stattools import coint
    pvals = pd.Series(np.nan, index=log_prices.index)
    for i in range(120, len(log_prices)):
        window = log_prices[[t1, t2]].iloc[i-60:i].dropna()
        try:
            _, p, _ = coint(window[t1], window[t2])
            pvals.iloc[i] = p
        except Exception:
            pass

    feat = pd.DataFrame({
        'z_score':         z,
        'z_momentum':      z - z.shift(5),
        'spread_vol_20':   spread_vol,
        'beta':            kalman_beta,
        'beta_change':     kalman_beta - kalman_beta.shift(10),
        'iv_rank_leg1':    ivrank_fn(vol_fn(np.exp(log_prices[t1]))),
        'iv_rank_leg2':    ivrank_fn(vol_fn(np.exp(log_prices[t2]))),
        'corr_60':         corr_60,
        'half_life':       hl.clip(0, 500) / 500,    # normalize to [0,1]
        'regime':          regime_series.reindex(log_prices.index).ffill().fillna(0),
        'days_since_cross': days_since.clip(0, 60) / 60,
        'pvalue_rolling':  pvals.clip(0, 1),
    }).ffill().fillna(0)

    return feat


def build_datasets(log_prices, screened_pairs, all_pairs,
                   kalman_betas, vol_fn, ivrank_fn, regime_series,
                   seq_len=20, horizon_rev=5, horizon_emer=63):
    """
    Build sequences and labels for both heads.

    screened_pairs : list of (t1,t2) that currently pass cointegration
    all_pairs      : all same-sector pairs (for emergence labeling)
    kalman_betas   : dict of {(t1,t2): pd.Series of beta values}
    """
    sequences, labels = [], []

    # Dataset A: reversion labels on cointegrated pairs
    for (t1, t2) in screened_pairs:
        beta = kalman_betas.get((t1,t2), pd.Series(1.0, index=log_prices.index))
        feat = build_feature_matrix(log_prices, t1, t2, beta,
                                    vol_fn, ivrank_fn, regime_series)
        z    = feat['z_score']
        rev_labels  = label_reversion(z, horizon=horizon_rev)
        emer_labels = label_emergence(log_prices, t1, t2, horizon=horizon_emer)

        feat_arr = feat.values
        for i in range(seq_len, len(feat_arr) - max(horizon_rev, horizon_emer)):
            seq = feat_arr[i-seq_len:i]
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            labels.append([rev_labels.iloc[i], emer_labels.iloc[i]])

    # Dataset B: emergence labels on non-cointegrated same-sector pairs
    non_coint = [p for p in all_pairs if p not in screened_pairs]
    for (t1, t2) in non_coint:
        beta = pd.Series(1.0, index=log_prices.index)   # no Kalman for non-coint pairs
        feat = build_feature_matrix(log_prices, t1, t2, beta,
                                    vol_fn, ivrank_fn, regime_series)
        emer_labels = label_emergence(log_prices, t1, t2, horizon=horizon_emer)

        feat_arr = feat.values
        for i in range(seq_len, len(feat_arr) - horizon_emer):
            seq = feat_arr[i-seq_len:i]
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            labels.append([0, emer_labels.iloc[i]])   # reversion label = 0 (not active)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)
```

---

## Module 2 — lstm_model.py

### Architecture: Two-Head LSTM (CPU-optimized, small and fast)

```python
import torch
import torch.nn as nn

class PairScorerLSTM(nn.Module):
    """
    Two-head LSTM for pair scoring.

    Input:  (batch, seq_len=20, n_features=12)
    Output: (batch, 2)
      output[:,0] = P(mean reversion within 5 days)   — reversion head
      output[:,1] = P(cointegration within 63 days)   — emergence head

    Kept small for CPU training:
      - 1 LSTM layer, hidden_dim=64
      - 2 FC layers per head
      - Dropout 0.3 for regularization on small dataset
      - Total params: ~25k (trains in <2 min on CPU)
    """
    def __init__(self, n_features=12, hidden_dim=64, seq_len=20, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0       # dropout only after LSTM with num_layers=1
        )
        self.dropout = nn.Dropout(dropout)

        # Shared representation layer
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Head 1: reversion probability
        self.reversion_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Head 2: emergence probability
        self.emergence_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        last_hidden  = self.dropout(lstm_out[:, -1, :])   # take last timestep
        shared       = self.shared_fc(last_hidden)

        p_reversion = self.reversion_head(shared)   # (batch, 1)
        p_emergence = self.emergence_head(shared)   # (batch, 1)

        return torch.cat([p_reversion, p_emergence], dim=1)   # (batch, 2)
```

---

## Module 3 — lstm_trainer.py

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from lstm_model   import PairScorerLSTM
from lstm_dataset import PairDataset

def train_model(sequences: np.ndarray,
                labels:    np.ndarray,
                save_path: str = 'models/lstm_pair_scorer.pt',
                epochs: int = 50,
                batch_size: int = 32,
                lr: float = 1e-3):
    """
    Train the two-head LSTM on CPU. Saves best model by validation loss.

    With ~500-2000 training examples (from walk-forward history):
    - 50 epochs at batch_size=32 takes ~1-3 min on laptop CPU
    - 80/20 train/val split
    - Early stopping patience=10
    """
    os.makedirs('models', exist_ok=True)

    dataset   = PairDataset(sequences, labels)
    val_size  = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = PairScorerLSTM(n_features=sequences.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Weighted BCE: handle class imbalance (reversions are rare)
    pos_weight_rev  = torch.tensor([(labels[:,0]==0).sum() / max((labels[:,0]==1).sum(), 1)])
    pos_weight_emer = torch.tensor([(labels[:,1]==0).sum() / max((labels[:,1]==1).sum(), 1)])

    criterion_rev  = nn.BCEWithLogitsLoss(pos_weight=pos_weight_rev)
    criterion_emer = nn.BCEWithLogitsLoss(pos_weight=pos_weight_emer)

    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10

    print(f"Training on {train_size} examples, validating on {val_size}")
    print(f"Class balance — reversion positives: {labels[:,0].mean():.1%}")
    print(f"Class balance — emergence positives: {labels[:,1].mean():.1%}\n")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)

            # Use raw logits for BCEWithLogitsLoss (more numerically stable)
            # Temporarily bypass sigmoid in heads for loss computation
            loss_rev  = criterion_rev( preds[:, 0].unsqueeze(1), y_batch[:, 0].unsqueeze(1))
            loss_emer = criterion_emer(preds[:, 1].unsqueeze(1), y_batch[:, 1].unsqueeze(1))
            loss = loss_rev + loss_emer

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss_rev  = criterion_rev( preds[:, 0].unsqueeze(1), y_batch[:, 0].unsqueeze(1))
                loss_emer = criterion_emer(preds[:, 1].unsqueeze(1), y_batch[:, 1].unsqueeze(1))
                val_loss += (loss_rev + loss_emer).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f}")

        # Early stopping + save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} — model saved to {save_path}")
    return model


def evaluate_model(model, sequences, labels):
    """Print classification metrics for both heads."""
    from sklearn.metrics import classification_report, roc_auc_score

    model.eval()
    X = torch.tensor(sequences, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).numpy()

    for i, head in enumerate(['Reversion', 'Emergence']):
        y_true = labels[:, i]
        y_prob = preds[:, i]
        y_pred = (y_prob > 0.5).astype(int)
        if y_true.sum() > 0:
            auc = roc_auc_score(y_true, y_prob)
            print(f"\n{head} Head — AUC: {auc:.3f}")
            print(classification_report(y_true, y_pred,
                                        target_names=['No', 'Yes'],
                                        zero_division=0))
        else:
            print(f"\n{head} Head — no positive examples in this split")
```

---

## Module 4 — lstm_predictor.py

```python
import torch
import numpy as np
import pandas as pd
from datetime import date, timedelta
import yfinance as yf

from lstm_model      import PairScorerLSTM
from lstm_dataset    import build_feature_matrix, FEATURES, SEQUENCE_LEN
from cointegration_screener import screen_pairs, SECTOR_MAP
from bs_vol_filter   import realized_vol, iv_rank
from itertools import combinations

MODEL_PATH = 'models/lstm_pair_scorer.pt'

def load_model(n_features=12) -> PairScorerLSTM:
    model = PairScorerLSTM(n_features=n_features)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model


def score_all_pairs(log_prices, kalman_betas, regime_series, model) -> pd.DataFrame:
    """
    Score every same-sector pair using the trained LSTM.
    Returns DataFrame with columns:
      pair, sector, p_reversion, p_emergence, z_score_now,
      half_life, is_cointegrated, composite_score
    """
    results = []
    universe = log_prices.columns.tolist()

    # Get currently cointegrated pairs
    coint_pairs = {p['pair'] for p in screen_pairs(log_prices)}

    for t1, t2 in combinations(universe, 2):
        if SECTOR_MAP.get(t1) != SECTOR_MAP.get(t2):
            continue

        beta = kalman_betas.get((t1, t2), pd.Series(1.0, index=log_prices.index))

        try:
            feat = build_feature_matrix(
                log_prices, t1, t2, beta,
                realized_vol, iv_rank, regime_series
            )
        except Exception:
            continue

        if len(feat) < SEQUENCE_LEN:
            continue

        seq = feat.values[-SEQUENCE_LEN:]
        if np.isnan(seq).any():
            continue

        X = torch.tensor(seq[np.newaxis], dtype=torch.float32)  # (1, 20, 12)
        with torch.no_grad():
            p_rev, p_emer = model(X)[0].numpy()

        z_now     = feat['z_score'].iloc[-1]
        hl        = feat['half_life'].iloc[-1] * 500   # un-normalize
        is_coint  = (t1, t2) in coint_pairs

        # Composite score: reversion matters more if already cointegrated
        composite = (0.7 * p_rev + 0.3 * p_emer) if is_coint else \
                    (0.2 * p_rev + 0.8 * p_emer)

        results.append({
            'pair':           f'{t1}/{t2}',
            't1':             t1,
            't2':             t2,
            'sector':         SECTOR_MAP.get(t1),
            'p_reversion':    float(p_rev),
            'p_emergence':    float(p_emer),
            'z_score_now':    float(z_now),
            'half_life':      float(hl),
            'is_cointegrated': is_coint,
            'composite_score': float(composite),
        })

    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.reset_index(drop=True)
```

---

## Module 5 — weekly_runner.py

### The $100/week recommendation engine

```python
# weekly_runner.py
# Run every Monday morning: python weekly_runner.py
# Outputs: weekly_recommendation.png + prints plain-English trade to terminal

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date, timedelta
import torch

from lstm_predictor      import load_model, score_all_pairs
from cointegration_screener import SECTOR_MAP
from bs_vol_filter       import realized_vol, iv_rank

BUDGET        = 100.00    # total capital per week
MIN_Z_ENTRY   = 1.8       # minimum |z-score| to enter (slightly below 2.0)
MIN_P_REV     = 0.45      # minimum reversion probability to trade
MIN_P_EMER    = 0.55      # minimum emergence probability to flag
UNIVERSE      = list(SECTOR_MAP.keys())
TRAIN_DAYS    = 504
END_DATE      = date.today().strftime('%Y-%m-%d')
START_DATE    = (date.today() - timedelta(days=TRAIN_DAYS + 60)).strftime('%Y-%m-%d')


def get_trade_sizing(t1, t2, beta, prices, budget=100.0):
    """
    Size $100 across two legs, dollar-neutral.
    Returns (shares_t1, shares_t2, cost_t1, cost_t2)
    With fractional shares for small budgets.
    """
    p1 = prices[t1].iloc[-1]
    p2 = prices[t2].iloc[-1]
    # Dollar-neutral: cost_t1 = cost_t2 = budget/2
    half = budget / 2
    shares_t1 = half / p1
    shares_t2 = (half * beta) / p2   # scale short leg by hedge ratio
    return round(shares_t1, 4), round(shares_t2, 4), round(half, 2), round(half, 2)


def run_weekly():
    print(f"\n{'='*60}")
    print(f"  WEEKLY PAIRS RECOMMENDATION — {END_DATE}")
    print(f"{'='*60}\n")

    # --- Fetch data ---
    print("Fetching market data...")
    raw = yf.download(UNIVERSE + ['^IRX'], start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)['Close'].ffill().dropna()

    prices     = raw.drop(columns=['^IRX'], errors='ignore')
    log_prices = np.log(prices)

    # --- Dummy regime (replace with hhm.py call after reading its interface) ---
    # TODO: replace with: from hhm import get_regime; regime = get_regime(log_prices)
    regime_series = pd.Series(0, index=log_prices.index)

    # --- Dummy Kalman betas (replace with actual kalman_hedger output) ---
    # TODO: replace with walk-forward Kalman betas from kalman_hedger.py
    from itertools import combinations
    kalman_betas = {}
    for t1, t2 in combinations(UNIVERSE, 2):
        if SECTOR_MAP.get(t1) == SECTOR_MAP.get(t2):
            kalman_betas[(t1, t2)] = pd.Series(1.0, index=log_prices.index)

    # --- Load model and score pairs ---
    try:
        model = load_model()
    except FileNotFoundError:
        print("ERROR: No trained model found at models/lstm_pair_scorer.pt")
        print("Run lstm_trainer.py first to train the model.")
        return

    scores = score_all_pairs(log_prices, kalman_betas, regime_series, model)

    # --- Find best trade ---
    tradeable = scores[
        scores['is_cointegrated'] &
        (scores['z_score_now'].abs() >= MIN_Z_ENTRY) &
        (scores['p_reversion'] >= MIN_P_REV)
    ]

    # --- Find emerging pairs to watch ---
    emerging = scores[
        ~scores['is_cointegrated'] &
        (scores['p_emergence'] >= MIN_P_EMER)
    ].head(3)

    # --- Print recommendation ---
    if tradeable.empty:
        print("NO TRADE THIS WEEK")
        print("No pairs meet all entry criteria right now.")
        print("(Need: cointegrated + |z| >= 1.8 + P(reversion) >= 45%)\n")
    else:
        best = tradeable.iloc[0]
        t1, t2 = best['t1'], best['t2']
        z  = best['z_score_now']
        beta_val = kalman_betas.get((t1,t2), pd.Series(1.0)).iloc[-1]

        shares1, shares2, cost1, cost2 = get_trade_sizing(
            t1, t2, beta_val, prices, BUDGET
        )

        long_leg  = t1 if z < 0 else t2
        short_leg = t2 if z < 0 else t1
        long_shr  = shares1 if z < 0 else shares2
        short_shr = shares2 if z < 0 else shares1

        print(f"  TRADE THIS WEEK")
        print(f"  {'─'*45}")
        print(f"  Pair:         {t1} / {t2}  ({best['sector'].upper()})")
        print(f"  Signal:       z-score = {z:+.2f}σ")
        print(f"  Action:       BUY  {long_shr:.4f} shares of {long_leg}  (${cost1:.2f})")
        print(f"                SELL {short_shr:.4f} shares of {short_leg} (${cost2:.2f})")
        print(f"  Total cost:   ${BUDGET:.2f}")
        print(f"  Exit when:    |z-score| < 0.5  OR  5 trading days elapsed")
        print(f"  P(reversion): {best['p_reversion']:.1%}")
        print(f"  Half-life:    {best['half_life']:.0f} days")
        print(f"  {'─'*45}\n")

    if not emerging.empty:
        print(f"  PAIRS TO WATCH (not yet tradeable but emerging)")
        print(f"  {'─'*45}")
        for _, row in emerging.iterrows():
            print(f"  {row['pair']:<14} P(cointegration): {row['p_emergence']:.1%}  "
                  f"sector: {row['sector']}")
        print()

    # --- Generate figure ---
    _generate_figure(scores, tradeable, emerging, prices, log_prices)


def _generate_figure(scores, tradeable, emerging, prices, log_prices):
    """
    Clean, non-technical figure anyone can read.
    Panel 1: Ranking bar chart — all scored pairs
    Panel 2: Spread z-score for top trade (if any)
    Panel 3: Watch list — emerging pairs
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Weekly Pairs Recommendation — {END_DATE}',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_rank   = fig.add_subplot(gs[0, :])   # full top row
    ax_spread = fig.add_subplot(gs[1, 0])
    ax_watch  = fig.add_subplot(gs[1, 1])

    # --- Panel 1: composite score bar chart (top 10 pairs) ---
    top10 = scores.head(10)
    colors = ['#16a34a' if row['is_cointegrated'] else '#2563EB'
              for _, row in top10.iterrows()]
    bars = ax_rank.barh(top10['pair'][::-1], top10['composite_score'][::-1],
                        color=colors[::-1], alpha=0.85)
    ax_rank.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Entry threshold')
    ax_rank.set_xlabel('LSTM Score (0 = no signal, 1 = strong signal)')
    ax_rank.set_title('All Pairs Ranked by LSTM Score', fontweight='bold')
    ax_rank.legend(fontsize=9)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor='#16a34a', label='Currently cointegrated (tradeable)'),
                  Patch(facecolor='#2563EB', label='Emerging (watch list)')]
    ax_rank.legend(handles=legend_els, fontsize=9, loc='lower right')
    ax_rank.grid(True, axis='x', alpha=0.3)

    # --- Panel 2: spread z-score for best trade ---
    if not tradeable.empty:
        best = tradeable.iloc[0]
        t1, t2 = best['t1'], best['t2']
        spread = log_prices[t1] - log_prices[t2]
        z = (spread - spread.rolling(20).mean()) / (spread.rolling(20).std() + 1e-8)
        z_plot = z.dropna().iloc[-63:]   # last quarter

        ax_spread.plot(z_plot.index, z_plot.values, color='gray', linewidth=1)
        ax_spread.axhline( 2.0, color='red',   linestyle='--', linewidth=1)
        ax_spread.axhline(-2.0, color='green', linestyle='--', linewidth=1)
        ax_spread.axhline( 0.0, color='black', linestyle='-',  linewidth=0.5, alpha=0.4)
        ax_spread.fill_between(z_plot.index, 2.0,  z_plot,
                               where=z_plot > 2.0,  color='red',   alpha=0.2)
        ax_spread.fill_between(z_plot.index, -2.0, z_plot,
                               where=z_plot < -2.0, color='green', alpha=0.2)

        current_z = z_plot.iloc[-1]
        direction = f"BUY {t1} / SELL {t2}" if current_z < 0 else f"SELL {t1} / BUY {t2}"
        ax_spread.set_title(f'THIS WEEK: {t1}/{t2}\n{direction}  (z={current_z:+.2f}σ)',
                            fontweight='bold',
                            color='green' if current_z < 0 else 'red')
        ax_spread.set_ylabel('Spread z-score')
        ax_spread.set_ylim(-4, 4)
        ax_spread.grid(True, alpha=0.3)
        ax_spread.tick_params(axis='x', rotation=30)
    else:
        ax_spread.text(0.5, 0.5, 'NO TRADE\nTHIS WEEK',
                       ha='center', va='center', fontsize=16,
                       color='gray', transform=ax_spread.transAxes)
        ax_spread.set_title('This Week\'s Trade', fontweight='bold')
        ax_spread.axis('off')

    # --- Panel 3: watch list ---
    if not emerging.empty:
        watch_pairs  = emerging['pair'].tolist()
        watch_scores = emerging['p_emergence'].tolist()
        ax_watch.barh(watch_pairs[::-1], watch_scores[::-1],
                      color='#7c3aed', alpha=0.75)
        ax_watch.axvline(MIN_P_EMER, color='gray', linestyle='--',
                         linewidth=1, label=f'Threshold ({MIN_P_EMER:.0%})')
        ax_watch.set_xlim(0, 1)
        ax_watch.set_xlabel('P(becomes tradeable within 3 months)')
        ax_watch.set_title('Emerging Pairs — Watch List', fontweight='bold')
        ax_watch.legend(fontsize=9)
        ax_watch.grid(True, axis='x', alpha=0.3)
    else:
        ax_watch.text(0.5, 0.5, 'No emerging\npairs flagged',
                      ha='center', va='center', fontsize=13,
                      color='gray', transform=ax_watch.transAxes)
        ax_watch.set_title('Emerging Pairs — Watch List', fontweight='bold')
        ax_watch.axis('off')

    plt.savefig('weekly_recommendation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: weekly_recommendation.png")


if __name__ == '__main__':
    run_weekly()
```

---

## Implementation Order

Work through in this exact sequence. Confirm each step before moving on.

```
Step 1  Read existing files (mandatory — see Step 0)
Step 2  Create models/ folder with .gitkeep
Step 3  Build lstm_dataset.py
        → test: call build_datasets() on 2 years of data
        → confirm sequences.shape = (N, 20, 12), labels.shape = (N, 2)
        → print class balance for both heads
        → if N < 200: expand universe to 40 tickers (add more per sector)
Step 4  Build lstm_model.py
        → test: model(torch.randn(4, 20, 12)).shape == (4, 2)
Step 5  Build lstm_trainer.py
        → train on full historical data (2010-today)
        → target: val AUC > 0.55 on both heads (random = 0.50)
        → if AUC < 0.52 after 50 epochs: print dataset stats and stop
Step 6  Build lstm_predictor.py
        → test: score_all_pairs() returns DataFrame with all expected columns
Step 7  Build weekly_runner.py
        → wire in real Kalman betas from kalman_hedger.py (replace TODO)
        → wire in real HMM regime from hhm.py (replace TODO)
        → run end-to-end, confirm figure generates
Step 8  Run python weekly_runner.py and review recommendation
```

---

## Realism Targets

| Metric | Minimum to proceed | Good result |
|---|---|---|
| Training examples N | > 200 | > 500 |
| Reversion head AUC | > 0.52 | > 0.60 |
| Emergence head AUC | > 0.52 | > 0.58 |
| Weekly recommendation | Fires on ≥1 pair | Fires on 1-3 pairs |
| P(reversion) on trade | > 0.45 | > 0.60 |

If either AUC is below 0.52 after training, the dataset is too small or the
features are not predictive. Do not proceed to live recommendations — report
the dataset stats and stop for review.

---

## $100 Sizing Logic

At $100 total ($50 per leg):
- Fractional shares required — use a broker that supports them
  (Robinhood, Webull, IBKR Lite, Fidelity)
- Transaction costs: target zero-commission broker
  At $50/leg, even 1bps = $0.05 — negligible
- Expected return per trade at 0.5 Sharpe: ~$0.40-$1.20 per week
  (This is a learning system — P&L is secondary to signal quality)
- Exit rule: whichever comes first —
  1. |z-score| < 0.5 (mean reversion achieved)
  2. 5 trading days elapsed (time stop)
  3. |z-score| > 3.5 (hard stop-loss, spread blowing out)

---

## Known Pitfalls to Avoid

1. Do NOT train the LSTM on the same data used for the walk-forward backtest
   without a proper time split. Training cutoff = end of walk-forward period.
   Scoring = only on dates after training cutoff.

2. Do NOT use future z-scores to build features. Every feature must use only
   data available at bar t. The label looks forward; the features look backward.

3. If the model always outputs p ≈ 0.5, the LSTM is not learning — check that
   features are being normalized (half_life / 500, days_since / 60, etc.)

4. The emergence head will have very few positive examples early on.
   The weighted BCE loss handles this — do not manually oversample.
