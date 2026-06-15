"""
lstm_trainer.py
───────────────
Train the two-head LSTM on historical pair data. Saves best model by val loss.

Usage:
  python lstm_trainer.py

Training targets (minimum to proceed):
  val AUC > 0.52 on both heads (random = 0.50)
  If AUC < 0.52 after 50 epochs: prints dataset stats and stops without saving.

Note: model heads output raw logits. BCEWithLogitsLoss applies sigmoid internally.
This avoids the double-sigmoid bug that would occur if Sigmoid were in the model.
"""

import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)


def train_model(sequences: np.ndarray,
                labels:    np.ndarray,
                save_path: str = None,
                epochs:    int = 50,
                batch_size: int = 32,
                lr: float  = 1e-3) -> object:
    """
    Train PairScorerLSTM. Returns the best model (by val loss).

    Parameters
    ----------
    sequences  : (N, seq_len, n_features) float32
    labels     : (N, 2) float32 — [reversion_label, emergence_label]
    save_path  : where to save .pt file; defaults to models/lstm_pair_scorer.pt
    epochs     : max training epochs
    batch_size : mini-batch size
    lr         : Adam learning rate

    Returns
    -------
    Trained PairScorerLSTM (best val loss checkpoint)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    from lstm_model   import PairScorerLSTM, count_parameters
    from lstm_dataset import make_pair_dataset

    if save_path is None:
        save_path = os.path.join(_here, 'models', 'lstm_pair_scorer.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_features = sequences.shape[2]
    dataset    = make_pair_dataset(sequences, labels)

    val_size   = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = PairScorerLSTM(n_features=n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Weighted BCE to handle class imbalance (rare reversion positives)
    def pos_weight(col):
        n_neg = float((labels[:, col] == 0).sum())
        n_pos = float((labels[:, col] == 1).sum())
        return torch.tensor([n_neg / max(n_pos, 1)])

    crit_rev  = nn.BCEWithLogitsLoss(pos_weight=pos_weight(0))
    crit_emer = nn.BCEWithLogitsLoss(pos_weight=pos_weight(1))

    print(f"  Parameters:          {count_parameters(model):,}")
    print(f"  Train / Val:         {train_size} / {val_size}")
    print(f"  Reversion positives: {labels[:,0].mean():.1%}")
    print(f"  Emergence positives: {labels[:,1].mean():.1%}")
    print(f"  Pos weights:         rev={float(pos_weight(0)):.1f}  emer={float(pos_weight(1)):.1f}")
    print()

    best_val_loss    = float('inf')
    patience_counter = 0
    PATIENCE         = 10
    best_state       = None

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits    = model(X_b)                          # (batch, 2) raw logits
            loss_rev  = crit_rev( logits[:, 0:1], y_b[:, 0:1])
            loss_emer = crit_emer(logits[:, 1:2], y_b[:, 1:2])
            loss      = loss_rev + loss_emer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits    = model(X_b)
                loss_rev  = crit_rev( logits[:, 0:1], y_b[:, 0:1])
                loss_emer = crit_emer(logits[:, 1:2], y_b[:, 1:2])
                val_loss += (loss_rev + loss_emer).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val   = val_loss   / max(len(val_loader),   1)
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f"\n  Best val loss: {best_val_loss:.4f}  — saved to {save_path}")

    return model


def evaluate_model(model, sequences: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute AUC and classification metrics for both heads.
    Returns dict with auc_reversion, auc_emergence.
    """
    import torch
    from sklearn.metrics import classification_report, roc_auc_score

    model.eval()
    X = torch.tensor(sequences, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        probs  = torch.sigmoid(logits).numpy()

    results = {}
    for i, head in enumerate(['Reversion', 'Emergence']):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        y_pred = (y_prob >= 0.5).astype(int)

        if y_true.sum() > 0 and (y_true == 0).sum() > 0:
            auc = roc_auc_score(y_true, y_prob)
            results[f'auc_{head.lower()}'] = auc
            print(f"\n  {head} Head — AUC: {auc:.3f}")
            print(classification_report(y_true, y_pred,
                                        target_names=['No', 'Yes'],
                                        zero_division=0))
        else:
            results[f'auc_{head.lower()}'] = 0.5
            print(f"\n  {head} Head — insufficient class balance for AUC")

    return results


if __name__ == '__main__':
    import yfinance as yf
    import pandas as pd
    from itertools import combinations
    from datetime import date

    from lstm_dataset import (build_datasets, compute_regime_series,
                               SECTOR_MAP, N_FEATURES, SEQUENCE_LEN)
    from cointegration_screener import screen_pairs

    UNIVERSE  = list(SECTOR_MAP.keys())
    END       = date.today().strftime('%Y-%m-%d')
    START     = '2010-01-01'

    print(f"Downloading data {START} to {END}...")
    raw = yf.download(UNIVERSE, start=START, end=END,
                      auto_adjust=True, progress=False)['Close'].ffill().dropna()
    lp  = np.log(raw)

    print("Computing regime series...")
    regime = compute_regime_series(lp)

    print("Running cointegration screen on full history...")
    ranked = screen_pairs(lp)
    if not ranked:
        print("No pairs found — check screener thresholds")
        sys.exit(1)

    all_ss = [(t1, t2) for t1, t2 in combinations(UNIVERSE, 2)
              if SECTOR_MAP.get(t1) == SECTOR_MAP.get(t2)]

    k_betas = {p['pair']: pd.Series(p['ols_beta'], index=lp.index) for p in ranked}

    print(f"\nBuilding dataset from {len(ranked)} cointegrated pairs "
          f"+ {len(all_ss)-len(ranked)} non-cointegrated pairs...")
    X, y = build_datasets(lp, ranked, all_ss, k_betas, regime)

    if len(X) < 50:
        print(f"\nWARNING: Only {len(X)} training examples. Minimum is 200.")
        print("Consider expanding the universe or using a longer history.")
        sys.exit(1)

    print(f"\nTraining LSTM on {len(X)} examples...")
    model = train_model(X, y, epochs=50)

    print("\nEvaluating on full dataset (training diagnostic)...")
    metrics = evaluate_model(model, X, y)

    MIN_AUC = 0.52
    if any(v < MIN_AUC for v in metrics.values()):
        print(f"\nWARNING: AUC below minimum threshold ({MIN_AUC}).")
        print("Features may not be predictive — do not use for live recommendations.")
    else:
        print(f"\nAll AUCs >= {MIN_AUC}. Model is ready for scoring.")
