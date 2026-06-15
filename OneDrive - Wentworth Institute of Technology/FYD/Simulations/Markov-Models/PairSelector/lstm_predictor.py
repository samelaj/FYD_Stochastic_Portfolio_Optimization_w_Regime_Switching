"""
lstm_predictor.py
─────────────────
Load trained model and score all same-sector pairs.

Usage (standalone):
  from lstm_predictor import load_model, score_all_pairs
  model  = load_model()
  scores = score_all_pairs(log_prices, kalman_betas, regime_series, model)

score_all_pairs() returns a DataFrame sorted by composite_score descending:
  pair, t1, t2, sector, p_reversion, p_emergence, z_score_now,
  half_life, is_cointegrated, composite_score
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from cointegration_screener import screen_pairs, SECTOR_MAP
from lstm_dataset import build_feature_matrix, SEQUENCE_LEN, N_FEATURES
from lstm_model   import PairScorerLSTM

MODEL_PATH = os.path.join(_here, 'models', 'lstm_pair_scorer.pt')


def load_model(model_path: str = MODEL_PATH,
               n_features: int = N_FEATURES) -> PairScorerLSTM:
    """Load saved model weights. Raises FileNotFoundError if not trained yet."""
    import torch
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. "
            "Run lstm_trainer.py first to train the model."
        )
    model = PairScorerLSTM(n_features=n_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu',
                                     weights_only=True))
    model.eval()
    return model


def score_all_pairs(log_prices: pd.DataFrame,
                    kalman_betas: dict,
                    regime_series: pd.Series,
                    model: PairScorerLSTM,
                    pvalue_threshold: float = 0.15) -> pd.DataFrame:
    """
    Score every same-sector pair using the trained LSTM.

    Parameters
    ----------
    log_prices       : full log-price history (dates x tickers)
    kalman_betas     : {(t1,t2): pd.Series of time-varying beta}
    regime_series    : daily regime Series (0=risk-on, 1=risk-off)
    model            : loaded PairScorerLSTM
    pvalue_threshold : EG p-value cutoff for is_cointegrated flag

    Returns
    -------
    DataFrame sorted by composite_score descending.
    """
    import torch

    universe    = log_prices.columns.tolist()
    coint_info  = screen_pairs(log_prices, pvalue_threshold=pvalue_threshold)
    coint_dict  = {p['pair']: p for p in coint_info}

    results = []

    for t1, t2 in combinations(universe, 2):
        if SECTOR_MAP.get(t1) != SECTOR_MAP.get(t2):
            continue
        if t1 not in log_prices.columns or t2 not in log_prices.columns:
            continue

        # Use OLS parameters from screener if pair is cointegrated, else unit beta
        if (t1, t2) in coint_dict:
            info      = coint_dict[(t1, t2)]
            alpha_ols = info.get('ols_alpha', 0.0)
            beta_ols  = info['ols_beta']
            half_life = info['half_life']
        else:
            alpha_ols = 0.0
            beta_ols  = 1.0
            half_life = 0.0

        k_beta = kalman_betas.get((t1, t2),
                                  pd.Series(beta_ols, index=log_prices.index))

        try:
            feat = build_feature_matrix(
                log_prices, t1, t2, k_beta, regime_series, alpha_ols, beta_ols
            )
        except Exception:
            continue

        if len(feat) < SEQUENCE_LEN:
            continue

        seq = feat.values[-SEQUENCE_LEN:]
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue

        X = torch.tensor(seq[np.newaxis], dtype=torch.float32)   # (1, 20, 12)
        p_rev, p_emer = model.predict_pair(X)

        z_now        = float(feat['z_score'].iloc[-1])
        is_coint     = (t1, t2) in coint_dict

        # Composite: reversion signal dominates when already cointegrated;
        # emergence signal dominates when watching a not-yet-cointegrated pair
        composite = (0.7 * p_rev + 0.3 * p_emer) if is_coint else \
                    (0.2 * p_rev + 0.8 * p_emer)

        results.append({
            'pair':            f'{t1}/{t2}',
            't1':              t1,
            't2':              t2,
            'sector':          SECTOR_MAP.get(t1, 'unknown'),
            'p_reversion':     p_rev,
            'p_emergence':     p_emer,
            'z_score_now':     z_now,
            'half_life':       half_life,
            'ols_beta':        beta_ols,
            'is_cointegrated': is_coint,
            'composite_score': composite,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values('composite_score',
                                              ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    import yfinance as yf
    from datetime import date, timedelta
    from lstm_dataset import compute_regime_series

    UNIVERSE   = list(SECTOR_MAP.keys())
    END_DATE   = date.today().strftime('%Y-%m-%d')
    START_DATE = (date.today() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"Fetching data {START_DATE} to {END_DATE}...")
    raw = yf.download(UNIVERSE, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)['Close'].ffill().dropna()
    lp  = np.log(raw)

    regime = compute_regime_series(lp)

    # Stub Kalman betas (replace with real Kalman betas for production)
    k_betas = {}

    try:
        model  = load_model()
        scores = score_all_pairs(lp, k_betas, regime, model)
        if scores.empty:
            print("No pairs scored.")
        else:
            print(f"\nTop 5 pairs by composite score:")
            print(scores[['pair', 'sector', 'p_reversion', 'p_emergence',
                           'z_score_now', 'composite_score']].head(5).to_string(index=False))
    except FileNotFoundError as e:
        print(e)
