# Stochastic Portfolio Optimization with Regime Switching

**Final Year Design Project — Wentworth Institute of Technology**
**Jake Samela**

---

## Overview

This project builds a **regime-aware statistical arbitrage system** that combines Hidden Markov Model (HMM) regime detection, Engle-Granger cointegration screening, Kalman filter dynamic hedging, and a two-head LSTM pair scorer into a fully automated weekly trade recommendation engine.

The core motivation: standard Black-Scholes assumes constant volatility and normally distributed returns. In practice, markets cycle through distinct regimes (low-vol bull, high-vol bear). This system detects those regimes in real time and uses them to filter and size pairs trades that exploit mean-reverting cointegrated spreads.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │        Market Data (yfinance)        │
                    │   25 tickers, 5 sectors, 2010–now   │
                    └──────────────┬──────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
     ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
     │  HMM Regime     │  │  Cointegration   │  │  Kalman Filter   │
     │  Detection      │  │  Screener        │  │  Hedge Ratio     │
     │  (hhm.py)       │  │  (EG + sector)   │  │  (dynamic_hedger)│
     └────────┬────────┘  └───────┬──────────┘  └────────┬─────────┘
              │                   │                       │
              └───────────────────┼───────────────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │  LSTM Pair Scorer      │
                     │  (two-head: reversion  │
                     │   + emergence)         │
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │  Weekly Recommendation │
                     │  weekly_runner.py      │
                     │  $100 dollar-neutral   │
                     └────────────────────────┘
```

---

## Project Structure

```
FYD/
├── Data_Pipeline/
│   ├── intial_data_pipeline.py          # Market data ingestion & feature engineering
│   └── black_scholes_overview.md        # Theory: GBM, Ito's Lemma, BS PDE, Greeks
├── Simulations/
│   ├── Black-Scholes/
│   │   └── black_scholes_simulator.py   # Live BS pricing, GBM simulator, Greeks, surfaces
│   ├── Brownian_Motion/
│   │   ├── simulating_brownian_motion.py
│   │   └── simulating_multidimensional_brownian_motion.py
│   └── Markov-Models/
│       ├── hhm.py                       # HMM regime detection (Baum-Welch EM, Viterbi)
│       ├── regime_hedger.py             # Regime-aware BS pricing + delta hedging (KO/PEP)
│       ├── kalman_hedger.py             # Regime-switching Kalman hedge ratio (KO/PEP)
│       ├── walk_forward.py              # Walk-forward ML model for KO/PEP spread direction
│       ├── model_validation.py          # Ablation, cross-pair, cost, bootstrap AUC tests
│       └── PairSelector/                # ← PAIRS TRADING SYSTEM (25-ticker universe)
│           ├── main.py                  # Walk-forward backtest entry point (51 folds)
│           ├── cointegration_screener.py# Engle-Granger screen + sector filter
│           ├── dynamic_hedger.py        # Kalman hedge ratio + OLS z-score signals + P&L
│           ├── bs_vol_filter.py         # IV rank vol gate (blocks entries at high IV)
│           ├── pair_universe.py         # Data download wrapper
│           ├── performance_report.py    # Sharpe, drawdown, fold-level report
│           ├── current_pairs.py         # Standalone: today's live pair recommendation
│           ├── debug_screener.py        # Diagnostic: per-filter funnel breakdown
│           ├── _sanity_fold.py          # Single-fold quick test
│           ├── lstm_dataset.py          # Feature engineering + reversion/emergence labels
│           ├── lstm_model.py            # Two-head LSTM (~23k params, CPU-optimized)
│           ├── lstm_trainer.py          # Training pipeline (BCEWithLogitsLoss, early stopping)
│           ├── lstm_predictor.py        # Inference: score all same-sector pairs
│           ├── weekly_runner.py         # End-to-end weekly recommendation + figure
│           └── models/
│               └── lstm_pair_scorer.pt  # Trained model weights (after running lstm_trainer.py)
├── Summarys/
│   ├── first_meeting_finding.md         # Stochastic processes, SDEs, BS foundations
│   └── second_meeting_finding.md        # Ito integrals, filtration, martingales
└── Sources/Literature/                  # Academic references (not tracked)
```

---

## Modules

### 1. HMM Regime Detection (`Simulations/Markov-Models/hhm.py`)

Detects latent market regimes from an equal-weighted portfolio of the 25-ticker universe using the Baum-Welch EM algorithm. Outputs a daily binary state: **0 = risk-on (bull)**, **1 = risk-off (bear)**. Regime state is fed into both the volatility gate and the LSTM feature matrix.

> **Note:** `hhm.py` has top-level execution code and cannot be imported directly. The rest of the system uses `regime_hedger.py` which wraps the same Baum-Welch functions with an `if __name__ == '__main__'` guard.

---

### 2. KO/PEP Prototype (`Simulations/Markov-Models/`)

Before scaling to 25 tickers, the core pipeline was prototyped on a single pair (KO / PEP). These four files document that progression:

| File | What it does |
|---|---|
| `regime_hedger.py` | Fits Gaussian HMM on weekly returns → bull/bear σ, μ, transition matrix. Prices ATM protective puts at regime-specific vols and simulates regime-switching GBM vs constant-vol GBM. |
| `kalman_hedger.py` | Regime-switching Kalman filter: slow process noise Q in bull, fast Q in bear. Outputs time-varying β(t) with 95% CI, spread z-score signals, and cumulative P&L vs fixed OLS beta. |
| `walk_forward.py` | Walk-forward predictive model (Logistic Regression + MLP) on 10 Kalman/HMM features. Strict time-ordered 70/30 split — no data leakage. Predicts spread direction 5 days ahead. |
| `model_validation.py` | Four robustness tests: (1) z-score ablation, (2) cross-pair generalisation (XOM/CVX, JPM/BAC), (3) transaction cost sensitivity, (4) block bootstrap AUC confidence interval. |

These prototypes informed the design of the full 25-ticker `PairSelector` system.

---

### 3. Cointegration Screener (`PairSelector/cointegration_screener.py`)

Screens all same-sector pairs for statistical mean-reversion using a multi-stage filter:

| Stage | Filter | Threshold |
|---|---|---|
| 1 | Same economic sector | Hard filter |
| 2 | Rolling 60-day log-return correlation | ≥ 0.40 |
| 3 | Engle-Granger p-value | < 0.15 |
| 4 | AR(1) coefficient \|φ\| | ≥ 0.003 |
| 5 | Half-life of mean reversion | 5–500 days |

**Sector universe (25 tickers, 5 sectors):**
| Sector | Tickers |
|---|---|
| Consumer Staples | KO, PEP, MCD, YUM, SBUX |
| Energy | XOM, CVX, COP, SLB, BKR |
| Financials | JPM, BAC, WFC, GS, MS |
| Healthcare | JNJ, PFE, MRK, ABT, BMY |
| Utilities | NEE, DUK, SO, D, AEP |

**Critical implementation detail:** The OLS spread must subtract *both* the intercept (α) and slope (β):
```python
spread = log(Y) - β·log(X) - α
```
Omitting α biases the AR(1) coefficient φ → 0 (infinite half-life) for all pairs.

---

### 4. Dynamic Hedger (`PairSelector/dynamic_hedger.py`)

Runs the full walk-forward backtest engine:

- **Kalman filter** on log prices → time-varying hedge ratio β(t)
- **OLS cointegrating spread** (fixed training α, β) → z-score signal series
- **Training-period z-score stats** (not rolling 20-day) → prevents spurious signals on short OOS windows
- **State machine signals:** LONG spread at z < −2, SHORT spread at z > +2, exit at |z| < 0.5, stop-loss at |z| > 3.5
- **P&L formula:** `signal[t−1] × (dlogY[t] − β[t−1] × dlogX[t]) × notional`

**Walk-forward structure:**
- Training window: 504 trading days (2 years)
- Test window: 63 trading days (1 quarter)
- Total folds: ~51 over 2010–present

**Sanity-check result** (2012–2014 single fold):
- Fold Sharpe: **2.48** | Avg hold: **7.9 days** | Cum P&L: **+$277.87**

---

### 5. Vol Gate (`PairSelector/bs_vol_filter.py`)

Blocks new pair entries when IV rank > 0.80 (top 20% of realized vol history). Prevents entering mean-reversion trades during volatility spikes when spreads are most likely to blow out.

---

### 6. LSTM Pair Scorer (`PairSelector/lstm_*.py`)

A two-head LSTM that learns to predict:
- **Reversion head:** P(spread reverts to zero within 5 days) — guides entry timing
- **Emergence head:** P(non-cointegrated pair becomes cointegrated within 63 days) — guides watch list

**Architecture:**
```
Input (batch, 20, 12) → LSTM(hidden=64) → FC(32, ReLU) → [head_rev(16→1), head_emer(16→1)]
Output: (batch, 2) raw logits   ~23k parameters   trains in <5 min on CPU
```

**12 features per timestep:**
z-score, z-momentum, spread vol (20d), Kalman β, β-change, IV rank leg1, IV rank leg2, rolling correlation (60d), half-life, HMM regime, days since zero-cross, rolling EG p-value

**Training:** BCEWithLogitsLoss with class-imbalance pos_weight, ReduceLROnPlateau scheduler, early stopping patience=10, 80/20 train/val split.

**Composite score:**
```
composite = 0.7 × P(reversion) + 0.3 × P(emergence)   [if cointegrated]
composite = 0.2 × P(reversion) + 0.8 × P(emergence)   [if emerging]
```

To train: `python lstm_trainer.py` (target AUC > 0.52 on both heads)

---

### 7. Weekly Runner (`PairSelector/weekly_runner.py`)

End-to-end pipeline run every Monday morning:

```bash
python weekly_runner.py
```

1. Downloads 2+ years of data through today
2. Computes HMM regime series
3. Computes Kalman betas for all same-sector pairs
4. Loads trained LSTM and scores all pairs
5. Prints plain-English trade recommendation to terminal
6. Saves `weekly_recommendation.png` (3-panel: LSTM ranking, spread z-score, watch list)

**Sizing:** $100 total, $50 per leg, dollar-neutral, fractional shares. Exit: |z| < 0.5 OR 5 trading days OR |z| > 3.5.

---

### 8. Current Pairs (`PairSelector/current_pairs.py`)

Standalone script that generates a snapshot of today's live pair recommendations without running the full backtest:

```bash
python current_pairs.py
```

Outputs `current_pairs_recommendation.png` — normalized price chart + z-score with BUY/SELL/HOLD signal.

---

## Mathematical Foundation

1. **Stochastic Processes / Random Walks** → foundational probability framework
2. **Brownian Motion** → $dW_t \sim \mathcal{N}(0, dt)$, continuous limit of random walk
3. **Geometric Brownian Motion** → $dS = \mu S\, dt + \sigma S\, dW_t$
4. **Ito's Lemma** → stochastic chain rule; adds $\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}$ correction
5. **Black-Scholes PDE** → riskless hedging; drift μ drops out under risk-neutral measure
6. **Hidden Markov Models** → Baum-Welch EM for regime detection; time-varying σ and μ per state
7. **Engle-Granger Cointegration** → two-step OLS test for stationary linear combination of log prices
8. **Kalman Filter** → online Bayesian update of hedge ratio β(t) as prices evolve
9. **AR(1) Mean Reversion** → half-life $= -\ln(2)/\ln(|\varphi|)$ where $\Delta s_t = \varphi s_{t-1} + \epsilon_t$
10. **LSTM Neural Networks** → sequence model for non-linear regime + spread feature interactions

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib scipy yfinance hmmlearn statsmodels torch scikit-learn

# 2. Run single-fold sanity check
python PairSelector/_sanity_fold.py

# 3. Train LSTM (one-time, ~5 min on CPU)
python PairSelector/lstm_trainer.py

# 4. Run full walk-forward backtest (~30 min)
python PairSelector/main.py

# 5. Get this week's trade recommendation
python PairSelector/weekly_runner.py
```

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
yfinance
hmmlearn
statsmodels
torch          # CPU build: pip install torch --index-url https://download.pytorch.org/whl/cpu
scikit-learn
```

---

## Academic References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Øksendal, B. *Stochastic Differential Equations* (6th ed.)
- Merton, R.C. (1971). *Optimum Consumption and Portfolio Rules in a Continuous-Time Model*
- Ang, A. & Bekaert, G. *International Asset Allocation with Regime Switches*
- Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and Error Correction*
- Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems*
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
