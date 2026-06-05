# Stochastic Portfolio Optimization with Regime Switching

**Final Year Design Project — Wentworth Institute of Technology**
**Jake Samela**

---

## Overview

This project builds a regime-aware portfolio optimization framework using Hidden Markov Models (HMM) to detect distinct market states (e.g. bull, bear, high-volatility) and applies Black-Scholes option pricing with regime-specific volatility and drift parameters. The core motivation: the Black-Scholes constant-volatility assumption breaks down in practice — markets have distinct regimes, and using a single volatility figure across all conditions leads to mispriced options and poor allocation decisions.

---

## Project Structure

```
FYD/
├── Data_Pipeline/
│   ├── intial_data_pipeline.py       # Market data ingestion & feature engineering
│   └── black_scholes_overview.md     # Theory: GBM, Ito's Lemma, BS PDE, Greeks
├── Simulations/
│   ├── Black-Scholes/
│   │   └── black_scholes_simulator.py    # Live BS pricing, GBM simulator, Greeks, surfaces
│   ├── Brownian_Motion/
│   │   ├── simulating_brownian_motion.py              # Standard Brownian motion
│   │   └── simulating_multidimensional_brownian_motion.py  # Correlated multi-asset BM
│   └── Markov-Models/
│       └── hhm.py                        # Hidden Markov Model regime detection
├── Summarys/
│   ├── first_meeting_finding.md     # Stochastic processes, SDEs, Black-Scholes foundations
│   └── second_meeting_finding.md    # Ito integrals, filtration, martingales
└── Sources/Literature/              # Academic references (not tracked)
```

---

## Modules

### Data Pipeline (`Data_Pipeline/intial_data_pipeline.py`)

Fetches 15+ years of daily market data (2010–present) across four asset classes using `yfinance` and engineers the features that feed the HMM.

**Asset Universe:**
| Class | Tickers |
|---|---|
| Equities | SPY, QQQ, IWM, EFA, EEM |
| Fixed Income | TLT, IEF, LQD, HYG |
| Commodities | GLD, SLV, USO, DBA |
| Volatility | ^VIX |

**Engineered Features:**
- `mkt_return` — daily log return of SPY
- `mkt_vol` — 21-day rolling annualized volatility
- `vol_of_vol` — volatility of volatility (regime instability signal)
- `avg_corr` — rolling average cross-asset correlation
- `vix` — cumulative VIX log return

---

### Black-Scholes Simulator (`Simulations/Black-Scholes/black_scholes_simulator.py`)

A fully live Black-Scholes pricing engine calibrated to real market data. Pulls current prices and volatility directly from `yfinance` and the 3-month T-bill rate (^IRX) for the risk-free rate.

**Capabilities:**
- European call/put pricing via closed-form Black-Scholes
- Monte Carlo call price (verifies analytical solution)
- Put-call parity check
- Full Greeks: Delta, Gamma, Vega, Theta, Rho
- GBM path simulation (500 paths, exact discretization)
- 3D option price surface (strike × time to expiry)
- Greeks sensitivity plots across stock price range

**Black-Scholes Formula:**

$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**GBM (Exact Discretization):**

$$S(t + \Delta t) = S(t) \cdot \exp\!\left(\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right), \quad Z \sim \mathcal{N}(0,1)$$

---

### Brownian Motion Simulations (`Simulations/Brownian_Motion/`)

Foundational stochastic process implementations that underpin the entire modeling framework.

- **`simulating_brownian_motion.py`** — Simulates 5 independent standard Brownian motion paths (1000 steps, dt = 0.01)
- **`simulating_multidimensional_brownian_motion.py`** — Correlated multi-asset Brownian motion using a volatility matrix; validates empirical vs. theoretical covariance

---

### Hidden Markov Model (`Simulations/Markov-Models/hhm.py`)

Detects latent market regimes (e.g. low-vol bull, high-vol bear) from the engineered feature set. Each detected regime carries its own volatility (σ) and drift (μ), which are then passed into the Black-Scholes and portfolio optimization layers for regime-aware pricing and allocation.

---

## Mathematical Foundation

The project builds on the following theoretical chain:

1. **Stochastic Processes / Random Walks** → foundational probability framework
2. **Brownian Motion** → $dW_t \sim \mathcal{N}(0, dt)$, continuous limit of random walk
3. **Geometric Brownian Motion** → $dS = \mu S\, dt + \sigma S\, dW_t$ (stock price model)
4. **Ito's Lemma** → stochastic chain rule; adds $\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}$ correction term
5. **Black-Scholes PDE** → riskless hedging portfolio eliminates drift; $\mu$ drops out (risk-neutral pricing)
6. **Hidden Markov Models** → regime detection; provides time-varying $\sigma$ and $\mu$ per state
7. **Portfolio Optimization** → allocate across asset classes using regime-specific parameters

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
yfinance
hmmlearn
```

Install with:
```bash
pip install numpy pandas matplotlib scipy yfinance hmmlearn
```

---

## Academic References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Øksendal, B. *Stochastic Differential Equations* (6th ed.)
- Merton, R.C. (1971). *Optimum Consumption and Portfolio Rules in a Continuous-Time Model*
- Ang, A. & Bekaert, G. *International Asset Allocation with Regime Switches*
