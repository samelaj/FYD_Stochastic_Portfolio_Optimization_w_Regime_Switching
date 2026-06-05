import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


# ── Real Market Data ─────────────────────────────────────────────────────────

def fetch_risk_free_rate() -> float:
    """3-month T-bill annualised rate from yfinance (^IRX), as a decimal."""
    try:
        irx = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        rate = float(irx["Close"].dropna().iloc[-1]) / 100.0
        return rate
    except Exception:
        return 0.05   # fallback: 5%


def market_inputs(ticker: str, vol_window: int = 21) -> dict:
    """
    Pull live market data for a ticker and return the inputs BS needs.

    Parameters
    ----------
    ticker     : e.g. "SPY", "QQQ", "GLD"
    vol_window : trading days for realised volatility (default 21 ~ 1 month)

    Returns
    -------
    dict with keys: ticker, S, sigma, r, prices, log_returns
    """
    raw = yf.download(ticker, start="2020-01-01",
                      end=pd.Timestamp.today().strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    # yfinance sometimes returns a DataFrame with ticker as column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    price_series = close.dropna().ffill().bfill()

    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    S     = float(price_series.iloc[-1])
    sigma = float(log_returns.iloc[-vol_window:].std() * np.sqrt(252))
    r     = fetch_risk_free_rate()

    print(f"\n[Market Inputs] {ticker}")
    print(f"  Current price S : ${S:.2f}")
    print(f"  Realised vol  sigma ({vol_window}d) : {sigma:.2%}")
    print(f"  Risk-free rate r : {r:.2%}")

    return {
        "ticker":      ticker,
        "S":           S,
        "sigma":       sigma,
        "r":           r,
        "prices":      price_series,
        "log_returns": log_returns,
    }


# ── Pricing ───────────────────────────────────────────────────────────────────

def _d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, r, sigma, T):
    return _d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

def bs_call(S, K, r, sigma, T):
    """European call: C = S*N(d1) - K*e^(-rT)*N(d2)"""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1(S, K, r, sigma, T), _d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    """European put via put-call parity: P = C - S + K*e^(-rT)"""
    return bs_call(S, K, r, sigma, T) - S + K * np.exp(-r * T)


# ── Greeks ────────────────────────────────────────────────────────────────────

def bs_greeks(S, K, r, sigma, T):
    """
    Delta  dC/dS  — hedge ratio, how much the option moves per $1 in stock
    Gamma  d2C/dS2 — rate of change of delta, key for dynamic hedging
    Vega   dC/dσ  — sensitivity to vol (important: σ changes per regime later)
    Theta  dC/dt  — time decay per calendar day
    Rho    dC/dr  — sensitivity to interest rate
    """
    d1, d2  = _d1(S, K, r, sigma, T), _d2(S, K, r, sigma, T)
    sqrt_T  = np.sqrt(T)

    delta_call = norm.cdf(d1)
    delta_put  = delta_call - 1.0
    gamma      = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega       = S * norm.pdf(d1) * sqrt_T / 100           # per 1% vol move
    theta_call = (
        -S * norm.pdf(d1) * sigma / (2 * sqrt_T)
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    ) / 365                                                 # per calendar day
    rho_call   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    return {
        "delta_call": delta_call,
        "delta_put":  delta_put,
        "gamma":      gamma,
        "vega":       vega,
        "theta_call": theta_call,
        "rho_call":   rho_call,
    }


# ── GBM Simulator ─────────────────────────────────────────────────────────────

def simulate_gbm(S0, mu, sigma, T, n_steps=252, n_paths=500, seed=42):
    """
    Simulate Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW_t

    Exact discretization:
      S(t + dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
      where Z ~ N(0,1)

    Returns array of shape (n_paths, n_steps+1).
    """
    rng        = np.random.default_rng(seed)
    dt         = T / n_steps
    Z          = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    paths       = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
    return paths


def mc_call_price(S0, K, r, sigma, T, n_paths=100_000, seed=42):
    """Monte Carlo call price — should match bs_call() closely."""
    rng     = np.random.default_rng(seed)
    Z       = rng.standard_normal(n_paths)
    ST      = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * payoffs.mean()


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_paths(paths, T, title="GBM Simulated Paths", n_show=30):
    t = np.linspace(0, T, paths.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, paths[:n_show].T, alpha=0.4, linewidth=0.7)
    ax.plot(t, np.median(paths, axis=0), color="black", linewidth=2, label="Median")
    ax.fill_between(t,
        np.percentile(paths,  5, axis=0),
        np.percentile(paths, 95, axis=0),
        alpha=0.15, color="steelblue", label="5–95th percentile")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price ($)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_option_surface(S0, r, sigma, option="call"):
    K_range = np.linspace(S0 * 0.6, S0 * 1.4, 50)
    T_range = np.linspace(0.02, 2.0, 50)
    K_grid, T_grid = np.meshgrid(K_range, T_range)

    fn = bs_call if option == "call" else bs_put
    Z  = np.vectorize(lambda k, t: fn(S0, k, r, sigma, t))(K_grid, T_grid)

    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(K_grid, T_grid, Z, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Strike K ($)")
    ax.set_ylabel("Time to Expiry (years)")
    ax.set_zlabel("Option Price ($)")
    ax.set_title(
        f"Black-Scholes {option.capitalize()} Surface  "
        f"(S={S0:.0f}, sigma={sigma:.1%}, r={r:.1%})"
    )
    fig.colorbar(surf, shrink=0.5)
    plt.tight_layout()
    return fig


def plot_greeks(S0, K, r, sigma, T):
    S_range    = np.linspace(S0 * 0.5, S0 * 1.5, 300)
    greek_vals = [bs_greeks(s, K, r, sigma, T) for s in S_range]

    keys = [
        ("delta_call", "Delta (Call)",       "Hedge ratio (0 to 1)"),
        ("delta_put",  "Delta (Put)",        "Hedge ratio (-1 to 0)"),
        ("gamma",      "Gamma",              "Rate of change of Delta (1 / $)"),
        ("vega",       "Vega (per 1% vol)",  "Option price change per 1% vol move ($)"),
        ("theta_call", "Theta (Call)",       "Option price decay per day ($)"),
        ("rho_call",   "Rho (Call)",         "Option price change per 1% rate move ($)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, label, ylabel) in zip(axes.flat, keys):
        ax.plot(S_range, [g[key] for g in greek_vals], linewidth=1.8)
        ax.axvline(K,  linestyle="--", color="grey", alpha=0.6, label=f"K={K}")
        ax.axvline(S0, linestyle=":",  color="red",  alpha=0.6, label=f"S={S0:.0f}")
        ax.set_title(label)
        ax.set_xlabel("Stock Price S ($)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    plt.suptitle(
        f"Black-Scholes Greeks  (K={K}, S={S0}, σ={sigma:.0%}, r={r:.1%}, T={T}y)",
        y=1.01,
    )
    plt.tight_layout()
    return fig


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Black-Scholes Simulator — Live Market Data")
    print("=" * 50)

    # ── Pull real inputs from the data pipeline
    mkt   = market_inputs("SPY", vol_window=21)
    S0    = mkt["S"]
    sigma = mkt["sigma"]
    r     = mkt["r"]

    # ── User-defined contract parameters
    K  = round(S0)   # at-the-money strike
    T  = 0.25        # 3 months to expiry
    mu = 0.08        # expected drift for GBM paths (not used in BS price)

    # ── Pricing
    call   = bs_call(S0, K, r, sigma, T)
    put    = bs_put(S0,  K, r, sigma, T)
    mc_val = mc_call_price(S0, K, r, sigma, T)
    parity = call - put - S0 + K * np.exp(-r * T)

    print(f"\nPricing  (S={S0:.2f}, K={K}, r={r:.2%}, sigma={sigma:.2%}, T={T}y)")
    print(f"  BS Call  : ${call:.4f}")
    print(f"  BS Put   : ${put:.4f}")
    print(f"  MC Call  : ${mc_val:.4f}  <- should match BS Call")
    print(f"  Put-call parity residual: {parity:.2e}")

    g = bs_greeks(S0, K, r, sigma, T)
    print(f"\nGreeks at S={S0:.2f}, K={K}:")
    for name, val in g.items():
        print(f"  {name:<15s}: {val:+.6f}")

    # ── Simulate GBM paths calibrated to real vol
    paths = simulate_gbm(S0, mu, sigma, T, n_steps=63, n_paths=500)

    fig1 = plot_paths(paths, T,
        f"SPY GBM Paths  (sigma={sigma:.1%}, T={T}y)")
    fig2 = plot_option_surface(S0, r, sigma, option="call")
    fig3 = plot_greeks(S0, K, r, sigma, T)

    plt.show()
