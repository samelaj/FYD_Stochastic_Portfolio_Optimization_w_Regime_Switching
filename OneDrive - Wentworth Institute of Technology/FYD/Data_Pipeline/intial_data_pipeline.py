import yfinance as yf
import numpy as np
import pandas as pd

TICKERS = {
    "equitites" : ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "fixed_income" : ["TLT", "IEF", "LQD", "HYG"],
    "commodities" : ["GLD", "SLV", "USO", "DBA"],
    "volitility" : ["^VIX"]
}

ALL_TICKERS = [t for group in TICKERS.values() for t in group]
start_date = "2010-01-01"
end_date = "2026-05-15"

def fetch_prices(tickers:list[str], start:str, end:str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress = False, threads = True)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    thresh = int(len(prices) * 0.95)
    prices = prices.dropna(axis = 1, thresh = thresh)

    prices = prices.ffill().bfill()
    return prices


def build_features(prices: pd.DataFrame, vol_window : int = 21) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1)).dropna()

    features = pd.DataFrame(index=log_returns.index)
    # Market return/volatility
    features["mkt_return"] = log_returns["SPY"]
    features["mkt_vol"] = log_returns["SPY"].rolling(vol_window).std() * np.sqrt(252)
    features["vol_of_vol"] = features["mkt_vol"].rolling(vol_window).std()

    features["avg_corr"] = (
        log_returns.drop(columns = ["^VIX"], errors = "ignore")
    .rolling(vol_window).corr().groupby(level=0).apply(lambda m : (m.values.sum() - len(m)) / (len(m) ** 2 - len(m)))
    )

    if "^VIX" in log_returns.columns:
        features["vix"] = log_returns["^VIX"].cumsum()
    
    return features.dropna()


prices = fetch_prices(ALL_TICKERS, start_date, end_date)

log_returns = np.log(prices/ prices.shift(1)).dropna()
features = build_features(prices)



print(f"[3] Features built: {features.shape[0]} rows x {features.shape[1]} columns")
print(f"    Columns : {list(features.columns)}")
print(f"    Date range: {features.index[0].date()} → {features.index[-1].date()}")
print(f"\n    Sample (last 5 rows):")
print(features.tail())




