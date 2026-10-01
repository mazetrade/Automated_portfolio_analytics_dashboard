import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for a list of tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices.dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns from prices."""
    return prices.pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute daily portfolio returns from asset returns and weights."""
    w = pd.Series(weights)
    w = w / w.sum()
    return returns[w.index].dot(w)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "SPY"]
    prices = download_prices(tickers, "2020-01-01", "2025-12-31")
    rets = compute_returns(prices)
    port = portfolio_returns(rets, {"AAPL": 0.4, "MSFT": 0.3, "JPM": 0.3})

    print(prices.tail())
    print(rets.describe())
    print(port.head())