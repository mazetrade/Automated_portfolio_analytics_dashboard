import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Cumulative returns

def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Growth of $1 minus 1: the total return up to each date."""
    return (1 + returns).cumprod() - 1

# Annualized returns

def annualized_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Geometric annualized return (CAGR)."""
    total_growth = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return total_growth ** (1 / n_years) - 1

# Annualized volatility

def annualized_volatility(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Standard deviation of returns, scaled to a yearly figure."""
    return returns.std() * np.sqrt(TRADING_DAYS)

# Sharpe ratio 

def sharpe_ratio(returns: pd.Series | pd.DataFrame, rf: float = 0.0) -> float | pd.Series:
    """Annualized excess return per unit of total volatility."""
    excess = returns - rf / TRADING_DAYS
    return (excess.mean() * TRADING_DAYS) / (excess.std() * np.sqrt(TRADING_DAYS))

# Sortino ratio

def sortino_ratio(returns: pd.Series | pd.DataFrame, rf: float = 0.0) -> float | pd.Series:
    """Annualized excess return per unit of downside deviation."""
    excess = returns - rf / TRADING_DAYS
    downside = excess.clip(upper=0)
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return (excess.mean() * TRADING_DAYS) / downside_dev

# Drowdown series 

def drawdown(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Percentage decline from the running peak at each date."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1

# Maximum drowdown 

def max_drawdown(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Worst peak-to-trough decline over the period."""
    return drawdown(returns).min()

# Test block 

if __name__ == "__main__":
    from src.data import download_prices, compute_returns, portfolio_returns

    tickers = ["AAPL", "MSFT", "JPM", "SPY"]
    prices = download_prices(tickers, "2020-01-01", "2025-12-31")
    rets = compute_returns(prices)
    rets["Portfolio"] = portfolio_returns(rets, {"AAPL": 0.4, "MSFT": 0.3, "JPM": 0.3})

    RF = 0.025

    summary = pd.DataFrame({
        "Ann. Return (%)": annualized_return(rets) * 100,
        "Ann. Volatility (%)": annualized_volatility(rets) * 100,
        "Sharpe": sharpe_ratio(rets, rf=RF),
        "Sortino": sortino_ratio(rets, rf=RF),
        "Max Drawdown (%)": max_drawdown(rets) * 100,
    })
    print(summary.round(2))
    print(drawdown(rets["Portfolio"]).idxmin())