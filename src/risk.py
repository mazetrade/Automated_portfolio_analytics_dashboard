import pandas as pd
from scipy.stats import norm

# Beta 

def beta(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """Sensitivity to benchmark moves: Cov(asset, market) / Var(market)."""
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: col.cov(benchmark)) / benchmark.var()
    return returns.cov(benchmark) / benchmark.var()

# Historical VaR 

def var_historical(returns: pd.Series | pd.DataFrame, level: float = 0.95) -> float | pd.Series:
    """Daily loss not exceeded with probability `level`, from actual returns."""
    return -returns.quantile(1 - level)

# Parametric VaR 

def var_parametric(returns: pd.Series | pd.DataFrame, level: float = 0.95) -> float | pd.Series:
    """Daily VaR assuming returns are normally distributed."""
    z = norm.ppf(1 - level)
    return -(returns.mean() + z * returns.std())

# CVaR 

def cvar_historical(returns: pd.Series | pd.DataFrame, level: float = 0.95) -> float | pd.Series:
    """Average loss on the days that are worse than the VaR."""
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: cvar_historical(col, level))
    threshold = returns.quantile(1 - level)
    return -returns[returns <= threshold].mean()

# Test block 

if __name__ == "__main__":
    from src.data import download_prices, compute_returns, portfolio_returns

    tickers = ["AAPL", "MSFT", "JPM", "SPY"]
    prices = download_prices(tickers, "2020-01-01", "2025-12-31")
    rets = compute_returns(prices)
    rets["Portfolio"] = portfolio_returns(rets, {"AAPL": 0.4, "MSFT": 0.3, "JPM": 0.3})
    benchmark = rets["SPY"]

    LEVEL = 0.95
    summary = pd.DataFrame({
        "Beta": beta(rets, benchmark),
        "VaR Hist (%)": var_historical(rets, LEVEL) * 100,
        "VaR Param (%)": var_parametric(rets, LEVEL) * 100,
        "CVaR Hist (%)": cvar_historical(rets, LEVEL) * 100,
    })
    print(summary.round(2))