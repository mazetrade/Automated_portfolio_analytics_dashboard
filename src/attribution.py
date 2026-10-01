import numpy as np
import pandas as pd

from src.factors import factor_regression
from src.metrics import TRADING_DAYS


def _normalize(weights: dict[str, float]) -> pd.Series:
    """Turn a weights dict into a Series that sums to 1."""
    w = pd.Series(weights, dtype=float)
    return w / w.sum()

# Return contribution 

def return_contribution(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Annualized contribution of each asset to the portfolio's mean return."""
    w = _normalize(weights)
    return w * returns[w.index].mean() * TRADING_DAYS

# Risk contribution 

def risk_contribution(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Each asset's contribution to annualized portfolio volatility (Euler decomposition)."""
    w = _normalize(weights)
    cov = returns[w.index].cov() * TRADING_DAYS
    port_vol = np.sqrt(w @ cov @ w)
    marginal = cov @ w / port_vol
    return w * marginal

# Factor attribution 

def factor_attribution(returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Split annualized mean excess return into alpha + contribution of each factor."""
    model = factor_regression(returns, factors)
    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    contrib = model.params * X.mean() * TRADING_DAYS
    return contrib.rename({"const": "Alpha"})

# Test block 

if __name__ == "__main__":
    from src.data import download_prices, compute_returns, portfolio_returns
    from src.factors import load_factors
    from src.metrics import annualized_volatility

    weights = {"AAPL": 0.4, "MSFT": 0.3, "JPM": 0.3}
    prices = download_prices(list(weights) + ["SPY"], "2020-01-01", "2025-12-31")
    rets = compute_returns(prices)
    port = portfolio_returns(rets, weights)

    table = pd.DataFrame({
        "Weight (%)": _normalize(weights) * 100,
        "Return contrib (%)": return_contribution(rets, weights) * 100,
        "Risk contrib (%)": risk_contribution(rets, weights) * 100,
    })
    table["Share of risk (%)"] = table["Risk contrib (%)"] / table["Risk contrib (%)"].sum() * 100
    table.loc["Total"] = table.sum()
    print(table.round(2))

    print(f"\nCheck - portfolio arithmetic return: {port.mean() * TRADING_DAYS * 100:.2f}%")
    print(f"Check - portfolio volatility:        {annualized_volatility(port) * 100:.2f}%")

    print("\nFactor attribution (annualized % of excess return):")
    fa = factor_attribution(port, load_factors()) * 100
    fa["Total"] = fa.sum()
    print(fa.round(2))