import io
import urllib.request
import zipfile

import pandas as pd
import statsmodels.api as sm

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# Download and parse a French data file 

def _download_french(url: str) -> pd.DataFrame:
    """Download a daily factor file from Ken French's library, as decimals."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        raw = response.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(","))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), index_col=0)

    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()]
    df.columns = df.columns.str.strip()
    return df.astype(float) / 100

# Load all factors 

def load_factors() -> pd.DataFrame:
    """Daily Fama-French 5 factors + Momentum + risk-free rate."""
    ff5 = _download_french(FF5_URL)
    mom = _download_french(MOM_URL)
    return ff5.join(mom, how="inner")

# The regression 

def factor_regression(returns: pd.Series, factors: pd.DataFrame):
    """Regress excess returns on factors; returns a fitted statsmodels OLS model."""
    data = pd.concat([returns.rename("asset"), factors], axis=1, join="inner").dropna()
    y = data["asset"] - data["RF"]
    X = sm.add_constant(data[FACTOR_COLS])
    return sm.OLS(y, X).fit(cov_type="HC3")

# Summary table for assets 

def exposures_table(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Factor loadings, annualized alpha and R² for each column of returns."""
    rows = {}
    for name in returns.columns:
        model = factor_regression(returns[name], factors)
        row = model.params.rename({"const": "Alpha (ann. %)"})
        row["Alpha (ann. %)"] *= 252 * 100
        row["Alpha t-stat"] = model.tvalues["const"]
        row["R²"] = model.rsquared
        rows[name] = row
    return pd.DataFrame(rows).T

# Test block 

if __name__ == "__main__":
    from src.data import download_prices, compute_returns, portfolio_returns

    tickers = ["AAPL", "MSFT", "JPM", "SPY"]
    prices = download_prices(tickers, "2020-01-01", "2025-12-31")
    rets = compute_returns(prices)
    rets["Portfolio"] = portfolio_returns(rets, {"AAPL": 0.4, "MSFT": 0.3, "JPM": 0.3})

    factors = load_factors()
    print(f"Factor data from {factors.index.min().date()} to {factors.index.max().date()}")

    print(exposures_table(rets, factors).round(3))
    print(factor_regression(rets["Portfolio"], factors).summary())