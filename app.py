from datetime import date
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
from src.data import download_prices, compute_returns, portfolio_returns
from src.metrics import (
    TRADING_DAYS, annualized_return, annualized_volatility, cumulative_returns,
    drawdown, max_drawdown, sharpe_ratio, sortino_ratio,
)
from src.risk import beta, var_historical, var_parametric, cvar_historical
from src.factors import load_factors, factor_regression, FACTOR_COLS
from src.attribution import return_contribution, risk_contribution, factor_attribution
 
st.set_page_config(page_title="Portfolio Analytics", page_icon="📊", layout="wide")

# Sidebar inputs 

st.sidebar.header("Portfolio settings")
 
tickers_text = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, JPM")
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
if not tickers:
    st.error("Enter at least one ticker.")
    st.stop()
 
st.sidebar.subheader("Weights (%)")
weights = {
    t: st.sidebar.number_input(t, min_value=0.0, max_value=100.0, value=100 / len(tickers),
                               step=5.0, format="%.1f", key=f"w_{t}")
    for t in tickers
}
 
benchmark = st.sidebar.text_input("Benchmark", "SPY").strip().upper()
start = st.sidebar.date_input("Start date", date(2020, 1, 1))
end = st.sidebar.date_input("End date", date.today())
rf = st.sidebar.number_input("Risk-free rate (% per year)", value=2.8, step=0.25) / 100
level = st.sidebar.slider("VaR / CVaR confidence", 0.90, 0.99, 0.95, 0.01)
window = st.sidebar.select_slider("Rolling window (days)", options=[21, 63, 126, 252], value=63)

# Input validation 

total_weight = sum(weights.values())
if total_weight == 0:
    st.error("Weights cannot all be zero.")
    st.stop()
if abs(total_weight - 100) > 0.01:
    st.sidebar.warning(f"Weights sum to {total_weight:.1f}%. They will be rescaled to 100%.")
if start >= end:
    st.error("Start date must be before end date.")
    st.stop()
    
# Data loaded (cached)

@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    return download_prices(tickers, str(start), str(end))
 
 
@st.cache_data(ttl=86400, show_spinner="Downloading Fama-French factors...")
def get_factors() -> pd.DataFrame:
    return load_factors()
 
 
all_tickers = list(dict.fromkeys(tickers + [benchmark]))
prices = load_prices(all_tickers, start, end)
 
missing = [t for t in all_tickers if t not in prices.columns or prices[t].isna().all()]
if missing:
    st.error(f"No data found for: {', '.join(missing)}. Check the ticker symbols.")
    st.stop()
 
rets = compute_returns(prices)
port = portfolio_returns(rets, weights)
bench = rets[benchmark]
compare = pd.DataFrame({"Portfolio": port, benchmark: bench})

# Header and KPI cards 

st.title("📊 Portfolio Analytics Dashboard")
st.caption(f"{rets.index.min():%d %b %Y} → {rets.index.max():%d %b %Y} · "
           f"{len(rets)} trading days · benchmark: {benchmark}")
 
p_ret, b_ret = annualized_return(port), annualized_return(bench)
p_vol, b_vol = annualized_volatility(port), annualized_volatility(bench)
p_sh, b_sh = sharpe_ratio(port, rf), sharpe_ratio(bench, rf)
p_so, b_so = sortino_ratio(port, rf), sortino_ratio(bench, rf)
p_dd, b_dd = max_drawdown(port), max_drawdown(bench)
 
k = st.columns(6)
k[0].metric("Ann. return", f"{p_ret:.2%}", f"{p_ret - b_ret:+.2%} vs bench")
k[1].metric("Volatility", f"{p_vol:.2%}", f"{p_vol - b_vol:+.2%}", delta_color="inverse")
k[2].metric("Sharpe", f"{p_sh:.2f}", f"{p_sh - b_sh:+.2f}")
k[3].metric("Sortino", f"{p_so:.2f}", f"{p_so - b_so:+.2f}")
k[4].metric("Max drawdown", f"{p_dd:.2%}", f"{p_dd - b_dd:+.2%}")
k[5].metric("Beta", f"{beta(port, bench):.2f}")
 
tab_perf, tab_risk, tab_factors, tab_attr = st.tabs(["Performance", "Risk", "Factors", "Attribution"])

# Tab 1: Performance 

with tab_perf:
    st.subheader("Cumulative return")
    fig = px.line(cumulative_returns(compare) * 100,
                  labels={"value": "Return (%)", "Date": "", "variable": ""})
    st.plotly_chart(fig)
 
    st.subheader("Drawdown")
    fig = px.line(drawdown(compare) * 100,
                  labels={"value": "Drawdown (%)", "Date": "", "variable": ""})
    fig.update_traces(fill="tozeroy")
    st.plotly_chart(fig)
 
    st.subheader("Summary statistics")
    stats = pd.DataFrame({
        "Ann. return": annualized_return(compare),
        "Volatility": annualized_volatility(compare),
        "Sharpe": sharpe_ratio(compare, rf),
        "Sortino": sortino_ratio(compare, rf),
        "Max drawdown": max_drawdown(compare),
    })
    st.dataframe(stats.style.format({
        "Ann. return": "{:.2%}", "Volatility": "{:.2%}",
        "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max drawdown": "{:.2%}",
    }))
    
# Tab 2: Risk 

with tab_risk:
    st.subheader(f"Daily risk measures at {level:.0%} confidence")
    risk_table = pd.DataFrame({
        "Beta": beta(compare, bench),
        "VaR (historical)": var_historical(compare, level),
        "VaR (parametric)": var_parametric(compare, level),
        "CVaR (historical)": cvar_historical(compare, level),
    })
    st.dataframe(risk_table.style.format("{:.2%}").format("{:.2f}", subset=["Beta"]))
 
    st.subheader("Distribution of daily portfolio returns")
    var_h, cvar_h = var_historical(port, level), cvar_historical(port, level)
    fig = px.histogram(x=port * 100, nbins=100, labels={"x": "Daily return (%)"})
    fig.add_vline(x=-var_h * 100, line_dash="dash", line_color="orange",
                  annotation_text=f"VaR {level:.0%}")
    fig.add_vline(x=-cvar_h * 100, line_dash="dash", line_color="red",
                  annotation_text=f"CVaR {level:.0%}")
    fig.update_layout(yaxis_title="Number of days")
    st.plotly_chart(fig)
 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Rolling volatility ({window} days)")
        roll_vol = compare.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
        fig = px.line(roll_vol.dropna(), labels={"value": "Volatility (%)", "Date": "", "variable": ""})
        st.plotly_chart(fig)
    with col2:
        st.subheader(f"Rolling beta ({window} days)")
        roll_beta = port.rolling(window).cov(bench) / bench.rolling(window).var()
        fig = px.line(roll_beta.dropna(), labels={"value": "Beta", "Date": ""})
        fig.add_hline(y=1, line_dash="dot")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
 
    st.subheader("Correlation matrix")
    fig = px.imshow(rets[all_tickers].corr(), text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig)
    
# Tab 3: Factors 

with tab_factors:
    try:
        factors = get_factors()
        n_common = len(port.index.intersection(factors.index))
    except Exception as e:
        factors, n_common = None, 0
        st.warning(f"Could not download factor data: {e}")
 
    if factors is not None and n_common < 60:
        st.warning("Not enough overlap with the Fama-French data (at least 60 days needed).")
    elif factors is not None:
        model = factor_regression(port, factors)
        st.caption(f"Fama-French 5 factors + Momentum · {int(model.nobs)} days · "
                   f"factor data available until {factors.index.max():%d %b %Y}")
 
        alpha_t = model.tvalues["const"]
        c = st.columns(3)
        c[0].metric("Alpha (annualized)", f"{model.params['const'] * TRADING_DAYS:.2%}")
        c[1].metric("Alpha t-stat", f"{alpha_t:.2f}",
                    "significant" if abs(alpha_t) > 2 else "not significant", delta_color="off")
        c[2].metric("R²", f"{model.rsquared:.1%}")
 
        conf = model.conf_int()
        fig = px.bar(x=FACTOR_COLS, y=model.params[FACTOR_COLS],
                     error_y=(model.params - conf[0])[FACTOR_COLS],
                     labels={"x": "", "y": "Loading"})
        fig.update_layout(title="Factor loadings with 95% confidence intervals")
        st.plotly_chart(fig)
 
        table = pd.DataFrame({"Loading": model.params, "t-stat": model.tvalues,
                              "p-value": model.pvalues}).rename(index={"const": "Alpha (daily)"})
        st.dataframe(table.style.format("{:.3f}"))
        
# Tab 4: Attribution
    
with tab_attr:
    attr = pd.DataFrame({
        "Weight": pd.Series(weights) / total_weight,
        "Return contribution": return_contribution(rets, weights),
        "Risk contribution": risk_contribution(rets, weights),
    })
    attr["Share of risk"] = attr["Risk contribution"] / attr["Risk contribution"].sum()
    attr.loc["Total"] = attr.sum()
 
    st.subheader("Contribution by asset (annualized)")
    st.dataframe(attr.style.format("{:.2%}"))
    st.caption("Return contributions use arithmetic returns, so their total is higher "
               "than the geometric (CAGR) return shown on the Performance tab.")
 
    st.subheader("Weight vs share of risk")
    fig = px.bar(attr.drop("Total")[["Weight", "Share of risk"]] * 100, barmode="group",
                 labels={"value": "%", "index": "", "variable": ""})
    st.plotly_chart(fig)
 
    if factors is not None and n_common >= 60:
        st.subheader("Where did the excess return come from?")
        fa = factor_attribution(port, factors) * 100
        values = list(fa.values) + [fa.sum()]
        fig = go.Figure(go.Waterfall(
            x=list(fa.index) + ["Total"],
            y=values,
            measure=["relative"] * len(fa) + ["total"],
            text=[f"{v:.2f}%" for v in values],
        ))
        fig.update_layout(yaxis_title="Annualized excess return (%)")
        st.plotly_chart(fig)