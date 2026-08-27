import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

from backend import analyze_portfolio, run_custom_stress_test


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PortIQ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(120,65,190,0.16), transparent 25%),
            #080f1d;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #07111f;
        border-right: 1px solid #1b2940;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #121d30, #0d1728);
        border: 1px solid #1e2c45;
        padding: 16px;
        border-radius: 14px;
        min-height: 105px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.20);
    }

    div[data-testid="stMetricLabel"] {
        color: #8d9ab5;
        font-size: 13px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 24px !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #1e2c45 !important;
        border-radius: 14px !important;
        background: #0e1829;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 10px;
        background: linear-gradient(90deg, #9b3fe8, #642de3);
        color: white;
        font-weight: 700;
        min-height: 44px;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #334461;
        background: #121d30;
        color: white;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background: #101c30 !important;
        border-color: #263856 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e2c45;
        border-radius: 12px;
        overflow: hidden;
    }

    h1, h2, h3 {
        color: #f5f7ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COMPANY NAME -> TICKER SEARCH
# Frontend helper only. Finance calculations remain in backend.py
# ============================================================

@st.cache_data(ttl=3600)
def search_company(company_name):
    text = company_name.strip()

    if not text:
        return None

    # Try direct ticker
    try:
        test = yf.download(
            text.upper(),
            period="5d",
            auto_adjust=True,
            progress=False,
        )
        if not test.empty:
            return text.upper()
    except Exception:
        pass

    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"

        response = requests.get(
            url,
            params={
                "q": text,
                "quotesCount": 10,
                "newsCount": 0,
            },
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10,
        )

        response.raise_for_status()
        results = response.json().get("quotes", [])

        # Prefer NSE
        for item in results:
            symbol = item.get("symbol", "")
            quote_type = item.get("quoteType", "")
            if (
                symbol.endswith(".NS")
                and quote_type in ["EQUITY", "ETF"]
            ):
                return symbol

        # Fallback
        for item in results:
            symbol = item.get("symbol", "")
            quote_type = item.get("quoteType", "")
            if symbol and quote_type in ["EQUITY", "ETF"]:
                return symbol

    except Exception:
        return None

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("PORTFOLIO ANALYTICS")
st.sidebar.caption("Frontend connected to backend.py")
st.sidebar.divider()

company_input = st.sidebar.text_area(
    "Companies / ETFs",
    value=(
        "Reliance Industries\n"
        "HDFC Bank\n"
        "TCS\n"
        "Infosys\n"
        "GoldBeES"
    ),
    height=160,
    help="Enter one company name or Yahoo Finance ticker per line.",
)

investment_amount = st.sidebar.number_input(
    "Investment Amount (₹)",
    min_value=1_000,
    value=100_000,
    step=10_000,
)

analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    ["1y", "3y", "5y", "10y"],
    index=1,
)

risk_free_rate = (
    st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.25,
    )
    / 100
)

w1, w2 = st.sidebar.columns(2)

with w1:
    min_weight_pct = st.number_input(
        "Min Weight %",
        min_value=0,
        max_value=40,
        value=5,
    )

with w2:
    max_weight_pct = st.number_input(
        "Max Weight %",
        min_value=10,
        max_value=100,
        value=40,
    )

min_weight = min_weight_pct / 100
max_weight = max_weight_pct / 100

analyze_button = st.sidebar.button(
    "ANALYZE PORTFOLIO"
)

st.sidebar.divider()
st.sidebar.caption("Market data: Yahoo Finance")


# ============================================================
# HEADER
# ============================================================

st.title("Portfolio Overview")

if not analyze_button:
    with st.container(border=True):
        st.subheader("Build your portfolio")
        st.write(
            "Enter at least two companies or ETFs in the sidebar "
            "and click **ANALYZE PORTFOLIO**."
        )
    st.stop()


# ============================================================
# RESOLVE TICKERS
# ============================================================

company_names = list(
    dict.fromkeys(
        [
            x.strip()
            for x in company_input.splitlines()
            if x.strip()
        ]
    )
)

if len(company_names) < 2:
    st.error("Please enter at least two companies or ETFs.")
    st.stop()


ticker_map = {}

with st.status(
    "Resolving company names...",
    expanded=False,
) as status:

    for company in company_names:
        ticker = search_company(company)

        if ticker:
            ticker_map[company] = ticker

        status.write(
            f"{company} → {ticker if ticker else 'Not found'}"
        )

    status.update(
        label="Company search completed",
        state="complete",
    )


tickers = list(
    dict.fromkeys(
        ticker_map.values()
    )
)

if len(tickers) < 2:
    st.error("At least two valid securities are required.")
    st.stop()


# ============================================================
# CALL THE REAL BACKEND
# ============================================================

try:
    with st.spinner(
        "Running portfolio model from backend.py..."
    ):
        result = analyze_portfolio(
            tickers=tickers,
            investment_amount=investment_amount,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            period=analysis_period,
        )

except Exception as exc:
    st.error(str(exc))
    st.stop()


# ============================================================
# RESULTS FROM BACKEND
# ============================================================

allocation = result["allocation"]
returns = result["returns"]
optimal_weights = result["optimal_weights"]
valid_tickers = result["valid_tickers"]

portfolio_return = result["portfolio_return"]
portfolio_volatility = result["portfolio_volatility"]
sharpe_ratio = result["sharpe_ratio"]
sortino_ratio = result["sortino_ratio"]

portfolio_value = result["portfolio_value"]
current_portfolio_value = result["current_portfolio_value"]
total_return = result["total_return"]
drawdown = result["drawdown"]
maximum_drawdown = result["maximum_drawdown"]

var_95 = result["var_95"]
expected_shortfall = result["expected_shortfall"]

beta = result["beta"]
alpha = result["alpha"]
active_return = result["active_return"]
tracking_error = result["tracking_error"]
information_ratio = result["information_ratio"]

portfolio_benchmark_return = result["portfolio_benchmark_return"]
nifty_return = result["nifty_return"]
comparison_growth = result["comparison_growth"]


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3 = st.columns(3)

with k1:
    st.metric(
        "Portfolio Value",
        f"₹{current_portfolio_value:,.0f}",
    )

with k2:
    st.metric(
        "Total Return",
        f"{total_return:.2%}",
    )

with k3:
    st.metric(
        "Expected Annual Return",
        f"{portfolio_return:.2%}",
    )


k4, k5, k6 = st.columns(3)

with k4:
    st.metric(
        "Annual Volatility",
        f"{portfolio_volatility:.2%}",
    )

with k5:
    st.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}",
    )

with k6:
    st.metric(
        "Maximum Drawdown",
        f"{maximum_drawdown:.2%}",
    )


# ============================================================
# PERFORMANCE + RISK
# ============================================================

left_main, right_main = st.columns(
    [2.7, 1.3]
)

with left_main:
    with st.container(border=True):
        st.subheader("Portfolio vs NIFTY 50")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=comparison_growth.index,
                y=comparison_growth["Portfolio"],
                name="Optimized Portfolio",
                mode="lines",
                line=dict(
                    color="#9d4edd",
                    width=2.5,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=comparison_growth.index,
                y=comparison_growth["NIFTY50"],
                name="NIFTY 50",
                mode="lines",
                line=dict(
                    color="#16d4d9",
                    width=2,
                ),
            )
        )

        fig.update_layout(
            height=400,
            paper_bgcolor="#0e1829",
            plot_bgcolor="#0e1829",
            font=dict(color="#98a6c1"),
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
            legend=dict(orientation="h"),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                tickprefix="₹",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


with right_main:
    with st.container(border=True):
        st.subheader("Risk Overview")

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Beta",
                f"{beta:.2f}" if pd.notna(beta) else "N/A",
            )

        with r2:
            st.metric(
                "Alpha",
                f"{alpha:.2%}" if pd.notna(alpha) else "N/A",
            )

        r3, r4 = st.columns(2)

        with r3:
            st.metric(
                "95% VaR",
                f"{var_95:.2%}",
            )

        with r4:
            st.metric(
                "Expected Shortfall",
                f"{expected_shortfall:.2%}",
            )

        r5, r6 = st.columns(2)

        with r5:
            st.metric(
                "Sortino Ratio",
                f"{sortino_ratio:.2f}"
                if pd.notna(sortino_ratio)
                else "N/A",
            )

        with r6:
            st.metric(
                "Information Ratio",
                f"{information_ratio:.2f}"
                if pd.notna(information_ratio)
                else "N/A",
            )


# ============================================================
# ALLOCATION + BENCHMARK
# ============================================================

allocation_col, benchmark_col = st.columns(
    [1.2, 1.8]
)

with allocation_col:
    with st.container(border=True):
        st.subheader("Allocation Summary")

        donut = go.Figure(
            data=[
                go.Pie(
                    labels=allocation["Asset"],
                    values=allocation["Optimal Weight (%)"],
                    hole=0.65,
                    textinfo="percent",
                    hovertemplate=(
                        "%{label}<br>"
                        "Weight: %{value:.2f}%"
                        "<extra></extra>"
                    ),
                )
            ]
        )

        donut.update_layout(
            height=340,
            paper_bgcolor="#0e1829",
            plot_bgcolor="#0e1829",
            font=dict(color="#aab5ca"),
            margin=dict(l=5, r=5, t=10, b=5),
            legend=dict(orientation="h"),
        )

        st.plotly_chart(
            donut,
            use_container_width=True,
        )


with benchmark_col:
    with st.container(border=True):
        st.subheader("Benchmark Summary")

        b1, b2 = st.columns(2)

        with b1:
            st.metric(
                "Portfolio Return",
                f"{portfolio_benchmark_return:.2%}",
            )

        with b2:
            st.metric(
                "NIFTY 50 Return",
                f"{nifty_return:.2%}",
            )

        b3, b4 = st.columns(2)

        with b3:
            st.metric(
                "Active Return",
                f"{active_return:.2%}",
            )

        with b4:
            st.metric(
                "Tracking Error",
                f"{tracking_error:.2%}",
            )


# ============================================================
# TOP HOLDINGS
# ============================================================

with st.container(border=True):
    st.subheader("Top Holdings")

    st.dataframe(
        allocation.sort_values(
            "Optimal Weight (%)",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TABS
# ============================================================

performance_tab, risk_tab, correlation_tab, stress_tab, data_tab = st.tabs(
    [
        "Performance",
        "Risk Analysis",
        "Correlation",
        "Stress Test",
        "Portfolio Data",
    ]
)


with performance_tab:
    with st.container(border=True):
        st.subheader("Portfolio Growth")

        growth_fig = go.Figure()

        growth_fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                fill="tozeroy",
                mode="lines",
                line=dict(
                    color="#14ced1",
                    width=2,
                ),
                fillcolor="rgba(20,206,209,0.08)",
            )
        )

        growth_fig.update_layout(
            height=380,
            paper_bgcolor="#0e1829",
            plot_bgcolor="#0e1829",
            font=dict(color="#9ba8c2"),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                tickprefix="₹",
            ),
        )

        st.plotly_chart(
            growth_fig,
            use_container_width=True,
        )


with risk_tab:
    with st.container(border=True):
        st.subheader("Historical Drawdown")

        drawdown_fig = go.Figure()

        drawdown_fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                fill="tozeroy",
                mode="lines",
                line=dict(
                    color="#ff3e63",
                    width=2,
                ),
                fillcolor="rgba(255,62,99,0.10)",
            )
        )

        drawdown_fig.update_layout(
            height=380,
            paper_bgcolor="#0e1829",
            plot_bgcolor="#0e1829",
            font=dict(color="#9ba8c2"),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                title="Drawdown (%)",
                gridcolor="rgba(255,255,255,0.05)",
            ),
        )

        st.plotly_chart(
            drawdown_fig,
            use_container_width=True,
        )


with correlation_tab:
    with st.container(border=True):
        st.subheader("Asset Correlation Matrix")

        correlation = returns.corr()

        corr_fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[
                "#111a31",
                "#14c6cf",
                "#a53deb",
            ],
        )

        corr_fig.update_layout(
            height=500,
            paper_bgcolor="#0e1829",
            plot_bgcolor="#0e1829",
            font=dict(color="#aab5ca"),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(
            corr_fig,
            use_container_width=True,
        )


with stress_tab:
    st.subheader("Portfolio Stress Test")

    sleft, sright = st.columns(2)

    with sleft:
        equity_shock_pct = st.slider(
            "Equity Shock (%)",
            min_value=-50,
            max_value=30,
            value=-20,
        )

    with sright:
        gold_shock_pct = st.slider(
            "Gold Shock (%)",
            min_value=-30,
            max_value=40,
            value=10,
        )

    custom_stress = run_custom_stress_test(
        optimal_weights=optimal_weights,
        valid_tickers=valid_tickers,
        investment_amount=investment_amount,
        equity_shock=equity_shock_pct / 100,
        gold_shock=gold_shock_pct / 100,
    )

    s1, s2 = st.columns(2)

    with s1:
        st.metric(
            "Portfolio Impact",
            f"{custom_stress['stress_impact']:.2%}",
        )

    with s2:
        st.metric(
            "Value Before",
            f"₹{investment_amount:,.0f}",
        )

    s3, s4 = st.columns(2)

    with s3:
        st.metric(
            "Value After",
            f"₹{custom_stress['stressed_value']:,.0f}",
        )

    with s4:
        st.metric(
            "Estimated Loss / Gain",
            f"₹{custom_stress['loss_gain']:,.0f}",
        )


with data_tab:
    st.subheader("Resolved Securities")

    ticker_table = pd.DataFrame(
        {
            "Company Input": list(ticker_map.keys()),
            "Ticker": list(ticker_map.values()),
        }
    )

    st.dataframe(
        ticker_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Backend Output - Optimized Portfolio")

    st.dataframe(
        allocation,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EXPORT
# ============================================================

st.divider()

csv_data = allocation.to_csv(
    index=False
)

st.download_button(
    "Download Portfolio CSV",
    csv_data,
    file_name="optimized_portfolio.csv",
    mime="text/csv",
)

st.success(
    "Connected analysis completed successfully."
)

st.caption(
    "Disclaimer: This model uses historical market data "
    "and portfolio optimization assumptions. Past performance "
    "does not guarantee future results. For educational and "
    "analytical purposes only."
)
