import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


def _download_close(tickers, start_date=None, end_date=None, period=None):
    """Download adjusted closing prices from Yahoo Finance."""
    kwargs = {
        "auto_adjust": True,
        "progress": False,
    }

    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start_date
        kwargs["end"] = end_date

    data = yf.download(tickers, **kwargs)

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        prices = data["Close"]
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        prices = data[["Close"]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices


def analyze_portfolio(
    tickers,
    investment_amount=100000,
    risk_free_rate=0.06,
    min_weight=0.05,
    max_weight=0.40,
    start_date=None,
    end_date=None,
    period="3y",
    benchmark_ticker="^NSEI",
):
    """
    Portfolio analytics engine based on the user's Finance Model notebook.

    Core methodology preserved:
    - Yahoo Finance historical prices
    - Daily returns
    - Annualized return and volatility (252 trading days)
    - Annualized covariance matrix
    - Maximum-Sharpe optimization using SLSQP
    - Long-only min/max allocation constraints
    - Maximum drawdown
    - Historical 95% VaR
    - 95% Expected Shortfall
    - NIFTY 50 benchmark comparison
    - Beta, CAPM-style alpha, active return
    - Equity crash / gold hedge stress scenario
    """

    tickers = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))

    if len(tickers) < 2:
        raise ValueError("At least two tickers are required.")

    if min_weight < 0 or max_weight <= 0:
        raise ValueError("Weights must be non-negative and maximum weight must be positive.")

    if min_weight > max_weight:
        raise ValueError("Minimum weight cannot be greater than maximum weight.")

    if min_weight * len(tickers) > 1:
        raise ValueError("Minimum weight constraint is too high for the number of assets.")

    if max_weight * len(tickers) < 1:
        raise ValueError("Maximum weight constraint is too low for the number of assets.")

    # --------------------------------------------------
    # 1. DOWNLOAD MARKET DATA
    # --------------------------------------------------
    if start_date is not None or end_date is not None:
        if start_date is None:
            start_date = "2023-01-01"
        prices = _download_close(
            tickers,
            start_date=start_date,
            end_date=end_date,
            period=None,
        )
    else:
        prices = _download_close(
            tickers,
            period=period,
        )

    if prices.empty:
        raise ValueError("No market data was returned for the selected assets.")

    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna()

    valid_tickers = list(prices.columns)

    if len(valid_tickers) < 2:
        raise ValueError("Not enough valid assets with historical data.")

    # Re-check bounds if invalid tickers were removed
    n_assets = len(valid_tickers)

    if min_weight * n_assets > 1:
        raise ValueError("Minimum weight is too high after invalid assets were removed.")

    if max_weight * n_assets < 1:
        raise ValueError("Maximum weight is too low after invalid assets were removed.")

    # --------------------------------------------------
    # 2. CALCULATE RETURNS
    # --------------------------------------------------
    returns = prices.pct_change().dropna()

    if len(returns) < 30:
        raise ValueError("Not enough historical observations to analyze the portfolio.")

    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    covariance = returns.cov() * 252

    expected_returns = annual_returns.values

    # --------------------------------------------------
    # 3. PORTFOLIO OPTIMIZATION
    # --------------------------------------------------
    def portfolio_performance(weights):
        portfolio_return = np.dot(weights, expected_returns)

        portfolio_volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(covariance.values, weights)
            )
        )

        return portfolio_return, portfolio_volatility

    def negative_sharpe(weights):
        portfolio_return, portfolio_volatility = portfolio_performance(weights)

        if portfolio_volatility <= 0:
            return 1_000_000

        return -(
            portfolio_return - risk_free_rate
        ) / portfolio_volatility

    initial_weights = np.array(
        [1 / n_assets] * n_assets
    )

    bounds = tuple(
        (min_weight, max_weight)
        for _ in range(n_assets)
    )

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    optimization_result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not optimization_result.success:
        raise RuntimeError(
            f"Portfolio optimization failed: {optimization_result.message}"
        )

    optimal_weights = optimization_result.x

    portfolio_return, portfolio_volatility = (
        portfolio_performance(optimal_weights)
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_volatility

    # --------------------------------------------------
    # 4. PORTFOLIO DAILY RETURNS / DRAWDOWN / VAR / ES
    # --------------------------------------------------
    portfolio_daily_returns = returns.dot(optimal_weights)

    portfolio_growth = (
        1 + portfolio_daily_returns
    ).cumprod()

    portfolio_value = (
        portfolio_growth * investment_amount
    )

    running_peak = portfolio_value.cummax()

    drawdown = (
        portfolio_value - running_peak
    ) / running_peak

    maximum_drawdown = drawdown.min()

    var_95 = portfolio_daily_returns.quantile(0.05)

    worst_returns = portfolio_daily_returns[
        portfolio_daily_returns <= var_95
    ]

    expected_shortfall = worst_returns.mean()

    # Sortino ratio - dashboard extension
    downside_returns = portfolio_daily_returns[
        portfolio_daily_returns < 0
    ]

    downside_deviation = (
        downside_returns.std() * np.sqrt(252)
        if len(downside_returns) > 1
        else np.nan
    )

    if pd.notna(downside_deviation) and downside_deviation > 0:
        sortino_ratio = (
            portfolio_return - risk_free_rate
        ) / downside_deviation
    else:
        sortino_ratio = np.nan

    # --------------------------------------------------
    # 5. BENCHMARK
    # --------------------------------------------------
    if start_date is not None or end_date is not None:
        benchmark_prices = _download_close(
            [benchmark_ticker],
            start_date=returns.index.min(),
            end_date=returns.index.max(),
            period=None,
        )
    else:
        benchmark_prices = _download_close(
            [benchmark_ticker],
            period=period,
        )

    if benchmark_prices.empty:
        raise ValueError("Unable to retrieve benchmark data.")

    benchmark = benchmark_prices.squeeze().ffill().dropna()

    benchmark_returns = (
        benchmark.pct_change()
        .dropna()
    )

    comparison = pd.concat(
        [
            portfolio_daily_returns,
            benchmark_returns
        ],
        axis=1,
        join="inner"
    )

    comparison.columns = [
        "Portfolio",
        "NIFTY50"
    ]

    if comparison.empty:
        raise ValueError("Portfolio and benchmark dates could not be aligned.")

    portfolio_benchmark_return = (
        comparison["Portfolio"].mean() * 252
    )

    nifty_return = (
        comparison["NIFTY50"].mean() * 252
    )

    nifty_variance = (
        comparison["NIFTY50"].var()
    )

    if nifty_variance <= 0:
        beta = np.nan
        alpha = np.nan
    else:
        beta = (
            comparison["Portfolio"].cov(
                comparison["NIFTY50"]
            )
            /
            nifty_variance
        )

        alpha = (
            portfolio_benchmark_return
            -
            (
                risk_free_rate
                +
                beta
                * (
                    nifty_return
                    - risk_free_rate
                )
            )
        )

    active_return = (
        portfolio_benchmark_return
        - nifty_return
    )

    active_daily_returns = (
        comparison["Portfolio"]
        - comparison["NIFTY50"]
    )

    tracking_error = (
        active_daily_returns.std()
        * np.sqrt(252)
    )

    if tracking_error > 0:
        information_ratio = (
            active_return / tracking_error
        )
    else:
        information_ratio = np.nan

    comparison_growth = (
        1 + comparison
    ).cumprod() * investment_amount

    # --------------------------------------------------
    # 6. STRESS TEST
    # --------------------------------------------------
    stress_returns = {}

    for asset in valid_tickers:
        if "GOLD" in asset.upper():
            stress_returns[asset] = 0.10
        else:
            stress_returns[asset] = -0.20

    stress_impact = sum(
        optimal_weights[i]
        * stress_returns[asset]
        for i, asset in enumerate(valid_tickers)
    )

    stress_value = (
        investment_amount
        * (1 + stress_impact)
    )

    # --------------------------------------------------
    # 7. ALLOCATION TABLE
    # --------------------------------------------------
    allocation = pd.DataFrame({
        "Asset": valid_tickers,
        "Optimal Weight (%)":
            optimal_weights * 100,
        "Investment (₹)":
            optimal_weights * investment_amount,
        "Annual Return (%)":
            annual_returns.values * 100,
        "Annual Volatility (%)":
            annual_volatility.values * 100
    })

    allocation["Optimal Weight (%)"] = allocation["Optimal Weight (%)"].round(2)
    allocation["Investment (₹)"] = allocation["Investment (₹)"].round(0)
    allocation["Annual Return (%)"] = allocation["Annual Return (%)"].round(2)
    allocation["Annual Volatility (%)"] = allocation["Annual Volatility (%)"].round(2)

    # --------------------------------------------------
    # 8. RETURN RESULTS TO FRONTEND
    # --------------------------------------------------
    return {
        "valid_tickers": valid_tickers,
        "prices": prices,
        "returns": returns,
        "annual_returns": annual_returns,
        "annual_volatility": annual_volatility,
        "covariance": covariance,
        "allocation": allocation,
        "optimal_weights": optimal_weights,
        "portfolio_return": portfolio_return,
        "portfolio_volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "portfolio_daily_returns": portfolio_daily_returns,
        "portfolio_growth": portfolio_growth,
        "portfolio_value": portfolio_value,
        "current_portfolio_value": portfolio_value.iloc[-1],
        "total_return": portfolio_value.iloc[-1] / investment_amount - 1,
        "drawdown": drawdown,
        "maximum_drawdown": maximum_drawdown,
        "var_95": var_95,
        "expected_shortfall": expected_shortfall,
        "comparison": comparison,
        "comparison_growth": comparison_growth,
        "portfolio_benchmark_return": portfolio_benchmark_return,
        "nifty_return": nifty_return,
        "beta": beta,
        "alpha": alpha,
        "active_return": active_return,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "stress_impact": stress_impact,
        "stress_value": stress_value,
        "investment_amount": investment_amount,
        "risk_free_rate": risk_free_rate,
        "benchmark_ticker": benchmark_ticker,
    }


def run_custom_stress_test(
    optimal_weights,
    valid_tickers,
    investment_amount,
    equity_shock=-0.20,
    gold_shock=0.10,
):
    """Run a custom equity/gold stress scenario on an analyzed portfolio."""
    stress_impact = 0.0

    for i, asset in enumerate(valid_tickers):
        shock = gold_shock if "GOLD" in asset.upper() else equity_shock
        stress_impact += optimal_weights[i] * shock

    stressed_value = investment_amount * (1 + stress_impact)

    return {
        "stress_impact": stress_impact,
        "stressed_value": stressed_value,
        "loss_gain": stressed_value - investment_amount,
    }
