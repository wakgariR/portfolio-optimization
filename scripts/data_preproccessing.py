"""
Reusable data preprocessing and EDA utilities for GMF portfolio optimization project
"""

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def fetch_data(tickers, start_date, end_date):
    """Fetch historical data from Yahoo Finance."""
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df["Asset"] = ticker
        data[ticker] = df
    combined_df = pd.concat(data.values())
    combined_df.reset_index(inplace=True)
    return combined_df


def clean_data(df):
    """Clean and prepare raw financial data."""
    df = df.sort_values(["Asset", "Date"])
    df = df.fillna(method="ffill")
    df = df.dropna()
    return df


def add_features(df, volatility_window=30):
    """Add daily returns and rolling volatility."""
    df["Daily_Return"] = df.groupby("Asset")["Adj Close"].pct_change()
    df["Rolling_Volatility"] = (
        df.groupby("Asset")["Daily_Return"]
        .rolling(window=volatility_window)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df


def adf_test(series):
    """Run Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
    }


def calculate_var(returns, confidence=0.95):
    """Calculate historical Value at Risk."""
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate historical Sharpe Ratio."""
    daily_rf = risk_free_rate / 252
    return (returns.mean() - daily_rf) / returns.std()


def save_processed_data(df, path):
    """Save processed dataset to disk."""
    df.to_csv(path, index=False)
