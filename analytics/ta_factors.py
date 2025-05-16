import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings

def calculate_rsi(prices, length=14):
    """
    Calculate RSI for all tickers in the prices DataFrame.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    length : int
        Period for RSI calculation
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with RSI values for each ticker
    """
    rsi_df = pd.DataFrame()
    
    for col in prices.columns:
        rsi_df[col] = ta.rsi(prices[col], length=length)
    
    return rsi_df

def calculate_momentum_values(prices):
    """
    Calculate momentum values for multiple timeframes (1, 3, 6, and 12 months).
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    
    Returns:
    -------
    dict of pandas.DataFrames
        Dictionary with keys '1m', '3m', '6m', '12m' containing momentum values
    """
    # Define lookback periods (approximate trading days)
    lookbacks = {
        '1m': 21,   # ~1 month of trading days
        '3m': 63,   # ~3 months of trading days
        '6m': 126,  # ~6 months of trading days
        '12m': 252  # ~12 months of trading days
    }
    
    # Calculate returns for each lookback period
    momentum_data = {}
    
    for period_name, lookback in lookbacks.items():
        if len(prices) < lookback:
            # If not enough data, use what we have
            adjusted_lookback = len(prices) - 1
            warnings.warn(f"lookback period of {lookback} days exceeds available data; using {adjusted_lookback} days instead")
            lookback = adjusted_lookback
        
        # Calculate momentum values (percentage change)
        momentum_values = prices.pct_change(lookback).iloc[-1] * 100
        
        # Sort values in descending order (higher momentum first)
        momentum_values = momentum_values.sort_values(ascending=False)
        
        # Store in the dictionary
        momentum_data[period_name] = momentum_values
    
    return momentum_data


def calculate_momentum_rank(prices, lookback=252):
    """
    Calculate momentum rank based on lookback period returns (legacy function).
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    lookback : int
        Lookback period (252=1yr for daily data)
        
    Returns:
    -------
    pandas.Series
        Series with momentum ranks for each ticker
    """
    # Use the new function but return only the 12m ranks for backwards compatibility
    momentum_data = calculate_momentum_values(prices)
    
    # Get the 12-month momentum values and convert to ranks (lower rank = higher momentum)
    momentum_values = momentum_data['12m']
    ranks = (-momentum_values).rank()  # Negate values so higher momentum gets lower rank
    
    return ranks

def calculate_rolling_beta(prices, benchmark="SPY", window=52):
    """
    Calculate rolling beta for all tickers relative to benchmark.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    benchmark : str
        Ticker symbol for the benchmark
    window : int
        Window size for beta calculation
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with rolling beta values for each ticker
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark ticker '{benchmark}' not found in price data")
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    benchmark_rets = returns[benchmark]
    
    beta_df = pd.DataFrame(index=returns.index)
    
    for col in returns.columns:
        if col != benchmark:
            # Calculate rolling covariance
            rolling_cov = returns[col].rolling(window=window).cov(benchmark_rets)
            
            # Calculate rolling variance of benchmark
            rolling_var = benchmark_rets.rolling(window=window).var()
            
            # Calculate rolling beta
            beta_df[col] = rolling_cov / rolling_var
    
    return beta_df

def calculate_rolling_sharpe(prices, risk_free_rate=0.0, window=52):
    """
    Calculate rolling Sharpe ratio for all tickers.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    risk_free_rate : float
        Annual risk-free rate
    window : int
        Window size for Sharpe calculation
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with rolling Sharpe values for each ticker
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Convert annual risk-free rate to match return frequency
    if len(returns.index) >= 252:  # Daily data
        rf_daily = (1 + risk_free_rate) ** (1/252) - 1
        periods_per_year = 252
    elif len(returns.index) >= 52:  # Weekly data
        rf_daily = (1 + risk_free_rate) ** (1/52) - 1
        periods_per_year = 52
    else:  # Monthly data
        rf_daily = (1 + risk_free_rate) ** (1/12) - 1
        periods_per_year = 12
    
    sharpe_df = pd.DataFrame(index=returns.index)
    
    for col in returns.columns:
        # Calculate rolling mean returns
        rolling_mean = returns[col].rolling(window=window).mean()
        
        # Calculate rolling standard deviation
        rolling_std = returns[col].rolling(window=window).std()
        
        # Calculate rolling Sharpe ratio
        sharpe_df[col] = (rolling_mean - rf_daily) / rolling_std * np.sqrt(periods_per_year)
    
    return sharpe_df

def calculate_rolling_moments(prices, window=52):
    """
    Calculate rolling skewness and kurtosis for all tickers.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    window : int
        Window size for calculation
        
    Returns:
    -------
    tuple of pandas.DataFrame
        (skew_df, kurt_df) containing rolling skewness and kurtosis for each ticker
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    skew_df = pd.DataFrame(index=returns.index)
    kurt_df = pd.DataFrame(index=returns.index)
    
    for col in returns.columns:
        # Calculate rolling skewness
        skew_df[col] = returns[col].rolling(window=window).skew()
        
        # Calculate rolling kurtosis
        kurt_df[col] = returns[col].rolling(window=window).kurt()
    
    return skew_df, kurt_df

def calculate_drawdowns(prices):
    """
    Calculate drawdowns for all tickers.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with drawdowns for each ticker
    """
    drawdown_df = pd.DataFrame(index=prices.index)
    
    for col in prices.columns:
        # Calculate running maximum
        running_max = prices[col].cummax()
        
        # Calculate drawdown
        drawdown_df[col] = (prices[col] / running_max) - 1
    
    return drawdown_df
