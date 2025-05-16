"""
Implementation of the Hurst exponent for time series analysis.
The Hurst exponent measures the long-term memory of a time series and can
determine if a time series is mean-reverting, random walk, or trending.
"""

import numpy as np
import pandas as pd


def calculate_hurst_exponent(time_series, max_lag=20):
    """
    Calculate the Hurst exponent for a time series.
    
    The Hurst exponent (H) interpretation:
    - H < 0.5: Mean-reverting (anti-persistent) behavior
    - H = 0.5: Random walk (Brownian motion)
    - H > 0.5: Trending (persistent) behavior
    
    Parameters
    ----------
    time_series : pandas.Series or numpy.ndarray
        The time series data to analyze
    max_lag : int, optional
        Maximum lag for calculating the Hurst exponent, by default 20
        
    Returns
    -------
    float
        The Hurst exponent value
    """
    # Convert to numpy array if Series
    if isinstance(time_series, pd.Series):
        time_series = time_series.values
    
    # Remove NaN values
    time_series = time_series[~np.isnan(time_series)]
    
    # Return NaN if not enough data points
    if len(time_series) < max_lag * 2:
        return np.nan
    
    # Calculate returns
    returns = np.log(time_series[1:] / time_series[:-1])
    
    # Calculate the variance of the returns
    tau = np.arange(1, max_lag + 1)
    var = np.zeros(max_lag)
    
    for lag in range(1, max_lag + 1):
        # For each lag, calculate the rescaled range
        # Segment the returns into non-overlapping blocks of length lag
        segments = len(returns) // lag
        if segments < 1:
            # Not enough data for this lag
            var[lag-1] = np.nan
            continue
            
        # Calculate the rescaled range for each segment and average
        r_s_values = []
        for i in range(segments):
            segment = returns[i*lag:(i+1)*lag]
            # Mean-adjusted segment
            z = segment - np.mean(segment)
            # Cumulative sum
            z_cumsum = np.cumsum(z)
            # Range
            r = np.max(z_cumsum) - np.min(z_cumsum)
            # Standard deviation
            s = np.std(segment)
            if s > 0:
                r_s_values.append(r / s)
        
        if r_s_values:
            var[lag-1] = np.mean(r_s_values)
    
    # Filter out NaN values
    valid_indices = ~np.isnan(var)
    if np.sum(valid_indices) < 4:  # Need at least a few points for regression
        return np.nan
    
    # Linear regression on log-log scale
    log_tau = np.log(tau[valid_indices])
    log_var = np.log(var[valid_indices])
    
    # Calculate Hurst exponent as slope of regression line
    hurst = np.polyfit(log_tau, log_var, 1)[0]
    
    return hurst


def calculate_rolling_hurst(time_series, window=252, step=1, max_lag=20):
    """
    Calculate rolling Hurst exponent for a time series.
    
    Parameters
    ----------
    time_series : pandas.Series
        The time series data to analyze
    window : int, optional
        Window size for rolling calculation, by default 252 (one year of trading days)
    step : int, optional
        Step size for moving the window, by default 1
    max_lag : int, optional
        Maximum lag for calculating the Hurst exponent, by default 20
        
    Returns
    -------
    pandas.Series
        Series of rolling Hurst exponent values with the same index as the input
    """
    # Initialize Series to store results
    hurst_values = pd.Series(index=time_series.index, dtype=float)
    hurst_values[:] = np.nan
    
    # Calculate rolling Hurst exponent
    for i in range(window, len(time_series) + 1, step):
        # Get the window of data
        window_data = time_series.iloc[i-window:i]
        # Calculate Hurst exponent for this window
        hurst = calculate_hurst_exponent(window_data, max_lag=max_lag)
        # Store the result at the end of the window
        hurst_values.iloc[i-1] = hurst
    
    return hurst_values
