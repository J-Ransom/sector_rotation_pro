import pandas as pd
import numpy as np
import hurst
import warnings

def calculate_rolling_hurst(prices, column="SPY", window=252):
    """
    Calculate rolling Hurst exponent for a price series.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices
    column : str
        Column to calculate Hurst exponent for (typically the benchmark)
    window : int
        Window size for rolling calculation
        
    Returns:
    -------
    pandas.Series
        Series with rolling Hurst exponent values
    """
    if column not in prices.columns:
        raise ValueError(f"Column '{column}' not found in price data")
    
    # Get the price series
    price_series = prices[column]
    
    # Initialize a Series to store Hurst exponents
    hurst_series = pd.Series(index=price_series.index, dtype=float)
    
    # Calculate rolling Hurst exponent
    for i in range(window, len(price_series)):
        try:
            # Get window of data
            window_data = price_series.iloc[i-window:i].values
            
            # Calculate Hurst exponent
            h, _, _ = hurst.compute_Hc(window_data)
            
            # Store result
            hurst_series.iloc[i] = h
        except:
            # Set NaN if calculation fails
            hurst_series.iloc[i] = np.nan
    
    return hurst_series

def classify_hurst(hurst_values):
    """
    Classify Hurst exponent values into regimes.
    
    Parameters:
    ----------
    hurst_values : pandas.Series
        Series with Hurst exponent values
        
    Returns:
    -------
    pandas.Series
        Series with regime classifications
    """
    # Initialize regime series
    regimes = pd.Series(index=hurst_values.index, dtype=str)
    
    # Classify regimes based on Hurst exponent
    regimes[hurst_values < 0.4] = 'Mean-Reverting'
    regimes[(hurst_values >= 0.4) & (hurst_values <= 0.6)] = 'Random Walk'
    regimes[hurst_values > 0.6] = 'Trending'
    
    return regimes

def classify_vix(vix_series):
    """
    Classify VIX values into volatility regimes using tertiles.
    
    Parameters:
    ----------
    vix_series : pandas.Series
        Series with VIX index values
        
    Returns:
    -------
    pandas.Series
        Series with volatility regime classifications
    """
    # Calculate tertiles
    low_tertile = vix_series.quantile(0.33)
    high_tertile = vix_series.quantile(0.67)
    
    # Initialize volatility regime series
    vol_regimes = pd.Series(index=vix_series.index, dtype=str)
    
    # Classify volatility regimes
    vol_regimes[vix_series <= low_tertile] = 'Low Volatility'
    vol_regimes[(vix_series > low_tertile) & (vix_series <= high_tertile)] = 'Normal Volatility'
    vol_regimes[vix_series > high_tertile] = 'High Volatility'
    
    return vol_regimes

def generate_regimes(spy_prices, vix_series, window=252):
    """
    Generate combined regime labels based on Hurst exponent and VIX.
    
    Parameters:
    ----------
    spy_prices : pandas.Series
        Series with SPY prices
    vix_series : pandas.Series
        Series with VIX index values
    window : int
        Window size for Hurst calculation
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with Hurst values, VIX values, and combined regime labels
    """
    # Ensure indices are aligned
    if not spy_prices.index.equals(vix_series.index):
        common_idx = spy_prices.index.intersection(vix_series.index)
        spy_prices = spy_prices.loc[common_idx]
        vix_series = vix_series.loc[common_idx]
        warnings.warn("Input series had different indices; using intersection")
    
    # Calculate rolling Hurst exponent
    prices_df = pd.DataFrame(spy_prices)
    prices_df.columns = ['SPY']
    hurst_values = calculate_rolling_hurst(prices_df, 'SPY', window)
    
    # Classify Hurst regimes
    hurst_regimes = classify_hurst(hurst_values)
    
    # Classify VIX regimes
    vix_regimes = classify_vix(vix_series)
    
    # Combine regimes
    combined_regimes = pd.DataFrame({
        'Hurst': hurst_values,
        'Hurst_Regime': hurst_regimes,
        'VIX': vix_series,
        'VIX_Regime': vix_regimes
    })
    
    # Create combined regime label
    combined_regimes['Combined_Regime'] = combined_regimes['Hurst_Regime'] + ' | ' + combined_regimes['VIX_Regime']
    
    return combined_regimes
