import pandas as pd
from scipy.stats import zscore

def calc_rrg(prices, benchmark="SPY", window=10):
    """
    Calculate Relative Rotation Graph data.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    benchmark : str
        Ticker symbol for the benchmark
    window : int
        Window size for the relative strength momentum calculation
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with RS_Ratio and RS_Mom columns for each ticker
    """
    # Extract benchmark prices
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark ticker '{benchmark}' not found in price data")
    
    benchmark_prices = prices[benchmark]
    
    # Calculate the ratio of each ticker to the benchmark
    ratio_df = pd.DataFrame()
    
    for col in prices.columns:
        if col != benchmark:
            ratio = prices[col] / benchmark_prices
            ratio_df[col] = ratio
    
    # Calculate the RS_Ratio (relative strength)
    rs_ratio = ratio_df / ratio_df.rolling(window=window*2).mean()
    
    # Calculate the RS_Momentum
    rs_mom = rs_ratio / rs_ratio.shift(window) - 1
    
    # Create the final RRG DataFrame
    rrg_df = pd.DataFrame()
    
    for col in ratio_df.columns:
        rrg_df.loc[col, 'RS_Ratio'] = rs_ratio[col].iloc[-1]
        rrg_df.loc[col, 'RS_Mom'] = rs_mom[col].iloc[-1]
    
    # Z-score normalization
    rrg_df[['RS_Ratio', 'RS_Mom']] = rrg_df[['RS_Ratio', 'RS_Mom']].apply(zscore)
    
    # Add the historical trail data
    rrg_df_with_trail = add_trail_data(prices, benchmark, window)
    
    # Merge the trail data with the current RRG data
    for col in rrg_df.index:
        rrg_df_with_trail.loc[rrg_df_with_trail['ticker'] == col, 'RS_Ratio_zscore'] = rrg_df.loc[col, 'RS_Ratio']
        rrg_df_with_trail.loc[rrg_df_with_trail['ticker'] == col, 'RS_Mom_zscore'] = rrg_df.loc[col, 'RS_Mom']
    
    return rrg_df_with_trail

def add_trail_data(prices, benchmark="SPY", window=10):
    """
    Add historical trail data for the RRG visualization.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    benchmark : str
        Ticker symbol for the benchmark
    window : int
        Window size for the relative strength momentum calculation
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with historical RS_Ratio and RS_Mom for each ticker
    """
    # Extract benchmark prices
    benchmark_prices = prices[benchmark]
    
    # Create a DataFrame to store trail data
    trail_data = []
    
    # For better animation, we need to ensure we have data at different dates
    # Let's use sliding windows through the price history instead of just offsets
    total_periods = min(8, len(prices) // (window // 2) - 2)  # Ensure we have enough data
    
    # Make sure we have at least 1 period
    if total_periods < 1:
        total_periods = 1
    
    for i in range(total_periods):  # Trail of the last N periods
        # Calculate sliding window indices
        # For each step, move back in time by half a window
        end_idx = -1 - (i * window // 2)
        if abs(end_idx) >= len(prices):
            # Skip if we don't have enough data
            continue
            
        # Ensure we have enough historical data for calculations
        # We need at least 3x the window size for meaningful calculations
        start_idx = max(0, end_idx - window * 3)
        
        # Get the actual data for this time period
        period_prices = prices.iloc[start_idx:end_idx]
        period_benchmark = benchmark_prices.iloc[start_idx:end_idx]
        
        # Calculate metrics for each ticker
        for ticker in prices.columns:
            if ticker != benchmark:
                # Skip if we don't have enough data
                if len(period_prices) < window * 2:
                    continue
                    
                # Calculate the relative ratio of the ticker to the benchmark
                ratio = period_prices[ticker] / period_benchmark
                
                # Calculate the RS-Ratio (relative strength)
                # This compares the current ratio to its historical average
                rs_ratio = ratio.iloc[-1] / ratio.iloc[:-window].mean()
                
                # Calculate the RS-Momentum
                # This shows how the relative strength is changing
                rs_mom = rs_ratio / (ratio.iloc[-window] / period_benchmark.iloc[-window]) - 1
                
                # Add this point to our trail data
                trail_data.append({
                    'ticker': ticker,
                    'period': total_periods - i,  # Use actual period for better sequencing
                    'RS_Ratio': rs_ratio,
                    'RS_Mom': rs_mom,
                    'week': i,  # Animation frame index
                    'end_date': prices.index[end_idx].strftime('%Y-%m-%d')  # Add date for reference
                })
    
    # Convert to DataFrame
    trail_df = pd.DataFrame(trail_data)
    
    # Z-score normalization for the entire dataset
    if not trail_df.empty:
        trail_df['RS_Ratio_zscore'] = zscore(trail_df['RS_Ratio'])
        trail_df['RS_Mom_zscore'] = zscore(trail_df['RS_Mom'])
    
    return trail_df
