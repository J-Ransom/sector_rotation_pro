import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import ccf, grangercausalitytests

def calculate_correlation_matrix(returns):
    """
    Calculate correlation matrix for all tickers.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return values for each ticker
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with correlation values
    """
    return returns.corr()

def calculate_lead_lag(returns, benchmark="SPY", maxlag=20):
    """
    Calculate lead-lag relationship (cross-correlation) between each ticker and the benchmark.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return values for each ticker
    benchmark : str
        Ticker symbol for the benchmark
    maxlag : int
        Maximum lag to compute cross-correlation
        
    Returns:
    -------
    dict
        Dictionary with ticker as key and dictionary of lags and cross-correlation values as value
    """
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark ticker '{benchmark}' not found in returns data")
    
    benchmark_rets = returns[benchmark]
    lead_lag_results = {}
    
    for col in returns.columns:
        if col != benchmark:
            # Calculate cross-correlation function
            ccf_values = ccf(returns[col], benchmark_rets, adjusted=False)
            
            # Store the results
            lead_lag_results[col] = {
                'lags': list(range(-maxlag, maxlag + 1)),
                'ccf': list(ccf_values)
            }
            
            # Find peak correlation and corresponding lag
            max_idx = np.argmax(np.abs(ccf_values))
            max_lag = max_idx - maxlag
            max_corr = ccf_values[max_idx]
            
            lead_lag_results[col]['peak_lag'] = max_lag
            lead_lag_results[col]['peak_corr'] = max_corr
    
    return lead_lag_results

def calculate_granger_causality(returns, benchmark="SPY", maxlag=5):
    """
    Test for Granger causality between each ticker and the benchmark.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return values for each ticker
    benchmark : str
        Ticker symbol for the benchmark
    maxlag : int
        Maximum lag to test for Granger causality
        
    Returns:
    -------
    dict
        Dictionary with ticker as key and dictionary of p-values for each lag as value
    """
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark ticker '{benchmark}' not found in returns data")
    
    benchmark_rets = returns[benchmark]
    granger_results = {}
    
    for col in returns.columns:
        if col != benchmark:
            # Prepare data for Granger causality test
            data = pd.concat([returns[col], benchmark_rets], axis=1).dropna()
            
            # Test ticker Granger-causes benchmark
            try:
                gc_benchmark = grangercausalitytests(data, maxlag, verbose=False)
                ticker_to_benchmark = {lag: gc_benchmark[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag + 1)}
            except:
                ticker_to_benchmark = {lag: np.nan for lag in range(1, maxlag + 1)}
            
            # Test benchmark Granger-causes ticker
            try:
                gc_ticker = grangercausalitytests(data.iloc[:, [1, 0]], maxlag, verbose=False)
                benchmark_to_ticker = {lag: gc_ticker[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag + 1)}
            except:
                benchmark_to_ticker = {lag: np.nan for lag in range(1, maxlag + 1)}
            
            # Store the results
            granger_results[col] = {
                'ticker_causes_benchmark': ticker_to_benchmark,
                'benchmark_causes_ticker': benchmark_to_ticker
            }
    
    return granger_results

def generate_network_data(corr_matrix, threshold=0.6):
    """
    Generate network data from correlation matrix for visualization.
    
    Parameters:
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    threshold : float
        Correlation threshold for creating edges
        
    Returns:
    -------
    tuple
        (nodes, edges) for network visualization
    """
    # Create nodes
    nodes = [{'id': ticker, 'name': ticker} for ticker in corr_matrix.columns]
    
    # Create edges based on correlation threshold
    edges = []
    
    for i, ticker1 in enumerate(corr_matrix.columns):
        for j, ticker2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates and self-loops
                corr = corr_matrix.loc[ticker1, ticker2]
                if abs(corr) >= threshold:
                    edges.append({
                        'source': ticker1,
                        'target': ticker2,
                        'weight': abs(corr),
                        'color': 'green' if corr > 0 else 'red'
                    })
    
    return nodes, edges
