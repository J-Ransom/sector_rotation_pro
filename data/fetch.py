import streamlit as st
import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file in the sector_rotation_pro folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Display message if API key is not found
if not ALPHA_VANTAGE_API_KEY:
    st.error(
        "Alpha Vantage API key not found in .env file. " +
        "Please add your key to the .env file as ALPHA_VANTAGE_API_KEY=your_key_here"
    )
    
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Set default rate limit delay (seconds) - Alpha Vantage free tier allows 5 calls/min
RATE_LIMIT_DELAY = 12  # 12 seconds between calls to respect rate limits

def get_alpha_vantage_data(symbol, function, outputsize="full"):
    """
    Fetch data from Alpha Vantage API.
    
    Parameters:
    ----------
    symbol : str
        Ticker symbol
    function : str
        API function to call (TIME_SERIES_DAILY_ADJUSTED, TIME_SERIES_WEEKLY_ADJUSTED, TIME_SERIES_MONTHLY_ADJUSTED)
    outputsize : str, optional
        Size of output data (compact or full)
        
    Returns:
    -------
    dict or None
        JSON response from API or None if error
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None
        
    params = {
        "function": function,
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            st.warning(f"Alpha Vantage API error for {symbol}: {data['Error Message']}")
            return None
        
        # Check for API limit messages
        if "Note" in data and "call frequency" in data["Note"]:
            st.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            time.sleep(15)  # Wait before next call
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return None

def parse_alpha_vantage_daily(data, ticker):
    """
    Parse the response data from Alpha Vantage daily API.
    
    Parameters:
    ----------
    data : dict
        Data from Alpha Vantage API
    ticker : str
        Ticker symbol
        
    Returns:
    -------
    pandas.Series
        Series with adjusted close price data
    """
    try:
        # Check if we're using TIME_SERIES_DAILY or TIME_SERIES_DAILY_ADJUSTED
        if "Time Series (Daily)" in data:
            time_series_key = "Time Series (Daily)"
            adjusted_close_key = "4. close"  # Regular daily has just close
        elif "Time Series (Daily Adjusted)" in data:
            time_series_key = "Time Series (Daily Adjusted)"
            adjusted_close_key = "5. adjusted close"  # Adjusted daily has adjusted close
        else:
            # No valid time series data found
            st.warning(f"No valid time series data found for {ticker}. API returned: {list(data.keys())}")
            return pd.Series(name=ticker)
        
        # Get the time series data
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            st.warning(f"Empty time series data for {ticker}")
            return pd.Series(name=ticker)
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Debugging information
        if df.empty:
            st.warning(f"DataFrame is empty for {ticker}")
            return pd.Series(name=ticker)
        
        # Check if the adjusted close column exists
        available_columns = df.columns.tolist()
        if adjusted_close_key not in available_columns:
            st.warning(f"Column '{adjusted_close_key}' not found for {ticker}. Available columns: {available_columns}")
            # Try to use regular close if adjusted close is not available
            adjusted_close_key = "4. close" if "4. close" in available_columns else available_columns[0]
        
        # Rename columns and convert to numeric
        df.index = pd.to_datetime(df.index)
        
        # Ensure the index is sorted
        df = df.sort_index()
        
        # Set frequency explicitly to daily
        # For daily data, we can directly infer the frequency if data is complete
        inferred_freq = pd.infer_freq(df.index)
        
        # Rename the column and extract the price series
        df = df.rename(columns={adjusted_close_key: ticker})
        df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
        
        # Check for missing values
        missing_count = df[ticker].isna().sum()
        if missing_count > 0 and missing_count < len(df) * 0.5:  # If less than 50% missing
            # Fill missing values with forward fill then backward fill
            # Using ffill and bfill directly to avoid deprecation warnings
            df[ticker] = df[ticker].ffill().bfill()
            st.info(f"Filled {missing_count} missing values for {ticker}")
        
        # Create the final series
        series = df[ticker].dropna()
        
        # Only set frequency if we could infer it
        if inferred_freq:
            series.index.freq = inferred_freq
        
        return series
    
    except Exception as e:
        st.error(f"Error parsing Alpha Vantage daily data for {ticker}: {e}")
        return pd.Series(name=ticker)

def parse_alpha_vantage_weekly(data, ticker):
    """
    Parse weekly data from Alpha Vantage API response.
    
    Parameters:
    ----------
    data : dict
        API response from Alpha Vantage
    ticker : str
        Ticker symbol
        
    Returns:
    -------
    pandas.Series
        Series with adjusted close prices
    """
    try:
        # Check if the weekly adjusted data exists
        if "Weekly Adjusted Time Series" in data:
            time_series_key = "Weekly Adjusted Time Series"
            adjusted_close_key = "5. adjusted close"
        else:
            # No valid time series data found
            st.warning(f"No valid weekly time series data found for {ticker}. API returned: {list(data.keys())}")
            return pd.Series(name=ticker)
            
        # Get the time series data
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            st.warning(f"Empty weekly time series data for {ticker}")
            return pd.Series(name=ticker)
        
        # Create a dictionary for the series
        df_dict = {}
        
        for date, values in time_series.items():
            # Check if the adjusted close key exists
            if adjusted_close_key in values:
                try:
                    df_dict[date] = float(values[adjusted_close_key])
                except (ValueError, TypeError):
                    st.warning(f"Invalid value for {ticker} on {date}: {values[adjusted_close_key]}")
            else:
                st.warning(f"Missing adjusted close key for {ticker} on {date}. Keys: {list(values.keys())}")
        
        if not df_dict:
            st.warning(f"No valid price data extracted for {ticker}")
            return pd.Series(name=ticker)
        
        # Create the series
        series = pd.Series(df_dict, name=ticker)
        series.index = pd.to_datetime(series.index)
        
        # Sort by date
        series = series.sort_index()
        
        # Check the inferred frequency - Alpha Vantage weekly data typically ends on Sundays or Fridays
        inferred_freq = pd.infer_freq(series.index)
        
        # If we can't infer the frequency directly, analyze the day of week patterns
        if inferred_freq is None and len(series) > 5:
            # Count the occurrences of each day of the week
            day_counts = series.index.dayofweek.value_counts()
            if not day_counts.empty:
                # Find the most common day of the week in the data
                most_common_day = day_counts.idxmax()
                # Map to the correct weekly frequency format
                day_map = {0: 'W-MON', 1: 'W-TUE', 2: 'W-WED', 3: 'W-THU', 
                          4: 'W-FRI', 5: 'W-SAT', 6: 'W-SUN'}
                inferred_freq = day_map.get(most_common_day)
                if inferred_freq:
                    st.info(f"Inferred weekly frequency for {ticker}: {inferred_freq}")
        
        # Only set the frequency if we could infer it
        if inferred_freq:
            series.index.freq = inferred_freq
            
        return series
    except Exception as e:
        st.error(f"Error parsing Alpha Vantage weekly data for {ticker}: {e}")
        return pd.Series(name=ticker)

def parse_alpha_vantage_monthly(data, ticker):
    """
    Parse monthly data from Alpha Vantage API response.
    
    Parameters:
    ----------
    data : dict
        API response from Alpha Vantage
    ticker : str
        Ticker symbol
        
    Returns:
    -------
    pandas.Series
        Series with adjusted close prices
    """
    try:
        # Check if the monthly adjusted data exists
        if "Monthly Adjusted Time Series" in data:
            time_series_key = "Monthly Adjusted Time Series"
            adjusted_close_key = "5. adjusted close"
        else:
            # No valid time series data found
            st.warning(f"No valid monthly time series data found for {ticker}. API returned: {list(data.keys())}")
            return pd.Series(name=ticker)
            
        # Get the time series data
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            st.warning(f"Empty monthly time series data for {ticker}")
            return pd.Series(name=ticker)
        
        # Create a dictionary for the series
        df_dict = {}
        
        for date, values in time_series.items():
            # Check if the adjusted close key exists
            if adjusted_close_key in values:
                try:
                    df_dict[date] = float(values[adjusted_close_key])
                except (ValueError, TypeError):
                    st.warning(f"Invalid value for {ticker} on {date}: {values[adjusted_close_key]}")
            else:
                st.warning(f"Missing adjusted close key for {ticker} on {date}. Keys: {list(values.keys())}")
        
        if not df_dict:
            st.warning(f"No valid monthly price data extracted for {ticker}")
            return pd.Series(name=ticker)
        
        # Create the series
        series = pd.Series(df_dict, name=ticker)
        series.index = pd.to_datetime(series.index)
        
        # Sort by date
        series = series.sort_index()
        
        # Try to infer the frequency
        inferred_freq = pd.infer_freq(series.index)
        
        # Only set the frequency if we could infer it
        if inferred_freq:
            series.index.freq = inferred_freq
            
        return series
    except Exception as e:
        st.error(f"Error parsing Alpha Vantage monthly data for {ticker}: {e}")
        return pd.Series(name=ticker)

@st.cache_data(ttl=86400)
def get_prices(tickers, start, end, freq='D'):
    """
    Fetch and cache stock price data from Alpha Vantage for the given tickers and time period.
    
    Parameters:
    ----------
    tickers : list or str
        List of ticker symbols or a single string of space-separated symbols
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format
    freq : str, optional
        Frequency of data: 'D' (daily), 'W' (weekly), or 'M' (monthly)
        
    Returns:
    -------
    pandas.DataFrame
        DataFrame with adjusted close prices for each ticker
    """
    # Display clear messages about the frequency being used
    if freq == 'W':
        st.info("Getting weekly data. Note: Alpha Vantage weekly data may end on different days of the week.")
    elif freq == 'M':
        st.info("Getting monthly data from Alpha Vantage.")
    else:
        st.info("Getting daily data from Alpha Vantage.")
    
    # Ensure API key is provided
    if not ALPHA_VANTAGE_API_KEY:
        st.error("Alpha Vantage API key is required. Please add your key to the .env file.")
        return pd.DataFrame()
    
    # Convert start/end to datetime if they're strings
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)
    
    # Convert tickers to list if it's a string
    if isinstance(tickers, str):
        ticker_list = tickers.split()
    else:
        ticker_list = tickers
    
    # Create an empty DataFrame to store results
    result_df = pd.DataFrame()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch data for each ticker
    for i, ticker in enumerate(ticker_list):
        status_text.text(f"Fetching data for {ticker}...")
        progress_bar.progress((i) / len(ticker_list))
        
        series = pd.Series(name=ticker)  # Initialize empty series
        
        if freq == 'D':
            # Get daily adjusted data
            function = "TIME_SERIES_DAILY_ADJUSTED"
            data = get_alpha_vantage_data(ticker, function, outputsize="full")
            if data:
                series = parse_alpha_vantage_daily(data, ticker)
        
        elif freq == 'W':
            # First try to get weekly adjusted data
            function = "TIME_SERIES_WEEKLY_ADJUSTED"
            data = get_alpha_vantage_data(ticker, function)
            if data:
                series = parse_alpha_vantage_weekly(data, ticker)
                
            # If we failed to get proper weekly data, try to create from daily
            if series.empty:
                st.warning(f"Could not get weekly data for {ticker} directly. Trying to create from daily data...")
                daily_function = "TIME_SERIES_DAILY_ADJUSTED"
                daily_data = get_alpha_vantage_data(ticker, daily_function, outputsize="full")
                if daily_data:
                    daily_series = parse_alpha_vantage_daily(daily_data, ticker)
                    if not daily_series.empty:
                        # Resample to weekly (using the last trading day of each week)
                        series = daily_series.resample('W').last()
                        st.info(f"Successfully created weekly data for {ticker} from daily data")
        
        elif freq == 'M':
            # Get monthly adjusted data
            function = "TIME_SERIES_MONTHLY_ADJUSTED"
            data = get_alpha_vantage_data(ticker, function)
            if data:
                series = parse_alpha_vantage_monthly(data, ticker)
                
            # If we failed to get proper monthly data, try to create from daily
            if series.empty:
                st.warning(f"Could not get monthly data for {ticker} directly. Trying to create from daily data...")
                daily_function = "TIME_SERIES_DAILY_ADJUSTED"
                daily_data = get_alpha_vantage_data(ticker, daily_function, outputsize="full")
                if daily_data:
                    daily_series = parse_alpha_vantage_daily(daily_data, ticker)
                    if not daily_series.empty:
                        # Resample to monthly (using the last trading day of each month)
                        series = daily_series.resample('M').last()
                        st.info(f"Successfully created monthly data for {ticker} from daily data")
        
        # Filter by date range
        if not series.empty:
            series = series[(series.index >= start) & (series.index <= end)]
            
            # Add to result DataFrame
            if result_df.empty:
                result_df = pd.DataFrame(series)
            else:
                result_df[ticker] = series
        else:
            st.warning(f"No data available for {ticker}")
        
        # Rate limit (respect Alpha Vantage's API limits)
        if i < len(ticker_list) - 1:  # Don't wait after the last request
            time.sleep(RATE_LIMIT_DELAY)
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Data fetching completed!")
    
    if result_df.empty:
        st.error("Could not fetch price data. Please check ticker symbols and try again.")
    else:
        st.success("Data successfully fetched from Alpha Vantage API")
        
    return result_df.dropna()

@st.cache_data(ttl=86400)
def get_vix_data(start, end, freq='D'):
    """
    Simulate VIX data from Alpha Vantage using an ETF proxy (VXX).
    
    Parameters:
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str
        End date in 'YYYY-MM-DD' format
    freq : str, optional
        Frequency of data: 'D' (daily), 'W' (weekly), or 'M' (monthly)
        
    Returns:
    -------
    pandas.Series
        Series with VIX proxy values
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.error("Alpha Vantage API key is required. Please add your key to the .env file.")
        return pd.Series()
    
    st.info("Using VXX ETF as a proxy for VIX data from Alpha Vantage")
    
    # Uses VXX (iPath Series B S&P 500 VIX Short-Term Futures ETN) as a proxy for VIX
    vix_proxy = "VXX"
    
    try:
        vix_df = pd.Series(name="VIX")  # Initialize empty series
        
        if freq == 'D':
            # Get daily adjusted data
            function = "TIME_SERIES_DAILY_ADJUSTED"
            data = get_alpha_vantage_data(vix_proxy, function, outputsize="full")
            if data:
                vix_df = parse_alpha_vantage_daily(data, vix_proxy)
        
        elif freq == 'W':
            # First try weekly data
            function = "TIME_SERIES_WEEKLY_ADJUSTED"
            data = get_alpha_vantage_data(vix_proxy, function)
            if data:
                vix_df = parse_alpha_vantage_weekly(data, vix_proxy)
                
            # If weekly fails, try to resample from daily
            if vix_df.empty:
                st.warning("Could not get weekly VIX data directly. Trying daily data instead...")
                daily_function = "TIME_SERIES_DAILY_ADJUSTED"
                daily_data = get_alpha_vantage_data(vix_proxy, daily_function, outputsize="full")
                if daily_data:
                    daily_series = parse_alpha_vantage_daily(daily_data, vix_proxy)
                    if not daily_series.empty:
                        vix_df = daily_series.resample('W').last()
                        st.info("Successfully created weekly VIX data from daily data")
        
        elif freq == 'M':
            # First try monthly data
            function = "TIME_SERIES_MONTHLY_ADJUSTED"
            data = get_alpha_vantage_data(vix_proxy, function)
            if data:
                vix_df = parse_alpha_vantage_monthly(data, vix_proxy)
                
            # If monthly fails, try to resample from daily
            if vix_df.empty:
                st.warning("Could not get monthly VIX data directly. Trying daily data instead...")
                daily_function = "TIME_SERIES_DAILY_ADJUSTED"
                daily_data = get_alpha_vantage_data(vix_proxy, daily_function, outputsize="full")
                if daily_data:
                    daily_series = parse_alpha_vantage_daily(daily_data, vix_proxy)
                    if not daily_series.empty:
                        vix_df = daily_series.resample('M').last()
                        st.info("Successfully created monthly VIX data from daily data")
        
        # Filter to the requested date range
        if not vix_df.empty:
            vix_df = vix_df[(vix_df.index >= start) & (vix_df.index <= end)]
            
            # Don't force a specific frequency as it can cause errors
            # Instead, try to infer the frequency and use it only if valid
            inferred_freq = pd.infer_freq(vix_df.index)
            if inferred_freq:
                vix_df.index.freq = inferred_freq
            
            # Rename to VIX
            vix_df.name = "VIX"
            
            # Return as Series
            return vix_df
        else:
            st.warning("No VIX proxy data available for the selected period")
            return pd.Series(name="VIX")
    
    except Exception as e:
        st.error(f"Error fetching VIX proxy data: {e}")
        return pd.Series(name="VIX")

def query_alpha_vantage(function, ticker, outputsize="compact"):
    """Helper function to query Alpha Vantage API with rate limiting."""
    # Check if we've already hit rate limits
    global ALPHA_VANTAGE_RATE_LIMIT_HIT
    if ALPHA_VANTAGE_RATE_LIMIT_HIT:
        st.warning("API rate limit already reached. Waiting before trying again.")
        time.sleep(60)  # Wait a full minute
        ALPHA_VANTAGE_RATE_LIMIT_HIT = False
    
    # Make the API call
    try:
        params = {
            "function": function,
            "symbol": ticker,
            "outputsize": outputsize,
            "datatype": "json",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        data = response.json()
        
        # Check for error messages
        if "Error Message" in data:
            st.warning(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
            return None
        
        # Check for API limit messages
        if "Note" in data and "call frequency" in data["Note"]:
            st.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            ALPHA_VANTAGE_RATE_LIMIT_HIT = True
            time.sleep(15)  # Wait before next call
            return None
        
        return data
    except Exception as e:
        st.error(f"Error querying Alpha Vantage: {e}")
        return None

# Initialize rate limit flag
ALPHA_VANTAGE_RATE_LIMIT_HIT = False
