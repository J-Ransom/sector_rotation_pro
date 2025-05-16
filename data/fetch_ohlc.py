"""
Module for fetching OHLC (Open, High, Low, Close) data from Alpha Vantage.
"""

import pandas as pd
import os
import streamlit as st
from datetime import datetime
from .fetch import query_alpha_vantage, ALPHA_VANTAGE_API_KEY

def get_ohlc_data(ticker, start_date, end_date):
    """
    Fetch OHLC (Open, High, Low, Close) data for a single ticker from Alpha Vantage.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    start_date : datetime.date or str
        Start date for data
    end_date : datetime.date or str
        End date for data
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with OHLC data and volume
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.error(
            "Alpha Vantage API key not found. Please add your key to the .env file."
        )
        return None
    
    # Convert dates to strings if they aren't already
    if not isinstance(start_date, str):
        start_date = start_date.strftime("%Y-%m-%d")
    
    if not isinstance(end_date, str):
        end_date = end_date.strftime("%Y-%m-%d")
    
    try:
        # Query Alpha Vantage for daily adjusted data (includes adjusted OHLC)
        data = query_alpha_vantage("TIME_SERIES_DAILY_ADJUSTED", ticker, outputsize="full")
        
        if "Error Message" in data:
            st.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None
        
        if "Time Series (Daily Adjusted)" not in data:
            st.warning(f"No OHLC data found for {ticker}. Response keys: {list(data.keys())}")
            return None
        
        # Extract time series data
        time_series = data["Time Series (Daily Adjusted)"]
        
        # Create DataFrame from the time series data
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort the index in ascending order
        df = df.sort_index()
        
        # Filter data by date range
        df = df.loc[start_date:end_date]
        
        if df.empty:
            st.warning(f"No data available for {ticker} in the specified date range.")
            return None
        
        # Rename columns for consistency
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adjusted Close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split"
        })
        
        # Convert string values to float
        for col in ["Open", "High", "Low", "Close", "Adjusted Close", "Volume", "Dividend", "Split"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Use adjusted prices for more accurate analysis
        df["Open"] = df["Open"] * (df["Adjusted Close"] / df["Close"])
        df["High"] = df["High"] * (df["Adjusted Close"] / df["Close"])
        df["Low"] = df["Low"] * (df["Adjusted Close"] / df["Close"])
        df["Close"] = df["Adjusted Close"]
        
        # Keep only OHLCV columns
        ohlc_columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df[ohlc_columns]
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching OHLC data for {ticker}: {str(e)}")
        return None
