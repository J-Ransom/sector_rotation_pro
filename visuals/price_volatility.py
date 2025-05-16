"""
Visualizations for price and volatility analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_price_candlestick(price_data, ticker, title=None, height=600):
    """
    Create a log-scale candlestick chart for a single ticker.
    
    Parameters
    ----------
    price_data : pandas.DataFrame
        DataFrame with open, high, low, close prices
    ticker : str
        Ticker symbol to plot
    title : str, optional
        Chart title, by default None
    height : int, optional
        Chart height, by default 600
        
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with candlestick chart
    """
    if title is None:
        title = f"{ticker} Price History (Log Scale)"
    
    # Create subplot for price chart
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name=ticker,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
    )
    
    # Configure y-axis for log scale
    fig.update_yaxes(type="log")
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title="Price (Log Scale)",
        template="plotly_white",
        legend_title="Ticker",
        xaxis_rangeslider_visible=False,  # Hide range slider
    )
    
    return fig


def create_volatility_chart(price_data, ticker, window=20, ma_window=50, title=None, height=300):
    """
    Create a volatility chart with simple moving average.
    
    Parameters
    ----------
    price_data : pandas.DataFrame
        DataFrame with close prices
    ticker : str
        Ticker symbol to plot
    window : int, optional
        Window for volatility calculation, by default 20
    ma_window : int, optional
        Window for moving average, by default 50
    title : str, optional
        Chart title, by default None
    height : int, optional
        Chart height, by default 300
        
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with volatility chart
    """
    if title is None:
        title = f"{ticker} {window}-day Volatility"
    
    # Calculate log returns
    returns = np.log(price_data['Close'] / price_data['Close'].shift(1))
    
    # Calculate rolling volatility (annualized)
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Calculate moving average of volatility
    vol_ma = volatility.rolling(window=ma_window).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add volatility trace
    fig.add_trace(
        go.Scatter(
            x=volatility.index,
            y=volatility,
            mode='lines',
            name=f'{window}-day Volatility',
            line=dict(color='#2962ff', width=1.5)
        )
    )
    
    # Add moving average trace
    fig.add_trace(
        go.Scatter(
            x=vol_ma.index,
            y=vol_ma,
            mode='lines',
            name=f'{ma_window}-day MA',
            line=dict(color='#ff6d00', width=1.5, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        template="plotly_white",
        legend_title="Metric",
        hovermode="x unified"
    )
    
    return fig


def create_hurst_chart(hurst_data, ticker, title=None, height=300):
    """
    Create a Hurst exponent chart.
    
    Parameters
    ----------
    hurst_data : pandas.Series
        Series with Hurst exponent values
    ticker : str
        Ticker symbol
    title : str, optional
        Chart title, by default None
    height : int, optional
        Chart height, by default 300
        
    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure with Hurst exponent chart
    """
    if title is None:
        title = f"{ticker} Hurst Exponent (Rolling 252-day)"
    
    # Create figure
    fig = go.Figure()
    
    # Add Hurst exponent trace
    fig.add_trace(
        go.Scatter(
            x=hurst_data.index,
            y=hurst_data,
            mode='lines',
            name='Hurst Exponent',
            line=dict(color='#8e24aa', width=1.5)
        )
    )
    
    # Add reference lines
    fig.add_shape(
        type="line",
        x0=hurst_data.index[0],
        y0=0.5,
        x1=hurst_data.index[-1],
        y1=0.5,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    # Add annotations
    fig.add_annotation(
        x=hurst_data.index[len(hurst_data)//2],
        y=0.55,
        text="Trending (H > 0.5)",
        showarrow=False,
        yanchor="bottom",
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=hurst_data.index[len(hurst_data)//2],
        y=0.45,
        text="Mean-reverting (H < 0.5)",
        showarrow=False,
        yanchor="top",
        font=dict(size=10)
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title="Hurst Exponent",
        template="plotly_white",
        legend_title="Metric",
        hovermode="x unified",
        yaxis=dict(range=[0, 1])  # Force y-axis range from 0 to 1
    )
    
    return fig


def create_price_volatility_dashboard(ohlc_data, ticker, hurst_data=None):
    """
    Create a dashboard with price, volatility, and Hurst exponent charts.
    
    Parameters
    ----------
    ohlc_data : pandas.DataFrame
        DataFrame with open, high, low, close prices
    ticker : str
        Ticker symbol
    hurst_data : pandas.Series, optional
        Series with Hurst exponent values, by default None
        
    Returns
    -------
    tuple
        (price_fig, volatility_fig, hurst_fig) - Tuple of Plotly figures
    """
    # Create price chart
    price_fig = create_price_candlestick(ohlc_data, ticker)
    
    # Create volatility chart
    volatility_fig = create_volatility_chart(ohlc_data, ticker)
    
    # Create Hurst exponent chart if data is provided
    hurst_fig = None
    if hurst_data is not None:
        hurst_fig = create_hurst_chart(hurst_data, ticker)
    
    return price_fig, volatility_fig, hurst_fig
