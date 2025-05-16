import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

def create_correlation_heatmap(corr_matrix, title="Correlation Matrix"):
    """
    Create a correlation heatmap using Plotly.
    
    Parameters:
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Correlation heatmap
    """
    # Create the heatmap
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title
    )
    
    # Add correlation values as text
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{corr_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white", size=9)
            )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=700,
        width=700,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title="",
        yaxis_title="",
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300
        )
    )
    
    return fig

def create_rsi_time_series(rsi_df, title="RSI (14) Time Series", height=600):
    """
    Create a time series chart of RSI values for multiple tickers.
    
    Parameters:
    ----------
    rsi_df : pandas.DataFrame
        DataFrame with RSI values for each ticker over time
    title : str
        Title for the figure
    height : int
        Height of the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Time series chart of RSI values
    """
    # Create figure
    fig = go.Figure()
    
    # Define colors for each line
    # We'll use a colorful palette to distinguish between sectors
    colors = px.colors.qualitative.Plotly
    
    # Add traces for each ticker
    for i, col in enumerate(rsi_df.columns):
        color_idx = i % len(colors)  # Cycle through colors if more tickers than colors
        
        fig.add_trace(
            go.Scatter(
                x=rsi_df.index,
                y=rsi_df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[color_idx]),
                hovertemplate='%{y:.1f}'
            )
        )
    
    # Add reference lines for overbought (70) and oversold (30) levels
    fig.add_shape(
        type="line",
        x0=rsi_df.index[0],
        y0=70,
        x1=rsi_df.index[-1],
        y1=70,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=rsi_df.index[0],
        y0=30,
        x1=rsi_df.index[-1],
        y1=30,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    # Add annotations for overbought and oversold levels
    fig.add_annotation(
        x=rsi_df.index[0],
        y=70,
        text="Overbought",
        showarrow=False,
        font=dict(color="red"),
        xanchor="right"
    )
    
    fig.add_annotation(
        x=rsi_df.index[0],
        y=30,
        text="Oversold",
        showarrow=False,
        font=dict(color="green"),
        xanchor="right"
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="RSI (14)",
        height=height,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            range=[0, 100]  # RSI ranges from 0 to 100
        )
    )
    
    return fig


def create_rsi_heatmap(rsi_values, title="RSI Dashboard"):
    """
    Create a RSI heatmap/bargrid using Plotly.
    
    Parameters:
    ----------
    rsi_values : pandas.DataFrame
        DataFrame with RSI values for each ticker (latest only)
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        RSI heatmap/bargrid
    """
    # Extract the latest RSI values if a DataFrame with time series is provided
    if isinstance(rsi_values, pd.DataFrame) and rsi_values.shape[0] > 1:
        latest_rsi = rsi_values.iloc[-1]
    else:
        latest_rsi = rsi_values
    
    # Sort values from highest to lowest
    sorted_rsi = latest_rsi.sort_values(ascending=False)
    
    # Create color scale for RSI values
    colors = []
    for value in sorted_rsi:
        if value >= 70:
            colors.append("red")  # Overbought
        elif value <= 30:
            colors.append("green")  # Oversold
        else:
            # Gradient from green to yellow to red as RSI goes from 30 to 70
            if 30 < value < 50:
                # Green to yellow gradient
                intensity = (value - 30) / 20  # 0 at RSI=30, 1 at RSI=50
                colors.append(f"rgb({int(255 * intensity)}, 255, 0)")
            else:
                # Yellow to red gradient
                intensity = (value - 50) / 20  # 0 at RSI=50, 1 at RSI=70
                colors.append(f"rgb(255, {int(255 * (1 - intensity))}, 0)")
    
    # Create the bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=sorted_rsi.index,
            y=sorted_rsi.values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in sorted_rsi.values],
            textposition="auto"
        )
    )
    
    # Add reference lines for overbought/oversold levels
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(sorted_rsi) - 0.5,
        y0=70,
        y1=70,
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(sorted_rsi) - 0.5,
        y0=30,
        y1=30,
        line=dict(color="green", width=2, dash="dash")
    )
    
    # Add annotations for overbought/oversold levels
    fig.add_annotation(
        x=len(sorted_rsi) - 1,
        y=72,
        text="Overbought (70)",
        showarrow=False,
        font=dict(color="red"),
        xanchor="right"
    )
    fig.add_annotation(
        x=len(sorted_rsi) - 1,
        y=28,
        text="Oversold (30)",
        showarrow=False,
        font=dict(color="green"),
        xanchor="right"
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="",
        yaxis_title="RSI (14)",
        yaxis=dict(range=[0, 100]),
        height=500
    )
    
    return fig

def create_seasonality_heatmap(returns, title="Monthly Seasonality"):
    """
    Create a seasonality heatmap of monthly returns.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Seasonality heatmap
    """
    # Ensure returns have a DatetimeIndex
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns DataFrame must have a DatetimeIndex")
    
    # Create month and year columns
    monthly_returns = returns.copy()
    monthly_returns['Month'] = monthly_returns.index.month
    monthly_returns['Year'] = monthly_returns.index.year
    
    # Calculate average returns by month for each ticker
    avg_monthly_returns = {}
    
    for ticker in returns.columns:
        # Pivot to get returns by month and year
        pivot = pd.pivot_table(
            monthly_returns, 
            values=ticker, 
            index='Year', 
            columns='Month',
            aggfunc='sum'  # Sum daily/weekly returns to get monthly return
        )
        
        # Calculate average return for each month across years
        avg_monthly = pivot.mean()
        
        # Store in the dictionary
        avg_monthly_returns[ticker] = avg_monthly
    
    # Convert to DataFrame
    seasonality_df = pd.DataFrame(avg_monthly_returns)
    
    # Replace month numbers with month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonality_df.index = [month_names[i-1] for i in seasonality_df.index]
    
    # Create the heatmap using Plotly
    fig = px.imshow(
        seasonality_df.T * 100,  # Convert to percentage
        x=seasonality_df.index,
        y=seasonality_df.columns,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title=title
    )
    
    # Add values as text
    for i, ticker in enumerate(seasonality_df.columns):
        for j, month in enumerate(seasonality_df.index):
            value = seasonality_df.loc[month, ticker] * 100
            color = "black" if abs(value) < 5 else "white"
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{value:.1f}%",
                showarrow=False,
                font=dict(color=color, size=9)
            )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Ticker",
        height=600,
        coloraxis_colorbar=dict(
            title="Avg Monthly Return (%)",
            thicknessmode="pixels",
            thickness=20,
            lenmode="pixels",
            len=300
        )
    )
    
    return fig

def corr_heat(returns):
    """
    Create correlation and RSI heatmaps.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
        
    Returns:
    -------
    tuple of plotly.graph_objects.Figure
        (corr_heatmap, rsi_heatmap, seasonality_heatmap)
    """
    # Import here to avoid circular imports
    from ..analytics.stats_tests import calculate_correlation_matrix
    from ..analytics.ta_factors import calculate_rsi
    
    # Get prices DataFrame from the same index as returns
    try:
        # Assumes returns were calculated from prices
        prices = returns.copy()
        prices.iloc[0] = 100  # Set initial price to 100
        for i in range(1, len(returns)):
            prices.iloc[i] = prices.iloc[i-1] * (1 + returns.iloc[i])
    except:
        # If the above fails, we'll use returns as is for correlation
        prices = returns
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(returns)
    
    # Calculate RSI values
    rsi_values = calculate_rsi(prices)
    
    # Create the figures
    corr_heatmap = create_correlation_heatmap(corr_matrix, title="Correlation Matrix")
    rsi_heatmap = create_rsi_heatmap(rsi_values, title="RSI (14) Dashboard")
    seasonality_heatmap = create_seasonality_heatmap(returns, title="Monthly Seasonality Heatmap")
    
    return corr_heatmap, rsi_heatmap, seasonality_heatmap
