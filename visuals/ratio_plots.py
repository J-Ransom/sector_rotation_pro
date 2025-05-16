import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_ratio_plot(prices, ratio_pairs, title="Sector Ratio Analysis"):
    """
    Create a plot of sector ratios (e.g., XLY/XLP for risk-on/risk-off).
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with price data
    ratio_pairs : dict
        Dictionary mapping ratio names to tuples of (numerator, denominator)
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Ratio plot
    """
    # Calculate ratios
    ratios = pd.DataFrame(index=prices.index)
    
    for ratio_name, (numerator, denominator) in ratio_pairs.items():
        if numerator in prices.columns and denominator in prices.columns:
            ratios[ratio_name] = prices[numerator] / prices[denominator]
    
    if ratios.empty:
        # Return empty figure with message if no valid ratios
        fig = go.Figure()
        fig.add_annotation(
            text="No valid ratio pairs available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create the figure
    fig = go.Figure()
    
    for ratio_name in ratios.columns:
        fig.add_trace(
            go.Scatter(
                x=ratios.index,
                y=ratios[ratio_name],
                mode='lines',
                name=ratio_name
            )
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Date",
        yaxis_title="Ratio Value",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_multi_ratio_plot(prices, ratio_pairs, title="Sector Ratio Comparison"):
    """
    Create a multi-panel plot of sector ratios with individual scaling.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with price data
    ratio_pairs : dict
        Dictionary mapping ratio names to tuples of (numerator, denominator)
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Multi-panel ratio plot
    """
    # Calculate ratios
    ratios = pd.DataFrame(index=prices.index)
    valid_ratios = []
    
    for ratio_name, (numerator, denominator) in ratio_pairs.items():
        if numerator in prices.columns and denominator in prices.columns:
            ratios[ratio_name] = prices[numerator] / prices[denominator]
            valid_ratios.append(ratio_name)
    
    if ratios.empty:
        # Return empty figure with message if no valid ratios
        fig = go.Figure()
        fig.add_annotation(
            text="No valid ratio pairs available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create subplots, one for each ratio
    fig = make_subplots(
        rows=len(valid_ratios),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=valid_ratios
    )
    
    # Add traces to subplots
    for i, ratio_name in enumerate(valid_ratios):
        fig.add_trace(
            go.Scatter(
                x=ratios.index,
                y=ratios[ratio_name],
                mode='lines',
                name=ratio_name
            ),
            row=i+1,
            col=1
        )
        
        # Calculate and add 200-day moving average
        if len(ratios) >= 200:
            ma = ratios[ratio_name].rolling(window=200).mean()
            fig.add_trace(
                go.Scatter(
                    x=ratios.index,
                    y=ma,
                    mode='lines',
                    name=f"{ratio_name} 200-day MA",
                    line=dict(dash='dash', color='red')
                ),
                row=i+1,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=300 * len(valid_ratios),
        showlegend=False
    )
    
    # Update x-axis titles (only for the bottom plot)
    fig.update_xaxes(title_text="Date", row=len(valid_ratios), col=1)
    
    return fig

def ratio_plots(prices):
    """
    Create various ratio plots.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with price data
        
    Returns:
    -------
    tuple of plotly.graph_objects.Figure
        (combined_ratio_plot, multi_ratio_plot)
    """
    # Define common ratio pairs to analyze
    ratio_pairs = {
        "XLY/XLP": ("XLY", "XLP"),  # Consumer Discretionary / Consumer Staples (risk-on/risk-off)
        "XLK/XLU": ("XLK", "XLU"),  # Technology / Utilities (growth/defensive)
        "XLF/XLV": ("XLF", "XLV"),  # Financials / Healthcare
        "XLI/XLB": ("XLI", "XLB"),  # Industrials / Materials
        "XLC/XLRE": ("XLC", "XLRE")  # Communication Services / Real Estate
    }
    
    # Create the plots
    combined_ratio_plot = create_ratio_plot(
        prices, 
        {k: ratio_pairs[k] for k in ["XLY/XLP", "XLK/XLU"]},  # Only the key risk pairs
        title="Risk-On/Risk-Off Ratio Analysis"
    )
    
    multi_ratio_plot = create_multi_ratio_plot(
        prices,
        ratio_pairs,
        title="Sector Ratio Comparison"
    )
    
    return combined_ratio_plot, multi_ratio_plot
