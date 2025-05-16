import plotly.express as px
import plotly.graph_objects as go

def create_rrg_figure(rrg_data, title="Relative Rotation Graph"):
    """
    Create an animated Relative Rotation Graph with trails.
    
    Parameters:
    ----------
    rrg_data : pandas.DataFrame
        DataFrame with RRG data including RS_Ratio_zscore, RS_Mom_zscore, ticker, and week columns
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive RRG plot
    """
    if rrg_data.empty:
        # Return empty figure with message if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No RRG data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Ensure we have the required columns
    required_cols = ['RS_Ratio_zscore', 'RS_Mom_zscore', 'ticker', 'week']
    if not all(col in rrg_data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in rrg_data.columns]
        raise ValueError(f"RRG data is missing required columns: {missing}")
    
    # Create the animated scatter plot
    fig = px.scatter(
        rrg_data, 
        x='RS_Ratio_zscore', 
        y='RS_Mom_zscore',
        color='ticker',
        hover_name='ticker',
        size_max=10,
        size=[10] * len(rrg_data),  # Consistent size
        animation_frame='week',
        animation_group='ticker',
        range_x=[-4, 4],
        range_y=[-4, 4],
        title=title
    )
    
    # Add lines connecting the points for each ticker to show trails
    for ticker in rrg_data['ticker'].unique():
        ticker_data = rrg_data[rrg_data['ticker'] == ticker].sort_values('week')
        
        fig.add_trace(
            go.Scatter(
                x=ticker_data['RS_Ratio_zscore'],
                y=ticker_data['RS_Mom_zscore'],
                mode='lines',
                line=dict(width=1, color='rgba(0,0,0,0.3)'),
                name=f"{ticker} Trail",
                showlegend=False,
                hoverinfo='none'
            )
        )
    
    # Add quadrant labels and lines
    fig.add_shape(
        type="line", x0=-4, y0=0, x1=4, y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    fig.add_shape(
        type="line", x0=0, y0=-4, x1=0, y1=4,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Add quadrant annotations
    quadrant_annotations = [
        dict(x=2, y=2, text="Leading<br>(Improving + Strong)", showarrow=False),
        dict(x=-2, y=2, text="Improving<br>(Improving + Weak)", showarrow=False),
        dict(x=-2, y=-2, text="Lagging<br>(Weakening + Weak)", showarrow=False),
        dict(x=2, y=-2, text="Weakening<br>(Weakening + Strong)", showarrow=False)
    ]
    for annotation in quadrant_annotations:
        fig.add_annotation(annotation)
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="RS-Ratio (Relative Strength)",
        yaxis_title="RS-Momentum",
        legend_title="Sectors",
        height=700,
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14
        )
    )
    
    # Update animation settings
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Week: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f"{week}"],
                        {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }
                    ],
                    "label": str(week),
                    "method": "animate"
                }
                for week in sorted(rrg_data['week'].unique())
            ]
        }]
    )
    
    return fig

def create_momentum_values_figure(momentum_data, title="Sector Momentum"):
    """
    Create a multi-timeframe momentum values visualization.
    
    Parameters:
    ----------
    momentum_data : dict
        Dictionary with keys '3m', '6m', '12m' containing momentum values for each period
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Three-panel figure showing momentum values for different timeframes
    """
    # Create figure with 3 subplots (one for each timeframe)
    fig = go.Figure()
    
    # Define colors for the bars - green for positive, red for negative
    colors = {
        '1m': ["green" if x >= 0 else "red" for x in momentum_data['1m'].values],
        '3m': ["green" if x >= 0 else "red" for x in momentum_data['3m'].values],
        '6m': ["green" if x >= 0 else "red" for x in momentum_data['6m'].values],
        '12m': ["green" if x >= 0 else "red" for x in momentum_data['12m'].values]
    }
    
    # Create subplot for 1-month momentum - shown by default
    fig.add_trace(go.Bar(
        y=momentum_data['1m'].index,
        x=momentum_data['1m'].values,
        name="1-Month",
        orientation='h',
        marker_color=colors['1m'],
        marker_line_width=0,
        opacity=0.8,
        visible=True
    ))
    
    # Create subplot for 3-month momentum
    fig.add_trace(go.Bar(
        y=momentum_data['3m'].index,
        x=momentum_data['3m'].values,
        name="3-Month",
        orientation='h',
        marker_color=colors['3m'],
        marker_line_width=0,
        opacity=0.8,
        visible=False
    ))
    
    # Create subplot for 6-month momentum
    fig.add_trace(go.Bar(
        y=momentum_data['6m'].index,
        x=momentum_data['6m'].values,
        name="6-Month",
        orientation='h',
        marker_color=colors['6m'],
        marker_line_width=0,
        opacity=0.8,
        visible=False
    ))
    
    # Create subplot for 12-month momentum
    fig.add_trace(go.Bar(
        y=momentum_data['12m'].index,
        x=momentum_data['12m'].values,
        name="12-Month",
        orientation='h',
        marker_color=colors['12m'],
        marker_line_width=0,
        opacity=0.8,
        visible=False
    ))
    
    # Create buttons for the timeframe selection
    buttons = [
        dict(
            label="1-Month",
            method="update",
            args=[
                {"visible": [True, False, False, False]},
                {"title": "1-Month Sector Momentum"}
            ]
        ),
        dict(
            label="3-Month",
            method="update",
            args=[
                {"visible": [False, True, False, False]},
                {"title": "3-Month Sector Momentum"}
            ]
        ),
        dict(
            label="6-Month",
            method="update",
            args=[
                {"visible": [False, False, True, False]},
                {"title": "6-Month Sector Momentum"}
            ]
        ),
        dict(
            label="12-Month",
            method="update",
            args=[
                {"visible": [False, False, False, True]},
                {"title": "12-Month Sector Momentum"}
            ]
        )
    ]
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ],
        annotations=[
            dict(
                text="Timeframe:",
                x=0,
                y=1.15,
                xref="paper",
                yref="paper",
                align="left",
                showarrow=False
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=600,
        title="1-Month Sector Momentum",  # Initial title
        xaxis_title="Momentum (%)",
        yaxis_title="Sector",
        hoverlabel=dict(
            bgcolor="white",
            font_size=14
        ),
        # Add a vertical line at x=0
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=0,
                y0=0,
                x1=0,
                y1=1,
                line=dict(
                    color="black",
                    width=1,
                    dash="dash"
                )
            )
        ]
    )
    
    return fig


def create_momentum_rank_figure(momentum_ranks, title="Momentum Ranking"):
    """
    Legacy function to create a bar chart for momentum rankings.
    Kept for backward compatibility.
    
    Parameters:
    ----------
    momentum_ranks : pandas.Series
        Series with momentum ranks for each ticker
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Bar chart of momentum rankings
    """
    # Create a simple DataFrame with the ranks
    import pandas as pd
    
    momentum_data = pd.DataFrame(momentum_ranks).reset_index()
    momentum_data.columns = ['ticker', 'rank']
    momentum_data = momentum_data.sort_values('rank')
    
    # Create the bar chart
    fig = px.bar(
        momentum_data,
        x='rank',
        y='ticker',
        orientation='h',
        title=title,
        labels={
            'rank': 'Rank (1 = Highest Momentum)',
            'ticker': 'Ticker'
        }
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=500,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14
        )
    )
    
    return fig

def rrg_fig(prices, returns, benchmark="SPY", window=10):
    """
    Create both RRG and momentum rank figures.
    
    Parameters:
    ----------
    prices : pandas.DataFrame
        DataFrame with price data
    returns : pandas.DataFrame
        DataFrame with return data
    benchmark : str
        Ticker symbol for the benchmark
    window : int
        Window size for RS calculations
        
    Returns:
    -------
    tuple of plotly.graph_objects.Figure
        (rrg_figure, momentum_rank_figure)
    """
    # Import here to avoid circular imports
    from ..analytics.rrg import calc_rrg
    from ..analytics.ta_factors import calculate_momentum_rank
    
    # Calculate RRG data
    rrg_data = calc_rrg(prices, benchmark, window)
    
    # Calculate momentum rankings
    momentum_ranks = calculate_momentum_rank(prices)
    
    # Create the figures
    rrg_figure = create_rrg_figure(rrg_data, title="Relative Rotation Graph (RRG)")
    momentum_figure = create_momentum_rank_figure(momentum_ranks, title="12-Month Momentum Ranking")
    
    return rrg_figure, momentum_figure
