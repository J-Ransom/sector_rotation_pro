import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def create_forecast_figure(historical_data, forecast_results, title="Price Forecast"):
    """
    Create a visualization of historical data and Prophet forecast.
    
    Parameters:
    ----------
    historical_data : pandas.Series
        Historical price data
    forecast_results : dict
        Dictionary with Prophet forecast results
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Figure with historical data and forecast
    """
    # Check if we have any forecast results
    if not forecast_results or 'prophet' not in forecast_results:
        # If no forecast results, just plot historical data
        st.warning("No Prophet forecast data available - showing historical data only")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        return fig
    
    # Get Prophet results
    prophet_data = forecast_results['prophet']
    
    # Create a main figure for the forecast
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add forecast
    if 'forecast' in prophet_data:
        fig.add_trace(
            go.Scatter(
                x=prophet_data['forecast'].index,
                y=prophet_data['forecast'].values,
                mode='lines',
                name='Prophet Forecast',
                line=dict(color='green', width=2)
            )
        )
        
        # Add confidence intervals
        if 'lower_bound' in prophet_data and 'upper_bound' in prophet_data:
            # Add upper bound
            fig.add_trace(
                go.Scatter(
                    x=prophet_data['upper_bound'].index,
                    y=prophet_data['upper_bound'].values,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            # Add lower bound with fill
            fig.add_trace(
                go.Scatter(
                    x=prophet_data['lower_bound'].index,
                    y=prophet_data['lower_bound'].values,
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(0, 128, 0, 0.2)',  # Light green
                    fill='tonexty',
                    name='90% Confidence Interval'
                )
            )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_prophet_components_figure(forecast_results, title="Prophet Components"):
    """
    Create a visualization of Prophet forecast components.
    
    Parameters:
    ----------
    forecast_results : dict
        Dictionary with Prophet forecast results including components
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Figure with Prophet components
    """
    # Check if we have forecast results with components
    if (not forecast_results or 
        'prophet' not in forecast_results or
        'components' not in forecast_results['prophet'] or
        not forecast_results['prophet']['components']):
        st.warning("No Prophet components available")
        return None
    
    # Get components with validation
    try:
        components = forecast_results['prophet']['components']
        if not isinstance(components, dict):
            st.warning(f"Components data is not a dictionary - found {type(components)}")
            return None
            
        # Remove any invalid component data
        valid_components = {}
        for name, data in components.items():
            if isinstance(data, pd.Series) and len(data) > 0:
                valid_components[name] = data
            else:
                st.warning(f"Component '{name}' has invalid data format or is empty - skipping")
        
        components = valid_components
        num_components = len(components)
        
        if num_components == 0:
            st.warning("No valid Prophet components found")
            return None
    except Exception as e:
        st.error(f"Error processing components: {e}")
        return None
    
    # Create subplot titles
    subplot_titles = []
    for comp_name in components.keys():
        subplot_titles.append(f"{comp_name.capitalize()} Component")
    
    # Create subplots - one for each component
    fig = make_subplots(
        rows=num_components,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=subplot_titles
    )
    
    # Component colors
    component_colors = {
        'trend': 'red',
        'yearly': 'purple',
        'weekly': 'orange',
        'daily': 'brown',
        'seasonal': 'blue'
    }
    
    # Add component plots
    row_idx = 1
    for comp_name, comp_data in components.items():
        # Get color for this component
        color = component_colors.get(comp_name, 'gray')
        
        # Add component line
        fig.add_trace(
            go.Scatter(
                x=comp_data.index,
                y=comp_data.values,
                mode='lines',
                name=f"{comp_name.capitalize()}",
                line=dict(color=color, width=2)
            ),
            row=row_idx,
            col=1
        )
        
        # Add zero reference line with better visibility
        fig.add_shape(
            type="line",
            x0=comp_data.index[0],
            y0=0,
            x1=comp_data.index[-1],
            y1=0,
            line=dict(color="rgba(100, 100, 100, 0.6)", width=1.5, dash="dash"),
            row=row_idx,
            col=1
        )
        
        # Add a text annotation for zero line for better context
        max_val = comp_data.max()
        min_val = comp_data.min()
        range_size = max_val - min_val
        
        # Only add annotation if there's significant range in the data
        if range_size > 0.001:
            fig.add_annotation(
                x=comp_data.index[0],
                y=0,
                text="Zero",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=10, color="rgba(100, 100, 100, 0.8)"),
                row=row_idx,
                col=1
            )
        
        # Update y-axis title
        fig.update_yaxes(
            title_text=f"{comp_name.capitalize()} Effect",
            row=row_idx,
            col=1
        )
        
        row_idx += 1
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=250 * num_components,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis title (only for the bottom plot)
    fig.update_xaxes(title_text="Date", row=num_components, col=1)
    
    return fig

def add_regime_overlay(fig, regime_data, historical_data):
    """
    Add regime overlay to a forecast figure.
    
    Parameters:
    ----------
    fig : plotly.graph_objects.Figure
        The figure to add the overlay to
    regime_data : pandas.Series
        Series with regime data (0=bear, 1=bull)
    historical_data : pandas.Series
        Historical price data used to align the regimes
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Figure with regime overlay
    """
    if regime_data is None or len(regime_data) == 0:
        return fig
    
    # Get min and max y values for the shapes
    if not fig.data:
        return fig
        
    y_vals = []
    for trace in fig.data:
        if 'y' in trace and trace.y is not None and len(trace.y) > 0:
            y_vals.extend(trace.y)
    
    if not y_vals:
        return fig
        
    y_min = min(y_vals) * 0.95
    y_max = max(y_vals) * 1.05
    
    # Add colored background for regimes
    for i in range(len(regime_data)):
        if i + 1 >= len(regime_data):
            continue
            
        # Get current regime
        regime = int(regime_data.iloc[i])
        
        # Get start and end dates
        start_date = regime_data.index[i]
        end_date = regime_data.index[i + 1]
        
        # Skip if dates are outside historical data range
        if end_date < historical_data.index[0] or start_date > historical_data.index[-1]:
            continue
        
        # Set color based on regime (green for bull, red for bear)
        color = "rgba(0, 128, 0, 0.1)" if regime == 1 else "rgba(255, 0, 0, 0.1)"
        
        # Add shape
        fig.add_shape(
            type="rect",
            x0=start_date,
            y0=y_min,
            x1=end_date,
            y1=y_max,
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        )
    
    return fig
