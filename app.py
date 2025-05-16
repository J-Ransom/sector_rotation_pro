import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import plotly.graph_objects as go
import plotly.io as pio

# Set default Plotly template
pio.templates.default = "plotly_white"

# Import local modules
from data.fetch import get_prices, get_vix_data
from data.fetch_ohlc import get_ohlc_data
from analytics.rrg import calc_rrg
from analytics.ta_factors import (
    calculate_rsi, calculate_momentum_values, calculate_rolling_beta,
    calculate_rolling_sharpe, calculate_rolling_moments, calculate_drawdowns
)
from analytics.stats_tests import (
    calculate_correlation_matrix, calculate_lead_lag,
    calculate_granger_causality
)
from analytics.regimes import generate_regimes
from analytics.forecasting import generate_forecasts
from analytics.hurst import calculate_rolling_hurst

from visuals.rrg_fig import create_rrg_figure, create_momentum_values_figure
from visuals.heatmaps import (
    create_correlation_heatmap, create_rsi_heatmap, create_seasonality_heatmap,
    create_rsi_time_series
)
from visuals.network import create_network_graph
from visuals.pca_cluster import create_pca_figure
from visuals.ratio_plots import create_ratio_plot, create_multi_ratio_plot
from visuals.forecast_fig import (
    create_forecast_figure, create_prophet_components_figure, add_regime_overlay
)
from visuals.price_volatility import (
    create_price_candlestick, create_volatility_chart, create_hurst_chart,
    create_price_volatility_dashboard
)

# Page configuration
st.set_page_config(
    page_title="Sector-Rotation Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("Sector-Rotation Pro Dashboard")
st.markdown("A comprehensive dashboard for sector rotation analysis with multiple analytical components and visualizations.")

# Sidebar configuration
st.sidebar.title("Settings")

# Date range selector
today = datetime.datetime.now().date()
default_start = today - timedelta(days=365*3)  # 3 years ago by default

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    max_value=today
)

end_date = st.sidebar.date_input(
    "End Date",
    value=today,
    max_value=today
)

# Always use daily data for more accurate analysis
freq = "D"  # Daily data only
st.sidebar.info("Analysis using daily data for maximum accuracy")

# Ticker selector
st.sidebar.subheader("ETF Selection")

# Hardcoded sector ETFs
all_tickers = {
    "SPY": "SPY",  # S&P 500 (benchmark)
    "XLK": "Technology",
    "XLV": "Health Care",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLE": "Energy"
}

selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=list(all_tickers.keys()),
    default=list(all_tickers.keys()),
    format_func=lambda x: f"{x} ({all_tickers[x]})"
)

# Ensure SPY is always selected as it's the benchmark
if "SPY" not in selected_tickers:
    st.sidebar.warning("SPY must be included as the benchmark. Adding it back.")
    selected_tickers = ["SPY"] + selected_tickers

# Module selection
st.sidebar.subheader("Dashboard Modules")
all_modules = [
    "Price & Volatility Analysis",  # New tab added as first option
    "Relative Rotation Graph (RRG)",
    "Correlation Analysis",
    "Technical Indicators",
    "Ratio Analysis",
    "Network Analysis",
    "PCA & Clustering",
    "Forecasting"
]

# Set default modules - ensure the new Price & Volatility tab is always included
default_modules = ["Price & Volatility Analysis"]
if len(all_modules) > 1:
    default_modules.extend(all_modules[1:])

selected_modules = st.sidebar.multiselect(
    "Select Modules to Display",
    options=all_modules,
    default=default_modules
)

# Advanced settings
st.sidebar.subheader("Advanced Settings")

# Network threshold
network_threshold = st.sidebar.slider(
    "Correlation Threshold for Network",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05
)

# Set fixed forecast horizon to 30 days (1 month)
forecast_horizon = 30

# Show regime overlay
show_regimes = st.sidebar.checkbox("Show Regime Overlay", value=True)

# Fetch data
@st.cache_data(ttl=86400)
def load_data(tickers, start, end):
    """Load and cache price data using daily frequency"""
    ticker_str = " ".join(tickers)
    # Always use daily frequency for all operations
    prices = get_prices(ticker_str, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), freq="D")
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Get VIX data if regimes are enabled
    vix = None
    if show_regimes:
        vix = get_vix_data(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), freq)
    
    return prices, returns, vix

@st.cache_data(ttl=86400)
def load_ohlc_data(ticker, start, end):
    """Load and cache OHLC data for a single ticker using Alpha Vantage"""
    try:
        # Use Alpha Vantage to get OHLC data
        data = get_ohlc_data(ticker, start, end)
        if data is None or data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching OHLC data for {ticker}: {e}")
        return None

@st.cache_data(ttl=86400)
def calculate_hurst_data(price_series, window=252):
    """Calculate and cache Hurst exponent data"""
    try:
        # Calculate rolling Hurst exponent
        hurst_series = calculate_rolling_hurst(price_series, window=window)
        return hurst_series
    except Exception as e:
        st.error(f"Error calculating Hurst exponent: {e}")
        return None

try:
    # Loading spinner
    with st.spinner("Loading data..."):
        prices, returns, vix = load_data(selected_tickers, start_date, end_date)
        
        if prices.empty:
            st.error("No data available for the selected tickers and date range.")
            st.stop()
            
        # Calculate regime data if enabled
        regime_data = None
        if show_regimes and vix is not None and "SPY" in prices.columns:
            regime_data = generate_regimes(prices["SPY"], vix, window=52)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Create tabs for each selected module
tab_names = selected_modules
tabs = st.tabs(tab_names)

# Populate tabs with visualizations
for i, tab_name in enumerate(tab_names):
    with tabs[i]:
        if tab_name == "Price & Volatility Analysis":
            st.subheader("Price & Volatility Analysis")
            
            # Create a ticker selector for this tab
            price_vol_ticker = st.selectbox(
                "Select Ticker for Analysis",
                options=selected_tickers,
                index=0,  # Default to first ticker (often SPY)
                format_func=lambda x: f"{x} ({all_tickers.get(x, '')})"
            )
            
            # Load OHLC data for the selected ticker
            with st.spinner(f"Loading OHLC data for {price_vol_ticker}..."):
                ohlc_data = load_ohlc_data(price_vol_ticker, start_date, end_date)
                
                if ohlc_data is None or ohlc_data.empty:
                    st.error(f"No OHLC data available for {price_vol_ticker}. Please select another ticker.")
                else:
                    # Calculate Hurst exponent
                    with st.spinner("Calculating Hurst exponent..."):
                        hurst_data = calculate_hurst_data(ohlc_data['Close'])
                    
                    # Create two columns for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Calculate current volatility (20-day)
                        returns = np.log(ohlc_data['Close'] / ohlc_data['Close'].shift(1))
                        current_vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
                        st.metric("Current Volatility (Annualized)", f"{current_vol:.2%}")
                        
                    with col2:
                        # Calculate current Hurst exponent if available
                        if hurst_data is not None and not hurst_data.empty:
                            latest_hurst = hurst_data.dropna().iloc[-1] if not hurst_data.dropna().empty else np.nan
                            if not np.isnan(latest_hurst):
                                regime = "Trending" if latest_hurst > 0.5 else "Mean-Reverting"
                                st.metric("Current Hurst Exponent", f"{latest_hurst:.3f}", 
                                         delta=regime, delta_color="normal")
                            else:
                                st.metric("Current Hurst Exponent", "N/A")
                        else:
                            st.metric("Current Hurst Exponent", "N/A")
                    
                    # Create log candlestick chart
                    price_fig = create_price_candlestick(ohlc_data, price_vol_ticker)
                    st.plotly_chart(price_fig, use_container_width=True)
                    
                    # Create volatility chart
                    vol_fig = create_volatility_chart(ohlc_data, price_vol_ticker)
                    st.plotly_chart(vol_fig, use_container_width=True)
                    
                    # Create Hurst exponent chart if available
                    if hurst_data is not None and not hurst_data.empty and not hurst_data.dropna().empty:
                        hurst_fig = create_hurst_chart(hurst_data, price_vol_ticker)
                        st.plotly_chart(hurst_fig, use_container_width=True)
                        
                        # Add explanation of Hurst exponent
                        with st.expander("About the Hurst Exponent"):
                            st.markdown("""
                            **The Hurst Exponent (H)** measures the long-term memory or persistence of a time series. It ranges from 0 to 1 and helps identify the market regime:
                            
                            - **H < 0.5**: Mean-reverting (anti-persistent) behavior - price tends to reverse direction
                            - **H = 0.5**: Random walk - price changes are independent of each other
                            - **H > 0.5**: Trending (persistent) behavior - price tends to continue in the same direction
                            
                            The Hurst exponent is calculated on a rolling window of 252 trading days (approximately one year).
                            """)
                    else:
                        st.warning("Insufficient data to calculate Hurst exponent. This requires at least 252 data points.")
        
        elif tab_name == "Relative Rotation Graph (RRG)":
            st.subheader("Relative Rotation Graph (RRG)")
            
            # Calculate RRG data
            try:
                rrg_data = calc_rrg(prices, benchmark="SPY", window=10)
                rrg_fig = create_rrg_figure(rrg_data, title="Relative Rotation Graph (RRG)")
                st.plotly_chart(rrg_fig, use_container_width=True)
                
                # Calculate momentum values for multiple timeframes
                momentum_data = calculate_momentum_values(prices)
                momentum_fig = create_momentum_values_figure(momentum_data, title="Sector Momentum")
                st.plotly_chart(momentum_fig, use_container_width=True)
                
                # Add download buttons
                rrg_html = rrg_fig.to_html(include_plotlyjs="cdn")
                st.download_button(
                    label="Download RRG as HTML",
                    data=rrg_html,
                    file_name="rrg_chart.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating RRG: {e}")
        
        elif tab_name == "Correlation Analysis":
            st.subheader("Correlation Analysis")
            
            # Calculate correlation matrix
            try:
                corr_matrix = calculate_correlation_matrix(returns)
                corr_fig = create_correlation_heatmap(corr_matrix, title="Correlation Matrix")
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Seasonality heatmap
                seasonality_fig = create_seasonality_heatmap(returns, title="Monthly Seasonality Heatmap")
                st.plotly_chart(seasonality_fig, use_container_width=True)
                
                # Lead-lag analysis
                if st.checkbox("Show Lead-Lag Analysis", value=False):
                    st.subheader("Lead-Lag Analysis")
                    lead_lag_results = calculate_lead_lag(returns, benchmark="SPY")
                    
                    # Create a DataFrame with peak lags and correlations
                    lead_lag_df = pd.DataFrame(columns=["Ticker", "Peak Lag", "Peak Correlation"])
                    
                    for ticker, result in lead_lag_results.items():
                        lead_lag_df = pd.concat([
                            lead_lag_df,
                            pd.DataFrame({
                                "Ticker": [ticker],
                                "Peak Lag": [result["peak_lag"]],
                                "Peak Correlation": [result["peak_corr"]]
                            })
                        ])
                    
                    # Display lead-lag table
                    st.dataframe(lead_lag_df.sort_values("Peak Correlation", ascending=False))
                
                # Granger causality
                if st.checkbox("Show Granger Causality", value=False):
                    st.subheader("Granger Causality Analysis")
                    granger_results = calculate_granger_causality(returns, benchmark="SPY")
                    
                    # Create a DataFrame with Granger causality p-values
                    granger_df = pd.DataFrame(
                        columns=["Ticker", "Ticker→SPY (p-value)", "SPY→Ticker (p-value)"]
                    )
                    
                    for ticker, result in granger_results.items():
                        granger_df = pd.concat([
                            granger_df,
                            pd.DataFrame({
                                "Ticker": [ticker],
                                "Ticker→SPY (p-value)": [result["ticker_causes_benchmark"][1]],
                                "SPY→Ticker (p-value)": [result["benchmark_causes_ticker"][1]]
                            })
                        ])
                    
                    # Display Granger causality table
                    st.dataframe(
                        granger_df.sort_values("SPY→Ticker (p-value)")
                    )
            except Exception as e:
                st.error(f"Error generating correlation analysis: {e}")
        
        elif tab_name == "Technical Indicators":
            st.subheader("Technical Indicators")
            
            try:
                # RSI time series for all sectors
                rsi_values = calculate_rsi(prices)
                
                # First show the time series chart for all sectors
                st.subheader("RSI (14) Time Series")
                
                # Allow filtering time range for better visibility
                rsi_timeframe = st.slider(
                    "RSI History Period (Days)", 
                    min_value=30, 
                    max_value=365,
                    value=90,
                    step=30,
                    help="Filter RSI history to focus on recent data"
                )
                
                # Filter the RSI values to the selected timeframe
                cutoff_date = rsi_values.index[-1] - pd.Timedelta(days=rsi_timeframe)
                filtered_rsi = rsi_values[rsi_values.index >= cutoff_date]
                
                # Create and display RSI time series chart
                rsi_ts_fig = create_rsi_time_series(filtered_rsi, title=f"RSI (14) - Last {rsi_timeframe} Days")
                st.plotly_chart(rsi_ts_fig, use_container_width=True)
                
                # Create a section with some explanation
                with st.expander("About RSI (Relative Strength Index)"):
                    st.markdown("""
                    **Relative Strength Index (RSI)** is a momentum oscillator that measures the speed and change of price movements. 
                    The RSI ranges from 0 to 100 and is typically used to identify overbought or oversold conditions:
                    
                    - **RSI > 70**: Potentially overbought condition (red line)
                    - **RSI < 30**: Potentially oversold condition (green line)
                    - **RSI = 50**: Neutral momentum
                    
                    RSI can be used to identify potential trend reversals, divergences, and market conditions.
                    """)
                
                # Still show the current RSI values as a heatmap/bar chart for quick reference
                st.subheader("Current RSI Values")
                rsi_current_fig = create_rsi_heatmap(rsi_values.iloc[-1], title="RSI (14) Current Values")
                st.plotly_chart(rsi_current_fig, use_container_width=True)
                
                # Beta and Sharpe
                st.subheader("Rolling Beta and Sharpe Ratio")
                col1, col2 = st.columns(2)
                
                # Calculate rolling beta
                beta_values = calculate_rolling_beta(prices, benchmark="SPY")
                
                # Select ticker for beta plot
                with col1:
                    beta_ticker = st.selectbox(
                        "Select Ticker for Beta Analysis",
                        options=[t for t in beta_values.columns if t != "SPY"],
                        key="beta_ticker"
                    )
                    
                    # Create beta plot
                    if beta_ticker:
                        beta_fig = go.Figure()
                        beta_fig.add_trace(go.Scatter(
                            x=beta_values.index,
                            y=beta_values[beta_ticker],
                            mode='lines',
                            name='Beta'
                        ))
                        beta_fig.update_layout(
                            title=f"Rolling Beta: {beta_ticker} vs SPY",
                            xaxis_title="Date",
                            yaxis_title="Beta",
                            template="plotly_white"
                        )
                        st.plotly_chart(beta_fig, use_container_width=True)
                
                # Calculate Sharpe ratio
                sharpe_values = calculate_rolling_sharpe(prices)
                
                # Select ticker for Sharpe plot
                with col2:
                    sharpe_ticker = st.selectbox(
                        "Select Ticker for Sharpe Ratio Analysis",
                        options=sharpe_values.columns,
                        key="sharpe_ticker"
                    )
                    
                    # Create Sharpe plot
                    if sharpe_ticker:
                        sharpe_fig = go.Figure()
                        sharpe_fig.add_trace(go.Scatter(
                            x=sharpe_values.index,
                            y=sharpe_values[sharpe_ticker],
                            mode='lines',
                            name='Sharpe Ratio'
                        ))
                        sharpe_fig.update_layout(
                            title=f"Rolling Sharpe Ratio: {sharpe_ticker}",
                            xaxis_title="Date",
                            yaxis_title="Sharpe Ratio",
                            template="plotly_white"
                        )
                        st.plotly_chart(sharpe_fig, use_container_width=True)
                
                # Drawdowns
                if st.checkbox("Show Drawdowns", value=False):
                    st.subheader("Drawdowns")
                    
                    # Calculate drawdowns
                    drawdowns = calculate_drawdowns(prices)
                    
                    # Select ticker for drawdown plot
                    drawdown_ticker = st.selectbox(
                        "Select Ticker for Drawdown Analysis",
                        options=drawdowns.columns,
                        key="drawdown_ticker"
                    )
                    
                    # Create drawdown plot
                    if drawdown_ticker:
                        drawdown_fig = go.Figure()
                        drawdown_fig.add_trace(go.Scatter(
                            x=drawdowns.index,
                            y=drawdowns[drawdown_ticker] * 100,  # Convert to percentage
                            mode='lines',
                            name='Drawdown',
                            fill='tozeroy',
                            fillcolor='rgba(255, 0, 0, 0.3)'
                        ))
                        drawdown_fig.update_layout(
                            title=f"Drawdown: {drawdown_ticker}",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            template="plotly_white",
                            yaxis=dict(tickformat=".1f")
                        )
                        st.plotly_chart(drawdown_fig, use_container_width=True)
                
                # Skewness and Kurtosis
                if st.checkbox("Show Skewness and Kurtosis", value=False):
                    st.subheader("Skewness and Kurtosis")
                    
                    # Calculate skewness and kurtosis
                    skew_values, kurt_values = calculate_rolling_moments(prices)
                    
                    col1, col2 = st.columns(2)
                    
                    # Select ticker for skewness plot
                    with col1:
                        skew_ticker = st.selectbox(
                            "Select Ticker for Skewness Analysis",
                            options=skew_values.columns,
                            key="skew_ticker"
                        )
                        
                        # Create skewness plot
                        if skew_ticker:
                            skew_fig = go.Figure()
                            skew_fig.add_trace(go.Scatter(
                                x=skew_values.index,
                                y=skew_values[skew_ticker],
                                mode='lines',
                                name='Skewness'
                            ))
                            skew_fig.update_layout(
                                title=f"Rolling Skewness: {skew_ticker}",
                                xaxis_title="Date",
                                yaxis_title="Skewness",
                                template="plotly_white"
                            )
                            st.plotly_chart(skew_fig, use_container_width=True)
                    
                    # Select ticker for kurtosis plot
                    with col2:
                        kurt_ticker = st.selectbox(
                            "Select Ticker for Kurtosis Analysis",
                            options=kurt_values.columns,
                            key="kurt_ticker"
                        )
                        
                        # Create kurtosis plot
                        if kurt_ticker:
                            kurt_fig = go.Figure()
                            kurt_fig.add_trace(go.Scatter(
                                x=kurt_values.index,
                                y=kurt_values[kurt_ticker],
                                mode='lines',
                                name='Kurtosis'
                            ))
                            kurt_fig.update_layout(
                                title=f"Rolling Kurtosis: {kurt_ticker}",
                                xaxis_title="Date",
                                yaxis_title="Kurtosis",
                                template="plotly_white"
                            )
                            st.plotly_chart(kurt_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating technical indicators: {e}")
        
        elif tab_name == "Ratio Analysis":
            st.subheader("Ratio Analysis")
            
            try:
                # Define key risk-on/risk-off ratio pairs to keep at the top
                key_ratio_pairs = {
                    "XLY/XLP": ("XLY", "XLP"),  # Consumer Discretionary / Consumer Staples (risk-on/risk-off)
                    "XLK/XLU": ("XLK", "XLU"),  # Technology / Utilities (growth/defensive)
                }
                
                # Create sector vs SPY ratio pairs for all available sectors
                sector_spy_ratios = {}
                
                # All available sector ETFs to compare against SPY
                sector_etfs = [
                    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", 
                    "XLP", "XLRE", "XLU", "XLV", "XLY"
                ]
                
                # Create sector vs SPY ratio pairs
                for sector in sector_etfs:
                    if sector in prices.columns and "SPY" in prices.columns:
                        sector_spy_ratios[f"{sector}/SPY"] = (sector, "SPY")
                
                # Filter key ratio pairs based on available tickers
                available_key_pairs = {}
                for name, (num, denom) in key_ratio_pairs.items():
                    if num in prices.columns and denom in prices.columns:
                        available_key_pairs[name] = (num, denom)
                
                # Display the key ratios at the top if available
                if available_key_pairs:
                    risk_fig = create_ratio_plot(
                        prices, 
                        available_key_pairs, 
                        title="Risk-On/Risk-Off Ratio Analysis"
                    )
                    st.plotly_chart(risk_fig, use_container_width=True)
                    
                    st.info("""
                        **Key Ratio Definitions:**
                        - **XLY/XLP:** Consumer Discretionary vs Consumer Staples (risk-on/risk-off indicator)
                        - **XLK/XLU:** Technology vs Utilities (growth vs defensive indicator)
                    """)
                
                # Display the sector vs SPY ratios
                if sector_spy_ratios:
                    st.subheader("Sector vs SPY Ratios")
                    st.markdown("Each sector's relative performance compared to the S&P 500 (SPY)")
                    
                    # Display all sector vs SPY ratios in a multi-panel plot
                    spy_ratios_fig = create_multi_ratio_plot(
                        prices, 
                        sector_spy_ratios, 
                        title="Sector vs SPY Relative Performance"
                    )
                    st.plotly_chart(spy_ratios_fig, use_container_width=True)
                    
                    # Additional info about interpreting these ratios
                    st.info("""
                        **How to interpret:** Rising lines indicate the sector is outperforming SPY, 
                        while falling lines indicate underperformance relative to the broader market.
                    """)
                
                if not available_key_pairs and not sector_spy_ratios:
                    st.warning("No valid ratio pairs available. Please ensure you've selected sector ETFs and SPY.")
            except Exception as e:
                st.error(f"Error generating ratio analysis: {e}")
        
        elif tab_name == "Network Analysis":
            st.subheader("Network Analysis")
            
            try:
                # Calculate correlation matrix
                corr_matrix = calculate_correlation_matrix(returns)
                
                # Create network graph
                network_fig = create_network_graph(
                    corr_matrix,
                    threshold=network_threshold,
                    title="Sector Correlation Network"
                )
                
                st.plotly_chart(network_fig, use_container_width=True)
                
                st.info(f"""
                    This network graph shows correlations above the threshold of {network_threshold}.
                    - Green lines represent positive correlations
                    - Red lines represent negative correlations
                    - Line thickness indicates correlation strength
                    - Node size indicates connectivity (number of correlations above threshold)
                """)
            except Exception as e:
                st.error(f"Error generating network analysis: {e}")
        
        elif tab_name == "PCA & Clustering":
            st.subheader("PCA with Spectral Clustering")
            
            try:
                # Create PCA figure (get both but only use the clustered version)
                _, pca_cluster_fig = create_pca_figure(returns, title="PCA with Spectral Clustering")
                
                # Display only the clustered PCA (skip the basic PCA as requested)
                st.plotly_chart(pca_cluster_fig, use_container_width=True)
                
                st.info("""
                    Principal Component Analysis (PCA) reduces the dimensionality of the return data to 2 dimensions,
                    allowing us to visualize the relationships between sectors. Sectors that are close together on the
                    plot tend to move similarly. 
                    
                    The spectral clustering automatically identifies groups of sectors with similar behavior patterns,
                    making it easier to spot which sectors tend to move together during different market environments.
                """)
            except Exception as e:
                st.error(f"Error generating PCA analysis: {e}")
        
        elif tab_name == "Forecasting":
            st.subheader("Price Forecasting")
            
            try:
                # Select ticker for forecasting
                forecast_ticker = st.selectbox(
                    "Select Ticker for Forecasting",
                    options=prices.columns,
                    key="forecast_ticker"
                )
                
                if forecast_ticker:
                    # Already using daily data for everything
                    forecast_data = prices[forecast_ticker]
                    
                    # Generate forecasts with fixed 30-day horizon using Prophet
                    with st.spinner("Generating Prophet forecast... This may take a moment."):
                        forecast_results = generate_forecasts(
                            forecast_data, 
                            forecast_horizon=30  # Fixed to 30 days (1 month)
                        )
                    
                    if forecast_results and 'prophet' in forecast_results:
                        # Create forecast figure
                        forecast_fig = create_forecast_figure(
                            forecast_data,
                            forecast_results,
                            title=f"{forecast_ticker} 30-Day Prophet Forecast"
                        )
                        
                        # Add regime overlay if enabled
                        if show_regimes and regime_data is not None:
                            forecast_fig = add_regime_overlay(
                                forecast_fig,
                                regime_data,
                                prices[forecast_ticker]
                            )
                        
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Display Prophet components
                        st.subheader("Prophet Forecast Components")
                        st.markdown("""
                        The components below show how different factors contribute to the Prophet forecast:
                        - **Trend**: The overall long-term direction of the price
                        - **Yearly**: Annual seasonal patterns in the data
                        - **Weekly**: Weekly patterns that repeat
                        - **Daily**: Day-of-week effects (if present)
                        """)
                        
                        # Create components figure
                        components_fig = create_prophet_components_figure(
                            forecast_results,
                            title=f"{forecast_ticker} Forecast Components"
                        )
                        
                        if components_fig:  # Only display if we have components
                            st.plotly_chart(components_fig, use_container_width=True)
                    else:
                        st.warning("Prophet forecasting failed to generate valid results. Please try a different ticker or timeframe.")
                    
                    # Display forecast statistics
                    if forecast_results and 'prophet' in forecast_results and 'forecast' in forecast_results['prophet']:
                        prophet_results = forecast_results['prophet']
                        
                        # Show forecast statistics
                        st.subheader("Prophet Forecast Statistics")
                        
                        # Create a DataFrame with forecast statistics with robust error handling
                        try:
                            # Get the last historical price value
                            if isinstance(forecast_data, pd.Series) and len(forecast_data) > 0:
                                last_price = float(forecast_data.iloc[-1])
                            else:
                                st.warning("Historical data is not available or in unexpected format")
                                last_price = np.nan
                            
                            # Get the forecast price value
                            if ('forecast' in prophet_results and 
                                isinstance(prophet_results['forecast'], pd.Series) and 
                                len(prophet_results['forecast']) > 0):
                                # Safely convert to float to avoid type errors
                                forecast_price = float(prophet_results['forecast'].iloc[-1])
                            else:
                                st.warning("Forecast data is not available or in unexpected format")
                                forecast_price = np.nan
                                
                            # Calculate change and percentage
                            if not np.isnan(last_price) and not np.isnan(forecast_price):
                                change = forecast_price - last_price
                                pct_change = (change / last_price) * 100 if last_price != 0 else np.nan
                            else:
                                change = np.nan
                                pct_change = np.nan
                                
                            st.info(f"Using last price: ${last_price:.2f} and forecast: ${forecast_price:.2f}")
                        except Exception as stats_err:
                            st.error(f"Error calculating forecast statistics: {stats_err}")
                            last_price = np.nan
                            forecast_price = np.nan
                            change = np.nan
                            pct_change = np.nan
                        
                        # Create columns with clear names for metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if not np.isnan(last_price):
                                st.metric("Current Price", f"${last_price:.2f}")
                            else:
                                st.metric("Current Price", "N/A")
                        
                        with col2:
                            if not np.isnan(forecast_price):
                                st.metric("Forecast (30d)", f"${forecast_price:.2f}")
                            else:
                                st.metric("Forecast (30d)", "N/A")
                        
                        with col3:
                            if not np.isnan(change):
                                st.metric("Change", f"${change:.2f}", delta=change)
                            else:
                                st.metric("Change", "N/A")
                        
                        with col4:
                            if not np.isnan(pct_change):
                                st.metric("% Change", f"{pct_change:.2f}%", delta=pct_change)
                            else:
                                st.metric("% Change", "N/A")
                        
                        # Show forecast characteristics
                        st.markdown("### Forecast Characteristics")
                        
                        if 'components' in prophet_results:
                            components = prophet_results['components']
                            if 'trend' in components:
                                trend_direction = "Upward" if components['trend'].iloc[-1] > components['trend'].iloc[0] else "Downward"
                                trend_strength = abs(components['trend'].iloc[-1] - components['trend'].iloc[0])
                                st.markdown(f"- **Trend Direction**: {trend_direction}")
                                st.markdown(f"- **Trend Strength**: {trend_strength:.2f}")
                            
                            has_seasonality = any(k for k in components.keys() if k != 'trend')
                            st.markdown(f"- **Seasonal Patterns**: {'Present' if has_seasonality else 'Not significant'}")
                            
                            for comp_name in components.keys():
                                if comp_name != 'trend':
                                    # Safely calculate amplitude with validation
                                    try:
                                        comp_data = components[comp_name]
                                        if len(comp_data) > 0 and not comp_data.isna().all():
                                            # Filter out NaN values for safe calculation
                                            valid_data = comp_data.dropna()
                                            if len(valid_data) > 0:
                                                amplitude = valid_data.max() - valid_data.min()
                                                st.markdown(f"  - **{comp_name.capitalize()} Amplitude**: {amplitude:.4f}")
                                            else:
                                                st.markdown(f"  - **{comp_name.capitalize()} Amplitude**: Not available (no valid data points)")
                                        else:
                                            st.markdown(f"  - **{comp_name.capitalize()} Amplitude**: Not available (component is empty)")
                                    except Exception as e:
                                        st.markdown(f"  - **{comp_name.capitalize()} Amplitude**: Error calculating ({str(e)})")
                        
                        # Add confidence interval information with improved validation
                        if 'lower_bound' in prophet_results and 'upper_bound' in prophet_results:
                            try:
                                # Make sure we have data available to calculate the interval
                                if (len(prophet_results['lower_bound']) > 0 and 
                                    len(prophet_results['upper_bound']) > 0 and
                                    not np.isnan(forecast_price) and
                                    forecast_price != 0):
                                    
                                    # Calculate interval width with NaN handling
                                    lower_val = prophet_results['lower_bound'].iloc[-1] 
                                    upper_val = prophet_results['upper_bound'].iloc[-1]
                                    
                                    if not (np.isnan(lower_val) or np.isnan(upper_val)):
                                        interval_width = upper_val - lower_val
                                        interval_pct = (interval_width / forecast_price) * 100
                                        st.markdown(f"- **Forecast Uncertainty**: ±{interval_pct/2:.2f}%")
                                    else:
                                        st.markdown("- **Forecast Uncertainty**: Not available (NaN values in bounds)")
                                else:
                                    st.markdown("- **Forecast Uncertainty**: Not available (insufficient data)")
                            except Exception as e:
                                st.markdown(f"- **Forecast Uncertainty**: Error calculating ({str(e)})")
                        else:
                            st.markdown("- **Forecast Uncertainty**: Not available (bounds missing)")
                        
                        # Add trading recommendation based on forecast with NaN handling
                        st.markdown("### Trading Signal")
                        
                        # Make sure pct_change is a valid number before making a recommendation
                        if np.isnan(pct_change):
                            signal = "Unknown"
                            confidence = "None"
                            st.markdown("- **Signal**: Unable to determine (insufficient data)")
                            st.markdown("- **Confidence**: N/A")
                        else:
                            if pct_change > 5:
                                signal = "Strong Buy"
                                confidence = "High"
                            elif pct_change > 2:
                                signal = "Buy"
                                confidence = "Moderate"
                            elif pct_change < -5:
                                signal = "Strong Sell"
                                confidence = "High"
                            elif pct_change < -2:
                                signal = "Sell"
                                confidence = "Moderate"
                            else:
                                signal = "Hold"
                                confidence = "Low"
                                
                            st.markdown(f"- **Signal**: {signal}")
                            st.markdown(f"- **Confidence**: {confidence}")
                            
                        # Always display the forecast horizon
                        st.markdown(f"- **Forecast Horizon**: 30 days")
                        
                        st.warning("⚠️ This trading signal is for informational purposes only and should not be considered as financial advice.")
                        
                        # The forecast information is already displayed above with metrics
            except Exception as e:
                st.error(f"Error generating Prophet forecasts: {e}")
                st.markdown("Please try selecting a different ticker or adjusting the date range.")
                st.code(str(e), language="python")

# Footer
st.markdown("---")
st.markdown("**Sector-Rotation Pro** | A comprehensive Streamlit dashboard for sector rotation analysis")
st.markdown("Data source: Alpha Vantage API")
