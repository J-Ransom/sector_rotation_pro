import inspect
import numpy as np
import pandas as pd
import streamlit as st

from darts import TimeSeries
from darts.models import Prophet

def run_prophet_forecast(price_series, forecast_horizon=30):
    """
    Run Prophet forecasting using daily data and extract trend and seasonal components.
    
    Parameters:
    ----------
    price_series : pandas.Series
        Series with price data (must have datetime index with daily frequency)
    forecast_horizon : int
        Number of days to forecast (default: 30 days)
        
    Returns:
    -------
    dict
        Dictionary with forecast, confidence intervals, and component breakdowns
    """
    # Validate input
    if not isinstance(price_series, pd.Series):
        st.error("Forecasting requires a pandas Series with datetime index")
        return {}
        
    if len(price_series) < 30:  # Need at least 30 data points for reasonable forecasts
        st.error(f"Insufficient data for forecasting: need at least 30 data points, got {len(price_series)}")
        return {}
    
    # Ensure we're using the most recent 3 years of data (if available)
    end_date = price_series.index.max()
    start_date = end_date - pd.DateOffset(years=3)
    
    # Filter to the last 3 years if we have more than a year of data
    if len(price_series) > 252:  # 252 trading days in a year
        st.info(f"Using data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} for forecasting")
        # Filter to the most recent 3 years
        price_series = price_series[price_series.index >= start_date]
    
    try:
        # Convert to Darts TimeSeries
        series = None
        try:
            # Business days frequency for stock data
            series = TimeSeries.from_series(price_series, freq='B')
            st.info("Successfully created forecasting time series with frequency: B")
            
            # Calculate average interval between data points for validation
            time_diffs = np.diff(price_series.index)/np.timedelta64(1, 'D')
            avg_diff = np.mean(time_diffs)
            st.info(f"Average interval between data points: {avg_diff:.0f} days")
            
            if avg_diff > 3:
                st.warning("Data intervals suggest non-daily data. Results may be less accurate.")
        except Exception as e:
            st.error(f"Could not create TimeSeries: {e}. Forecasting will fail.")
            return {}
        
        # Create and train the Prophet model
        try:
            # Configure Prophet with appropriate settings for daily financial data
            # Note: darts Prophet wrapper has different parameters than direct Prophet API
            prophet = Prophet(
                n_changepoints=25,  # Number of potential changepoints
                yearly_seasonality=True,  # Include yearly seasonality
                weekly_seasonality=True,  # Include weekly seasonality
                daily_seasonality=False,  # No daily seasonality for stock data
                uncertainty_samples=1000  # More samples for better confidence intervals
            )
            
            # Fit the model
            prophet.fit(series)
            st.success("Successfully trained Prophet model")
            
            # Generate forecast with prediction intervals
            forecast = prophet.predict(forecast_horizon, num_samples=1000)
            
            # Handle different Darts versions - try multiple methods to convert TimeSeries to pandas
            try:
                # Method for newer Darts versions
                if hasattr(forecast, 'pd_dataframe'):
                    forecast_df = forecast.pd_dataframe()
                # Method for newer Darts versions
                elif hasattr(forecast, 'to_dataframe'):
                    forecast_df = forecast.to_dataframe()
                # Method for older Darts versions
                elif hasattr(forecast, 'data_array') and hasattr(forecast, 'time_index'):
                    forecast_df = pd.DataFrame(forecast.data_array().reshape(-1, 1), 
                                             index=forecast.time_index(), 
                                             columns=['value'])
                else:
                    raise AttributeError("Could not find a method to convert TimeSeries to DataFrame")
                    
                st.info(f"Successfully converted forecast to DataFrame with shape {forecast_df.shape}")
            except Exception as conv_err:
                st.error(f"Failed to convert forecast to DataFrame: {conv_err}")
                return {}
            
            # Create forecast dates (business days)
            last_date = price_series.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=forecast_horizon, 
                freq='B'
            )
            
            # Convert forecast to pandas Series with better error handling
            try:
                # If we have probabilistic forecast with multiple samples, use the mean
                if forecast_df.shape[1] > 1:
                    # Mean across all samples
                    forecast_values = forecast_df.mean(axis=1).values
                else:
                    # Single sample case
                    forecast_values = forecast_df.iloc[:, 0].values
                    
                # Create series with proper index
                forecast_series = pd.Series(forecast_values, index=forecast_dates)
                st.success(f"Created forecast series with {len(forecast_series)} points")
            except Exception as series_err:
                st.error(f"Failed to convert forecast to series: {series_err}")
                # Return empty result if conversion fails
                return {}
            
            # Get prediction intervals with multiple method handling
            try:
                # First try predict_interval method (older Darts versions)
                if hasattr(prophet, 'predict_interval'):
                    try:
                        lower_upper = prophet.predict_interval(forecast_horizon, coverage=0.9)
                        
                        # Convert lower bound TimeSeries to DataFrame
                        if hasattr(lower_upper[0], 'pd_dataframe'):
                            lower_df = lower_upper[0].pd_dataframe()
                        elif hasattr(lower_upper[0], 'to_dataframe'):
                            lower_df = lower_upper[0].to_dataframe()
                        elif hasattr(lower_upper[0], 'data_array') and hasattr(lower_upper[0], 'time_index'):
                            lower_df = pd.DataFrame(lower_upper[0].data_array().reshape(-1, 1),
                                                index=lower_upper[0].time_index(),
                                                columns=['value'])
                        else:
                            raise AttributeError("Could not convert lower bound to DataFrame")
                        
                        # Convert upper bound TimeSeries to DataFrame
                        if hasattr(lower_upper[1], 'pd_dataframe'):
                            upper_df = lower_upper[1].pd_dataframe()
                        elif hasattr(lower_upper[1], 'to_dataframe'):
                            upper_df = lower_upper[1].to_dataframe()
                        elif hasattr(lower_upper[1], 'data_array') and hasattr(lower_upper[1], 'time_index'):
                            upper_df = pd.DataFrame(lower_upper[1].data_array().reshape(-1, 1),
                                                index=lower_upper[1].time_index(),
                                                columns=['value'])
                        else:
                            raise AttributeError("Could not convert upper bound to DataFrame")
                        
                        lower_bound = pd.Series(lower_df.iloc[:, 0].values, index=forecast_dates)
                        upper_bound = pd.Series(upper_df.iloc[:, 0].values, index=forecast_dates)
                    except Exception as method_err:
                        raise AttributeError(f"predict_interval method failed: {method_err}")
                        
                # Next try predict with confidence (newer Darts versions)
                elif hasattr(prophet, 'predict') and 'num_samples' in inspect.signature(prophet.predict).parameters:
                    st.info("Using predict with num_samples to generate confidence intervals")
                    # We already generated forecast with num_samples=1000 earlier
                    # Now extract quantiles from the probabilistic forecast
                    if forecast_df.shape[1] > 1:  # Check if we have multiple samples
                        try:
                            # Calculate quantiles from the sample forecasts
                            lower_bound_values = np.quantile(forecast_df.values, 0.05, axis=1)
                            upper_bound_values = np.quantile(forecast_df.values, 0.95, axis=1)
                            
                            lower_bound = pd.Series(lower_bound_values, index=forecast_dates)
                            upper_bound = pd.Series(upper_bound_values, index=forecast_dates)
                            st.success("Successfully computed confidence intervals from samples")
                        except Exception as quant_err:
                            raise ValueError(f"Failed to calculate quantiles: {quant_err}")
                    else:
                        raise ValueError("Forecast doesn't contain multiple samples for interval calculation")
                else:
                    raise AttributeError("No compatible methods found for prediction intervals")
                    
            except Exception as interval_err:
                st.warning(f"Interval prediction failed with error: {interval_err}. Using approximation instead.")
                
                # Approximate prediction intervals (using mean prediction +/- 10%)
                try:
                    # Create dummy intervals (10% above and below the mean)
                    lower_bound = forecast_series * 0.9
                    upper_bound = forecast_series * 1.1
                    st.info("Using approximated confidence intervals (±10% of forecast)")
                except Exception as approx_err:
                    st.error(f"Failed to create approximate intervals: {approx_err}")
                    # Last resort - empty series with same index
                    lower_bound = pd.Series(index=forecast_dates)
                    upper_bound = pd.Series(index=forecast_dates)
            
            # Extract Prophet components
            components = {}
            try:
                # Check if the Prophet model is accessible
                if not hasattr(prophet, 'model') or prophet.model is None:
                    st.warning("Prophet model doesn't expose the internal model attribute, cannot extract components")
                    return {
                        'forecast': forecast_series,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'components': components  # Empty components dict
                    }
                
                # Get prophet model components (trend, seasonality)
                prophet_model = prophet.model
                
                # Format dates for Prophet's component extraction with error handling
                try:
                    prophet_dates = pd.DataFrame({'ds': forecast_dates})
                    
                    # Get component predictions
                    prophet_components = prophet_model.predict(prophet_dates)
                except Exception as date_err:
                    st.warning(f"Failed to prepare dates for component extraction: {date_err}")
                    return {
                        'forecast': forecast_series,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'components': components  # Empty components dict
                    }
                
                # Extract trend component
                if 'trend' in prophet_components.columns:
                    trend = pd.Series(prophet_components['trend'].values, index=forecast_dates)
                    components['trend'] = trend
                
                # Extract various seasonality components if available
                if 'yearly' in prophet_components.columns:
                    yearly = pd.Series(prophet_components['yearly'].values, index=forecast_dates)
                    components['yearly'] = yearly
                    
                if 'weekly' in prophet_components.columns:
                    weekly = pd.Series(prophet_components['weekly'].values, index=forecast_dates)
                    components['weekly'] = weekly
                    
                if 'daily' in prophet_components.columns:
                    daily = pd.Series(prophet_components['daily'].values, index=forecast_dates)
                    components['daily'] = daily
                
                # Extract additive or multiplicative seasonality
                if 'seasonal' in prophet_components.columns:
                    seasonal = pd.Series(prophet_components['seasonal'].values, index=forecast_dates)
                    components['seasonal'] = seasonal
                    
                # Extract holiday components if present
                if 'holidays' in prophet_components.columns:
                    holidays = pd.Series(prophet_components['holidays'].values, index=forecast_dates)
                    components['holidays'] = holidays
                
                # Look for any other components that might be present
                for col in prophet_components.columns:
                    if col not in ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'yearly', 'weekly', 'daily', 'seasonal', 'holidays']:
                        if 'seasonality' in col or '_delim_' in col:
                            comp_data = pd.Series(prophet_components[col].values, index=forecast_dates)
                            components[col] = comp_data
                
                # Prepare a detailed success message about extracted components
                component_names = list(components.keys())
                component_summary = ", ".join([name.capitalize() for name in component_names])
                if component_names:
                    st.success(f"Successfully extracted {len(components)} Prophet components: {component_summary}")
                else:
                    st.warning("No Prophet components were found in the model output")
            except Exception as comp_err:
                st.warning(f"Failed to extract Prophet components: {comp_err}")
            
            # Return forecast results with components
            return {
                'forecast': forecast_series,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'components': components
            }
            
        except Exception as model_err:
            st.error(f"Prophet model failed with error: {model_err}")
            return {}
            
    except Exception as e:
        st.error(f"Unexpected error in forecasting: {e}")
        return {}

def generate_forecasts(price_series, forecast_horizon=30):
    """
    Generate Prophet forecast for a price series using daily data.
    
    Parameters:
    ----------
    price_series : pandas.Series
        Series with price data
    forecast_horizon : int
        Number of days to forecast (default: 30 days)
        
    Returns:
    -------
    dict
        Dictionary with the Prophet forecast results
    """
    # Ensure forecast horizon is appropriate for daily data
    if forecast_horizon != 30:
        st.info(f"Using standard 30-day forecast horizon instead of {forecast_horizon}")
        forecast_horizon = 30
    
    # Run Prophet forecast
    try:
        prophet_results = run_prophet_forecast(price_series, forecast_horizon)
        
        if prophet_results and len(prophet_results) > 0:
            return {'prophet': prophet_results}
        else:
            st.warning("Prophet forecast did not produce valid results")
            return {}
    except Exception as e:
        st.error(f"Forecasting failed with error: {e}")
        return {}
