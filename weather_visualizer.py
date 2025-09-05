#!/usr/bin/env python3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import logging
import os
from weather_client import WeatherClient
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherVisualizer:
    """Class for visualizing weather data and its impact on demand forecasts"""
    
    def __init__(self, data_dir="data", default_location="New York"):
        """Initialize the visualizer with data directory and default location
        
        Args:
            data_dir (str): Directory containing data files
            default_location (str): Default location name for weather data
        """
        self.data_dir = data_dir
        self.default_location = default_location
        self.weather_client = WeatherClient(cache_session=True)
        self.cached_weather_data = {}
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        logger.info("Weather visualizer initialized")
    
    def get_weather_data(self, location=None, days_historical=30, days_forecast=7):
        """Get historical and forecast weather data for a location
        
        Args:
            location (str): Location name
            days_historical (int): Number of historical days to fetch
            days_forecast (int): Number of forecast days to fetch
            
        Returns:
            tuple: (historical_df, forecast_df) containing weather data
        """
        location = location or self.default_location
        cache_key = f"{location}_{days_historical}_{days_forecast}"
        
        # Check if data is already cached
        if cache_key in self.cached_weather_data:
            return self.cached_weather_data[cache_key]
        
        # Get location coordinates
        lat, lon = self.weather_client.get_location_by_name(location)
        
        # Historical dates - ensure we're using past dates
        now = datetime.now().date()
        
        # In case system date is in the future, use a fallback date
        # This handles cases where the system clock might be incorrectly set
        if now.year > 2024:  # If date appears to be in far future
            logger.warning(f"System date appears to be in the future: {now}. Using fallback date range.")
            end_date = "2023-08-31"  # Use a reliable past date
            start_date = "2023-08-01" if days_historical >= 30 else (datetime.strptime("2023-08-31", "%Y-%m-%d") - timedelta(days=days_historical)).strftime("%Y-%m-%d")
        else:
            end_date = now.strftime("%Y-%m-%d")
            start_date = (now - timedelta(days=days_historical)).strftime("%Y-%m-%d")
        
        try:
            # Get historical weather
            historical_df = self.weather_client.fetch_historical_weather(lat, lon, start_date, end_date)
            
            # Get forecast
            forecast_df = self.weather_client.fetch_forecast(lat, lon, days=days_forecast)
            
            # Add impact factors
            historical_df = self.weather_client.get_weather_impact_factors(historical_df)
            forecast_df = self.weather_client.get_weather_impact_factors(forecast_df)
            
            # Cache the data
            self.cached_weather_data[cache_key] = (historical_df, forecast_df)
            
            return historical_df, forecast_df
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def create_weather_overview(self, location=None, days_historical=30, days_forecast=7):
        """Create a comprehensive weather overview plot with historical and forecast data
        
        Args:
            location (str): Location name
            days_historical (int): Number of historical days to include
            days_forecast (int): Number of forecast days to include
            
        Returns:
            plotly.graph_objects.Figure: Figure object with the weather overview
        """
        try:
            historical_df, forecast_df = self.get_weather_data(location, days_historical, days_forecast)
            
            if historical_df.empty or forecast_df.empty:
                logger.warning("Empty weather data, cannot create overview")
                return go.Figure()
            
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                subplot_titles=(
                    f"Temperature and Precipitation Overview - {location or self.default_location}",
                    "Weather Impact on Demand"
                )
            )
            
            # Combined dataframe with forecast marked
            historical_df['data_type'] = 'Historical'
            forecast_df['data_type'] = 'Forecast'
            combined_df = pd.concat([historical_df, forecast_df])
            
            # Temperature traces for historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['temperature_max'],
                    name="Max Temp (Historical)",
                    line=dict(color='orangered', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=historical_df['date'],
                    y=historical_df['temperature_min'],
                    name="Min Temp (Historical)",
                    line=dict(color='royalblue', width=1),
                    opacity=0.7,
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.2)'
                ),
                row=1, col=1
            )
            
            # Temperature traces for forecast data
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['temperature_max'],
                    name="Max Temp (Forecast)",
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['temperature_min'],
                    name="Min Temp (Forecast)",
                    line=dict(color='blue', width=2),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.2)'
                ),
                row=1, col=1
            )
            
            # Add precipitation on secondary axis for historical data
            fig.add_trace(
                go.Bar(
                    x=historical_df['date'],
                    y=historical_df['precipitation'],
                    name="Precipitation (Historical)",
                    marker_color='rgba(158, 202, 225, 0.6)',
                    opacity=0.7
                ),
                row=1, col=1, secondary_y=True
            )
            
            # Add precipitation on secondary axis for forecast data
            fig.add_trace(
                go.Bar(
                    x=forecast_df['date'],
                    y=forecast_df['precipitation'],
                    name="Precipitation (Forecast)",
                    marker_color='rgba(8, 81, 156, 0.6)'
                ),
                row=1, col=1, secondary_y=True
            )
            
            # Add impact factors in bottom subplot
            fig.add_trace(
                go.Scatter(
                    x=combined_df['date'],
                    y=combined_df['overall_impact'],
                    name="Overall Weather Impact",
                    line=dict(color='purple', width=2),
                    hovertemplate='<b>Date</b>: %{x}<br><b>Impact Factor</b>: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add impact components
            impact_components = ['temperature_impact', 'rain_impact', 'snow_impact']
            colors = ['orange', 'cornflowerblue', 'lightblue']
            
            for impact, color in zip(impact_components, colors):
                fig.add_trace(
                    go.Scatter(
                        x=combined_df['date'],
                        y=combined_df[impact],
                        name=f"{impact.replace('_', ' ').title()}",
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Add reference line at 1.0 (no impact)
            fig.add_trace(
                go.Scatter(
                    x=[combined_df['date'].min(), combined_df['date'].max()],
                    y=[1, 1],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Baseline (No Impact)',
                    hoverinfo='none'
                ),
                row=2, col=1
            )
            
            # Add vertical line separating historical from forecast
            forecast_start = forecast_df['date'].min()
            for row in [1, 2]:
                fig.add_vline(
                    x=forecast_start, 
                    line_width=1, 
                    line_dash="dash", 
                    line_color="gray",
                    row=row, col=1
                )
                
                # Add "Forecast starts" annotation
                if row == 1:
                    fig.add_annotation(
                        x=forecast_start,
                        y=1.05,
                        yref="paper",
                        text="Forecast →",
                        showarrow=False,
                        font=dict(size=12, color="darkgray"),
                        row=row, col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=800,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=40, t=50, b=50)
            )
            
            # Set axis titles
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Impact Factor", range=[0.7, 1.4], row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            # Improve hover info
            fig.update_traces(
                hovertemplate='<b>Date</b>: %{x}<br><b>Value</b>: %{y}<extra>%{fullData.name}</extra>'
            )
            
            logger.info(f"Created weather overview visualization for {location or self.default_location}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating weather overview: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating weather visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
    
    def create_weather_demand_correlation(self, store_id, item_id, location=None, 
                                         days_historical=30, demand_df=None):
        """Create visualization showing correlation between weather and demand
        
        Args:
            store_id (str): Store ID
            item_id (str): Item ID
            location (str): Location name
            days_historical (int): Number of historical days to include
            demand_df (pandas.DataFrame): Optional demand data dataframe
            
        Returns:
            tuple: (correlation_fig, impact_fig) figures showing correlations and impacts
        """
        try:
            # Get historical weather data
            historical_df, _ = self.get_weather_data(location, days_historical, days_forecast=1)
            
            if historical_df.empty:
                logger.warning("Empty weather data, cannot create correlation")
                return go.Figure(), go.Figure()
            
            # If no demand data provided, try to load from file
            if demand_df is None:
                try:
                    # Assuming a file structure like data/store_123/item_456_demand.csv
                    demand_path = os.path.join(self.data_dir, f"store_{store_id}", f"item_{item_id}_demand.csv")
                    if not os.path.exists(demand_path):
                        # Alternative path - try combined data
                        demand_path = os.path.join(self.data_dir, "combined_pizza_data.csv")
                    
                    if os.path.exists(demand_path):
                        demand_df = pd.read_csv(demand_path)
                        
                        # Ensure date column exists and is in datetime format
                        date_cols = [col for col in demand_df.columns if 'date' in col.lower()]
                        if date_cols:
                            date_col = date_cols[0]
                            demand_df[date_col] = pd.to_datetime(demand_df[date_col])
                        else:
                            logger.warning("No date column found in demand data")
                            return go.Figure(), go.Figure()
                    else:
                        logger.warning(f"No demand data found for store {store_id}, item {item_id}")
                        return go.Figure(), go.Figure()
                        
                except Exception as e:
                    logger.error(f"Error loading demand data: {str(e)}")
                    return go.Figure(), go.Figure()
            
            # Find which column contains the demand data
            # Typically it would be named 'sales', 'demand', 'quantity', etc.
            potential_demand_cols = ['sales', 'demand', 'quantity', 'units', 'amount']
            demand_col = None
            
            for col in demand_df.columns:
                if any(term in col.lower() for term in potential_demand_cols):
                    demand_col = col
                    break
            
            if not demand_col:
                # If no obvious demand column, take first numeric column that's not a date
                for col in demand_df.columns:
                    if pd.api.types.is_numeric_dtype(demand_df[col]) and 'date' not in col.lower():
                        demand_col = col
                        break
            
            if not demand_col:
                logger.warning("Could not identify demand column in data")
                return go.Figure(), go.Figure()
            
            # Find date column in demand data
            date_col = None
            for col in demand_df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
                    
            if not date_col:
                logger.warning("No date column found in demand data")
                return go.Figure(), go.Figure()
            
            # Ensure date is in datetime format
            demand_df[date_col] = pd.to_datetime(demand_df[date_col])
            
            # Filter demand data to match the historical weather period
            min_date = historical_df['date'].min()
            max_date = historical_df['date'].max()
            
            filtered_demand = demand_df[
                (demand_df[date_col] >= min_date) & 
                (demand_df[date_col] <= max_date)
            ].copy()
            
            # If store_id column exists, filter by it
            if 'store' in demand_df.columns or 'store_id' in demand_df.columns:
                store_col = 'store' if 'store' in demand_df.columns else 'store_id'
                filtered_demand = filtered_demand[filtered_demand[store_col] == store_id]
            
            # If item_id column exists, filter by it
            if 'item' in demand_df.columns or 'item_id' in demand_df.columns:
                item_col = 'item' if 'item' in demand_df.columns else 'item_id'
                filtered_demand = filtered_demand[filtered_demand[item_col] == item_id]
            
            # If after filtering we have no data, return empty figures
            if filtered_demand.empty:
                logger.warning("No matching demand data found after filtering")
                return go.Figure(), go.Figure()
            
            # Rename columns for consistency
            filtered_demand = filtered_demand.rename(columns={date_col: 'date', demand_col: 'demand'})
            
            # Merge weather and demand data
            merged_df = pd.merge(historical_df, filtered_demand[['date', 'demand']], on='date', how='inner')
            
            if merged_df.empty:
                logger.warning("No matching dates between weather and demand data")
                return go.Figure(), go.Figure()
            
            # Calculate correlations
            weather_metrics = [
                'temperature_mean', 'temperature_max', 'temperature_min',
                'precipitation', 'rain', 'wind_speed_max', 'temperature_range'
            ]
            
            correlations = {}
            for metric in weather_metrics:
                if metric in merged_df.columns:
                    corr = merged_df['demand'].corr(merged_df[metric])
                    correlations[metric] = round(corr, 3)
            
            # Create correlation bar chart
            correlation_fig = go.Figure()
            
            # Sort by absolute correlation value
            sorted_correlations = sorted(
                correlations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Create bar chart
            correlation_fig.add_trace(go.Bar(
                x=[k.replace('_', ' ').title() for k, v in sorted_correlations],
                y=[v for k, v in sorted_correlations],
                text=[f"{v:.3f}" for k, v in sorted_correlations],
                textposition='auto',
                marker_color=[
                    'green' if v > 0 else 'crimson' 
                    for k, v in sorted_correlations
                ],
                name='Correlation Coefficient'
            ))
            
            correlation_fig.update_layout(
                title=f"Weather-Demand Correlation for Store {store_id}, Item {item_id}",
                xaxis_title="Weather Metric",
                yaxis_title="Correlation with Demand",
                yaxis=dict(range=[-1, 1]),
                template='plotly_white'
            )
            
            # Add reference lines
            for y, text, color in [
                (0.7, 'Strong Positive', 'rgba(0,100,0,0.3)'),
                (0.4, 'Moderate Positive', 'rgba(144,238,144,0.3)'),
                (-0.4, 'Moderate Negative', 'rgba(255,99,71,0.3)'),
                (-0.7, 'Strong Negative', 'rgba(139,0,0,0.3)')
            ]:
                correlation_fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=y,
                    x1=len(sorted_correlations) - 0.5,
                    y1=y,
                    line=dict(color=color, width=1, dash="dot")
                )
                correlation_fig.add_annotation(
                    x=len(sorted_correlations) - 0.5,
                    y=y,
                    text=text,
                    showarrow=False,
                    xshift=10,
                    align="left",
                    font=dict(size=10)
                )
            
            # Create scatter plot for the strongest correlation
            strongest_metric, strongest_corr = sorted_correlations[0]
            
            impact_fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                specs=[[{}], [{}]],
                subplot_titles=(
                    f"Impact of {strongest_metric.replace('_', ' ').title()} on Demand",
                    "Daily Demand and Weather Impact Factor"
                )
            )
            
            # Add scatter plot
            impact_fig.add_trace(
                go.Scatter(
                    x=merged_df[strongest_metric],
                    y=merged_df['demand'],
                    mode='markers',
                    marker=dict(
                        color=merged_df['date'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Date"),
                        size=10
                    ),
                    name='Demand',
                    hovertemplate=
                    '<b>Date</b>: %{text}<br>' +
                    f'<b>{strongest_metric.replace("_", " ").title()}</b>: %{{x}}<br>' +
                    '<b>Demand</b>: %{y}' +
                    '<extra></extra>',
                    text=[d.strftime('%Y-%m-%d') for d in merged_df['date']]
                ),
                row=1, col=1
            )
            
            # Add trend line
            z = np.polyfit(merged_df[strongest_metric], merged_df['demand'], 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(merged_df[strongest_metric].min(), merged_df[strongest_metric].max(), 100)
            
            impact_fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name=f'Trend (r={strongest_corr:.3f})'
                ),
                row=1, col=1
            )
            
            # Add time series
            impact_fig.add_trace(
                go.Scatter(
                    x=merged_df['date'],
                    y=merged_df['demand'],
                    mode='lines+markers',
                    name='Demand',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            impact_fig.add_trace(
                go.Scatter(
                    x=merged_df['date'],
                    y=merged_df['overall_impact'],
                    mode='lines',
                    name='Weather Impact',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # Update layout
            impact_fig.update_layout(
                height=700,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Set axis titles
            impact_fig.update_xaxes(
                title_text=strongest_metric.replace('_', ' ').title(),
                row=1, col=1
            )
            impact_fig.update_yaxes(title_text="Demand", row=1, col=1)
            impact_fig.update_xaxes(title_text="Date", row=2, col=1)
            impact_fig.update_yaxes(title_text="Value", row=2, col=1)
            
            logger.info(f"Created weather-demand correlation visualization for store {store_id}, item {item_id}")
            return correlation_fig, impact_fig
            
        except Exception as e:
            logger.error(f"Error creating weather-demand correlation: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating correlation visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig, fig

    def create_weather_adjusted_forecast(self, store_id, item_id, location=None,
                                        base_forecast=None, forecast_dates=None,
                                        days_forecast=14):
        """Create a visualization comparing base forecast and weather-adjusted forecast
        
        Args:
            store_id (str): Store ID
            item_id (str): Item ID
            location (str): Location name
            base_forecast (list): List of base demand forecast values
            forecast_dates (list): List of forecast dates as strings
            days_forecast (int): Number of forecast days
            
        Returns:
            plotly.graph_objects.Figure: Figure with forecast comparison
        """
        try:
            # Get forecast weather data
            _, forecast_df = self.get_weather_data(location, days_historical=1, days_forecast=days_forecast)
            
            if forecast_df.empty:
                logger.warning("Empty weather forecast data")
                return go.Figure()
            
            # Validate base forecast
            if base_forecast is None or not base_forecast:
                logger.warning("No base forecast provided")
                return go.Figure()
                
            if forecast_dates is None or not forecast_dates:
                # Generate forecast dates starting from tomorrow
                start_date = datetime.now().date() + timedelta(days=1)
                forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                                 for i in range(len(base_forecast))]
            
            # Ensure forecast_df has the same number of days as base_forecast
            if len(forecast_df) > len(base_forecast):
                forecast_df = forecast_df.iloc[:len(base_forecast)]
            
            # Get weather impact
            impact_factors = forecast_df['overall_impact'].values
            
            # Apply weather adjustment
            if len(impact_factors) < len(base_forecast):
                # Pad with 1.0 (no adjustment) if needed
                impact_factors = np.append(
                    impact_factors, 
                    np.ones(len(base_forecast) - len(impact_factors))
                )
                
            # Calculate adjusted forecast
            adjusted_forecast = [base * impact for base, impact in zip(base_forecast, impact_factors)]
            
            # Create visualization
            fig = go.Figure()
            
            # Convert forecast dates to datetime for proper ordering
            if isinstance(forecast_dates[0], str):
                dt_dates = [pd.to_datetime(d) for d in forecast_dates]
            else:
                dt_dates = forecast_dates
            
            # Base forecast line
            fig.add_trace(go.Scatter(
                x=dt_dates,
                y=base_forecast,
                mode='lines+markers',
                name='Base Forecast',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            
            # Adjusted forecast line
            fig.add_trace(go.Scatter(
                x=dt_dates,
                y=adjusted_forecast,
                mode='lines+markers',
                name='Weather-Adjusted',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ))
            
            # Add area showing the difference
            fig.add_trace(go.Scatter(
                x=dt_dates + dt_dates[::-1],
                y=adjusted_forecast + base_forecast[::-1],
                fill='toself',
                fillcolor='rgba(0,100,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Adjustment Range',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add impact factor on secondary y-axis
            fig.add_trace(go.Scatter(
                x=dt_dates,
                y=impact_factors,
                mode='lines',
                name='Weather Impact Factor',
                line=dict(color='purple', width=1, dash='dot'),
                yaxis='y2'
            ))
            
            # Weather indicators
            weather_markers = []
            
            for i, date in enumerate(dt_dates):
                if i < len(forecast_df):
                    # Check for rain or snow
                    rain = forecast_df.iloc[i]['rain'] > 1  # More than 1mm rain
                    snow = forecast_df.iloc[i]['snowfall'] > 0  # Any snow
                    temp = forecast_df.iloc[i]['temperature_mean']
                    
                    marker = None
                    if snow:
                        marker = '❄️'  # Snowflake
                    elif rain:
                        marker = '🌧️'  # Rain
                    elif temp > 25:
                        marker = '☀️'  # Hot
                    elif temp < 5:
                        marker = '❄'  # Cold
                    
                    if marker:
                        weather_markers.append({
                            'x': date,
                            'y': max(base_forecast[i], adjusted_forecast[i]) * 1.05,
                            'text': marker,
                            'showarrow': False,
                            'font': {'size': 16}
                        })
            
            # Add weather markers
            for marker in weather_markers:
                fig.add_annotation(marker)
            
            # Calculate percentage difference
            avg_base = sum(base_forecast) / len(base_forecast)
            avg_adjusted = sum(adjusted_forecast) / len(adjusted_forecast)
            pct_diff = (avg_adjusted - avg_base) / avg_base * 100
            
            # Add annotation explaining the adjustment
            fig.add_annotation(
                x=0.5,
                y=1.1,
                xref='paper',
                yref='paper',
                text=f"Weather impact: {'+' if pct_diff >= 0 else ''}{pct_diff:.1f}% overall forecast adjustment",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Update layout
            fig.update_layout(
                title=f"Weather-Adjusted Forecast for Store {store_id}, Item {item_id}",
                xaxis_title="Date",
                yaxis_title="Forecast Demand",
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified',
                yaxis2=dict(
                    title="Impact Factor",
                    overlaying='y',
                    side='right',
                    range=[0.7, 1.3],
                    showgrid=False
                )
            )
            
            logger.info(f"Created weather-adjusted forecast visualization for store {store_id}, item {item_id}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating weather-adjusted forecast: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating weather-adjusted forecast visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

    def create_weather_impact_summary(self, location=None):
        """Create a summary visualization of weather impact factors
        
        Args:
            location (str): Location name
            
        Returns:
            plotly.graph_objects.Figure: Figure with weather impact summary
        """
        try:
            # Get historical and forecast weather data
            historical_df, forecast_df = self.get_weather_data(
                location, days_historical=60, days_forecast=14
            )
            
            if historical_df.empty and forecast_df.empty:
                logger.warning("Empty weather data, cannot create impact summary")
                return go.Figure()
            
            # Combine historical and forecast data
            historical_df['data_type'] = 'Historical'
            forecast_df['data_type'] = 'Forecast'
            combined_df = pd.concat([historical_df, forecast_df])
            
            # Bin the data by temperature ranges
            temp_bins = [-20, 0, 10, 20, 30, 50]
            bin_labels = ['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot']
            
            combined_df['temp_category'] = pd.cut(
                combined_df['temperature_mean'],
                bins=temp_bins,
                labels=bin_labels,
                include_lowest=True
            )
            
            # Calculate average impact by temperature category
            impact_by_temp = combined_df.groupby('temp_category')['overall_impact'].mean().reset_index()
            
            # Create impact summary figure
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter", "colspan": 2}, None]],
                subplot_titles=(
                    "Impact by Temperature Range", 
                    "Impact by Precipitation Level",
                    "Weather Impact Distribution"
                )
            )
            
            # Impact by temperature category
            fig.add_trace(
                go.Bar(
                    x=impact_by_temp['temp_category'],
                    y=impact_by_temp['overall_impact'],
                    name="Temperature Impact",
                    marker_color='orange',
                    text=[f"{x:.2f}" for x in impact_by_temp['overall_impact']],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Bin the data by precipitation ranges
            precip_bins = [0, 0.1, 5, 15, 30, 100]
            precip_labels = ['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy']
            
            combined_df['precip_category'] = pd.cut(
                combined_df['precipitation'],
                bins=precip_bins,
                labels=precip_labels,
                include_lowest=True
            )
            
            # Calculate average impact by precipitation category
            impact_by_precip = combined_df.groupby('precip_category')['overall_impact'].mean().reset_index()
            
            # Impact by precipitation category
            fig.add_trace(
                go.Bar(
                    x=impact_by_precip['precip_category'],
                    y=impact_by_precip['overall_impact'],
                    name="Precipitation Impact",
                    marker_color='blue',
                    text=[f"{x:.2f}" for x in impact_by_precip['overall_impact']],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Distribution of impact factors
            fig.add_trace(
                go.Scatter(
                    x=combined_df[combined_df['data_type'] == 'Historical']['date'],
                    y=combined_df[combined_df['data_type'] == 'Historical']['overall_impact'],
                    mode='lines',
                    name="Historical Impact",
                    line=dict(color='gray', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=combined_df[combined_df['data_type'] == 'Forecast']['date'],
                    y=combined_df[combined_df['data_type'] == 'Forecast']['overall_impact'],
                    mode='lines',
                    name="Forecast Impact",
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # Add vertical line separating historical from forecast
            forecast_start = combined_df[combined_df['data_type'] == 'Forecast']['date'].min()
            fig.add_vline(
                x=forecast_start, 
                line_width=1, 
                line_dash="dash", 
                line_color="gray",
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=700,
                title_text=f"Weather Impact Factor Analysis - {location or self.default_location}",
                template='plotly_white',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update axes
            fig.update_yaxes(title_text="Average Impact Factor", range=[0.9, 1.2], row=1, col=1)
            fig.update_yaxes(title_text="Average Impact Factor", range=[0.9, 1.2], row=1, col=2)
            fig.update_yaxes(title_text="Impact Factor", range=[0.7, 1.3], row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            # Add reference line at 1.0
            for row, col in [(1, 1), (1, 2)]:
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=1,
                    x1=4.5,
                    y1=1,
                    line=dict(color="red", width=1, dash="dot"),
                    row=row, col=col
                )
            
            logger.info(f"Created weather impact summary visualization for {location or self.default_location}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating weather impact summary: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating weather impact summary: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig

# Example usage
if __name__ == "__main__":
    visualizer = WeatherVisualizer(data_dir="data", default_location="New York")
    
    try:
        # Create overview
        fig = visualizer.create_weather_overview(location="New York")
        fig.write_html("output/weather_overview.html")
        
        # Create impact summary
        impact_fig = visualizer.create_weather_impact_summary(location="New York")
        impact_fig.write_html("output/weather_impact_summary.html")
        
        print("Visualizations created successfully")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")