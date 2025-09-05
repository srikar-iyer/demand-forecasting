#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherClient:
    """Client for fetching weather data from OpenMeteo API"""
    
    def __init__(self, cache_session=True):
        """Initialize the weather client with caching and retry support
        
        Args:
            cache_session (bool): Whether to cache API requests
        """
        # Setup the Open-Meteo API client with cache and retry on error
        if cache_session:
            cache_session = requests_cache.CachedSession('.openmeteo_cache', expire_after=3600)
        else:
            cache_session = requests.Session()
            
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        logger.info("Weather client initialized")
        
    def fetch_historical_weather(self, latitude, longitude, start_date, end_date):
        """Fetch historical weather data for a specific location and time period
        
        Args:
            latitude (float): Location latitude
            longitude (float): Location longitude
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: DataFrame with daily weather data or empty DataFrame if error occurs
        """
        try:
            # Convert dates to datetime objects for validation
            from datetime import datetime
            
            # Parse the input dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            
            # Validate dates - historical API only works with past dates
            today = datetime.now().date()
            if start_dt > today or end_dt > today:
                logger.warning("Historical API only supports past dates. Adjusting to use only past dates.")
                # Adjust end_date to be today or earlier
                end_dt = min(end_dt, today)
                # Adjust start_date to be today or earlier, and maintain same duration if possible
                duration = (end_dt - start_dt).days
                start_dt = end_dt - timedelta(days=duration)
                # Convert back to string format
                start_date = start_dt.strftime("%Y-%m-%d")
                end_date = end_dt.strftime("%Y-%m-%d")
                
                logger.info(f"Adjusted date range to {start_date} - {end_date}")
            
            # Define API parameters
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                          "precipitation_sum", "rain_sum", "snowfall_sum", 
                          "precipitation_hours", "wind_speed_10m_max"],
                "timezone": "auto"
            }
            
            logger.info(f"Fetching historical weather data for coordinates ({latitude}, {longitude}) from {start_date} to {end_date}")
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Parse the response
            daily = response.Daily()
            
            # Get values and check lengths
            time_start = pd.Timestamp(daily.Time())
            time_end = pd.Timestamp(daily.TimeEnd())
            date_range = pd.date_range(start=time_start, end=time_end, freq=pd.Timedelta(days=1))
            
            # Get all variables and verify they have the same length as the date range
            temps_max = daily.Variables(0).ValuesAsNumpy()
            temps_min = daily.Variables(1).ValuesAsNumpy()
            temps_mean = daily.Variables(2).ValuesAsNumpy()
            precip = daily.Variables(3).ValuesAsNumpy()
            rain = daily.Variables(4).ValuesAsNumpy()
            snow = daily.Variables(5).ValuesAsNumpy()
            precip_hours = daily.Variables(6).ValuesAsNumpy()
            wind = daily.Variables(7).ValuesAsNumpy()
            
            # Debug info
            logger.debug(f"Date range length: {len(date_range)}")
            logger.debug(f"Temperature max length: {len(temps_max)}")
            logger.debug(f"Temperature min length: {len(temps_min)}")
            logger.debug(f"Temperature mean length: {len(temps_mean)}")
            logger.debug(f"Precipitation length: {len(precip)}")
            logger.debug(f"Rain length: {len(rain)}")
            logger.debug(f"Snow length: {len(snow)}")
            logger.debug(f"Precipitation hours length: {len(precip_hours)}")
            logger.debug(f"Wind length: {len(wind)}")
            
            # Check if any array length doesn't match date range
            expected_length = len(date_range)
            if (len(temps_max) != expected_length or len(temps_min) != expected_length or
                len(temps_mean) != expected_length or len(precip) != expected_length or
                len(rain) != expected_length or len(snow) != expected_length or
                len(precip_hours) != expected_length or len(wind) != expected_length):
                
                # Try to handle the mismatched lengths by truncating to shortest length
                min_length = min(len(date_range), len(temps_max), len(temps_min), len(temps_mean),
                                len(precip), len(rain), len(snow), len(precip_hours), len(wind))
                
                logger.warning(f"Array length mismatch detected. Truncating to shortest length: {min_length}")
                
                # Truncate arrays to the same length
                date_range = date_range[:min_length]
                temps_max = temps_max[:min_length]
                temps_min = temps_min[:min_length]
                temps_mean = temps_mean[:min_length]
                precip = precip[:min_length]
                rain = rain[:min_length]
                snow = snow[:min_length]
                precip_hours = precip_hours[:min_length]
                wind = wind[:min_length]
            
            # Create daily data dictionary with verified arrays
            daily_data = {
                "date": date_range,
                "temperature_max": temps_max,
                "temperature_min": temps_min,
                "temperature_mean": temps_mean,
                "precipitation": precip,
                "rain": rain,
                "snowfall": snow,
                "precipitation_hours": precip_hours,
                "wind_speed_max": wind
            }
            
            # Create DataFrame
            df = pd.DataFrame(daily_data)
            
            # Add derived features
            df["temperature_range"] = df["temperature_max"] - df["temperature_min"]
            df["is_rainy"] = df["precipitation"] > 0
            df["is_snowy"] = df["snowfall"] > 0
            
            logger.info(f"Successfully retrieved weather data: {len(df)} days")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical weather: {str(e)}")
            # Return empty DataFrame instead of raising exception
            return pd.DataFrame()
            
    def fetch_forecast(self, latitude, longitude, days=10):
        """Fetch weather forecast for a specific location
        
        Args:
            latitude (float): Location latitude
            longitude (float): Location longitude
            days (int): Number of days to forecast (max 14)
            
        Returns:
            pandas.DataFrame: DataFrame with forecast weather data
        """
        try:
            # Ensure days is within allowed range
            days = min(max(1, days), 14)
            
            # Define API parameters
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                          "precipitation_sum", "rain_sum", "snowfall_sum", 
                          "precipitation_hours", "wind_speed_10m_max"],
                "timezone": "auto",
                "forecast_days": days
            }
            
            logger.info(f"Fetching weather forecast for coordinates ({latitude}, {longitude}) for next {days} days")
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Parse the response
            daily = response.Daily()
            
            # Get values and check lengths
            time_start = pd.Timestamp(daily.Time())
            time_end = pd.Timestamp(daily.TimeEnd())
            date_range = pd.date_range(start=time_start, end=time_end, freq=pd.Timedelta(days=1))
            
            # Get all variables and verify they have the same length as the date range
            temps_max = daily.Variables(0).ValuesAsNumpy()
            temps_min = daily.Variables(1).ValuesAsNumpy()
            temps_mean = daily.Variables(2).ValuesAsNumpy()
            precip = daily.Variables(3).ValuesAsNumpy()
            rain = daily.Variables(4).ValuesAsNumpy()
            snow = daily.Variables(5).ValuesAsNumpy()
            precip_hours = daily.Variables(6).ValuesAsNumpy()
            wind = daily.Variables(7).ValuesAsNumpy()
            
            # Debug info
            logger.debug(f"Date range length: {len(date_range)}")
            logger.debug(f"Temperature max length: {len(temps_max)}")
            logger.debug(f"Temperature min length: {len(temps_min)}")
            logger.debug(f"Temperature mean length: {len(temps_mean)}")
            logger.debug(f"Precipitation length: {len(precip)}")
            logger.debug(f"Rain length: {len(rain)}")
            logger.debug(f"Snow length: {len(snow)}")
            logger.debug(f"Precipitation hours length: {len(precip_hours)}")
            logger.debug(f"Wind length: {len(wind)}")
            
            # Check if any array length doesn't match date range
            expected_length = len(date_range)
            if (len(temps_max) != expected_length or len(temps_min) != expected_length or
                len(temps_mean) != expected_length or len(precip) != expected_length or
                len(rain) != expected_length or len(snow) != expected_length or
                len(precip_hours) != expected_length or len(wind) != expected_length):
                
                # Try to handle the mismatched lengths by truncating to shortest length
                min_length = min(len(date_range), len(temps_max), len(temps_min), len(temps_mean),
                                len(precip), len(rain), len(snow), len(precip_hours), len(wind))
                
                logger.warning(f"Array length mismatch detected in forecast. Truncating to shortest length: {min_length}")
                
                # Truncate arrays to the same length
                date_range = date_range[:min_length]
                temps_max = temps_max[:min_length]
                temps_min = temps_min[:min_length]
                temps_mean = temps_mean[:min_length]
                precip = precip[:min_length]
                rain = rain[:min_length]
                snow = snow[:min_length]
                precip_hours = precip_hours[:min_length]
                wind = wind[:min_length]
            
            # Create daily data dictionary with verified arrays
            daily_data = {
                "date": date_range,
                "temperature_max": temps_max,
                "temperature_min": temps_min,
                "temperature_mean": temps_mean,
                "precipitation": precip,
                "rain": rain,
                "snowfall": snow,
                "precipitation_hours": precip_hours,
                "wind_speed_max": wind
            }
            
            # Create DataFrame
            df = pd.DataFrame(daily_data)
            
            # Add derived features
            df["temperature_range"] = df["temperature_max"] - df["temperature_min"]
            df["is_rainy"] = df["precipitation"] > 0
            df["is_snowy"] = df["snowfall"] > 0
            
            logger.info(f"Successfully retrieved forecast data: {len(df)} days")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {str(e)}")
            # Return empty DataFrame instead of raising exception
            return pd.DataFrame()
            
    def get_location_by_name(self, location_name):
        """Get latitude and longitude for a location name (simplified)
        
        Args:
            location_name (str): Name of the location (e.g., "New York")
            
        Returns:
            tuple: (latitude, longitude) coordinates
        """
        # Simplified location lookup - in a real app, use a geocoding API
        location_map = {
            "new york": (40.7128, -74.0060),
            "los angeles": (34.0522, -118.2437),
            "chicago": (41.8781, -87.6298),
            "houston": (29.7604, -95.3698),
            "phoenix": (33.4484, -112.0740),
            "philadelphia": (39.9526, -75.1652),
            "san antonio": (29.4241, -98.4936),
            "san diego": (32.7157, -117.1611),
            "dallas": (32.7767, -96.7970),
            "san francisco": (37.7749, -122.4194),
            "austin": (30.2672, -97.7431),
            "seattle": (47.6062, -122.3321),
            "boston": (42.3601, -71.0589),
            "miami": (25.7617, -80.1918),
            "denver": (39.7392, -104.9903)
        }
        
        location_key = location_name.lower()
        if location_key in location_map:
            return location_map[location_key]
        else:
            logger.warning(f"Location '{location_name}' not found in mapping. Using New York as default.")
            return location_map["new york"]

    def get_weather_impact_factors(self, weather_df):
        """Calculate impact factors of weather on demand based on weather metrics
        
        Args:
            weather_df (pandas.DataFrame): DataFrame with weather data
            
        Returns:
            pandas.DataFrame: DataFrame with original data and impact factors
        """
        df = weather_df.copy()
        
        # Define impact factors based on weather conditions
        # These are simplified factors for demonstration
        
        # Temperature impact - higher temps increase cold food demand
        # Scale from 0.8 (cold) to 1.3 (hot) - centered at 20°C
        df['temperature_impact'] = df['temperature_mean'].apply(
            lambda x: min(1.3, max(0.8, 1 + (x - 20) / 30))
        )
        
        # Rain impact - rainy days increase delivery/frozen food
        df['rain_impact'] = df['precipitation'].apply(
            lambda x: 1.15 if x > 10 else (1.05 if x > 0 else 1.0)
        )
        
        # Snow impact - snowy days strongly increase delivery/frozen food
        df['snow_impact'] = df['snowfall'].apply(
            lambda x: 1.25 if x > 5 else (1.15 if x > 0 else 1.0)
        )
        
        # Wind impact - very windy days slightly decrease shopping trips
        df['wind_impact'] = df['wind_speed_max'].apply(
            lambda x: 1.1 if x > 30 else (1.05 if x > 20 else 1.0)
        )
        
        # Calculate combined impact - product of individual impacts
        df['overall_impact'] = df['temperature_impact'] * df['rain_impact'] * \
                              df['snow_impact'] * df['wind_impact']
                              
        logger.info(f"Calculated weather impact factors for {len(df)} days")
        return df

    def analyze_weather_demand_correlation(self, weather_df, demand_df, date_field='date'):
        """Analyze correlation between weather factors and demand
        
        Args:
            weather_df (pandas.DataFrame): DataFrame with weather data
            demand_df (pandas.DataFrame): DataFrame with demand data
            date_field (str): Name of the date column in both dataframes
            
        Returns:
            dict: Dictionary with correlation metrics and analysis
        """
        try:
            # Ensure date field is in datetime format
            weather_df[date_field] = pd.to_datetime(weather_df[date_field])
            demand_df[date_field] = pd.to_datetime(demand_df[date_field])
            
            # Merge dataframes on date
            merged_df = pd.merge(weather_df, demand_df, on=date_field, how='inner')
            
            if merged_df.empty:
                logger.warning("No matching dates between weather and demand data")
                return {
                    "correlation": {},
                    "valid_data_points": 0,
                    "analysis": "No matching dates found between weather and demand data"
                }
                
            # Identify demand column - assume it's the numeric column that's not in weather_df
            demand_columns = [col for col in demand_df.columns 
                             if col not in weather_df.columns and col != date_field]
            
            if not demand_columns:
                logger.warning("No demand column identified for correlation analysis")
                return {
                    "correlation": {},
                    "valid_data_points": len(merged_df),
                    "analysis": "No suitable demand column found for correlation analysis"
                }
                
            demand_col = demand_columns[0]  # Use the first identified demand column
            
            # Calculate correlations between weather metrics and demand
            weather_metrics = ['temperature_mean', 'precipitation', 'wind_speed_max', 'temperature_range']
            correlations = {}
            
            for metric in weather_metrics:
                if metric in merged_df.columns:
                    corr = merged_df[metric].corr(merged_df[demand_col])
                    correlations[metric] = round(corr, 3)
            
            # Basic analysis of correlations
            analysis = []
            for metric, corr in correlations.items():
                if abs(corr) < 0.2:
                    analysis.append(f"No significant correlation between {metric} and demand")
                elif abs(corr) < 0.4:
                    direction = "positive" if corr > 0 else "negative"
                    analysis.append(f"Weak {direction} correlation between {metric} and demand")
                elif abs(corr) < 0.6:
                    direction = "positive" if corr > 0 else "negative"
                    analysis.append(f"Moderate {direction} correlation between {metric} and demand")
                else:
                    direction = "positive" if corr > 0 else "negative"
                    analysis.append(f"Strong {direction} correlation between {metric} and demand")
            
            result = {
                "correlation": correlations,
                "valid_data_points": len(merged_df),
                "analysis": "; ".join(analysis)
            }
            
            logger.info(f"Completed correlation analysis with {len(merged_df)} data points")
            return result
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {
                "correlation": {},
                "valid_data_points": 0,
                "analysis": f"Error in analysis: {str(e)}"
            }

    def forecast_with_weather_adjustment(self, base_forecast, weather_forecast):
        """Adjust demand forecast based on weather forecast
        
        Args:
            base_forecast (list): List of base demand forecast values
            weather_forecast (pandas.DataFrame): DataFrame with weather forecast
            
        Returns:
            list: Adjusted forecast values
        """
        try:
            if len(base_forecast) > len(weather_forecast):
                logger.warning(f"Base forecast ({len(base_forecast)} days) longer than weather forecast ({len(weather_forecast)} days)")
                weather_forecast = weather_forecast.iloc[:len(base_forecast)]
            elif len(base_forecast) < len(weather_forecast):
                weather_forecast = weather_forecast.iloc[:len(base_forecast)]
            
            # Get weather impact factors
            impact_df = self.get_weather_impact_factors(weather_forecast)
            
            # Apply impact to base forecast
            adjusted_forecast = [base * impact for base, impact in 
                                zip(base_forecast, impact_df['overall_impact'])]
            
            logger.info(f"Generated weather-adjusted forecast for {len(adjusted_forecast)} days")
            return adjusted_forecast
            
        except Exception as e:
            logger.error(f"Error adjusting forecast with weather: {str(e)}")
            return base_forecast  # Return original forecast in case of error

# Example usage
if __name__ == "__main__":
    # Test the weather client
    client = WeatherClient()
    
    # Example location: New York
    lat, lon = client.get_location_by_name("New York")
    
    # Historical weather for last 30 days - make sure we're using past dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Ensure we're not using future dates due to any system clock issues
    today_date = datetime.now().strftime("%Y-%m-%d")
    if end_date > today_date:
        print(f"Warning: System date appears to be set in the future. Adjusting to today's date.")
        end_date = "2023-08-31"  # Use a safe past date
        start_date = "2023-08-01"  # Use a safe past date
    
    try:
        # Get historical weather
        historical = client.fetch_historical_weather(lat, lon, start_date, end_date)
        print(f"Retrieved {len(historical)} days of historical weather data")
        print(historical.head())
        
        # Get forecast
        forecast = client.fetch_forecast(lat, lon, days=7)
        print(f"Retrieved {len(forecast)} days of forecast data")
        print(forecast.head())
        
        # Get weather impact factors
        impact = client.get_weather_impact_factors(forecast)
        print("\nWeather Impact Factors:")
        print(impact[['date', 'temperature_mean', 'precipitation', 'overall_impact']].head())
        
    except Exception as e:
        print(f"Error testing weather client: {str(e)}")