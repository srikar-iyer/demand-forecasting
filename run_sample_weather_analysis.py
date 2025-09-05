#!/usr/bin/env python3
"""
Sample script to demonstrate weather analysis functionality
without requiring the full Dash app to run.
This script generates sample weather visualizations and saves them as HTML files.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from weather_client import WeatherClient
from weather_visualizer import WeatherVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_analysis_sample.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_sample_demand_data():
    """Load and prepare sample demand data from the sales file"""
    try:
        # Load sales data
        sales_df = pd.read_csv('data/FrozenPizzaSales.csv')
        
        # Convert date column
        sales_df['date'] = pd.to_datetime(sales_df['Proc_date'])
        
        # Aggregate by date and store_id to get daily totals
        demand_df = sales_df.groupby(['store_id', 'date']).agg(
            demand=('Total_units', 'sum')
        ).reset_index()
        
        logger.info(f"Loaded sample demand data with {len(demand_df)} records")
        return demand_df
    
    except Exception as e:
        logger.error(f"Error loading sample demand data: {str(e)}")
        return pd.DataFrame()

def run_sample_analysis():
    """Generate sample weather visualizations"""
    try:
        # Initialize components
        weather_client = WeatherClient(cache_session=True)
        weather_visualizer = WeatherVisualizer(data_dir="data", default_location="New York")
        
        # Location to analyze
        location = "New York"
        
        # Generate visualizations
        logger.info(f"Generating weather overview for {location}")
        weather_overview = weather_visualizer.create_weather_overview(location)
        weather_overview.write_html("output/weather_overview.html")
        
        logger.info(f"Generating weather impact summary for {location}")
        impact_summary = weather_visualizer.create_weather_impact_summary(location)
        impact_summary.write_html("output/weather_impact_summary.html")
        
        # Sample forecast for adjustment
        base_forecast = [10, 12, 15, 18, 20, 17, 15, 14, 12, 10, 8, 9, 10, 12]
        start_date = datetime.now().date() + timedelta(days=1)
        forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(base_forecast))]
        
        logger.info("Generating weather-adjusted forecast")
        adjusted_forecast = weather_visualizer.create_weather_adjusted_forecast(
            store_id="104", 
            item_id="3913116850", 
            location=location,
            base_forecast=base_forecast,
            forecast_dates=forecast_dates
        )
        adjusted_forecast.write_html("output/weather_adjusted_forecast.html")
        
        # Load demand data for correlation analysis
        demand_df = load_sample_demand_data()
        
        if not demand_df.empty:
            logger.info("Generating weather-demand correlation analysis")
            correlation_fig, impact_fig = weather_visualizer.create_weather_demand_correlation(
                store_id="104", 
                item_id="3913116850",
                location=location,
                demand_df=demand_df
            )
            
            correlation_fig.write_html("output/weather_correlation.html")
            impact_fig.write_html("output/weather_impact.html")
        
        logger.info("Sample analysis complete. Output files saved to the output directory.")
        
        # Print links to the generated files
        print("\nGenerated visualization files:")
        print("------------------------------")
        print("1. Weather Overview:          output/weather_overview.html")
        print("2. Weather Impact Summary:    output/weather_impact_summary.html")
        print("3. Weather-Adjusted Forecast: output/weather_adjusted_forecast.html")
        
        if not demand_df.empty:
            print("4. Weather-Demand Correlation: output/weather_correlation.html")
            print("5. Weather Impact Analysis:    output/weather_impact.html")
            
    except Exception as e:
        logger.error(f"Error running sample analysis: {str(e)}")

if __name__ == "__main__":
    run_sample_analysis()