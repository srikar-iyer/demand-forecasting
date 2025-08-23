#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing/seasonality.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ItemSeasonalityAnalyzer:
    """Class for analyzing and visualizing item-specific seasonality patterns"""
    
    def __init__(self, data_dir: str = "../data"):
        """Initialize the analyzer with data directory path"""
        self.data_dir = data_dir
        self.sales_data = None
        self.purchase_data = None
        
        # Create output directory
        os.makedirs("testing/output", exist_ok=True)
        
        try:
            self._load_data()
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _load_data(self) -> None:
        """Load all required data files"""
        try:
            # Load raw data
            self.sales_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaSales.csv"))
            self.purchase_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaPurchases.csv"))
            
            # Convert dates to datetime objects
            if 'Proc_date' in self.sales_data.columns:
                self.sales_data['Proc_date'] = pd.to_datetime(self.sales_data['Proc_date'])
            if 'Proc_date' in self.purchase_data.columns:
                self.purchase_data['Proc_date'] = pd.to_datetime(self.purchase_data['Proc_date'])
            
        except Exception as e:
            logger.error(f"Error in _load_data: {str(e)}")
            raise
    
    def detect_seasonality(
        self, 
        store_id: str, 
        item_id: str, 
        test_periods: List[int] = [7, 14, 30],
        min_strength_threshold: float = 0.3
    ) -> Dict:
        """
        Detect seasonality patterns for a specific item
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            test_periods: List of periods (in days) to test for seasonality
            min_strength_threshold: Minimum autocorrelation to consider seasonal
            
        Returns:
            Dictionary with seasonality information
        """
        try:
            # Filter data for the specific store and item
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            if item_sales.empty:
                logger.warning(f"No sales data found for store {store_id}, item {item_id}")
                return {"has_seasonality": False}
            
            # Aggregate by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            daily_sales = daily_sales.set_index('Proc_date')
            
            # Need sufficient data for seasonality detection
            if len(daily_sales) < max(test_periods) * 2:
                logger.warning(f"Insufficient data for seasonality detection for store {store_id}, item {item_id}")
                return {
                    "has_seasonality": False, 
                    "reason": "insufficient_data",
                    "available_days": len(daily_sales),
                    "required_days": max(test_periods) * 2
                }
            
            # Fill missing dates with zeros
            idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
            daily_sales = daily_sales.reindex(idx, fill_value=0)
            
            # Calculate autocorrelation function for different lags
            n_lags = max(test_periods) + 1
            try:
                acf_values = acf(daily_sales['Total_units'], nlags=n_lags)
            except Exception as e:
                logger.error(f"Error calculating ACF: {str(e)}")
                return {"has_seasonality": False, "error": f"ACF calculation failed: {str(e)}"}
            
            # Test for different seasonal patterns
            seasonality_result = {}
            best_seasonality = None
            best_strength = 0
            
            for period in test_periods:
                if period < len(acf_values):
                    acf_value = acf_values[period]
                    
                    # If autocorrelation is strong enough, consider it seasonal
                    if acf_value > min_strength_threshold:
                        if acf_value > best_strength:
                            best_strength = acf_value
                            best_seasonality = period
                        
                        seasonality_result[f"{period}_day"] = {
                            "autocorrelation": acf_value,
                            "significance": "strong" if acf_value > 0.5 else "moderate"
                        }
            
            result = {
                "has_seasonality": best_seasonality is not None,
                "best_period": best_seasonality,
                "strength": best_strength if best_seasonality else 0,
                "tested_periods": seasonality_result
            }
            
            logger.info(f"Seasonality detection for store {store_id}, item {item_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in detect_seasonality: {str(e)}")
            return {"has_seasonality": False, "error": str(e)}
    
    def visualize_seasonality(
        self, 
        store_id: str, 
        item_id: str,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create visualization of seasonality components for a specific item
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object with seasonality decomposition
        """
        try:
            # Filter data for the specific store and item
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            if item_sales.empty:
                logger.warning(f"No sales data found for store {store_id}, item {item_id}")
                return None
            
            # Aggregate by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            daily_sales = daily_sales.set_index('Proc_date')
            
            # Get item description for title
            item_desc = item_sales['Item_Description'].iloc[0] if not item_sales.empty else f"Item {item_id}"
            
            # Fill missing dates with interpolation if possible, otherwise zeros
            idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
            daily_sales = daily_sales.reindex(idx)
            daily_sales['Total_units'] = daily_sales['Total_units'].interpolate(method='linear').fillna(0)
            
            # Detect seasonality first
            seasonality_info = self.detect_seasonality(store_id, item_id)
            
            if not seasonality_info.get("has_seasonality", False):
                # Create a simple time series plot if no seasonality detected
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_sales.index,
                    y=daily_sales['Total_units'],
                    mode='lines+markers',
                    name='Sales',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title=f"{item_desc} (Store {store_id}) - No Significant Seasonality Detected",
                    xaxis_title="Date",
                    yaxis_title="Units Sold",
                    height=600
                )
                
                # Save the figure if output file is provided
                if output_file:
                    try:
                        fig.write_html(f"testing/output/{output_file}.html")
                        fig.write_image(f"testing/output/{output_file}.png")
                        logger.info(f"Figure saved to testing/output/{output_file}.html and .png")
                    except Exception as e:
                        logger.error(f"Error saving figure: {str(e)}")
                
                return fig
            
            # Get the best seasonal period
            best_period = seasonality_info["best_period"]
            
            # Decompose the time series
            try:
                if len(daily_sales) >= best_period * 2:
                    # STL decomposition (Season-Trend-Loess)
                    decomposition = seasonal_decompose(
                        daily_sales['Total_units'], 
                        model='additive', 
                        period=best_period,
                        extrapolate_trend='freq'
                    )
                    
                    # Create figure with subplots
                    fig = make_subplots(
                        rows=4, 
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=(
                            "Original Time Series", 
                            "Trend Component", 
                            f"Seasonal Component (Period: {best_period} days)", 
                            "Residual Component"
                        ),
                        vertical_spacing=0.1,
                        row_heights=[0.4, 0.2, 0.2, 0.2]
                    )
                    
                    # Add original data
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sales.index,
                            y=daily_sales['Total_units'],
                            mode='lines',
                            name='Original',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add trend component
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sales.index,
                            y=decomposition.trend,
                            mode='lines',
                            name='Trend',
                            line=dict(color='green', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    # Add seasonal component
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sales.index,
                            y=decomposition.seasonal,
                            mode='lines',
                            name='Seasonal',
                            line=dict(color='red', width=2)
                        ),
                        row=3, col=1
                    )
                    
                    # Add residual component
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sales.index,
                            y=decomposition.resid,
                            mode='lines',
                            name='Residual',
                            line=dict(color='purple', width=2)
                        ),
                        row=4, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{item_desc} (Store {store_id}) - Seasonality Analysis (Period: {best_period} days)",
                        height=900,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    fig.update_xaxes(title_text="Date", row=4, col=1)
                    fig.update_yaxes(title_text="Units Sold", row=1, col=1)
                    fig.update_yaxes(title_text="Trend", row=2, col=1)
                    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
                    fig.update_yaxes(title_text="Residual", row=4, col=1)
                    
                    # Save the figure if output file is provided
                    if output_file:
                        try:
                            fig.write_html(f"testing/output/{output_file}.html")
                            fig.write_image(f"testing/output/{output_file}.png")
                            logger.info(f"Figure saved to testing/output/{output_file}.html and .png")
                        except Exception as e:
                            logger.error(f"Error saving figure: {str(e)}")
                    
                    return fig
                else:
                    logger.warning(f"Insufficient data for decomposition for store {store_id}, item {item_id}")
                    return None
            except Exception as e:
                logger.error(f"Error in seasonal decomposition: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in visualize_seasonality: {str(e)}")
            return None
    
    def apply_seasonal_adjustment(
        self, 
        store_id: str, 
        item_id: str, 
        forecasts_df: pd.DataFrame,
        date_column: str = 'Date',
        forecast_column: str = 'Forecast'
    ) -> pd.DataFrame:
        """
        Apply item-specific seasonality adjustments to forecasts
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            forecasts_df: DataFrame with forecasts
            date_column: Name of the date column
            forecast_column: Name of the forecast column
            
        Returns:
            DataFrame with seasonally adjusted forecasts
        """
        try:
            if forecasts_df is None or forecasts_df.empty:
                logger.warning(f"No forecast data provided for store {store_id}, item {item_id}")
                return forecasts_df
            
            # Detect seasonality
            seasonality_info = self.detect_seasonality(store_id, item_id)
            
            # If no seasonality detected, return original forecasts
            if not seasonality_info.get("has_seasonality", False):
                logger.info(f"No seasonality detected for store {store_id}, item {item_id}, returning original forecasts")
                forecasts_df['Seasonally_Adjusted'] = forecasts_df[forecast_column]
                return forecasts_df
            
            # Get the best seasonal period
            best_period = seasonality_info["best_period"]
            
            # Filter data for the specific store and item
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            # Aggregate by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            daily_sales = daily_sales.set_index('Proc_date')
            
            # Fill missing dates
            idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
            daily_sales = daily_sales.reindex(idx)
            daily_sales['Total_units'] = daily_sales['Total_units'].interpolate(method='linear').fillna(0)
            
            # Decompose to get seasonal factors
            try:
                decomposition = seasonal_decompose(
                    daily_sales['Total_units'], 
                    model='additive', 
                    period=best_period,
                    extrapolate_trend='freq'
                )
                
                # Extract seasonal pattern (normalize to have mean of 1)
                seasonal_pattern = decomposition.seasonal[-best_period:]
                
                # Create a dictionary mapping day of season to seasonal factor
                seasonal_factors = {}
                for i, factor in enumerate(seasonal_pattern):
                    seasonal_factors[i % best_period] = factor
                
                # Apply seasonal adjustment to forecasts
                adjusted_forecasts = forecasts_df.copy()
                
                # Convert forecast dates to datetime if needed
                if adjusted_forecasts[date_column].dtype != 'datetime64[ns]':
                    adjusted_forecasts[date_column] = pd.to_datetime(adjusted_forecasts[date_column])
                
                # Calculate reference date (start of the seasonality cycle)
                reference_date = daily_sales.index.min()
                
                # Apply seasonal factors
                for idx, row in adjusted_forecasts.iterrows():
                    forecast_date = row[date_column]
                    days_since_reference = (forecast_date - reference_date).days
                    seasonal_position = days_since_reference % best_period
                    seasonal_factor = seasonal_factors.get(seasonal_position, 0)
                    
                    # Apply adjustment
                    adjusted_forecasts.at[idx, 'Seasonally_Adjusted'] = row[forecast_column] + seasonal_factor
                
                logger.info(f"Seasonality adjustment applied to forecasts for store {store_id}, item {item_id}")
                return adjusted_forecasts
                
            except Exception as e:
                logger.error(f"Error applying seasonal adjustment: {str(e)}")
                # Return original forecasts with a placeholder column
                forecasts_df['Seasonally_Adjusted'] = forecasts_df[forecast_column]
                return forecasts_df
            
        except Exception as e:
            logger.error(f"Error in apply_seasonal_adjustment: {str(e)}")
            # Return original forecasts with a placeholder column
            if forecasts_df is not None:
                forecasts_df['Seasonally_Adjusted'] = forecasts_df[forecast_column]
            return forecasts_df

    def get_item_list(self) -> List[Tuple[str, str, str]]:
        """Return a list of unique items with store_id and description"""
        try:
            if self.sales_data is not None:
                items = self.sales_data[['store_id', 'item', 'Item_Description']].drop_duplicates()
                return [(row['store_id'], row['item'], row['Item_Description']) 
                        for _, row in items.iterrows()]
            return []
        except Exception as e:
            logger.error(f"Error in get_item_list: {str(e)}")
            return []

if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = ItemSeasonalityAnalyzer(data_dir="data")
        
        # Get list of items
        items = analyzer.get_item_list()
        logger.info(f"Found {len(items)} unique store-item combinations")
        
        # Process first 5 items as a sample
        for i, (store_id, item_id, desc) in enumerate(items[:5]):
            logger.info(f"Processing {i+1}/5: Store {store_id}, Item {item_id} ({desc})")
            
            # Detect seasonality
            seasonality = analyzer.detect_seasonality(store_id, item_id)
            logger.info(f"Seasonality for {desc}: {seasonality}")
            
            # Visualize seasonality if detected
            if seasonality.get("has_seasonality", False):
                analyzer.visualize_seasonality(
                    store_id, 
                    item_id,
                    output_file=f"seasonality_{store_id}_{item_id}"
                )
        
        logger.info("Sample seasonality analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")