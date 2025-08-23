#!/usr/bin/env python3
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Union, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ItemDataVisualizer:
    """Class for visualizing sales and purchase data for individual items"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the visualizer with data directory path"""
        self.data_dir = data_dir
        self.sales_data = None
        self.purchase_data = None
        self.stock_data = None
        # No forecast models in current implementation
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
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
            self.stock_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaStock.csv"))
            
            # Convert dates to datetime objects
            if 'Proc_date' in self.sales_data.columns:
                self.sales_data['Proc_date'] = pd.to_datetime(self.sales_data['Proc_date'])
            if 'Proc_date' in self.purchase_data.columns:
                self.purchase_data['Proc_date'] = pd.to_datetime(self.purchase_data['Proc_date'])
            
            # No forecast models in current implementation
            logger.info("Using only raw data, no forecast models implemented")
            
        except Exception as e:
            logger.error(f"Error in _load_data: {str(e)}")
            raise
    
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
    
    def visualize_item_sales_purchases(
        self, 
        store_id: str, 
        item_id: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create visualization of sales and purchases for a specific item
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            start_date: Start date for visualization (format: YYYY-MM-DD)
            end_date: End date for visualization (format: YYYY-MM-DD)
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object
        """
        try:
            # Filter data for the specific store and item
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            item_purchases = self.purchase_data[
                (self.purchase_data['store_id'] == store_id) & 
                (self.purchase_data['item'] == item_id)
            ].copy()
            
            if item_sales.empty and item_purchases.empty:
                logger.warning(f"No data found for store {store_id}, item {item_id}")
                return None
            
            # Apply date filters if provided
            if start_date:
                start_date = pd.Timestamp(start_date)
                item_sales = item_sales[item_sales['Proc_date'] >= start_date]
                item_purchases = item_purchases[item_purchases['Proc_date'] >= start_date]
                
            if end_date:
                end_date = pd.Timestamp(end_date)
                item_sales = item_sales[item_sales['Proc_date'] <= end_date]
                item_purchases = item_purchases[item_purchases['Proc_date'] <= end_date]
            
            # Get item description for title
            item_desc = item_sales['Item_Description'].iloc[0] if not item_sales.empty else item_purchases['Item_Description'].iloc[0] if not item_purchases.empty else f"Item {item_id}"
            
            # Aggregate by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            daily_purchases = item_purchases.groupby('Proc_date')['Total_units'].sum().reset_index()
            
            # Create figure
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True, 
                subplot_titles=(f"Sales: {item_desc}", f"Purchases: {item_desc}"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Add sales trace
            fig.add_trace(
                go.Scatter(
                    x=daily_sales['Proc_date'],
                    y=daily_sales['Total_units'],
                    mode='lines+markers',
                    name='Sales',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6),
                    hovertemplate='%{x|%Y-%m-%d}<br>Units: %{y}<extra>Sales</extra>'
                ),
                row=1, col=1
            )
            
            # Add purchases trace
            fig.add_trace(
                go.Scatter(
                    x=daily_purchases['Proc_date'],
                    y=daily_purchases['Total_units'],
                    mode='lines+markers',
                    name='Purchases',
                    line=dict(color='green', width=2),
                    marker=dict(size=6),
                    hovertemplate='%{x|%Y-%m-%d}<br>Units: %{y}<extra>Purchases</extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"{item_desc} (Store {store_id})",
                height=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified"
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Units Sold", row=1, col=1)
            fig.update_yaxes(title_text="Units Purchased", row=2, col=1)
            
            # Save the figure if output file is provided
            if output_file:
                try:
                    fig.write_html(f"output/{output_file}.html")
                    fig.write_image(f"output/{output_file}.png")
                    logger.info(f"Figure saved to output/{output_file}.html and .png")
                except Exception as e:
                    logger.error(f"Error saving figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in visualize_item_sales_purchases: {str(e)}")
            return None
    
    def detect_item_seasonality(self, store_id: str, item_id: str) -> Dict:
        """
        Detect seasonality patterns for a specific item
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            
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
            if len(daily_sales) < 30:
                logger.warning(f"Insufficient data for seasonality detection for store {store_id}, item {item_id}")
                return {"has_seasonality": False, "reason": "insufficient_data"}
            
            # Fill missing dates with zeros
            idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
            daily_sales = daily_sales.reindex(idx, fill_value=0)
            
            # Test for different seasonal patterns
            seasonality_result = {}
            best_seasonality = None
            best_strength = 0
            
            for period in [7, 14, 30]:
                if len(daily_sales) >= period * 2:  # Need at least 2 full cycles
                    try:
                        # Calculate autocorrelation at the seasonal lag
                        autocorr = daily_sales['Total_units'].autocorr(lag=period)
                        
                        # If autocorrelation is strong enough, consider it seasonal
                        if autocorr > 0.3:  # Threshold for seasonality
                            if autocorr > best_strength:
                                best_strength = autocorr
                                best_seasonality = period
                            
                            seasonality_result[f"{period}_day"] = {
                                "autocorrelation": autocorr,
                                "significance": "strong" if autocorr > 0.5 else "moderate"
                            }
                    except Exception as e:
                        logger.warning(f"Error calculating {period}-day seasonality: {str(e)}")
            
            result = {
                "has_seasonality": best_seasonality is not None,
                "best_period": best_seasonality,
                "strength": best_strength if best_seasonality else 0,
                "tested_periods": seasonality_result
            }
            
            logger.info(f"Seasonality detection for store {store_id}, item {item_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in detect_item_seasonality: {str(e)}")
            return {"has_seasonality": False, "error": str(e)}

    def visualize_sales_trends(
        self,
        store_id: str,
        item_id: str,
        timeframe: str = "1_month",  # "1_month", "3_months", "6_months", "all"
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create visualization of sales trends for a specific item with different timeframes
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            timeframe: Period for visualization ('1_month', '3_months', '6_months', 'all')
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object
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
            
            # Get item description
            item_desc = item_sales['Item_Description'].iloc[0] if not item_sales.empty else f"Item {item_id}"
            
            # Determine timeframe
            today = datetime.now()
            if timeframe == "1_month":
                start_date = today - timedelta(days=30)
                title_period = "Last Month"
            elif timeframe == "3_months":
                start_date = today - timedelta(days=90)
                title_period = "Last 3 Months"
            elif timeframe == "6_months":
                start_date = today - timedelta(days=180)
                title_period = "Last 6 Months"
            else:  # "all"
                start_date = None
                title_period = "All Time"
            
            # Filter by start date if specified
            if start_date:
                item_sales = item_sales[item_sales['Proc_date'] >= start_date]
            
            # Create figure
            fig = go.Figure()
            
            # Aggregate by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            
            # Add sales trace
            fig.add_trace(
                go.Scatter(
                    x=daily_sales['Proc_date'],
                    y=daily_sales['Total_units'],
                    mode='lines+markers',
                    name='Sales',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6),
                    hovertemplate='%{x|%Y-%m-%d}<br>Units Sold: %{y}<extra></extra>'
                )
            )
            
            # Calculate moving average if enough data
            if len(daily_sales) > 7:
                daily_sales['MA7'] = daily_sales['Total_units'].rolling(window=7).mean()
                fig.add_trace(
                    go.Scatter(
                        x=daily_sales['Proc_date'],
                        y=daily_sales['MA7'],
                        mode='lines',
                        name='7-Day Moving Average',
                        line=dict(color='red', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>MA7: %{y:.1f}<extra></extra>'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f"{item_desc} (Store {store_id}) - Sales Trend {title_period}",
                xaxis_title="Date",
                yaxis_title="Units Sold",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                height=600
            )
            
            # Save the figure if output file is provided
            if output_file:
                try:
                    fig.write_html(f"output/{output_file}.html")
                    fig.write_image(f"output/{output_file}.png")
                    logger.info(f"Figure saved to output/{output_file}.html and .png")
                except Exception as e:
                    logger.error(f"Error saving figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in visualize_sales_trends: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        # Initialize visualizer
        visualizer = ItemDataVisualizer(data_dir="data")
        
        # Get list of items
        items = visualizer.get_item_list()
        logger.info(f"Found {len(items)} unique store-item combinations")
        
        # Process first 5 items as a sample
        for i, (store_id, item_id, desc) in enumerate(items[:5]):
            logger.info(f"Processing {i+1}/5: Store {store_id}, Item {item_id} ({desc})")
            
            # Visualize sales and purchases
            vis_fig = visualizer.visualize_item_sales_purchases(
                store_id, 
                item_id,
                output_file=f"sales_purchases_{store_id}_{item_id}"
            )
            
            # Detect seasonality
            seasonality = visualizer.detect_item_seasonality(store_id, item_id)
            logger.info(f"Seasonality for {desc}: {seasonality}")
            
            # Visualize model predictions (2 weeks)
            pred_fig_2w = visualizer.visualize_model_predictions(
                store_id,
                item_id,
                forecast_period="2_weeks",
                output_file=f"predictions_2w_{store_id}_{item_id}"
            )
            
            # Visualize model predictions (2 months)
            pred_fig_2m = visualizer.visualize_model_predictions(
                store_id,
                item_id,
                forecast_period="2_months",
                output_file=f"predictions_2m_{store_id}_{item_id}"
            )
            
        logger.info("Sample visualizations completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")