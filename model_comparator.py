#!/usr/bin/env python3
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Union, Optional
from seasonality_analyzer import ItemSeasonalityAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelComparator:
    """Class for comparing different forecast models"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the comparator with data directory path"""
        self.data_dir = data_dir
        self.sales_data = None
        # No forecast models in current implementation
        
        # Initialize seasonality analyzer
        self.seasonality_analyzer = ItemSeasonalityAnalyzer(data_dir=data_dir)
        
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
            
            # Convert dates to datetime objects
            if 'Proc_date' in self.sales_data.columns:
                self.sales_data['Proc_date'] = pd.to_datetime(self.sales_data['Proc_date'])
            
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
    
    def get_sales_data(
        self, 
        store_id: str, 
        item_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get sales data for a specific item within date range
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            start_date: Start date for filtering data
            end_date: End date for filtering data
            
        Returns:
            DataFrame with sales data
        """
        try:
            # Filter data for the specific store and item
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            # Apply date filters
            if start_date and not pd.isna(start_date):
                item_sales = item_sales[item_sales['Proc_date'] >= start_date]
            if end_date and not pd.isna(end_date):
                item_sales = item_sales[item_sales['Proc_date'] <= end_date]
            
            # Aggregate by date
            if not item_sales.empty:
                daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
                return daily_sales
            else:
                return pd.DataFrame(columns=['Proc_date', 'Total_units'])
            
        except Exception as e:
            logger.error(f"Error in get_sales_data: {str(e)}")
            return pd.DataFrame(columns=['Proc_date', 'Total_units'])

    def create_sales_analysis(
        self,
        store_id: str,
        item_id: str,
        period: str = "all",  # "1_month", "3_months", "6_months", "all"
        with_seasonality: bool = True,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create visualization of sales data with seasonality indicators
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            period: Period for visualization ('1_month', '3_months', '6_months', 'all')
            with_seasonality: Whether to show seasonality indicators
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object
        """
        try:
            # Determine date range based on the period
            today = datetime.now()
            
            if period == "1_month":
                start_date = today - timedelta(days=30)
                title_period = "1 Month"
            elif period == "3_months":
                start_date = today - timedelta(days=90)
                title_period = "3 Months"
            elif period == "6_months":
                start_date = today - timedelta(days=180)
                title_period = "6 Months"
            else:  # "all"
                start_date = None
                title_period = "All Time"
            
            # Get sales data
            daily_sales = self.get_sales_data(
                store_id, 
                item_id, 
                start_date=start_date
            )
            
            # Check if we have any sales data
            if daily_sales.empty:
                logger.warning(f"No sales data found for store {store_id}, item {item_id}")
                return None
            
            # Get item description
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ]
            item_desc = item_sales['Item_Description'].iloc[0] if not item_sales.empty else f"Item {item_id}"
            
            # Create figure
            fig = go.Figure()
            
            # Add sales trace
            fig.add_trace(go.Scatter(
                x=daily_sales['Proc_date'],
                y=daily_sales['Total_units'],
                mode='lines+markers',
                name='Sales',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                hovertemplate='%{x|%Y-%m-%d}<br>Units: %{y}<extra>Sales</extra>'
            ))
            
            # Add moving averages
            if len(daily_sales) >= 7:
                # 7-day moving average
                daily_sales['MA7'] = daily_sales['Total_units'].rolling(window=7).mean()
                fig.add_trace(go.Scatter(
                    x=daily_sales['Proc_date'],
                    y=daily_sales['MA7'],
                    mode='lines',
                    name='7-Day MA',
                    line=dict(color='red', width=2, dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>7-Day MA: %{y:.1f}<extra>MA7</extra>'
                ))
            
            if len(daily_sales) >= 30:
                # 30-day moving average
                daily_sales['MA30'] = daily_sales['Total_units'].rolling(window=30).mean()
                fig.add_trace(go.Scatter(
                    x=daily_sales['Proc_date'],
                    y=daily_sales['MA30'],
                    mode='lines',
                    name='30-Day MA',
                    line=dict(color='green', width=2, dash='dot'),
                    hovertemplate='%{x|%Y-%m-%d}<br>30-Day MA: %{y:.1f}<extra>MA30</extra>'
                ))
            
            # Add seasonality indicators if requested
            if with_seasonality:
                seasonality_info = self.seasonality_analyzer.detect_seasonality(store_id, item_id)
                if seasonality_info.get('has_seasonality', False):
                    seasonality_period = seasonality_info.get('best_period')
                    
                    if seasonality_period:
                        # Start from the earliest date in the data
                        min_date = daily_sales['Proc_date'].min()
                        max_date = daily_sales['Proc_date'].max()
                        
                        # Add vertical lines at seasonal intervals
                        current_date = min_date
                        while current_date <= max_date:
                            fig.add_vline(
                                x=current_date, 
                                line_width=1, 
                                line_dash="dot", 
                                line_color="rgba(0, 0, 255, 0.3)"
                            )
                            current_date = current_date + timedelta(days=seasonality_period)
                        
                        # Add an annotation explaining the seasonal pattern
                        fig.add_annotation(
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            text=f"Seasonality detected: {seasonality_period}-day cycle",
                            showarrow=False,
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="rgba(0, 0, 255, 0.3)",
                            borderwidth=1,
                            borderpad=4
                        )
            
            # Update layout
            seasonality_note = " (with Seasonality Indicators)" if with_seasonality else ""
            fig.update_layout(
                title=f"{item_desc} (Store {store_id}) - Sales Analysis for {title_period}{seasonality_note}",
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
            logger.error(f"Error in create_sales_analysis: {str(e)}")
            return None
    
    def create_inventory_analysis(
        self,
        store_id: str,
        item_id: str,
        lookback_days: int = 30,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create visualization of inventory levels and stock-outs
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            lookback_days: Number of days to look back for analysis
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object
        """
        try:
            # Load stock data if not already loaded
            stock_data = None
            try:
                stock_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaStock.csv"))
            except Exception as e:
                logger.warning(f"Stock data not available: {str(e)}")
                
            # Get sales data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            sales_data = self.get_sales_data(
                store_id, 
                item_id, 
                start_date=start_date,
                end_date=end_date
            )
            
            if sales_data.empty:
                logger.warning(f"No sales data found for store {store_id}, item {item_id}")
                return None
            
            # Get item description
            item_desc = ""
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ]
            if not item_sales.empty:
                item_desc = item_sales['Item_Description'].iloc[0]
            else:
                item_desc = f"Item {item_id}"
            
            # Create figure
            fig = make_subplots(
                rows=1, 
                cols=1,
                subplot_titles=[f"Inventory Analysis for {item_desc} (Last {lookback_days} Days)"]
            )
            
            # Add sales data
            fig.add_trace(
                go.Bar(
                    x=sales_data['Proc_date'],
                    y=sales_data['Total_units'],
                    name='Units Sold',
                    marker_color='blue',
                    hovertemplate='%{x|%Y-%m-%d}<br>Units Sold: %{y}<extra>Sales</extra>'
                )
            )
            
            # Add stock data if available
            if stock_data is not None:
                try:
                    item_stock = stock_data[
                        (stock_data['store_id'] == store_id) & 
                        (stock_data['item'] == item_id)
                    ].copy()
                    
                    if 'date' in item_stock.columns and 'stock_level' in item_stock.columns:
                        # Convert date if needed
                        if item_stock['date'].dtype != 'datetime64[ns]':
                            item_stock['date'] = pd.to_datetime(item_stock['date'])
                        
                        # Filter to relevant dates
                        item_stock = item_stock[
                            (item_stock['date'] >= start_date) &
                            (item_stock['date'] <= end_date)
                        ]
                        
                        if not item_stock.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=item_stock['date'],
                                    y=item_stock['stock_level'],
                                    mode='lines',
                                    name='Stock Level',
                                    line=dict(color='red', width=2),
                                    hovertemplate='%{x|%Y-%m-%d}<br>Stock: %{y}<extra>Stock</extra>'
                                )
                            )
                except Exception as e:
                    logger.warning(f"Error processing stock data: {str(e)}")
            
            # Update layout
            fig.update_layout(
                title=f"{item_desc} (Store {store_id}) - Inventory Analysis",
                xaxis_title="Date",
                yaxis_title="Units",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                height=500
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
            logger.error(f"Error in create_inventory_analysis: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        # Initialize comparator
        comparator = ModelComparator(data_dir="data")
        
        # Get list of items
        items = comparator.get_item_list()
        logger.info(f"Found {len(items)} unique store-item combinations")
        
        # Process first 5 items as a sample
        for i, (store_id, item_id, desc) in enumerate(items[:5]):
            logger.info(f"Processing {i+1}/5: Store {store_id}, Item {item_id} ({desc})")
            
            # Create sales analysis with different periods
            comparator.create_sales_analysis(
                store_id,
                item_id,
                period="3_months",
                with_seasonality=True,
                output_file=f"sales_analysis_3m_{store_id}_{item_id}"
            )
            
            # Create inventory analysis
            comparator.create_inventory_analysis(
                store_id,
                item_id,
                lookback_days=30,
                output_file=f"inventory_analysis_{store_id}_{item_id}"
            )
            
            logger.info(f"Analysis completed for {desc}")
        
        logger.info("Sample analyses completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")