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
        logging.FileHandler("testing/visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ItemDataVisualizer:
    """Class for visualizing sales and purchase data for individual items"""
    
    def __init__(self, data_dir: str = "../data"):
        """Initialize the visualizer with data directory path"""
        self.data_dir = data_dir
        self.sales_data = None
        self.purchase_data = None
        self.stock_data = None
        self.rf_forecasts = None
        self.pytorch_forecasts = None
        self.arima_forecasts = None
        
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
            self.stock_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaStock.csv"))
            
            # Convert dates to datetime objects
            if 'Proc_date' in self.sales_data.columns:
                self.sales_data['Proc_date'] = pd.to_datetime(self.sales_data['Proc_date'])
            if 'Proc_date' in self.purchase_data.columns:
                self.purchase_data['Proc_date'] = pd.to_datetime(self.purchase_data['Proc_date'])
            
            # Load model forecasts if they exist
            try:
                self.rf_forecasts = pd.read_csv(os.path.join(self.data_dir, "processed/rf_forecasts.csv"))
                self.rf_forecasts['Date'] = pd.to_datetime(self.rf_forecasts['Date'])
            except FileNotFoundError:
                logger.warning("RF forecasts file not found")
                
            try:
                self.pytorch_forecasts = pd.read_csv(os.path.join(self.data_dir, "processed/pytorch_forecasts.csv"))
                self.pytorch_forecasts['Date'] = pd.to_datetime(self.pytorch_forecasts['Date'])
            except FileNotFoundError:
                logger.warning("PyTorch forecasts file not found")
                
            try:
                self.arima_forecasts = pd.read_csv(os.path.join(self.data_dir, "processed/arima_forecasts.csv"))
                self.arima_forecasts['Date'] = pd.to_datetime(self.arima_forecasts['Date'])
            except FileNotFoundError:
                logger.warning("ARIMA forecasts file not found")
            
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
                    fig.write_html(f"testing/output/{output_file}.html")
                    fig.write_image(f"testing/output/{output_file}.png")
                    logger.info(f"Figure saved to testing/output/{output_file}.html and .png")
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

    def visualize_model_predictions(
        self,
        store_id: str,
        item_id: str,
        forecast_period: str = "2_weeks",  # "2_weeks" or "2_months"
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create visualization of model predictions for a specific item
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            forecast_period: Period for forecast visualization ('2_weeks' or '2_months')
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object
        """
        try:
            # Filter forecasts for the specific store and item
            rf_item = None if self.rf_forecasts is None else self.rf_forecasts[
                (self.rf_forecasts['Store_Id'] == store_id) & 
                (self.rf_forecasts['Item'] == item_id)
            ].copy()
            
            pytorch_item = None if self.pytorch_forecasts is None else self.pytorch_forecasts[
                (self.pytorch_forecasts['Store_Id'] == store_id) & 
                (self.pytorch_forecasts['Item'] == item_id)
            ].copy()
            
            arima_item = None if self.arima_forecasts is None else self.arima_forecasts[
                (self.arima_forecasts['Store_Id'] == store_id) & 
                (self.arima_forecasts['Item'] == item_id)
            ].copy()
            
            # Check if we have any forecast data
            if (rf_item is None or rf_item.empty) and \
               (pytorch_item is None or pytorch_item.empty) and \
               (arima_item is None or arima_item.empty):
                logger.warning(f"No forecast data found for store {store_id}, item {item_id}")
                return None
            
            # Determine forecast dates based on the period
            today = datetime.now()
            if forecast_period == "2_weeks":
                end_date = today + timedelta(days=14)
                title_period = "2 Weeks"
            else:  # "2_months"
                end_date = today + timedelta(days=60)
                title_period = "2 Months"
            
            # Filter forecasts for the period
            if rf_item is not None:
                rf_item = rf_item[rf_item['Date'] <= end_date]
            if pytorch_item is not None:
                pytorch_item = pytorch_item[pytorch_item['Date'] <= end_date]
            if arima_item is not None:
                arima_item = arima_item[arima_item['Date'] <= end_date]
            
            # Get item description
            item_desc = ""
            for forecast_df in [rf_item, pytorch_item, arima_item]:
                if forecast_df is not None and not forecast_df.empty:
                    if 'Product' in forecast_df.columns:
                        item_desc = forecast_df['Product'].iloc[0]
                        break
            
            if not item_desc:
                # Try to get from sales data
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
                specs=[[{"secondary_y": False}]],
                subplot_titles=[f"{item_desc} Demand Forecast - Next {title_period}"]
            )
            
            # Add historical sales data
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            if not item_sales.empty:
                # Get the last 30 days of historical data
                cutoff_date = today - timedelta(days=30)
                recent_sales = item_sales[item_sales['Proc_date'] >= cutoff_date]
                
                if not recent_sales.empty:
                    # Aggregate by date
                    daily_sales = recent_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
                    
                    # Add historical sales trace
                    fig.add_trace(
                        go.Scatter(
                            x=daily_sales['Proc_date'],
                            y=daily_sales['Total_units'],
                            mode='lines',
                            name='Historical Sales',
                            line=dict(color='gray', width=2, dash='dot'),
                            hovertemplate='%{x|%Y-%m-%d}<br>Units: %{y}<extra>Historical</extra>'
                        )
                    )
            
            # Add RF forecast
            if rf_item is not None and not rf_item.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rf_item['Date'],
                        y=rf_item['Forecast'],
                        mode='lines',
                        name='Random Forest',
                        line=dict(color='blue', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra>RF</extra>'
                    )
                )
                
                # Add RF confidence interval
                if 'Lower_Bound' in rf_item.columns and 'Upper_Bound' in rf_item.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=rf_item['Date'].tolist() + rf_item['Date'].tolist()[::-1],
                            y=rf_item['Upper_Bound'].tolist() + rf_item['Lower_Bound'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0, 0, 255, 0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='RF Confidence',
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
            
            # Add PyTorch forecast
            if pytorch_item is not None and not pytorch_item.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pytorch_item['Date'],
                        y=pytorch_item['Forecast'],
                        mode='lines',
                        name='PyTorch',
                        line=dict(color='green', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra>PyTorch</extra>'
                    )
                )
                
                # Add PyTorch confidence interval
                if 'Lower_Bound' in pytorch_item.columns and 'Upper_Bound' in pytorch_item.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pytorch_item['Date'].tolist() + pytorch_item['Date'].tolist()[::-1],
                            y=pytorch_item['Upper_Bound'].tolist() + pytorch_item['Lower_Bound'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0, 255, 0, 0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='PyTorch Confidence',
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
            
            # Add ARIMA forecast
            if arima_item is not None and not arima_item.empty:
                fig.add_trace(
                    go.Scatter(
                        x=arima_item['Date'],
                        y=arima_item['Forecast'],
                        mode='lines',
                        name='ARIMA',
                        line=dict(color='red', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra>ARIMA</extra>'
                    )
                )
                
                # Add ARIMA confidence interval
                if 'Lower_Bound' in arima_item.columns and 'Upper_Bound' in arima_item.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=arima_item['Date'].tolist() + arima_item['Date'].tolist()[::-1],
                            y=arima_item['Upper_Bound'].tolist() + arima_item['Lower_Bound'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255, 0, 0, 0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='ARIMA Confidence',
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
            
            # Add vertical line for today
            fig.add_vline(
                x=today, 
                line_width=2, 
                line_dash="dash", 
                line_color="black",
                annotation_text="Today",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title=f"{item_desc} (Store {store_id}) - Model Predictions for Next {title_period}",
                xaxis_title="Date",
                yaxis_title="Predicted Units",
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
                    fig.write_html(f"testing/output/{output_file}.html")
                    fig.write_image(f"testing/output/{output_file}.png")
                    logger.info(f"Figure saved to testing/output/{output_file}.html and .png")
                except Exception as e:
                    logger.error(f"Error saving figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in visualize_model_predictions: {str(e)}")
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