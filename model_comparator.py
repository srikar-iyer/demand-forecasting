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
        logging.FileHandler("testing/model_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelComparator:
    """Class for comparing different forecast models"""
    
    def __init__(self, data_dir: str = "../data"):
        """Initialize the comparator with data directory path"""
        self.data_dir = data_dir
        self.sales_data = None
        self.rf_forecasts = None
        self.pytorch_forecasts = None
        self.arima_forecasts = None
        
        # Initialize seasonality analyzer
        self.seasonality_analyzer = ItemSeasonalityAnalyzer(data_dir=data_dir)
        
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
            
            # Convert dates to datetime objects
            if 'Proc_date' in self.sales_data.columns:
                self.sales_data['Proc_date'] = pd.to_datetime(self.sales_data['Proc_date'])
            
            # Load model forecasts if they exist
            try:
                self.rf_forecasts = pd.read_csv(os.path.join(self.data_dir, "processed/rf_forecasts.csv"))
                self.rf_forecasts['Date'] = pd.to_datetime(self.rf_forecasts['Date'])
                logger.info(f"Loaded RF forecasts with {len(self.rf_forecasts)} entries")
            except FileNotFoundError:
                logger.warning("RF forecasts file not found")
                
            try:
                self.pytorch_forecasts = pd.read_csv(os.path.join(self.data_dir, "processed/pytorch_forecasts.csv"))
                self.pytorch_forecasts['Date'] = pd.to_datetime(self.pytorch_forecasts['Date'])
                logger.info(f"Loaded PyTorch forecasts with {len(self.pytorch_forecasts)} entries")
            except FileNotFoundError:
                logger.warning("PyTorch forecasts file not found")
                
            try:
                self.arima_forecasts = pd.read_csv(os.path.join(self.data_dir, "processed/arima_forecasts.csv"))
                self.arima_forecasts['Date'] = pd.to_datetime(self.arima_forecasts['Date'])
                logger.info(f"Loaded ARIMA forecasts with {len(self.arima_forecasts)} entries")
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
    
    def get_model_forecasts(
        self, 
        store_id: str, 
        item_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get forecasts from all models for a specific item within date range
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            start_date: Start date for filtering forecasts
            end_date: End date for filtering forecasts
            
        Returns:
            Dictionary with model name as key and forecast DataFrame as value
        """
        try:
            results = {}
            
            # Filter RF forecasts
            if self.rf_forecasts is not None:
                rf_item = self.rf_forecasts[
                    (self.rf_forecasts['Store_Id'] == store_id) & 
                    (self.rf_forecasts['Item'] == item_id)
                ].copy()
                
                if not rf_item.empty:
                    if start_date:
                        rf_item = rf_item[rf_item['Date'] >= start_date]
                    if end_date:
                        rf_item = rf_item[rf_item['Date'] <= end_date]
                    
                    if not rf_item.empty:
                        results['RF'] = rf_item
            
            # Filter PyTorch forecasts
            if self.pytorch_forecasts is not None:
                pytorch_item = self.pytorch_forecasts[
                    (self.pytorch_forecasts['Store_Id'] == store_id) & 
                    (self.pytorch_forecasts['Item'] == item_id)
                ].copy()
                
                if not pytorch_item.empty:
                    if start_date:
                        pytorch_item = pytorch_item[pytorch_item['Date'] >= start_date]
                    if end_date:
                        pytorch_item = pytorch_item[pytorch_item['Date'] <= end_date]
                    
                    if not pytorch_item.empty:
                        results['PyTorch'] = pytorch_item
            
            # Filter ARIMA forecasts
            if self.arima_forecasts is not None:
                arima_item = self.arima_forecasts[
                    (self.arima_forecasts['Store_Id'] == store_id) & 
                    (self.arima_forecasts['Item'] == item_id)
                ].copy()
                
                if not arima_item.empty:
                    if start_date:
                        arima_item = arima_item[arima_item['Date'] >= start_date]
                    if end_date:
                        arima_item = arima_item[arima_item['Date'] <= end_date]
                    
                    if not arima_item.empty:
                        results['ARIMA'] = arima_item
            
            return results
            
        except Exception as e:
            logger.error(f"Error in get_model_forecasts: {str(e)}")
            return {}

    def create_forecast_comparison(
        self,
        store_id: str,
        item_id: str,
        period: str = "2_weeks",  # "2_weeks" or "2_months"
        with_confidence_intervals: bool = True,
        with_seasonality: bool = True,
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create comparison visualization of model forecasts for a specific item
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            period: Period for forecast visualization ('2_weeks' or '2_months')
            with_confidence_intervals: Whether to show confidence intervals
            with_seasonality: Whether to apply item-specific seasonality adjustments
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object
        """
        try:
            # Determine forecast dates based on the period
            today = datetime.now()
            start_date = today - timedelta(days=30)  # include some history
            
            if period == "2_weeks":
                end_date = today + timedelta(days=14)
                title_period = "2 Weeks"
            else:  # "2_months"
                end_date = today + timedelta(days=60)
                title_period = "2 Months"
            
            # Get forecasts from all models
            model_forecasts = self.get_model_forecasts(
                store_id, 
                item_id, 
                start_date=start_date,
                end_date=end_date
            )
            
            # Check if we have any forecast data
            if not model_forecasts:
                logger.warning(f"No forecast data found for store {store_id}, item {item_id}")
                return None
            
            # Get item description
            item_desc = ""
            for model_name, forecast_df in model_forecasts.items():
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
            
            # Apply seasonality adjustments if requested
            if with_seasonality:
                for model_name, forecast_df in model_forecasts.items():
                    try:
                        adjusted_df = self.seasonality_analyzer.apply_seasonal_adjustment(
                            store_id,
                            item_id,
                            forecast_df,
                            date_column='Date',
                            forecast_column='Forecast'
                        )
                        model_forecasts[model_name] = adjusted_df
                    except Exception as e:
                        logger.error(f"Error applying seasonality adjustment for {model_name}: {str(e)}")
            
            # Create figure
            fig = go.Figure()
            
            # Add historical sales data
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            if not item_sales.empty:
                # Filter to relevant date range
                recent_sales = item_sales[item_sales['Proc_date'] >= start_date]
                
                if not recent_sales.empty:
                    # Aggregate by date
                    daily_sales = recent_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
                    
                    # Add historical sales trace
                    fig.add_trace(go.Scatter(
                        x=daily_sales['Proc_date'],
                        y=daily_sales['Total_units'],
                        mode='lines+markers',
                        name='Historical Sales',
                        line=dict(color='black', width=3),
                        marker=dict(size=6),
                        hovertemplate='%{x|%Y-%m-%d}<br>Units: %{y}<extra>Historical</extra>'
                    ))
            
            # Define colors and styles for each model
            model_styles = {
                'RF': {
                    'color': 'blue',
                    'dash': 'solid',
                    'width': 2.5,
                    'ci_color': 'rgba(0, 0, 255, 0.1)'
                },
                'PyTorch': {
                    'color': 'green',
                    'dash': 'solid',
                    'width': 2.5,
                    'ci_color': 'rgba(0, 255, 0, 0.1)'
                },
                'ARIMA': {
                    'color': 'red',
                    'dash': 'solid',
                    'width': 2.5,
                    'ci_color': 'rgba(255, 0, 0, 0.1)'
                }
            }
            
            # Add model forecast traces
            for model_name, forecast_df in model_forecasts.items():
                style = model_styles.get(model_name, {
                    'color': 'purple',
                    'dash': 'solid',
                    'width': 2.5,
                    'ci_color': 'rgba(128, 0, 128, 0.1)'
                })
                
                # Determine which forecast column to use
                forecast_column = 'Seasonally_Adjusted' if with_seasonality and 'Seasonally_Adjusted' in forecast_df.columns else 'Forecast'
                
                # Add forecast trace
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df[forecast_column],
                    mode='lines',
                    name=f"{model_name} {forecast_column.replace('_', ' ')}",
                    line=dict(
                        color=style['color'],
                        dash=style['dash'],
                        width=style['width']
                    ),
                    hovertemplate='%{x|%Y-%m-%d}<br>Forecast: %{y:.1f}<extra>' + model_name + '</extra>'
                ))
                
                # Add confidence intervals if requested
                if with_confidence_intervals and 'Lower_Bound' in forecast_df.columns and 'Upper_Bound' in forecast_df.columns:
                    lb_column = 'Lower_Bound'
                    ub_column = 'Upper_Bound'
                    
                    # If using seasonally adjusted forecasts, apply the same adjustment to bounds
                    if with_seasonality and 'Seasonally_Adjusted' in forecast_df.columns:
                        # Calculate the adjustment factor for each row
                        adjustment = forecast_df['Seasonally_Adjusted'] - forecast_df['Forecast']
                        forecast_df['Adjusted_Lower_Bound'] = forecast_df['Lower_Bound'] + adjustment
                        forecast_df['Adjusted_Upper_Bound'] = forecast_df['Upper_Bound'] + adjustment
                        lb_column = 'Adjusted_Lower_Bound'
                        ub_column = 'Adjusted_Upper_Bound'
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                        y=forecast_df[ub_column].tolist() + forecast_df[lb_column].tolist()[::-1],
                        fill='toself',
                        fillcolor=style['ci_color'],
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f"{model_name} CI",
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
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
            seasonality_note = " (with Item-Specific Seasonality)" if with_seasonality else ""
            fig.update_layout(
                title=f"{item_desc} (Store {store_id}) - Model Comparison for Next {title_period}{seasonality_note}",
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
            logger.error(f"Error in create_forecast_comparison: {str(e)}")
            return None
    
    def create_forecast_error_analysis(
        self,
        store_id: str,
        item_id: str,
        lookback_days: int = 30,
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Analyze forecast errors for each model based on past predictions
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            lookback_days: Number of days to look back for analysis
            output_file: If provided, save the figure to this file
            
        Returns:
            Dictionary with error metrics for each model
        """
        try:
            # Get historical sales data
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id) & 
                (self.sales_data['item'] == item_id)
            ].copy()
            
            if item_sales.empty:
                logger.warning(f"No historical sales data found for store {store_id}, item {item_id}")
                return {}
            
            # Aggregate sales by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            daily_sales.rename(columns={'Proc_date': 'Date'}, inplace=True)
            
            # Define lookback period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Filter historical sales to lookback period
            historical_sales = daily_sales[
                (daily_sales['Date'] >= start_date) &
                (daily_sales['Date'] <= end_date)
            ]
            
            if historical_sales.empty:
                logger.warning(f"No historical sales in lookback period for store {store_id}, item {item_id}")
                return {}
            
            # Get forecasts from all models for the lookback period
            model_forecasts = self.get_model_forecasts(
                store_id, 
                item_id, 
                start_date=start_date,
                end_date=end_date
            )
            
            # Check if we have any forecast data
            if not model_forecasts:
                logger.warning(f"No forecast data found for lookback period for store {store_id}, item {item_id}")
                return {}
            
            # Calculate error metrics for each model
            error_metrics = {}
            
            for model_name, forecast_df in model_forecasts.items():
                # Merge forecasts with actual sales
                merged_data = pd.merge(
                    historical_sales,
                    forecast_df[['Date', 'Forecast']],
                    on='Date',
                    how='inner'
                )
                
                if merged_data.empty:
                    logger.warning(f"No overlapping data for error analysis for {model_name}")
                    continue
                
                # Calculate error metrics
                merged_data['Error'] = merged_data['Total_units'] - merged_data['Forecast']
                merged_data['AbsError'] = abs(merged_data['Error'])
                merged_data['SquaredError'] = merged_data['Error'] ** 2
                merged_data['PercentError'] = 100 * merged_data['Error'] / merged_data['Total_units'].replace(0, np.nan)
                
                metrics = {
                    'MAE': merged_data['AbsError'].mean(),
                    'MSE': merged_data['SquaredError'].mean(),
                    'RMSE': np.sqrt(merged_data['SquaredError'].mean()),
                    'MAPE': merged_data['PercentError'].abs().mean(),
                    'Bias': merged_data['Error'].mean(),
                    'StdDev': merged_data['Error'].std(),
                    'DataPoints': len(merged_data)
                }
                
                error_metrics[model_name] = metrics
            
            # Create error comparison visualization
            if error_metrics and output_file:
                try:
                    # Prepare data for plotting
                    models = list(error_metrics.keys())
                    mae_values = [error_metrics[model]['MAE'] for model in models]
                    rmse_values = [error_metrics[model]['RMSE'] for model in models]
                    bias_values = [error_metrics[model]['Bias'] for model in models]
                    
                    # Create subplots
                    fig = make_subplots(
                        rows=1, 
                        cols=3, 
                        subplot_titles=('Mean Absolute Error', 'Root Mean Squared Error', 'Bias'),
                        shared_yaxes=True
                    )
                    
                    # Add MAE bars
                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=mae_values,
                            name='MAE',
                            marker_color='blue',
                            text=[f"{val:.2f}" for val in mae_values],
                            textposition='auto'
                        ),
                        row=1, col=1
                    )
                    
                    # Add RMSE bars
                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=rmse_values,
                            name='RMSE',
                            marker_color='green',
                            text=[f"{val:.2f}" for val in rmse_values],
                            textposition='auto'
                        ),
                        row=1, col=2
                    )
                    
                    # Add Bias bars
                    fig.add_trace(
                        go.Bar(
                            x=models,
                            y=bias_values,
                            name='Bias',
                            marker_color='red',
                            text=[f"{val:.2f}" for val in bias_values],
                            textposition='auto'
                        ),
                        row=1, col=3
                    )
                    
                    # Get item description
                    item_desc = item_sales['Item_Description'].iloc[0] if not item_sales.empty else f"Item {item_id}"
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{item_desc} (Store {store_id}) - Forecast Error Analysis (Last {lookback_days} Days)",
                        showlegend=False,
                        height=400
                    )
                    
                    # Save the figure
                    fig.write_html(f"testing/output/{output_file}.html")
                    fig.write_image(f"testing/output/{output_file}.png")
                    logger.info(f"Error analysis figure saved to testing/output/{output_file}.html and .png")
                
                except Exception as e:
                    logger.error(f"Error creating error analysis visualization: {str(e)}")
            
            return error_metrics
            
        except Exception as e:
            logger.error(f"Error in create_forecast_error_analysis: {str(e)}")
            return {}

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
            
            # Create 2-week forecast comparison
            comparator.create_forecast_comparison(
                store_id,
                item_id,
                period="2_weeks",
                with_confidence_intervals=True,
                with_seasonality=True,
                output_file=f"comparison_2w_{store_id}_{item_id}"
            )
            
            # Create 2-month forecast comparison
            comparator.create_forecast_comparison(
                store_id,
                item_id,
                period="2_months",
                with_confidence_intervals=True,
                with_seasonality=True,
                output_file=f"comparison_2m_{store_id}_{item_id}"
            )
            
            # Create error analysis
            error_metrics = comparator.create_forecast_error_analysis(
                store_id,
                item_id,
                lookback_days=30,
                output_file=f"error_analysis_{store_id}_{item_id}"
            )
            
            logger.info(f"Error metrics for {desc}: {error_metrics}")
        
        logger.info("Sample model comparisons completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")