#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from demand_forecaster import DemandForecaster
from seasonality_analyzer import ItemSeasonalityAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forecast_visualizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForecastVisualizer:
    """Visualize demand forecasts with time series plots"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the visualizer with data directory path"""
        self.data_dir = data_dir
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        try:
            self.forecaster = DemandForecaster(data_dir=data_dir)
            self.seasonality_analyzer = ItemSeasonalityAnalyzer(data_dir=data_dir)
            logger.info("Forecast visualizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing forecast visualizer: {str(e)}")
            raise
    
    def create_forecast_time_series(
        self, 
        store_id: str, 
        item_id: str, 
        forecast_weeks: List[int] = [1, 2],
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create time series plot showing historical sales + forecasted demand
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            forecast_weeks: List of weeks to forecast
            output_file: Optional filename for saving
            
        Returns:
            Plotly figure with time series and forecasts
        """
        try:
            # Generate forecast
            forecast_result = self.forecaster.generate_forecast(store_id, item_id, forecast_weeks)
            
            if "error" in forecast_result:
                logger.error(f"Forecast error: {forecast_result['error']}")
                return self._create_error_plot(f"Forecast Error: {forecast_result['error']}")
            
            # Get historical sales data
            sales_data = self.forecaster._prepare_daily_sales(store_id, item_id)
            
            if sales_data.empty:
                return self._create_error_plot("No historical sales data available")
            
            # Get item description
            items = self.forecaster.get_item_list()
            item_desc = next((desc for s, i, desc in items if s == store_id and i == item_id), f"Item {item_id}")
            
            # Create figure with subplots for each forecast period
            fig = make_subplots(
                rows=len(forecast_weeks), 
                cols=1,
                shared_xaxes=True,
                subplot_titles=[f"{weeks}-Week Demand Forecast" for weeks in forecast_weeks],
                vertical_spacing=0.2
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors for each forecast
            
            for idx, weeks in enumerate(forecast_weeks):
                row_idx = idx + 1
                color = colors[idx % len(colors)]
                
                # Historical data
                fig.add_trace(
                    go.Scatter(
                        x=sales_data['Date'],
                        y=sales_data['Sales'],
                        mode='lines+markers',
                        name=f'Historical Sales',
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        showlegend=(idx == 0)  # Only show legend for first subplot
                    ),
                    row=row_idx, col=1
                )
                
                # Forecast data
                if f"{weeks}_week" in forecast_result["forecasts"]:
                    forecast_info = forecast_result["forecasts"][f"{weeks}_week"]
                    daily_forecasts = forecast_info["daily_forecast"]
                    
                    # Create forecast dates
                    last_date = sales_data['Date'].max()
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(daily_forecasts))]
                    
                    # Add forecast line
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_dates,
                            y=daily_forecasts,
                            mode='lines+markers',
                            name=f'{weeks}-Week Forecast',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=4, symbol='diamond'),
                            fill='tonexty' if idx == 0 else None,
                            fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.1])}',
                            showlegend=(idx == 0)
                        ),
                        row=row_idx, col=1
                    )
                    
                    # Add connection line between historical and forecast
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, forecast_dates[0]],
                            y=[sales_data['Sales'].iloc[-1], daily_forecasts[0]],
                            mode='lines',
                            line=dict(color=color, width=1, dash='dot'),
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
                    
                    # Add weekly totals as annotations
                    weekly_totals = forecast_info["weekly_totals"]
                    for week_idx, total in enumerate(weekly_totals):
                        week_start = forecast_dates[week_idx * 7]
                        week_end = forecast_dates[min((week_idx + 1) * 7 - 1, len(forecast_dates) - 1)]
                        
                        # Add weekly totals as legend items instead of annotations
                        # Skip adding annotations here as we'll add them to the legend
                    
                    # Add confidence interval if available
                    confidence = forecast_info.get("confidence", 0.5)
                    if confidence > 0:
                        # Create confidence bands
                        upper_bound = [f * (1 + (1 - confidence) * 0.5) for f in daily_forecasts]
                        lower_bound = [f * (1 - (1 - confidence) * 0.5) for f in daily_forecasts]
                        
                        # Upper confidence bound
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=upper_bound,
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=row_idx, col=1
                        )
                        
                        # Lower confidence bound
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=lower_bound,
                                mode='lines',
                                fill='tonexty',
                                fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.1])}',
                                line=dict(width=0),
                                name=f'Confidence ({confidence:.0%})' if idx == 0 else None,
                                showlegend=(idx == 0),
                                hoverinfo='skip'
                            ),
                            row=row_idx, col=1
                        )
                
                # Update axes labels
                fig.update_yaxes(title_text="Units Sold", row=row_idx, col=1)
                if row_idx == len(forecast_weeks):
                    fig.update_xaxes(title_text="Date", row=row_idx, col=1)
            
            # Add seasonality indicators if detected
            seasonality_result = self.seasonality_analyzer.detect_seasonality(store_id, item_id)
            if seasonality_result.get('has_seasonality', False):
                period = seasonality_result['best_period']
                
                # Add vertical lines for seasonal patterns
                last_date = sales_data['Date'].max()
                for days_ahead in range(0, max(forecast_weeks) * 7, period):
                    seasonal_date = last_date + timedelta(days=days_ahead)
                    for row_idx in range(1, len(forecast_weeks) + 1):
                        fig.add_vline(
                            x=seasonal_date,
                            line_width=1,
                            line_dash="dot",
                            line_color="rgba(128, 128, 128, 0.3)",
                            row=row_idx, col=1
                        )
            
            # Add weekly totals as legend items
            for idx, weeks in enumerate(forecast_weeks):
                if f"{weeks}_week" in forecast_result["forecasts"]:
                    forecast_info = forecast_result["forecasts"][f"{weeks}_week"]
                    weekly_totals = forecast_info["weekly_totals"]
                    
                    for week_idx, total in enumerate(weekly_totals):
                        # Add invisible trace just to show in legend
                        # For week 2, show the combined total instead of individual weeks
                        legend_text = f"Week {week_idx + 1}: {total:.1f} units"
                        if weeks == 2:
                            total_sum = sum(weekly_totals)
                            legend_text = f"Week 2 (Combined): {total_sum:.1f} units"
                            # Only show one entry for week 2 combined
                            if week_idx > 0:
                                continue
                                
                        fig.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode='markers',
                                marker=dict(color='#7030A0' if weeks == 1 else '#00B050'),
                                name=legend_text,
                                showlegend=True
                            ),
                            row=idx+1, col=1
                        )
            
            # Update layout
            fig.update_layout(
                title=f"{item_desc} - Demand Forecast (Store {store_id})",
                title_y=0.98,
                height=300 * len(forecast_weeks) + 250,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1
                ),
                margin=dict(t=80, b=50, l=50, r=50),
                template="plotly_white"
            )
            
            # Add forecast summary as annotation
            summary_text = self._create_forecast_summary(forecast_result)
            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
            
            # Save figure if output file provided
            if output_file:
                try:
                    fig.write_html(f"output/{output_file}.html")
                    fig.write_image(f"output/{output_file}.png")
                    logger.info(f"Forecast visualization saved to output/{output_file}")
                except Exception as e:
                    logger.error(f"Error saving forecast visualization: {str(e)}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating forecast time series: {str(e)}")
            return self._create_error_plot(f"Visualization Error: {str(e)}")
    
    def _create_forecast_summary(self, forecast_result: Dict) -> str:
        """Create a text summary of the forecast results"""
        try:
            summary_lines = []
            
            # Data info
            summary_lines.append(f"Data Points: {forecast_result.get('data_points', 'N/A')}")
            
            # Averages
            averages = forecast_result.get('averages', {})
            if 'last_week' in averages:
                summary_lines.append(f"Last Week Avg: {averages['last_week']:.1f} (per day)")
            if 'last_2_weeks' in averages:
                summary_lines.append(f"Last 2 Weeks Avg: {averages['last_2_weeks']:.1f} (per day)")
            
            # Forecasts
            for period, forecast_info in forecast_result.get('forecasts', {}).items():
                predicted = forecast_info.get('total_predicted', 0)
                practical = forecast_info.get('total_practical', 0)
                confidence = forecast_info.get('confidence', 0)
                
                summary_lines.append(f"{period.replace('_', ' ').title()}: {predicted:.1f} predicted, {practical} practical (total for period)")
                summary_lines.append(f"Confidence: {confidence:.0%}")
            
            return "<br>".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error creating forecast summary: {str(e)}")
            return "Summary unavailable"
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot when data is unavailable"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=2
        )
        
        fig.update_layout(
            title="Forecast Visualization Error",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_forecast_comparison_table(
        self, 
        store_id: str, 
        item_id: str,
        forecast_weeks: List[int] = [1, 2]
    ) -> pd.DataFrame:
        """
        Create a comparison table of predicted vs practical forecasts
        
        Returns:
            DataFrame with forecast comparison data
        """
        try:
            # Generate forecast
            forecast_result = self.forecaster.generate_forecast(store_id, item_id, forecast_weeks)
            
            if "error" in forecast_result:
                logger.error(f"Forecast error: {forecast_result['error']}")
                return pd.DataFrame()
            
            # Create comparison table
            table_data = []
            
            for period, forecast_info in forecast_result.get('forecasts', {}).items():
                period_name = period.replace('_', ' ').title()
                predicted = forecast_info.get('total_predicted', 0)
                practical = forecast_info.get('total_practical', 0)
                confidence = forecast_info.get('confidence', 0)
                explanation = forecast_info.get('explanations', [''])[0]
                
                table_data.append({
                    'Period': period_name + ' (total)',
                    'Predicted Forecast': float(predicted) if isinstance(predicted, (int, float)) else predicted,
                    'Practical Forecast': int(practical) if isinstance(practical, (int, float)) else practical,
                    'Confidence': float(confidence) if isinstance(confidence, (int, float)) else confidence,
                    'Explanation': str(explanation) if explanation is not None else ''
                })
            
            # Add averages for context
            averages = forecast_result.get('averages', {})
            context_data = []
            for avg_name, avg_value in averages.items():
                context_data.append({
                    'Period': avg_name.replace('_', ' ').title() + ' (per day)',
                    'Predicted Forecast': float(avg_value) if isinstance(avg_value, (int, float)) else avg_value,
                    'Practical Forecast': 'Historical',
                    'Confidence': 'Actual',
                    'Explanation': 'Historical average for reference'
                })
            
            # Combine tables
            comparison_df = pd.DataFrame(table_data)
            context_df = pd.DataFrame(context_data)
            
            if not context_df.empty:
                comparison_df = pd.concat([context_df, comparison_df], ignore_index=True)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error creating forecast comparison table: {str(e)}")
            return pd.DataFrame()
    
    def get_item_list(self) -> List[tuple]:
        """Return a list of unique items"""
        return self.forecaster.get_item_list()

if __name__ == "__main__":
    try:
        # Initialize visualizer
        visualizer = ForecastVisualizer(data_dir="data")
        
        # Get list of items
        items = visualizer.get_item_list()
        logger.info(f"Found {len(items)} unique store-item combinations")
        
        # Test visualization with first item
        if items:
            store_id, item_id, desc = items[0]
            logger.info(f"Creating forecast visualization for: Store {store_id}, Item {item_id} ({desc})")
            
            # Create time series forecast
            fig = visualizer.create_forecast_time_series(
                store_id, 
                item_id, 
                forecast_weeks=[1, 2],
                output_file=f"forecast_{store_id}_{item_id}"
            )
            
            # Create comparison table
            comparison_table = visualizer.create_forecast_comparison_table(store_id, item_id)
            if not comparison_table.empty:
                logger.info(f"Forecast comparison table:\n{comparison_table}")
            
        logger.info("Forecast visualization testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")