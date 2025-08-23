#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from demand_forecaster import DemandForecaster
from forecast_visualizer import ForecastVisualizer
from datetime import datetime, timedelta
import logging
from flask import Flask

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forecast_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Initialize forecasting components
try:
    forecaster = DemandForecaster(data_dir="data")
    visualizer = ForecastVisualizer(data_dir="data")
    items = forecaster.get_item_list()
    
    # Extract unique stores
    stores = sorted(list(set([item[0] for item in items])))
    
    logger.info(f"Forecast app initialized. Found {len(items)} items across {len(stores)} stores.")
except Exception as e:
    logger.error(f"Error initializing forecast components: {str(e)}")
    forecaster = None
    visualizer = None
    items = []
    stores = []

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Define the app layout with tabs
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Demand Forecasting Dashboard", className="text-center mb-4"),
            html.P("Generate 1-week and 2-week demand forecasts using local trajectory weighting", 
                  className="text-center text-muted mb-4")
        ])
    ]),
    
    # Filter controls
    dbc.Card([
        dbc.CardHeader("Forecast Parameters"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Store:", className="fw-bold"),
                    dcc.Dropdown(
                        id="store-dropdown",
                        options=[{"label": f"Store {s}", "value": s} for s in stores],
                        value=stores[0] if stores else None,
                        clearable=False
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Item:", className="fw-bold"),
                    dcc.Dropdown(
                        id="item-dropdown",
                        clearable=False
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-chart-line me-2"), "Generate Forecast"],
                        id="generate-forecast-btn",
                        color="primary",
                        size="lg",
                        className="mt-4"
                    )
                ], width=2)
            ])
        ])
    ], className="mb-4"),
    
    # Tabs for different views
    dbc.Tabs([
        dbc.Tab(
            label="Time Series Forecasts", 
            tab_id="tab-timeseries",
            active_tab_style={"textTransform": "none"}
        ),
        dbc.Tab(
            label="Numerical Forecasts", 
            tab_id="tab-numerical",
            active_tab_style={"textTransform": "none"}
        )
    ], id="forecast-tabs", active_tab="tab-timeseries", className="mb-4"),
    
    # Content area
    html.Div(id="tab-content"),
    
    # Loading overlay
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="loading-output")
    )
], fluid=True)

# Callback to populate item dropdown based on store selection
@app.callback(
    Output("item-dropdown", "options"),
    Output("item-dropdown", "value"),
    Input("store-dropdown", "value")
)
def update_item_dropdown(selected_store):
    if not selected_store or not items:
        return [], None
    
    store_items = [
        {"label": f"{desc} ({item_id})", "value": item_id} 
        for store_id, item_id, desc in items 
        if store_id == selected_store
    ]
    
    default_value = store_items[0]["value"] if store_items else None
    return store_items, default_value

# Main callback to update content based on active tab
@app.callback(
    Output("tab-content", "children"),
    [Input("forecast-tabs", "active_tab"),
     Input("generate-forecast-btn", "n_clicks")],
    [State("store-dropdown", "value"),
     State("item-dropdown", "value")]
)
def update_tab_content(active_tab, n_clicks, store_id, item_id):
    if not n_clicks or not store_id or not item_id or not forecaster:
        return dbc.Alert(
            [
                html.I(className="fas fa-info-circle me-2"),
                "Select a store and item, then click 'Generate Forecast' to begin."
            ],
            color="info",
            className="text-center"
        )
    
    try:
        # Generate forecast data
        forecast_result = forecaster.generate_forecast(store_id, item_id, [1, 2])
        
        if "error" in forecast_result:
            return dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Forecast Error: {forecast_result['error']}"
                ],
                color="warning"
            )
        
        # Get item description
        item_desc = next((desc for s, i, desc in items if s == store_id and i == item_id), f"Item {item_id}")
        
        if active_tab == "tab-timeseries":
            return create_timeseries_tab_content(store_id, item_id, item_desc, forecast_result)
        elif active_tab == "tab-numerical":
            return create_numerical_tab_content(store_id, item_id, item_desc, forecast_result)
            
    except Exception as e:
        logger.error(f"Error updating tab content: {str(e)}")
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-circle me-2"),
                f"An error occurred: {str(e)}"
            ],
            color="danger"
        )

def create_timeseries_tab_content(store_id, item_id, item_desc, forecast_result):
    """Create the time series tab content"""
    try:
        # Generate the time series plot
        fig = visualizer.create_forecast_time_series(store_id, item_id, [1, 2])
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4(f"{item_desc}", className="mb-3"),
                    html.P(f"Store {store_id} • Item {item_id}", className="text-muted mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("1-Week & 2-Week Demand Forecasts"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=fig,
                                style={"height": "700px"}
                            )
                        ])
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    create_forecast_summary_cards(forecast_result)
                ], className="mt-4")
            ])
        ]
        
    except Exception as e:
        logger.error(f"Error creating time series content: {str(e)}")
        return dbc.Alert(f"Error creating visualization: {str(e)}", color="danger")

def create_numerical_tab_content(store_id, item_id, item_desc, forecast_result):
    """Create the numerical forecasts tab content"""
    try:
        # Create forecast comparison table
        comparison_df = visualizer.create_forecast_comparison_table(store_id, item_id, [1, 2])
        
        if comparison_df.empty:
            return dbc.Alert("No forecast data available", color="warning")
        
        # Separate historical and forecast data
        forecast_rows = comparison_df[comparison_df['Period'].str.contains('Week')].copy()
        historical_rows = comparison_df[~comparison_df['Period'].str.contains('Week')].copy()
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4(f"{item_desc}", className="mb-3"),
                    html.P(f"Store {store_id} • Item {item_id}", className="text-muted mb-4")
                ])
            ]),
            
            # Forecast Results
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Forecast Results"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_forecast_table(forecast_rows)
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Historical Context
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-history me-2"),
                                "Historical Averages (for context)"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            create_historical_table(historical_rows)
                        ])
                    ])
                ])
            ]),
            
            # Detailed explanations
            dbc.Row([
                dbc.Col(
                    create_detailed_explanations(forecast_result),
                    className="mt-4"
                )
            ])
        ]
        
    except Exception as e:
        logger.error(f"Error creating numerical content: {str(e)}")
        return dbc.Alert(f"Error creating table: {str(e)}", color="danger")

def create_forecast_table(forecast_df):
    """Create formatted forecast results table"""
    if forecast_df.empty:
        return html.P("No forecast data available")
    
    # Convert all values to native types before passing to DataTable
    forecast_df_processed = forecast_df.copy()
    if 'Predicted Forecast' in forecast_df_processed.columns:
        forecast_df_processed['Predicted Forecast'] = pd.to_numeric(forecast_df_processed['Predicted Forecast'], errors='coerce')
    if 'Practical Forecast' in forecast_df_processed.columns:
        forecast_df_processed['Practical Forecast'] = pd.to_numeric(forecast_df_processed['Practical Forecast'], errors='coerce')
    if 'Confidence' in forecast_df_processed.columns:
        # Remove percentage symbol if present and convert to numeric
        # Handle different types of confidence values properly
        forecast_df_processed['Confidence'] = forecast_df_processed['Confidence'].apply(
            lambda x: float(str(x).rstrip('%'))/100 if isinstance(x, str) and x != 'Actual' else 
            (x if isinstance(x, (int, float)) or x == 'Actual' else str(x))
        )
        
    return dash_table.DataTable(
        data=forecast_df_processed.to_dict('records'),
        columns=[
            {"name": "Forecast Period", "id": "Period"},
            {"name": "Predicted Units", "id": "Predicted Forecast", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Practical Order", "id": "Practical Forecast", "type": "numeric"},
            {"name": "Confidence", "id": "Confidence", "format": {"specifier": ".0%"}},
            {"name": "Explanation", "id": "Explanation"}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'font-family': 'Arial'
        },
        style_header={
            'backgroundColor': '#e9ecef',
            'fontWeight': 'bold',
            'border': '1px solid #dee2e6'
        },
        style_data={
            'border': '1px solid #dee2e6',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 0},  # First row (1-week)
                'backgroundColor': '#f8f9fa'
            }
        ]
    )

def create_historical_table(historical_df):
    """Create formatted historical averages table"""
    if historical_df.empty:
        return html.P("No historical data available")
    
    # Convert all values to native types before passing to DataTable
    historical_df_processed = historical_df.copy()
    if 'Predicted Forecast' in historical_df_processed.columns:
        historical_df_processed['Predicted Forecast'] = pd.to_numeric(historical_df_processed['Predicted Forecast'], errors='coerce')
    
    # Ensure all non-numeric columns are treated as strings
    if 'Confidence' in historical_df_processed.columns:
        historical_df_processed['Confidence'] = historical_df_processed['Confidence'].apply(
            lambda x: str(x) if x is not None else ''
        )
    
    if 'Practical Forecast' in historical_df_processed.columns:
        historical_df_processed['Practical Forecast'] = historical_df_processed['Practical Forecast'].apply(
            lambda x: str(x) if x is not None else ''
        )
    
    return dash_table.DataTable(
        data=historical_df_processed.to_dict('records'),
        columns=[
            {"name": "Time Period", "id": "Period"},
            {"name": "Average Units", "id": "Predicted Forecast", "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Type", "id": "Practical Forecast"},
            {"name": "Status", "id": "Confidence"},
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'font-family': 'Arial'
        },
        style_header={
            'backgroundColor': '#e9ecef',
            'fontWeight': 'bold',
            'border': '1px solid #dee2e6'
        },
        style_data={
            'border': '1px solid #dee2e6',
            'backgroundColor': '#fafafa'
        }
    )

def create_forecast_summary_cards(forecast_result):
    """Create summary cards for forecast metrics"""
    try:
        cards = []
        
        # Data quality card
        data_points = forecast_result.get('data_points', 0)
        purchase_patterns = forecast_result.get('purchase_patterns', {})
        
        data_card = dbc.Card([
            dbc.CardBody([
                html.H6("Data Quality", className="card-title"),
                html.P([
                    html.Strong("Data Points: "), f"{data_points} days",
                    html.Br(),
                    html.Strong("Order Multiple: "), f"{purchase_patterns.get('multiple', 1)} units",
                    html.Br(),
                    html.Strong("Min Order: "), f"{purchase_patterns.get('min_order', 1)} units"
                ], className="card-text")
            ])
        ], color="info", outline=True)
        
        cards.append(dbc.Col([data_card], width=4))
        
        # Forecast cards for each period
        forecasts = forecast_result.get('forecasts', {})
        colors = ["success", "warning"]
        
        for idx, (period, forecast_info) in enumerate(forecasts.items()):
            color = colors[idx % len(colors)]
            period_name = period.replace('_', ' ').title()
            
            predicted = forecast_info.get('total_predicted', 0)
            practical = forecast_info.get('total_practical', 0)
            confidence = forecast_info.get('confidence', 0)
            
            forecast_card = dbc.Card([
                dbc.CardBody([
                    html.H6(f"{period_name} Forecast", className="card-title"),
                    html.P([
                        html.Strong("Predicted: "), f"{predicted:.1f} units",
                        html.Br(),
                        html.Strong("Practical: "), f"{practical} units",
                        html.Br(),
                        html.Strong("Confidence: "), f"{confidence:.0%}"
                    ], className="card-text")
                ])
            ], color=color, outline=True)
            
            cards.append(dbc.Col([forecast_card], width=4))
        
        return dbc.Row(cards)
        
    except Exception as e:
        logger.error(f"Error creating summary cards: {str(e)}")
        return html.P("Summary unavailable")

def create_detailed_explanations(forecast_result):
    """Create detailed explanations section"""
    try:
        explanations = []
        
        # Methodology explanation
        methodology_card = dbc.Card([
            dbc.CardHeader("Forecasting Methodology"),
            dbc.CardBody([
                html.Ul([
                    html.Li("Uses exponential smoothing with local trajectory weighting"),
                    html.Li("Recent 1-2 weeks heavily weighted vs older data"),
                    html.Li("Maximum 30% deviation from recent averages enforced"),
                    html.Li("Pattern preservation: sparse data → sparse forecasts"),
                    html.Li("Trend dampening applied towards historical mean"),
                    html.Li("Purchase multiples considered for practical ordering")
                ])
            ])
        ], className="mb-3")
        
        explanations.append(methodology_card)
        
        # Forecast explanations
        forecasts = forecast_result.get('forecasts', {})
        
        for period, forecast_info in forecasts.items():
            period_name = period.replace('_', ' ').title()
            period_explanations = forecast_info.get('explanations', [])
            
            if period_explanations:
                explanation_items = []
                for exp in period_explanations:
                    if exp:
                        explanation_items.append(html.Li(exp))
                        
                explanation_card = dbc.Card([
                    dbc.CardHeader(f"{period_name} Forecast Explanation"),
                    dbc.CardBody([
                        html.Ul(explanation_items)
                    ])
                ], className="mb-3")
                
                explanations.append(explanation_card)
        
        return explanations
        
    except Exception as e:
        logger.error(f"Error creating explanations: {str(e)}")
        return [dbc.Alert("Explanations unavailable", color="info")]

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)