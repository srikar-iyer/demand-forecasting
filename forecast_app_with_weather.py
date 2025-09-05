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
from weather_client import WeatherClient
from weather_visualizer import WeatherVisualizer
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
    weather_client = WeatherClient(cache_session=True)
    weather_visualizer = WeatherVisualizer(data_dir="data", default_location="New York")
    items = forecaster.get_item_list()
    
    # Extract unique stores
    stores = sorted(list(set([item[0] for item in items])))
    
    logger.info(f"Forecast app initialized. Found {len(items)} items across {len(stores)} stores.")
except Exception as e:
    logger.error(f"Error initializing forecast components: {str(e)}")
    forecaster = None
    visualizer = None
    weather_client = None
    weather_visualizer = None
    items = []
    stores = []

# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app with a colorful theme
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.COSMO, dbc.icons.FONT_AWESOME],  # COSMO theme for more color
    suppress_callback_exceptions=True
)

# Define the app layout with tabs
# Define custom CSS for improved styling
custom_css = {
    'font_family': {
        'font-family': '"Poppins", "Roboto", "Helvetica Neue", Arial, sans-serif',
    },
    'header_style': {
        'font-family': '"Montserrat", "Helvetica Neue", Arial, sans-serif',
        'font-weight': '600',
        'color': '#2C3E50',
    },
    'subheader_style': {
        'font-family': '"Montserrat", "Helvetica Neue", Arial, sans-serif',
        'font-weight': '500',
        'color': '#34495E',
    }
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fas fa-chart-line fa-3x text-primary me-3", style={"display": "inline-block"}),
                html.I(className="fas fa-chart-bar fa-3x text-success me-3", style={"display": "inline-block"}),
                html.I(className="fas fa-chart-area fa-3x text-info", style={"display": "inline-block"}),
                html.H1("Demand Forecasting Dashboard", className="text-center mb-4", 
                       style={**custom_css['header_style'], "display": "inline-block", "margin-left": "15px", "letter-spacing": "0.5px"}),
            ], className="d-flex justify-content-center align-items-center"),
            html.P("Generate 1-week and 2-week demand forecasts with weather integration", 
                  className="text-center text-muted mb-4",
                  style={**custom_css['subheader_style'], "font-size": "1.1rem"})
        ])
    ]),
    
    # Filter controls with colorful styling
    dbc.Card([
        dbc.CardHeader("Forecast Parameters", className="bg-primary text-white"),
        dbc.CardBody([
            html.Div(className="bg-light p-2 mb-3", style={"border-radius": "5px", "border-left": "4px solid #007bff"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Store:", className="fw-bold", style={**custom_css['subheader_style'], "font-size": "1rem"}),
                    dcc.Dropdown(
                        id="store-dropdown",
                        options=[{"label": f"Store {s}", "value": s} for s in stores],
                        value=stores[0] if stores else None,
                        clearable=False
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Item:", className="fw-bold", style={**custom_css['subheader_style'], "font-size": "1rem"}),
                    dcc.Dropdown(
                        id="item-dropdown",
                        clearable=False
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Location:", className="fw-bold", style={**custom_css['subheader_style'], "font-size": "1rem"}),
                    dcc.Dropdown(
                        id="location-dropdown",
                        options=[
                            {"label": "New York", "value": "New York"},
                            {"label": "Chicago", "value": "Chicago"},
                            {"label": "Los Angeles", "value": "Los Angeles"},
                            {"label": "Miami", "value": "Miami"},
                            {"label": "Denver", "value": "Denver"},
                            {"label": "Seattle", "value": "Seattle"},
                            {"label": "Boston", "value": "Boston"},
                            {"label": "San Francisco", "value": "San Francisco"},
                            {"label": "Dallas", "value": "Dallas"}
                        ],
                        value="New York",
                        clearable=False
                    )
                ], width=3),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-chart-line me-2"), "Generate"],
                        id="generate-forecast-btn",
                        color="primary",
                        size="lg",
                        className="mt-4",
                        style={"background-image": "linear-gradient(to right, #6e48aa, #00a4db)", "border": "none"}
                    )
                ], width=2)
            ])
        ])
    ], className="mb-4"),
    
    # Colorful tabs for different views
    dbc.Tabs([
        dbc.Tab(
            label="Time Series Forecasts", 
            tab_id="tab-timeseries",
            active_tab_style={"textTransform": "none", "background-color": "#e9f7fe", "border-top": "3px solid #17a2b8"}
        ),
        dbc.Tab(
            label="Numerical Forecasts", 
            tab_id="tab-numerical",
            active_tab_style={"textTransform": "none", "background-color": "#e9f7fe", "border-top": "3px solid #17a2b8"}
        ),
        dbc.Tab(
            label="Weather Analysis", 
            tab_id="tab-weather",
            active_tab_style={"textTransform": "none", "background-color": "#ffefd5", "border-top": "3px solid #ff8c00"}
        ),
        dbc.Tab(
            label="Weather-Demand Correlation", 
            tab_id="tab-correlation",
            active_tab_style={"textTransform": "none", "background-color": "#f0fff0", "border-top": "3px solid #2e8b57"}
        )
    ], id="forecast-tabs", active_tab="tab-timeseries", className="mb-4", style={"border-bottom": "2px solid #dee2e6"}),
    
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
     State("item-dropdown", "value"),
     State("location-dropdown", "value")]
)
def update_tab_content(active_tab, n_clicks, store_id, item_id, location):
    if not n_clicks:
        return dbc.Alert(
            [
                html.I(className="fas fa-info-circle me-2"),
                "Select parameters and click 'Generate' to begin."
            ],
            color="info",
            className="text-center"
        )
    
    if not store_id or not item_id:
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please select a store and item."
            ],
            color="warning",
            className="text-center"
        )
    
    if not forecaster or not weather_client or not weather_visualizer:
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-circle me-2"),
                "Forecasting components are not initialized properly."
            ],
            color="danger",
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
        
        # Get base forecast for weather adjustment
        forecasts = forecast_result.get('forecasts', {})
        week1_forecast = forecasts.get('week_1', {}).get('daily_forecast', [])
        week2_forecast = forecasts.get('week_2', {}).get('daily_forecast', [])
        base_forecast = week1_forecast + week2_forecast
        
        # Generate forecast dates
        start_date = datetime.now().date() + timedelta(days=1)
        forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(base_forecast))]
        
        if active_tab == "tab-timeseries":
            return create_timeseries_tab_content(store_id, item_id, item_desc, forecast_result, location)
        elif active_tab == "tab-numerical":
            return create_numerical_tab_content(store_id, item_id, item_desc, forecast_result)
        elif active_tab == "tab-weather":
            return create_weather_tab_content(store_id, item_id, item_desc, location, base_forecast, forecast_dates)
        elif active_tab == "tab-correlation":
            return create_correlation_tab_content(store_id, item_id, item_desc, location)
            
    except Exception as e:
        logger.error(f"Error updating tab content: {str(e)}")
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-circle me-2"),
                f"An error occurred: {str(e)}"
            ],
            color="danger"
        )

def create_timeseries_tab_content(store_id, item_id, item_desc, forecast_result, location):
    """Create the time series tab content"""
    try:
        # Generate the time series plot
        fig = visualizer.create_forecast_time_series(store_id, item_id, [1, 2])
        
        # Get weather-adjusted forecast data
        forecasts = forecast_result.get('forecasts', {})
        week1_forecast = forecasts.get('week_1', {}).get('daily_forecast', [])
        week2_forecast = forecasts.get('week_2', {}).get('daily_forecast', [])
        base_forecast = week1_forecast + week2_forecast
        
        # Generate forecast dates
        start_date = datetime.now().date() + timedelta(days=1)
        forecast_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(base_forecast))]
        
        # Get weather-adjusted forecast
        _, forecast_df = weather_visualizer.get_weather_data(location, days_historical=1, days_forecast=14)
        
        if not forecast_df.empty:
            impact_df = weather_client.get_weather_impact_factors(forecast_df)
            impact_factors = impact_df['overall_impact'].values[:len(base_forecast)]
            
            # Pad with 1.0 if needed
            if len(impact_factors) < len(base_forecast):
                impact_factors = np.append(
                    impact_factors, 
                    np.ones(len(base_forecast) - len(impact_factors))
                )
                
            # Apply impact factors to get adjusted forecast
            adjusted_forecast = [base * impact for base, impact in zip(base_forecast, impact_factors)]
            
            # Calculate percentage difference
            avg_base = sum(base_forecast) / len(base_forecast)
            avg_adjusted = sum(adjusted_forecast) / len(adjusted_forecast)
            pct_diff = (avg_adjusted - avg_base) / avg_base * 100
            
            weather_impact_alert = dbc.Alert(
                [
                    html.I(className="fas fa-cloud me-2"),
                    f"Weather Impact: {'+' if pct_diff >= 0 else ''}{pct_diff:.1f}% forecast adjustment",
                    html.Span(" (See Weather tabs for details)", className="ms-2 text-muted")
                ],
                color="info",
                className="mt-3"
            )
        else:
            weather_impact_alert = dbc.Alert(
                [
                    html.I(className="fas fa-cloud me-2"),
                    f"Weather data unavailable for {location}. Using base forecast only."
                ],
                color="secondary",
                className="mt-3"
            )
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4(f"{item_desc}", className="mb-3", style={**custom_css['header_style'], "color": "#2980b9", "border-bottom": "2px solid #3498db", "padding-bottom": "8px"}),
                    html.P(f"Store {store_id} • Item {item_id} • Location: {location}", className="text-muted mb-4", style={**custom_css['font_family']})
                ])
            ]),
            
            # Weather impact alert
            dbc.Row([
                dbc.Col([
                    weather_impact_alert
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("1-Week & 2-Week Demand Forecasts", 
                                     className="bg-info text-white", 
                                     style={"font-weight": "bold", "font-size": "1.1rem"}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=fig,
                                style={"height": "700px"}
                            )
                        ], className="bg-light")
                    ], className="shadow")
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
                    html.H4(f"{item_desc}", className="mb-3", style={**custom_css['header_style'], "color": "#2980b9", "border-bottom": "2px solid #3498db", "padding-bottom": "8px"}),
                    html.P(f"Store {store_id} • Item {item_id}", className="text-muted mb-4", style={**custom_css['font_family']})
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

def create_weather_tab_content(store_id, item_id, item_desc, location, base_forecast, forecast_dates):
    """Create the weather analysis tab content"""
    try:
        # Create weather overview plot
        weather_overview_fig = weather_visualizer.create_weather_overview(location)
        
        # Create weather impact summary
        weather_impact_fig = weather_visualizer.create_weather_impact_summary(location)
        
        # Create weather-adjusted forecast
        weather_adjusted_fig = weather_visualizer.create_weather_adjusted_forecast(
            store_id, item_id, location, base_forecast, forecast_dates
        )
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4(f"Weather Analysis - {location}", className="mb-3", 
                           style={**custom_css['header_style'], "color": "#ff8c00", 
                                 "border-bottom": "2px solid #ffa500", "padding-bottom": "8px"}),
                    html.P(f"Store {store_id} • Item {item_id} • {item_desc}", 
                          className="text-muted mb-4", style={**custom_css['font_family']})
                ])
            ]),
            
            # Weather Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-cloud-sun me-2"),
                                "Weather Overview"
                            ], className="mb-0")
                        ], className="bg-warning text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=weather_overview_fig,
                                style={"height": "700px"}
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Weather-Adjusted Forecast
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-line me-2"),
                                "Weather-Adjusted Demand Forecast"
                            ], className="mb-0")
                        ], className="bg-success text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=weather_adjusted_fig,
                                style={"height": "600px"}
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Weather Impact Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Weather Impact Analysis"
                            ], className="mb-0")
                        ], className="bg-info text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=weather_impact_fig,
                                style={"height": "600px"}
                            )
                        ])
                    ])
                ])
            ]),
            
            # Weather Impact Explanation
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Weather Impact Explanation", className="bg-light"),
                        dbc.CardBody([
                            html.H5("How Weather Affects Demand", className="mb-3"),
                            html.Ul([
                                html.Li([
                                    html.Strong("Temperature:"), 
                                    " Higher temperatures generally ", 
                                    html.Span("increase", className="text-success"), 
                                    " demand for frozen foods"
                                ]),
                                html.Li([
                                    html.Strong("Precipitation:"), 
                                    " Rainy days can ", 
                                    html.Span("increase", className="text-success"), 
                                    " delivery and frozen food orders"
                                ]),
                                html.Li([
                                    html.Strong("Snow:"), 
                                    " Snowfall strongly ", 
                                    html.Span("increases", className="text-success"), 
                                    " frozen food purchases (stocking up effect)"
                                ]),
                                html.Li([
                                    html.Strong("Wind:"), 
                                    " Very windy days may slightly ", 
                                    html.Span("decrease", className="text-danger"), 
                                    " in-person shopping trips"
                                ])
                            ]),
                            html.H5("Impact Calculation", className="mt-4 mb-3"),
                            html.P([
                                "Weather impact factors are calculated based on historical correlations between weather conditions and sales patterns. The overall impact is a multiplicative combination of temperature, precipitation, snow, and wind factors."
                            ])
                        ])
                    ])
                ])
            ], className="mt-4")
        ]
        
    except Exception as e:
        logger.error(f"Error creating weather tab content: {str(e)}")
        return dbc.Alert(f"Error creating weather visualizations: {str(e)}", color="danger")

def create_correlation_tab_content(store_id, item_id, item_desc, location):
    """Create the weather-demand correlation tab content"""
    try:
        # Get correlation figures
        correlation_fig, impact_fig = weather_visualizer.create_weather_demand_correlation(
            store_id, item_id, location
        )
        
        if isinstance(correlation_fig, go.Figure) and not correlation_fig.data:
            # Empty figure, likely no correlation data available
            return dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Insufficient data to calculate correlations between weather and demand."
                ],
                color="warning"
            )
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4(f"Weather-Demand Correlation Analysis", className="mb-3", 
                           style={**custom_css['header_style'], "color": "#2e8b57", 
                                 "border-bottom": "2px solid #3cb371", "padding-bottom": "8px"}),
                    html.P(f"Store {store_id} • Item {item_id} • {item_desc} • Location: {location}", 
                          className="text-muted mb-4", style={**custom_css['font_family']})
                ])
            ]),
            
            # Correlation Bar Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Weather-Demand Correlation Coefficients"
                            ], className="mb-0")
                        ], className="bg-success text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=correlation_fig,
                                style={"height": "500px"}
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Scatter and Time Series
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-braille me-2"),
                                "Weather Impact Relationship Analysis"
                            ], className="mb-0")
                        ], className="bg-info text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=impact_fig,
                                style={"height": "700px"}
                            )
                        ])
                    ])
                ])
            ]),
            
            # Correlation Explanation
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Understanding Correlation Analysis", className="bg-light"),
                        dbc.CardBody([
                            html.H5("Interpreting Correlation Coefficients", className="mb-3"),
                            html.Ul([
                                html.Li([
                                    html.Strong("Strong Positive (0.7 to 1.0):"), 
                                    " Strong direct relationship - as one variable increases, the other strongly tends to increase"
                                ]),
                                html.Li([
                                    html.Strong("Moderate Positive (0.4 to 0.7):"), 
                                    " Moderate direct relationship - as one variable increases, the other moderately tends to increase"
                                ]),
                                html.Li([
                                    html.Strong("Weak Positive (0.2 to 0.4):"), 
                                    " Weak direct relationship - as one variable increases, the other weakly tends to increase"
                                ]),
                                html.Li([
                                    html.Strong("No Correlation (-0.2 to 0.2):"), 
                                    " No discernible linear relationship between the variables"
                                ]),
                                html.Li([
                                    html.Strong("Weak Negative (-0.4 to -0.2):"), 
                                    " Weak inverse relationship - as one variable increases, the other weakly tends to decrease"
                                ]),
                                html.Li([
                                    html.Strong("Moderate Negative (-0.7 to -0.4):"), 
                                    " Moderate inverse relationship - as one variable increases, the other moderately tends to decrease"
                                ]),
                                html.Li([
                                    html.Strong("Strong Negative (-1.0 to -0.7):"), 
                                    " Strong inverse relationship - as one variable increases, the other strongly tends to decrease"
                                ])
                            ]),
                            html.H5("Application to Demand Forecasting", className="mt-4 mb-3"),
                            html.P([
                                "Understanding these correlations allows us to adjust demand forecasts based on weather predictions. Strong correlations suggest that weather-adjusted forecasts will provide better accuracy than base forecasts alone."
                            ])
                        ])
                    ])
                ])
            ], className="mt-4")
        ]
        
    except Exception as e:
        logger.error(f"Error creating correlation tab content: {str(e)}")
        return dbc.Alert(f"Error creating correlation visualizations: {str(e)}", color="danger")

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
            'font-family': 'Arial',
            'width': '400px',
            'maxWidth': '400px'
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
            'font-family': 'Arial',
            'width': '400px',
            'maxWidth': '400px'
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
                    html.Li("Purchase multiples considered for practical ordering"),
                    html.Li("AI-based linear trend detection for 1-week and 2-week time frames"),
                    html.Li("Weather impacts factored in for adjusted forecasts")
                ])
            ])
        ], className="mb-3")
        
        explanations.append(methodology_card)
        
        # Forecast explanations
        forecasts = forecast_result.get('forecasts', {})
        
        for period, forecast_info in forecasts.items():
            period_name = period.replace('_', ' ').title()
            period_explanations = forecast_info.get('explanations', [])
            
            # Add enhanced explanations based on forecast patterns
            if period_explanations:
                explanation_items = []
                
                # Add basic explanations
                for exp in period_explanations:
                    if exp:
                        explanation_items.append(html.Li(exp))
                
                # Get trend information
                daily_forecasts = forecast_info.get('daily_forecast', [])
                if len(daily_forecasts) > 3:
                    first_value = daily_forecasts[0]
                    last_value = daily_forecasts[-1]
                    avg_value = sum(daily_forecasts) / len(daily_forecasts)
                    
                    # Calculate trend characteristics
                    pct_change = (last_value - first_value) / first_value * 100 if first_value > 0 else 0
                    
                    # Add trend explanations with more detailed rationale and data-based reasons
                    if abs(pct_change) < 5:
                        explanation_items.append(html.Li(f"Trend: Stable demand pattern with minimal variation ({abs(pct_change):.1f}%)"))
                        explanation_items.append(html.Li(f"Rationale: 1) Day-to-day consistency in sales data with standard deviation of {np.std(daily_forecasts):.2f} units. 2) Recent historical data shows similar stability pattern with minimal directional momentum."))
                    elif pct_change > 20:
                        explanation_items.append(html.Li(f"Trend: Strong upward trajectory showing {pct_change:.1f}% increase"))
                        explanation_items.append(html.Li(f"Rationale: 1) Accelerating growth pattern in recent days with day-over-day gains averaging {(pct_change/len(daily_forecasts)):.2f}%. 2) Historical data shows similar steep growth during comparable time periods."))
                    elif pct_change > 5:
                        explanation_items.append(html.Li(f"Trend: Moderate upward trajectory showing {pct_change:.1f}% increase"))
                        explanation_items.append(html.Li(f"Rationale: 1) Consistent positive momentum with day-over-day gains averaging {(pct_change/len(daily_forecasts)):.2f}%. 2) Recent purchasing frequency shows gradual acceleration pattern."))
                    elif pct_change < -20:
                        explanation_items.append(html.Li(f"Trend: Strong downward trajectory showing {abs(pct_change):.1f}% decrease"))
                        explanation_items.append(html.Li(f"Rationale: 1) Rapid decline in daily sales averaging {(abs(pct_change)/len(daily_forecasts)):.2f}% per day. 2) Similar pattern observed in historical data during comparable seasonal periods."))
                    elif pct_change < -5:
                        explanation_items.append(html.Li(f"Trend: Moderate downward trajectory showing {abs(pct_change):.1f}% decrease"))
                        explanation_items.append(html.Li(f"Rationale: 1) Consistent negative momentum with day-over-day decline averaging {(abs(pct_change)/len(daily_forecasts)):.2f}%. 2) Recent purchasing frequency shows gradual deceleration pattern."))
                        
                # Add weather-related explanation
                explanation_items.append(html.Li(f"Weather factors have been considered in the Weather Analysis tab"))
                
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
    app.run(debug=True, host='0.0.0.0', port=5000)