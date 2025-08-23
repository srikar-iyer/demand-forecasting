#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from item_data_visualizer import ItemDataVisualizer
from seasonality_analyzer import ItemSeasonalityAnalyzer
from model_comparator import ModelComparator
from datetime import datetime, timedelta
import logging
from flask import Flask, render_template, jsonify, send_from_directory
import glob
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Initialize data processing classes
visualizer = ItemDataVisualizer(data_dir="data")
seasonality_analyzer = ItemSeasonalityAnalyzer(data_dir="data")
model_comparator = ModelComparator(data_dir="data")

# Load and prepare data
try:
    sales_data = pd.read_csv(os.path.join("data", "FrozenPizzaSales.csv"))
    purchase_data = pd.read_csv(os.path.join("data", "FrozenPizzaPurchases.csv"))
    stock_data = pd.read_csv(os.path.join("data", "FrozenPizzaStock.csv"))
    
    # Convert dates
    sales_data['Proc_date'] = pd.to_datetime(sales_data['Proc_date'])
    purchase_data['Proc_date'] = pd.to_datetime(purchase_data['Proc_date'])
    
    # Get unique items and stores
    items = visualizer.get_item_list()
    stores = sorted(sales_data['store_id'].unique())
    min_date = sales_data['Proc_date'].min().strftime('%Y-%m-%d')
    max_date = sales_data['Proc_date'].max().strftime('%Y-%m-%d')
    
    logger.info(f"Data loaded successfully. Found {len(items)} items across {len(stores)} stores.")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    sales_data = pd.DataFrame()
    purchase_data = pd.DataFrame()
    stock_data = pd.DataFrame()
    items = []
    stores = []
    min_date = datetime.now().strftime('%Y-%m-%d')
    max_date = datetime.now().strftime('%Y-%m-%d')

# Initialize Flask server
server = Flask(__name__)

# Define API routes for the HTML template
@server.route('/')
def index():
    return render_template('index.html')

@server.route('/api/stores')
def api_stores():
    try:
        store_list = []
        for store in stores:
            store_list.append({
                "store_id": str(store),
                "store_name": f"Store {store}"
            })
        return jsonify(store_list)
    except Exception as e:
        logger.error(f"Error in api_stores: {str(e)}")
        return jsonify([])

@server.route('/api/items')
def api_items():
    try:
        store_id = request.args.get('store_id', 'all')
        
        if store_id == 'all':
            item_list = []
            for store_id, item_id, desc in items:
                item_list.append({
                    "item_id": str(item_id),
                    "item_name": desc
                })
        else:
            item_list = []
            for s_id, item_id, desc in items:
                if s_id == store_id:
                    item_list.append({
                        "item_id": str(item_id),
                        "item_name": desc
                    })
        
        return jsonify(item_list)
    except Exception as e:
        logger.error(f"Error in api_items: {str(e)}")
        return jsonify([])

@server.route('/api/date-range')
def api_date_range():
    return jsonify({
        "min_date": min_date,
        "max_date": max_date
    })

@server.route('/api/categories')
def api_categories():
    return jsonify([
        "sales_purchases", 
        "seasonality", 
        "forecast"
    ])

@server.route('/api/images')
def api_images():
    try:
        store_id = request.args.get('store_id', 'all')
        item_id = request.args.get('item_id', 'all')
        category = request.args.get('category', 'all')
        
        # Get all image files in output directory
        image_files = glob.glob('output/*.png')
        filtered_images = []
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # Filter by category
            if category != 'all':
                if not filename.startswith(category):
                    continue
            
            # Filter by store_id and item_id
            if store_id != 'all' and item_id != 'all':
                if f"{store_id}_{item_id}" not in filename:
                    continue
            elif store_id != 'all':
                if f"{store_id}_" not in filename:
                    continue
            elif item_id != 'all':
                if f"_{item_id}" not in filename:
                    continue
            
            filtered_images.append({
                "url": f"/output/{filename}",
                "filename": filename
            })
        
        return jsonify(filtered_images)
    except Exception as e:
        logger.error(f"Error in api_images: {str(e)}")
        return jsonify([])

# Serve static files
@server.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory('output', filename)

# Initialize Dash app
dash_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dash/',
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Define the Dash layout
dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Demand Forecasting Dashboard", className="mb-4"),
            html.P("Interactive visualization of sales data and forecast predictions with seasonality detection.")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filter Options"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Store"),
                            dcc.Dropdown(
                                id="store-dropdown",
                                options=[{"label": f"Store {s}", "value": str(s)} for s in stores],
                                value=str(stores[0]) if stores else None
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Item"),
                            dcc.Dropdown(id="item-dropdown")
                        ], width=8)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=datetime.strptime(min_date, '%Y-%m-%d').date() if min_date else None,
                                max_date_allowed=datetime.strptime(max_date, '%Y-%m-%d').date() if max_date else None,
                                start_date=datetime.strptime(min_date, '%Y-%m-%d').date() if min_date else None,
                                end_date=datetime.strptime(max_date, '%Y-%m-%d').date() if max_date else None
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Display Type"),
                            dcc.RadioItems(
                                id="display-type",
                                options=[
                                    {"label": "Sales & Purchases", "value": "sales"},
                                    {"label": "Seasonality", "value": "seasonality"}
                                ],
                                value="sales",
                                inline=True
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Button("Update Visualization", id="update-button", color="primary")
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Spinner(
                dcc.Graph(id="main-graph", style={"height": "600px"}),
                color="primary",
                type="border"
            )
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Seasonality Information"),
                dbc.CardBody([
                    html.Div(id="seasonality-info")
                ])
            ])
        ], width=12)
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "View all visualizations in the ", 
                html.A("Image Gallery", href="/", target="_blank")
            ], className="text-center")
        ])
    ], className="mt-4")
], fluid=True)

# Callback to populate item dropdown based on store selection
@dash_app.callback(
    Output("item-dropdown", "options"),
    Output("item-dropdown", "value"),
    Input("store-dropdown", "value")
)
def update_item_dropdown(selected_store):
    if not selected_store:
        raise PreventUpdate
    
    store_items = [
        {"label": desc, "value": str(item_id)} 
        for s, item_id, desc in items 
        if s == selected_store
    ]
    
    # Select the first item by default
    default_value = store_items[0]["value"] if store_items else None
    
    return store_items, default_value

# Main callback to update the visualization
@dash_app.callback(
    Output("main-graph", "figure"),
    Output("seasonality-info", "children"),
    Input("update-button", "n_clicks"),
    State("store-dropdown", "value"),
    State("item-dropdown", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("display-type", "value")
)
def update_visualization(n_clicks, store_id, item_id, start_date, end_date, display_type):
    if not n_clicks or not store_id or not item_id:
        # Default empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Select parameters and click 'Update Visualization'",
            template="plotly_white"
        )
        return fig, "No data to display. Please select parameters above."
    
    try:
        # Check for seasonality
        seasonality_result = seasonality_analyzer.detect_seasonality(store_id, item_id)
        has_seasonality = seasonality_result.get('has_seasonality', False)
        seasonality_period = seasonality_result.get('best_period', None)
        
        # Format seasonality info
        if has_seasonality:
            seasonality_info = html.Div([
                html.H5(f"Seasonality Detected: {seasonality_period}-day cycle"),
                html.P(f"Strength: {seasonality_result.get('strength', 0):.2f}"),
                html.Ul([
                    html.Li(f"Pattern: {seasonality_period}-day cycle")
                ])
            ])
        else:
            seasonality_info = html.P("No significant seasonality detected in this item's data.")
        
        if display_type == "sales":
            # Create sales and purchases visualization
            fig = visualizer.visualize_item_sales_purchases(
                store_id,
                item_id,
                start_date=start_date,
                end_date=end_date
            )
            
            # Add vertical lines for seasonality if detected
            if has_seasonality and seasonality_period:
                # Start from the earliest date in the plot
                dates = []
                for trace_data in fig.data:
                    if hasattr(trace_data, 'x'):
                        dates.extend(trace_data.x)
                
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    
                    # Create vertical lines at seasonal intervals
                    current_date = min_date
                    while current_date <= max_date:
                        fig.add_vline(
                            x=current_date, 
                            line_width=1, 
                            line_dash="dot", 
                            line_color="rgba(0, 0, 255, 0.3)"
                        )
                        current_date = current_date + timedelta(days=seasonality_period)
            
        elif display_type == "seasonality" and has_seasonality:
            # Create seasonality visualization
            fig = seasonality_analyzer.visualize_seasonality(store_id, item_id)
        else:
            # Create a default visualization
            fig = go.Figure()
            fig.update_layout(
                title=f"No seasonality data available for this item",
                template="plotly_white"
            )
        
        # Update the layout to ensure it's responsive
        fig.update_layout(
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            template="plotly_white"
        )
        
        return fig, seasonality_info
        
    except Exception as e:
        logger.error(f"Error updating visualization: {str(e)}")
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: {str(e)}",
            template="plotly_white"
        )
        
        error_info = html.P(f"An error occurred: {str(e)}", className="text-danger")
        
        return fig, error_info

if __name__ == "__main__":
    # Run the Dash app
    server.run(debug=True, host='0.0.0.0', port=5000)