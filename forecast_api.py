#!/usr/bin/env python3
import os
import json
import numpy as np
from flask import Flask, request, jsonify
from demand_forecaster import DemandForecaster
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forecast_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize forecaster
forecaster = DemandForecaster(data_dir="data")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Forecasting API is running"}), 200

@app.route('/api/items', methods=['GET'])
def get_items():
    """Get list of available store-item combinations with predictions"""
    try:
        items = forecaster.get_item_list()
        item_list = []
        
        for store_id, item_id, desc in items:
            if store_id != "nan" and item_id != "nan":  # Exclude NaN store and items
                # Generate forecast for this item
                forecast_result = forecaster.generate_forecast(store_id, item_id, forecast_weeks=[1, 2])
                
                if "error" not in forecast_result:
                    # Get the rounded 1-week prediction
                    one_week_rounded = forecast_result["forecasts"].get("1_week", {}).get("total_practical", 0)
                    
                    # Get the rounded 2-week prediction (as sum of rounded values)
                    two_week_rounded = one_week_rounded + forecast_result["forecasts"].get("2_week", {}).get("total_practical", 0)
                    
                    # Convert NumPy int64 values to Python int for JSON serialization
                    if isinstance(one_week_rounded, (np.int64, np.int32)):
                        one_week_rounded = int(one_week_rounded)
                    if isinstance(two_week_rounded, (np.int64, np.int32)):
                        two_week_rounded = int(two_week_rounded)
                    
                    item_list.append({
                        "store_id": store_id,
                        "item_id": item_id,
                        "description": desc,
                        "1_week_prediction": one_week_rounded,
                        "2_week_prediction": two_week_rounded
                    })
                else:
                    # Include the item without predictions if there was an error
                    item_list.append({
                        "store_id": store_id,
                        "item_id": item_id,
                        "description": desc
                    })
            
        return jsonify({"items": item_list}), 200
    except Exception as e:
        logger.error(f"Error in get_items: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast/<store_id>/<item_id>', methods=['GET'])
def get_forecast(store_id, item_id):
    """Get forecast for a specific store-item combination"""
    try:
        # Skip if store_id or item_id is "nan"
        if store_id.lower() == "nan" or item_id.lower() == "nan":
            return jsonify({"error": "Invalid store or item ID"}), 400

        # Generate forecasts for 1-week and 2-week periods
        forecast_result = forecaster.generate_forecast(store_id, item_id, forecast_weeks=[1, 2])
        
        if "error" in forecast_result:
            return jsonify({"error": forecast_result["error"]}), 404
        
        # Get forecast values
        one_week = forecast_result["forecasts"].get("1_week", {}).get("total_practical", 0)
        two_week = forecast_result["forecasts"].get("2_week", {}).get("total_practical", 0)
        one_week_conf = forecast_result["forecasts"].get("1_week", {}).get("confidence", 0)
        two_week_conf = forecast_result["forecasts"].get("2_week", {}).get("confidence", 0)
        
        # Convert NumPy values to Python native types for JSON serialization
        if isinstance(one_week, (np.int64, np.int32)):
            one_week = int(one_week)
        if isinstance(two_week, (np.int64, np.int32)):
            two_week = int(two_week)
        if isinstance(one_week_conf, (np.float64, np.float32)):
            one_week_conf = float(one_week_conf)
        if isinstance(two_week_conf, (np.float64, np.float32)):
            two_week_conf = float(two_week_conf)
        
        # Extract just the practical forecasted units for each period
        forecasted_units = {
            "store_id": store_id,
            "item_id": item_id,
            "1_week": one_week,
            "2_week": two_week,
            "confidence": {
                "1_week": one_week_conf,
                "2_week": two_week_conf
            }
        }
        
        return jsonify(forecasted_units), 200
    
    except Exception as e:
        logger.error(f"Error in get_forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/bulk-forecast', methods=['POST'])
def bulk_forecast():
    """Get forecasts for multiple store-item combinations"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or not isinstance(request_data, list):
            return jsonify({"error": "Invalid request format. Expecting a list of store-item pairs"}), 400
        
        results = []
        
        for item in request_data:
            store_id = item.get("store_id")
            item_id = item.get("item_id")
            
            # Skip invalid entries
            if not store_id or not item_id or store_id.lower() == "nan" or item_id.lower() == "nan":
                continue
                
            # Generate forecast
            forecast_result = forecaster.generate_forecast(store_id, item_id, forecast_weeks=[1, 2])
            
            if "error" not in forecast_result:
                # Get forecast values
                one_week = forecast_result["forecasts"].get("1_week", {}).get("total_practical", 0)
                two_week = forecast_result["forecasts"].get("2_week", {}).get("total_practical", 0)
                one_week_conf = forecast_result["forecasts"].get("1_week", {}).get("confidence", 0)
                two_week_conf = forecast_result["forecasts"].get("2_week", {}).get("confidence", 0)
                
                # Convert NumPy values to Python native types for JSON serialization
                if isinstance(one_week, (np.int64, np.int32)):
                    one_week = int(one_week)
                if isinstance(two_week, (np.int64, np.int32)):
                    two_week = int(two_week)
                if isinstance(one_week_conf, (np.float64, np.float32)):
                    one_week_conf = float(one_week_conf)
                if isinstance(two_week_conf, (np.float64, np.float32)):
                    two_week_conf = float(two_week_conf)
                
                forecasted_units = {
                    "store_id": store_id,
                    "item_id": item_id,
                    "1_week": one_week,
                    "2_week": two_week,
                    "confidence": {
                        "1_week": one_week_conf,
                        "2_week": two_week_conf
                    }
                }
                results.append(forecasted_units)
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Error in bulk_forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/export-forecast-data', methods=['GET'])
def export_forecast_data():
    """Export forecast data for all valid items"""
    try:
        items = forecaster.get_item_list()
        all_forecasts = []
        
        for store_id, item_id, desc in items:
            # Skip NaN store and items
            if store_id.lower() == "nan" or item_id.lower() == "nan":
                continue
                
            forecast_result = forecaster.generate_forecast(store_id, item_id, forecast_weeks=[1, 2])
            
            if "error" not in forecast_result:
                # Get forecast values
                one_week = forecast_result["forecasts"].get("1_week", {}).get("total_practical", 0)
                two_week = forecast_result["forecasts"].get("2_week", {}).get("total_practical", 0)
                
                # Convert NumPy values to Python native types for JSON serialization
                if isinstance(one_week, (np.int64, np.int32)):
                    one_week = int(one_week)
                if isinstance(two_week, (np.int64, np.int32)):
                    two_week = int(two_week)
                
                forecast_entry = {
                    "store_id": store_id,
                    "item_id": item_id,
                    "description": desc,
                    "1_week": one_week,
                    "2_week": two_week
                }
                all_forecasts.append(forecast_entry)
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        # Write to JSON file
        with open("output/forecasted_units.json", "w") as f:
            json.dump(all_forecasts, f, indent=2)
            
        return jsonify({
            "message": "Forecast data exported successfully", 
            "file_path": "output/forecasted_units.json",
            "item_count": len(all_forecasts)
        }), 200
        
    except Exception as e:
        logger.error(f"Error exporting forecast data: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting Forecast API server...")
        # Ensure we're using port 5002 to match Docker config
        app.run(debug=False, host="0.0.0.0", port=5002)
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")