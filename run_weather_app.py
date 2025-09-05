#!/usr/bin/env python3
import os
import sys

# Modify port number to avoid conflict with existing server
from forecast_app_with_weather import app

if __name__ == "__main__":
    port = 5002
    app.run(debug=True, host='0.0.0.0', port=port)
    print(f"Running on port {port}")