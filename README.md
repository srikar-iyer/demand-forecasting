# Advanced Sales & Forecast Visualization Toolkit

This toolkit provides comprehensive visualization tools for sales data and forecast predictions with item-specific seasonality detection and comparison of multiple forecasting models.

## Image Visualization App

A Flask web application that allows users to visualize all forecasting images using customizable filters.

### Features

- View all visualization images in a responsive grid layout
- Filter images by:
  - Store
  - Item number/name
  - Date range (start date, end date)
  - Image category (ARIMA, profit, inventory, etc.)
- Interactive image viewer with zoom capability
- Export functionality for selected images

### Running the Image Visualization App

1. Ensure you have Python 3.8+ installed
2. Install required dependencies:

```bash
pip install flask pandas
```

3. Navigate to the testing directory:

```bash
cd testing
```

4. Run the application:

```bash
python app.py
```

5. Open your browser and navigate to:

```
http://localhost:5000
```

## Quick Start

```bash
# Basic usage with default parameters (processes first 5 items)
./main.py --data-dir ../data

# Process specific number of items
./main.py --data-dir ../data --limit 10

python testing/main.py --data-dir data/raw --limit 60

# Filter by store ID
./main.py --data-dir ../data --store 104.0

# Filter by item ID
./main.py --data-dir ../data --item 3913116850.0

# Custom output report name
./main.py --data-dir ../data --output custom_report.html

# Full example with all parameters
./main.py --data-dir ../data --limit 20 --store 104.0 --item 3913116850.0 --output detailed_report.html
```

## Script Usage Examples

### 1. Item Data Visualizer

The `item_data_visualizer.py` script provides visualizations of sales and purchase data for specific items.

```python
from item_data_visualizer import ItemDataVisualizer

# Initialize the visualizer
visualizer = ItemDataVisualizer(data_dir="../data")

# Get a list of all items
items = visualizer.get_item_list()
print(f"Found {len(items)} items")

# Visualize sales and purchases for a specific item
store_id = "104"
item_id = "3913116850"
visualizer.visualize_item_sales_purchases(
    store_id=store_id,
    item_id=item_id,
    start_date="2023-03-30",    # Optional: filter by start date
    end_date="2023-04-30",      # Optional: filter by end date
    output_file=f"sales_{store_id}_{item_id}"
)

# Visualize model predictions
visualizer.visualize_model_predictions(
    store_id=store_id,
    item_id=item_id,
    forecast_period="2_weeks",  # Options: "2_weeks" or "2_months"
    output_file=f"forecast_{store_id}_{item_id}"
)
```

### 2. Seasonality Analyzer

The `seasonality_analyzer.py` script detects and visualizes item-specific seasonality patterns.

```python
from seasonality_analyzer import ItemSeasonalityAnalyzer

# Initialize the analyzer
analyzer = ItemSeasonalityAnalyzer(data_dir="../data")

# Detect seasonality for a specific item
store_id = "104"
item_id = "3913116850"
seasonality = analyzer.detect_seasonality(
    store_id=store_id,
    item_id=item_id,
    test_periods=[7, 14, 30],           # Optional: periods to test (days)
    min_strength_threshold=0.3          # Optional: minimum correlation threshold
)
print(f"Seasonality result: {seasonality}")

# Visualize seasonality if detected
if seasonality.get("has_seasonality", False):
    analyzer.visualize_seasonality(
        store_id=store_id,
        item_id=item_id,
        output_file=f"seasonality_{store_id}_{item_id}"
    )

# Apply seasonal adjustments to forecasts
import pandas as pd
forecasts_df = pd.read_csv("../data/processed/rf_forecasts.csv")
item_forecasts = forecasts_df[
    (forecasts_df['Store_Id'] == 104.0) & 
    (forecasts_df['Item'] == 3913116850.0)
]
adjusted_forecasts = analyzer.apply_seasonal_adjustment(
    store_id=store_id,
    item_id=item_id,
    forecasts_df=item_forecasts,
    date_column="Date",                 # Optional: name of date column
    forecast_column="Forecast"          # Optional: name of forecast column
)
```

### 3. Model Comparator

The `model_comparator.py` script compares forecasts from different models (ARIMA, PyTorch, Random Forest).

```python
from model_comparator import ModelComparator

# Initialize the comparator
comparator = ModelComparator(data_dir="../data")

# Get forecasts for a specific item
store_id = "104"
item_id = "3913116850"
forecasts = comparator.get_model_forecasts(
    store_id=store_id,
    item_id=item_id,
    start_date=None,                    # Optional: filter by start date
    end_date=None                       # Optional: filter by end date
)

# Create forecast comparison visualization
comparator.create_forecast_comparison(
    store_id=store_id,
    item_id=item_id,
    period="2_weeks",                   # Options: "2_weeks" or "2_months"
    with_confidence_intervals=True,     # Optional: show confidence intervals
    with_seasonality=True,              # Optional: apply seasonality adjustments
    output_file=f"comparison_{store_id}_{item_id}"
)

# Analyze forecast errors
error_metrics = comparator.create_forecast_error_analysis(
    store_id=store_id,
    item_id=item_id,
    lookback_days=30,                   # Optional: days to analyze
    output_file=f"errors_{store_id}_{item_id}"
)
```

### 4. Error Handler

The `error_handler.py` provides decorators and utilities for robust error handling.

```python
from error_handler import ErrorHandler
import pandas as pd

# Use decorators for error handling
@ErrorHandler.handle_data_errors
def load_data(file_path):
    return pd.read_csv(file_path)

@ErrorHandler.handle_visualization_errors
def create_visualization(data, x_column, y_column):
    # Your visualization code here
    pass

# Validate data
data = load_data("data/raw/FrozenPizzaSales.csv")
is_valid, error_message = ErrorHandler.validate_data(
    data=data,
    required_columns=["store_id", "item", "Proc_date", "Total_units"],
    min_rows=10
)

# Create fallback visualizations for errors
if not is_valid:
    error_fig = ErrorHandler.create_error_plot(
        error_message=error_message,
        title="Data Validation Error",
        subtitle="Could not create visualization",
        output_file="error_viz"
    )
```

### 5. Visualization Orchestrator

The `main.py` script provides a unified interface for all visualization components.

```python
from main import VisualizationOrchestrator

# Initialize the orchestrator
orchestrator = VisualizationOrchestrator(data_dir="../data")

# Process a single item
results = orchestrator.process_item(
    store_id="104",
    item_id="3913116850",
    generate_sales_viz=True,            # Optional: generate sales visualization
    generate_seasonality_viz=True,      # Optional: generate seasonality visualization
    generate_short_forecast=True,       # Optional: generate 2-week forecast
    generate_long_forecast=True,        # Optional: generate 2-month forecast
    generate_error_analysis=True        # Optional: generate error analysis
)

# Process multiple items
all_results = orchestrator.process_all_items(
    limit=10,                           # Optional: limit number of items
    store_filter="104",                 # Optional: filter by store
    item_filter=None                    # Optional: filter by item
)

# Generate summary report
orchestrator.generate_summary_report(
    results=all_results,
    output_file="summary_report.html"   # Optional: output file name
)
```

## Data Requirements

The toolkit uses the following data files from your project:

### Raw Data Files (data/raw/)

1. **Sales Data** (`FrozenPizzaSales.csv`):
   ```csv
   store_id,item,Proc_date,Item_Description,Size,Total_units,Total_lbs,Total_Retail_$,Total_Cost_$
   104,3913116850,3/30/2023,BELLATORIA BBQ CHK PIZZA,15.51 OZ,1,0,6.99,5.5
   104,3913116850,4/4/2023,BELLATORIA BBQ CHK PIZZA,15.51 OZ,1,0,6.99,5.5
   ```

2. **Purchase Data** (`FrozenPizzaPurchases.csv`):
   Similar structure to sales data

3. **Stock Data** (`FrozenPizzaStock.csv`):
   Contains inventory levels by store and item

### Processed Data Files (data/processed/)

1. **Random Forest Forecasts** (`rf_forecasts.csv`):
   ```csv
   Date,Store_Id,Item,Forecast,Product,Size,Forecast_Generated,Days_In_Future,Std_Dev,Lower_Bound,Upper_Bound
   2025-07-09,104.0,3913116850.0,0.3462086332354963,BELLATORIA BBQ CHK PIZZA,15.51 OZ,2025-08-06 14:00:54.839516,1,0.45703543492270854,0.0,1.1896907720543393
   2025-07-10,104.0,3913116850.0,0.3857215335408816,BELLATORIA BBQ CHK PIZZA,15.51 OZ,2025-08-06 14:00:54.839516,2,0.5687164774303074,0.0,1.7436750739057785
   ```

2. **PyTorch Forecasts** (`pytorch_forecasts.csv`):
   ```csv
   Date,Store_Id,Item,Product,Day_Of_Week,Month,Price,Weather,Promotion,Stock_Level,Forecast,Std_Dev,Lower_Bound,Upper_Bound,Forecast_Generated,Days_In_Future
   2025-07-09,104.0,3913116850.0,BELLATORIA BBQ CHK PIZZA,2,7,7.99,Normal,0,0,0.89387816,0.0938572070002556,0.709918036186695,1.077838287627697,2025-08-06 14:12:21.773228,1
   2025-07-10,104.0,3913116850.0,BELLATORIA BBQ CHK PIZZA,3,7,7.99,Normal,0,0,0.90450305,0.09949533522129059,0.7094921904325485,1.0995139045000077,2025-08-06 14:12:21.773228,2
   ```

3. **ARIMA Forecasts** (`arima_forecasts.csv`):
   ```csv
   Date,Forecast,Lower_Bound,Upper_Bound,Store_Id,Item,Forecast_Generated,Days_In_Future,Product,Model,Forecast_Type
   2025-08-02,8.026061073366627,3.7789582322513358,12.273163914481916,1,1,2025-08-01 10:56:13.974258,1,Cheese Pizza,ARIMA,Time Series
   2025-08-03,10.892383686042603,6.6452808449273135,15.139486527157894,1,1,2025-08-01 10:56:13.974258,2,Cheese Pizza,ARIMA,Time Series
   ```

### Example Product Data

Here are some examples of actual products in the dataset that you can visualize:

1. **BELLATORIA BBQ CHK PIZZA** (Item: 3913116850, Size: 15.51 OZ)
2. **BELLATORIA ULT PEPPERONI PIZZA** (Item: 3913116852, Size: 17.31 OZ)
3. **BELLATORIA ULT SUPREME PIZZA** (Item: 3913116853, Size: 21.71 OZ)
4. **BELLATORIA GAR CHKN ALFR PIZZA** (Item: 3913116856, Size: 16.03 OZ)
5. **BELLATORIA SAUS ITALIA PIZZA** (Item: 3913116891, Size: 18.27 OZ)

## Output Directory Structure

All visualizations are saved to the `testing/output/` directory with the following naming convention:

- `sales_purchases_104_3913116850.html/.png`: Sales and purchase data for BELLATORIA BBQ CHK PIZZA
- `seasonality_104_3913116850.html/.png`: Seasonality analysis for BELLATORIA BBQ CHK PIZZA
- `forecast_2w_104_3913116850.html/.png`: 2-week forecast for BELLATORIA BBQ CHK PIZZA
- `forecast_2m_104_3913116850.html/.png`: 2-month forecast for BELLATORIA BBQ CHK PIZZA
- `errors_104_3913116850.html/.png`: Error analysis for BELLATORIA BBQ CHK PIZZA
- `summary_report.html`: Overall summary report with links to all visualizations

## Error Logs

The toolkit generates detailed logs for each component:

- `testing/visualization.log`: Item data visualizer logs
- `testing/seasonality.log`: Seasonality analyzer logs
- `testing/model_comparison.log`: Model comparator logs
- `testing/error_handling.log`: Error handler logs
- `testing/main.log`: Main orchestrator logs

## Notes on Seasonality Detection

The toolkit detects seasonality using autocorrelation at different lags (7, 14, and 30 days by default). For each item, it:

1. Tests multiple potential seasonal periods
2. Calculates autocorrelation for each period
3. Selects the period with the strongest autocorrelation (if above threshold)
4. Visualizes the detected seasonal pattern
5. Applies seasonal adjustments to forecasts

This item-specific approach ensures that seasonality is only applied where appropriate and with the right period for each individual item.