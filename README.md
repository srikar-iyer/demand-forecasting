# Advanced Demand Forecasting System

This system provides comprehensive demand forecasting with local trajectory weighting, focusing on recent sales patterns while maintaining constraint-based validation. The system uses exponential smoothing with heavy weighting on recent 1-2 week patterns to generate reliable forecasts.

## Core Features

- **Local Trajectory Forecasting**: Heavy weighting on recent 1-2 weeks vs historical data
- **Constraint-Based Validation**: Maximum 30% deviation from recent averages
- **Pattern Preservation**: Maintains characteristics of recent data (sparse → sparse, seasonal → seasonal)
- **Dual Forecast Types**: Predicted forecasts and practical order quantities
- **Interactive Dashboard**: Two-tab interface for time series and numerical forecasts
- **Explanation System**: Detailed reasoning for forecast predictions
- **Model Testing Suite**: Comprehensive backtesting and performance metrics

## Installation

1. Ensure you have Python 3.8+ installed
2. Clone this repository
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Requirements

The toolkit uses the following data files from your project:

### Data Files (data/)

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

4. **Additional Data** (`Price_Elasticity_Frozen_Input.csv`, `combined_pizza_data.csv`):
   Additional data files for extended analysis

## Quick Start - Demand Forecasting

### Run the Demand Forecasting Dashboard

```bash
# Launch the interactive demand forecasting app
python forecast_app.py
```

Access the dashboard at http://localhost:5000/ with these features:
- **Tab 1**: Time series plots showing historical sales + 1-week & 2-week forecasts
- **Tab 2**: Numerical forecast results with predicted and practical quantities
- **Interactive Controls**: Filter by store and item
- **Detailed Explanations**: Reasoning behind each forecast prediction

### Generate Individual Forecasts

```bash
# Test the core forecasting engine
python demand_forecaster.py

# Test forecast visualization
python forecast_visualizer.py

# Generate forecast explanations
python forecast_explainer.py
```

### Run Model Performance Testing

```bash
# Evaluate forecasting accuracy across multiple items
python model_testing_suite.py
```

## Demand Forecasting Methodology

### Core Algorithm: Exponential Smoothing with Local Trajectory Weighting

1. **Recent Data Emphasis**: Last 1-2 weeks weighted heavily (exponential decay)
2. **Trend Dampening**: Linear trends dampened towards historical mean
3. **Constraint Enforcement**: Maximum 30% deviation from recent averages
4. **Pattern Preservation**: Maintains sparsity and seasonality characteristics
5. **Purchase Optimization**: Rounds forecasts to practical ordering multiples

### Key Parameters

- **Maximum Deviation**: 30% from recent average sales (user requirement)
- **Local Weighting Period**: 14 days (recent data heavily weighted)
- **Minimum Data Requirement**: 14 days of historical sales
- **Forecast Horizons**: 1 week and 2 weeks
- **Alpha Tuning**: Automatic based on volatility (0.3-0.7 range)

## Forecasting Components

### 1. Demand Forecaster (`demand_forecaster.py`)

The core forecasting engine implementing exponential smoothing with local trajectory weighting.

```python
from demand_forecaster import DemandForecaster

# Initialize the forecaster
forecaster = DemandForecaster(data_dir="data")

# Generate 1-week and 2-week forecasts
results = forecaster.generate_forecast(
    store_id="104.0",
    item_id="3913116850.0",
    forecast_weeks=[1, 2]
)

print(f"1-week forecast: {results['forecasts']['1_week']['total_predicted']:.2f} units")
print(f"Practical order: {results['forecasts']['1_week']['total_practical']} units")
print(f"Confidence: {results['forecasts']['1_week']['confidence']:.0%}")
```

### 2. Forecast Visualizer (`forecast_visualizer.py`)

Creates interactive time series plots showing historical sales and forecasted demand.

```python
from forecast_visualizer import ForecastVisualizer

# Initialize the visualizer
visualizer = ForecastVisualizer(data_dir="data")

# Create time series forecast plot
fig = visualizer.create_forecast_time_series(
    store_id="104.0",
    item_id="3913116850.0",
    forecast_weeks=[1, 2],
    output_file="forecast_104_3913116850"
)

# Create numerical comparison table
comparison_df = visualizer.create_forecast_comparison_table(
    store_id="104.0",
    item_id="3913116850.0"
)
print(comparison_df)
```

### 3. Forecast Explainer (`forecast_explainer.py`)

Provides detailed explanations for why forecasts were generated with specific values.

```python
from forecast_explainer import ForecastExplainer

# Initialize the explainer
explainer = ForecastExplainer(data_dir="data")

# Generate detailed explanations
explanation = explainer.generate_forecast_explanation(
    store_id="104.0",
    item_id="3913116850.0",
    forecast_weeks=[1, 2]
)

# Access key factors
for period, details in explanation['forecast_explanations'].items():
    print(f"\n{period} forecast explanation:")
    for factor in details['key_factors']:
        print(f"  • {factor}")
    
    risk = details['risk_assessment']
    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Recommendation: {risk['recommendation']}")
```

### 4. Model Testing Suite (`model_testing_suite.py`)

Comprehensive backtesting and performance evaluation for the forecasting models.

```python
from model_testing_suite import ModelTestingSuite

# Initialize testing suite
testing_suite = ModelTestingSuite(data_dir="data")

# Evaluate performance across multiple items
results = testing_suite.evaluate_model_performance(
    max_items=10,
    store_filter="104.0"  # Optional: filter by store
)

# Generate performance report
report = testing_suite.generate_performance_report(results)
print(report)

# Perform detailed backtesting for specific item
backtest_result = testing_suite.backtest_forecast_accuracy(
    store_id="104.0",
    item_id="3913116850.0",
    test_periods=[7, 14],        # Test 7-day and 14-day forecasts
    lookback_windows=[30, 60, 90] # Use different training windows
)
```

## Legacy Visualization Components

### 1. Item Data Visualizer

The `item_data_visualizer.py` script provides visualizations of sales and purchase data for specific items.

```python
from item_data_visualizer import ItemDataVisualizer

# Initialize the visualizer
visualizer = ItemDataVisualizer(data_dir="data")

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
    output_file=f"sales_purchases_{store_id}_{item_id}"
)

# Visualize sales trends with different timeframes
visualizer.visualize_sales_trends(
    store_id=store_id,
    item_id=item_id,
    timeframe="3_months",      # Options: "1_month", "3_months", "6_months", "all"
    output_file=f"sales_trends_{store_id}_{item_id}"
)
```

### 2. Seasonality Analyzer

The `seasonality_analyzer.py` script detects and visualizes item-specific seasonality patterns.

```python
from seasonality_analyzer import ItemSeasonalityAnalyzer

# Initialize the analyzer
analyzer = ItemSeasonalityAnalyzer(data_dir="data")

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
```

### 3. Model Comparator (Sales Analysis)

The `model_comparator.py` script provides detailed sales analysis and inventory analysis.

```python
from model_comparator import ModelComparator

# Initialize the comparator
comparator = ModelComparator(data_dir="data")

# Create sales analysis with different time periods
comparator.create_sales_analysis(
    store_id="104",
    item_id="3913116850",
    period="3_months",             # Options: "1_month", "3_months", "6_months", "all"
    with_seasonality=True,         # Optional: show seasonality indicators
    output_file=f"sales_analysis_{store_id}_{item_id}"
)

# Create inventory analysis
comparator.create_inventory_analysis(
    store_id="104",
    item_id="3913116850",
    lookback_days=30,              # Optional: days to analyze
    output_file=f"inventory_{store_id}_{item_id}"
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
data = load_data("data/FrozenPizzaSales.csv")
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
orchestrator = VisualizationOrchestrator(data_dir="data")

# Process a single item
results = orchestrator.process_item(
    store_id="104",
    item_id="3913116850",
    generate_sales_viz=True,           # Optional: generate sales visualization
    generate_seasonality_viz=True,     # Optional: generate seasonality visualization
    generate_sales_analysis=True,      # Optional: generate sales analysis
    generate_inventory_analysis=True   # Optional: generate inventory analysis
)

# Process multiple items
all_results = orchestrator.process_all_items(
    limit=10,                          # Optional: limit number of items
    store_filter="104",                # Optional: filter by store
    item_filter=None                   # Optional: filter by item
)

# Generate summary report
orchestrator.generate_summary_report(
    results=all_results,
    output_file="summary_report.html"  # Optional: output file name
)
```

## Output Directory Structure

### Demand Forecasting Outputs (`output/` directory)

- `forecast_104_3913116850.html/.png`: Time series forecast plots with 1-week & 2-week predictions
- `explanation_104_3913116850.html`: Detailed forecast explanation visualizations
- `summary_report.html`: Overall summary report with links to all visualizations

### Model Testing Results (`model_testing_results/` directory)

- `performance_evaluation_YYYYMMDD_HHMMSS.json`: Comprehensive performance metrics
- `performance_report_YYYYMMDD_HHMMSS.txt`: Human-readable performance reports

### Legacy Visualization Outputs

- `sales_purchases_104_3913116850.html/.png`: Sales and purchase data visualizations
- `seasonality_104_3913116850.html/.png`: Seasonality analysis plots
- `sales_analysis_3m_104_3913116850.html/.png`: 3-month sales analysis
- `inventory_104_3913116850.html/.png`: Inventory analysis visualizations

## Logs

### Demand Forecasting Logs

- `demand_forecaster.log`: Core forecasting engine operations and results
- `forecast_visualizer.log`: Time series visualization generation
- `forecast_explainer.log`: Explanation generation and analysis
- `forecast_app.log`: Interactive dashboard operations
- `model_testing.log`: Performance testing and backtesting results

### Legacy Component Logs

- `visualization.log`: Item data visualizer logs
- `seasonality.log`: Seasonality analyzer logs
- `model_comparison.log`: Model comparator logs
- `error_handling.log`: Error handler logs
- `main.log`: Main orchestrator logs

## Demand Forecasting Constraints & Validation

The forecasting system implements several key constraints to ensure reliable predictions:

### 1. Maximum Deviation Constraint (30%)
- All forecasts are constrained to within 30% of recent average sales
- Prevents unrealistic predictions during volatile periods
- Uses most recent available averages (1 week > 2 weeks > 1 month priority)

### 2. Pattern Preservation Logic
- **Sparse Data**: If recent sales have >50% zero days, forecasts maintain sparsity
- **Consistent Data**: Low-volatility patterns generate smooth forecasts
- **Seasonal Data**: Applies detected seasonal adjustments when robust patterns exist

### 3. Local Trajectory Weighting
- Recent 14 days: Heavy weighting (exponential decay)
- Historical data: Lower weighting for context
- Volatility-based alpha adjustment (0.3-0.7 range)

### 4. Purchase Multiple Optimization
- Analyzes historical purchase patterns to identify ordering multiples
- Rounds predicted forecasts to practical order quantities
- Considers minimum order requirements from purchase history

## Notes on Seasonality Detection

The system detects seasonality using autocorrelation at different lags (7, 14, and 30 days by default). For each item, it:

1. Tests multiple potential seasonal periods
2. Calculates autocorrelation for each period  
3. Selects the period with the strongest autocorrelation (if above threshold)
4. Applies seasonal adjustments only when patterns are robust
5. Integrates seasonality with local trajectory weighting

This item-specific approach ensures seasonality is only applied where statistically significant and appropriate for each individual item's sales pattern.