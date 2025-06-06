# Retail Predictive Ordering System

A machine learning-based system for predicting optimal order quantities for retail products based on historical sales data, price elasticity, promotional effects, seasonality, and external factors.

## Overview

The Retail Predictive Ordering System helps retailers optimize their inventory management by providing data-driven order quantity recommendations. The system uses Random Forest machine learning models to analyze historical sales patterns and predict future demand while accounting for factors like:

- Price changes and elasticity
- Promotional activities
- Seasonal patterns
- Day-of-week effects
- Weather conditions
- Holiday impacts
- Lead time variations

## Features

- **Demand Forecasting**: Generate accurate 30-day demand forecasts for retail products
- **Order Quantity Recommendation**: Calculate optimal order quantities with configurable safety stock levels
- **Scenario Analysis**: Test different pricing, promotion, and environmental scenarios
- **Feature Importance Analysis**: Understand key factors driving demand for each product
- **Visual Analytics**: Interactive charts showing historical patterns, forecasts, and decision factors
- **Interactive UI**: User-friendly Gradio interface for making predictions
- **Comprehensive Testing**: Unit tests validating all system components

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
gradio
```

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/predictive-ordering.git
   cd predictive-ordering
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```
   mkdir -p static/images
   ```

## Usage

### Running the Full Application

```
python retail_forecast.py
```

This launches the complete application with comprehensive features including:
- Full model training on historical data
- Demand forecasting for all products
- Visualization generation
- Interactive UI

### Running the Lightweight Version

```
python standalone_retail_forecast_ui.py
```

Launches a simplified version with:
- Smaller sample data generation
- Faster model training
- Streamlined UI

### Running Tests

```
python -m unittest tests/test_forecast.py
```

Or use the test runner script:

```
python run_test.py
```

### Docker Deployment

Build and run the Docker container:

```
docker build -t retail-forecast .
docker run -p 7860:7860 retail-forecast
```

Then access the UI at http://localhost:7860

## Core Components

### Data Management

- **CSV Import**: Load and process real retail data from CSV files
- **Sample Data Generation**: Create realistic synthetic data when needed
- **Feature Engineering**: Transform raw data into ML-ready features including:
  - Temporal features (day of week, month, year)
  - Lag features capturing past sales
  - Rolling averages for trend detection
  - Cyclical encoding for seasonal patterns
  - One-hot encoding for categorical variables

### Machine Learning

- **Random Forest Model**: Ensemble learning for robust prediction with features importance analysis
- **Model Training**: Split data into training/testing sets for evaluation
- **Evaluation Metrics**: RMSE, MAE, and R² for model performance assessment

### Visualization

The system generates several types of visualizations:

1. **Historical Sales Trends**: Time series by product
2. **Seasonal Patterns**: Monthly sales patterns by product
3. **Feature Importance**: Top factors affecting sales predictions
4. **Prediction Accuracy**: Model accuracy visualization
5. **Scenario Comparison**: Impact of different scenarios on predicted sales
6. **Promotional Impact**: Effect of promotions on sales by product
7. **Demand Forecast**: 30-day demand forecast visualization

### User Interface

The Gradio-based UI allows users to:

- Select products for analysis
- Adjust pricing parameters
- Toggle promotional periods
- Set holiday conditions
- Select weather conditions
- Configure lead times and safety stock percentages
- View prediction results and explanatory charts

## File Structure

- `retail_forecast.py`: Main application with comprehensive features
- `standalone_retail_forecast_ui.py`: Lightweight version for quick demonstrations
- `run_test.py`: Test runner script
- `run_mini.py`: Minimal test script for quick UI validation
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration
- `tests/`: Unit tests
  - `test_forecast.py`: Test cases for the forecasting system
- `static/images/`: Generated visualizations and charts

## Output Examples

The system generates several output files:

- `testresult.txt`: Detailed test results and model evaluation metrics
- Visualization images in `static/images/`:
  - `historical_sales.png`
  - `seasonal_patterns.png`
  - `feature_importance.png`
  - `prediction_accuracy.png`
  - `scenario_comparison.png`
  - `promotion_impact.png`
  - `demand_forecast.png`

## Development

### Adding New Features

To add new predictive factors or enhance the model:

1. Update the feature engineering in `prepare_features()` function
2. Retrain the model to incorporate new features
3. Update UI components if needed to expose new parameters

### Customizing for Different Retail Domains

The system can be customized for specific retail domains by:

1. Adjusting the data preprocessing for domain-specific attributes
2. Modifying scenario tests for relevant business conditions
3. Tuning safety stock calculations for specific inventory policies
4. Adding domain-specific visualizations

## License

[Insert your license information here]

## Acknowledgments

- This project uses scikit-learn for machine learning capabilities
- Gradio for the interactive UI components
- Matplotlib and Seaborn for visualization