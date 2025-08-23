#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("demand_forecaster.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemandForecaster:
    """
    Local trajectory-weighted demand forecasting with exponential smoothing.
    Focuses on recent patterns while maintaining 30% deviation constraints.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the forecaster with data directory path"""
        self.data_dir = data_dir
        self.sales_data = None
        self.purchase_data = None
        
        # Forecasting parameters
        self.max_deviation_pct = 30  # Maximum deviation from recent averages
        self.recent_weight_days = 14  # Days to consider as "recent" for heavy weighting
        self.min_data_days = 14      # Minimum days of data required
        
        try:
            self._load_data()
            logger.info("Demand forecaster initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing demand forecaster: {str(e)}")
            raise
    
    def _load_data(self) -> None:
        """Load sales and purchase data"""
        try:
            # Load raw data
            self.sales_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaSales.csv"))
            self.purchase_data = pd.read_csv(os.path.join(self.data_dir, "FrozenPizzaPurchases.csv"))
            
            # Convert dates to datetime objects
            self.sales_data['Proc_date'] = pd.to_datetime(self.sales_data['Proc_date'])
            self.purchase_data['Proc_date'] = pd.to_datetime(self.purchase_data['Proc_date'])
            
        except Exception as e:
            logger.error(f"Error in _load_data: {str(e)}")
            raise
    
    def _prepare_daily_sales(self, store_id: str, item_id: str) -> pd.DataFrame:
        """Prepare daily sales data for a specific store-item combination"""
        try:
            # Convert inputs to match data types
            store_id_val = float(store_id)
            item_id_val = float(item_id)
            
            # Filter data for the specific store and item
            item_sales = self.sales_data[
                (self.sales_data['store_id'] == store_id_val) & 
                (self.sales_data['item'] == item_id_val)
            ].copy()
            
            if item_sales.empty:
                logger.warning(f"No sales data found for store {store_id}, item {item_id}")
                return pd.DataFrame()
            
            # Aggregate by date
            daily_sales = item_sales.groupby('Proc_date')['Total_units'].sum().reset_index()
            daily_sales = daily_sales.set_index('Proc_date').sort_index()
            
            # Fill missing dates with zeros (sparse pattern preservation)
            idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max())
            daily_sales = daily_sales.reindex(idx, fill_value=0)
            
            return daily_sales.reset_index().rename(columns={'index': 'Date', 'Total_units': 'Sales'})
            
        except Exception as e:
            logger.error(f"Error preparing daily sales: {str(e)}")
            return pd.DataFrame()
    
    def _get_purchase_patterns(self, store_id: str, item_id: str) -> Dict:
        """Analyze purchase patterns to determine ordering multiples"""
        try:
            # Convert inputs to match data types
            store_id_val = float(store_id)
            item_id_val = float(item_id)
            
            item_purchases = self.purchase_data[
                (self.purchase_data['store_id'] == store_id_val) & 
                (self.purchase_data['item'] == item_id_val)
            ].copy()
            
            if item_purchases.empty:
                return {"multiple": 1, "min_order": 1}
            
            # Find common multiples in purchase quantities
            quantities = item_purchases['Total_units'].values
            quantities = quantities[quantities > 0]
            
            if len(quantities) == 0:
                return {"multiple": 1, "min_order": 1}
            
            # Find the most common divisor (likely ordering multiple)
            common_divisors = []
            for q in quantities:
                for div in [12, 24, 6, 10, 5]:  # Common ordering multiples
                    if q % div == 0:
                        common_divisors.append(div)
            
            if common_divisors:
                most_common_multiple = max(set(common_divisors), key=common_divisors.count)
            else:
                most_common_multiple = 1
            
            min_order = min(quantities) if len(quantities) > 0 else 1
            
            return {
                "multiple": most_common_multiple,
                "min_order": min_order,
                "avg_order": np.mean(quantities)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing purchase patterns: {str(e)}")
            return {"multiple": 1, "min_order": 1}
    
    def _calculate_moving_averages(self, sales_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages with recent bias"""
        try:
            if sales_data.empty or len(sales_data) == 0:
                return {}
            
            sales_values = sales_data['Sales'].values
            dates = sales_data['Date']
            
            # Calculate different time period averages
            averages = {}
            
            # Last week average (7 days)
            if len(sales_values) >= 7:
                averages['last_week'] = np.mean(sales_values[-7:])
            
            # Last two weeks average (14 days)
            if len(sales_values) >= 14:
                averages['last_2_weeks'] = np.mean(sales_values[-14:])
            
            # Last month average (30 days)
            if len(sales_values) >= 30:
                averages['last_month'] = np.mean(sales_values[-30:])
            
            # Overall average
            averages['overall'] = np.mean(sales_values)
            
            # Weighted moving average (recent data weighted more heavily)
            if len(sales_values) >= 7:
                weights = np.exp(np.linspace(-2, 0, len(sales_values[-14:])))  # Exponential weights
                weights = weights / weights.sum()
                recent_data = sales_values[-len(weights):]
                averages['weighted_recent'] = np.average(recent_data, weights=weights)
            
            return averages
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return {}
    
    def _exponential_smoothing_forecast(self, sales_data: pd.DataFrame, periods: int, alpha: float = None) -> List[float]:
        """Apply exponential smoothing with automatic alpha tuning for local patterns"""
        try:
            if sales_data.empty or len(sales_data) < 3:
                return [0] * periods
            
            sales_values = sales_data['Sales'].values
            
            # Auto-tune alpha based on data volatility and recency bias
            if alpha is None:
                # Higher alpha for more recent bias, lower for stable patterns
                recent_volatility = np.std(sales_values[-7:]) if len(sales_values) >= 7 else np.std(sales_values)
                overall_volatility = np.std(sales_values)
                
                if recent_volatility > overall_volatility * 0.5:
                    alpha = 0.7  # High alpha for volatile recent data
                else:
                    alpha = 0.3  # Lower alpha for stable patterns
            
            # Initialize with first value
            smoothed = [sales_values[0]]
            
            # Apply exponential smoothing
            for i in range(1, len(sales_values)):
                smoothed_value = alpha * sales_values[i] + (1 - alpha) * smoothed[-1]
                smoothed.append(smoothed_value)
            
            # Generate forecasts
            forecast = []
            last_smoothed = smoothed[-1]
            
            # Apply trend dampening
            if len(sales_values) >= 7:
                recent_trend = (np.mean(sales_values[-3:]) - np.mean(sales_values[-7:-3])) if len(sales_values) >= 7 else 0
                trend_damping = 0.8  # Dampen trend towards mean
            else:
                recent_trend = 0
                trend_damping = 0.5
            
            for period in range(periods):
                # Apply dampened trend
                dampened_trend = recent_trend * (trend_damping ** period)
                forecast_value = last_smoothed + dampened_trend
                
                # Ensure non-negative
                forecast_value = max(0, forecast_value)
                forecast.append(forecast_value)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing: {str(e)}")
            return [0] * periods
    
    def _validate_forecast_deviation(self, forecast_values: List[float], averages: Dict[str, float]) -> Tuple[List[float], List[str]]:
        """Validate forecasts don't exceed 30% deviation from recent averages"""
        try:
            validated_forecasts = []
            explanations = []
            
            # Get reference average (prefer recent periods)
            if 'last_week' in averages:
                reference_avg = averages['last_week']
                reference_period = "last week"
            elif 'last_2_weeks' in averages:
                reference_avg = averages['last_2_weeks']
                reference_period = "last 2 weeks"
            else:
                reference_avg = averages.get('overall', 1)
                reference_period = "overall period"
            
            # Calculate allowed range
            max_allowed = reference_avg * (1 + self.max_deviation_pct / 100)
            min_allowed = reference_avg * (1 - self.max_deviation_pct / 100)
            
            for i, forecast_value in enumerate(forecast_values):
                if forecast_value > max_allowed:
                    validated_value = max_allowed
                    explanations.append(f"Forecast capped at 30% above {reference_period} average ({reference_avg:.1f})")
                elif forecast_value < min_allowed:
                    validated_value = min_allowed
                    explanations.append(f"Forecast floored at 30% below {reference_period} average ({reference_avg:.1f})")
                else:
                    validated_value = forecast_value
                    explanations.append(f"Forecast within 30% of {reference_period} average")
                
                validated_forecasts.append(validated_value)
            
            return validated_forecasts, explanations
            
        except Exception as e:
            logger.error(f"Error validating forecasts: {str(e)}")
            return forecast_values, ["Validation error occurred"]
    
    def _preserve_pattern_characteristics(self, sales_data: pd.DataFrame, forecast_values: List[float]) -> List[float]:
        """Preserve characteristics of recent sales patterns (sparse, seasonal, etc.)"""
        try:
            if sales_data.empty or len(forecast_values) == 0:
                return forecast_values
            
            recent_sales = sales_data['Sales'].values[-14:]  # Last 2 weeks
            
            # Check for sparsity (many zeros)
            zero_ratio = (recent_sales == 0).sum() / len(recent_sales)
            
            if zero_ratio > 0.5:  # If more than 50% zeros, maintain sparsity
                for i in range(len(forecast_values)):
                    if np.random.random() < zero_ratio * 0.8:  # Slightly less sparse in forecast
                        forecast_values[i] = 0
            
            # Preserve low-volatility patterns
            if len(recent_sales) > 0 and np.std(recent_sales) < np.mean(recent_sales) * 0.3:
                # Low volatility - smooth out forecasts
                for i in range(len(forecast_values)):
                    if i > 0:
                        forecast_values[i] = 0.7 * forecast_values[i] + 0.3 * forecast_values[i-1]
            
            return forecast_values
            
        except Exception as e:
            logger.error(f"Error preserving patterns: {str(e)}")
            return forecast_values
    
    def generate_forecast(self, store_id: str, item_id: str, forecast_weeks: List[int] = [1, 2]) -> Dict:
        """
        Generate demand forecast for specified periods
        
        Args:
            store_id: Store identifier
            item_id: Item identifier  
            forecast_weeks: List of weeks to forecast (e.g., [1, 2])
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Prepare sales data
            sales_data = self._prepare_daily_sales(store_id, item_id)
            
            if sales_data.empty:
                logger.warning(f"No sales data available for forecasting: store {store_id}, item {item_id}")
                return {"error": "No sales data available"}
            
            if len(sales_data) < self.min_data_days:
                logger.warning(f"Insufficient data for forecasting: {len(sales_data)} days (minimum {self.min_data_days})")
                return {"error": f"Insufficient data: {len(sales_data)} days"}
            
            # Get purchase patterns
            purchase_patterns = self._get_purchase_patterns(store_id, item_id)
            
            # Calculate moving averages
            averages = self._calculate_moving_averages(sales_data)
            
            results = {
                "store_id": store_id,
                "item_id": item_id,
                "data_points": len(sales_data),
                "averages": averages,
                "purchase_patterns": purchase_patterns,
                "forecasts": {}
            }
            
            # Generate forecasts for each requested period
            for weeks in forecast_weeks:
                forecast_days = weeks * 7
                
                # Generate raw forecast using exponential smoothing
                raw_forecast = self._exponential_smoothing_forecast(sales_data, forecast_days)
                
                # Preserve pattern characteristics
                pattern_adjusted = self._preserve_pattern_characteristics(sales_data, raw_forecast)
                
                # Validate against deviation constraints
                validated_forecast, explanations = self._validate_forecast_deviation(pattern_adjusted, averages)
                
                # Calculate weekly aggregations
                weekly_totals = []
                for week in range(weeks):
                    start_idx = week * 7
                    end_idx = (week + 1) * 7
                    week_total = sum(validated_forecast[start_idx:end_idx])
                    weekly_totals.append(week_total)
                
                # Calculate practical forecast (rounded to purchase multiples)
                total_forecast = sum(validated_forecast)
                practical_forecast = self._round_to_purchase_multiple(total_forecast, purchase_patterns)
                
                # Store results
                results["forecasts"][f"{weeks}_week"] = {
                    "daily_forecast": validated_forecast,
                    "weekly_totals": weekly_totals,
                    "total_predicted": total_forecast,
                    "total_practical": practical_forecast,
                    "explanations": explanations[:weeks],  # One explanation per week
                    "confidence": self._calculate_confidence(sales_data, validated_forecast)
                }
            
            logger.info(f"Forecast generated successfully for store {store_id}, item {item_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {"error": str(e)}
    
    def _round_to_purchase_multiple(self, forecast_value: float, purchase_patterns: Dict) -> int:
        """Round forecast to practical purchase multiples"""
        try:
            multiple = purchase_patterns.get("multiple", 1)
            min_order = purchase_patterns.get("min_order", 1)
            
            if forecast_value <= 0:
                return 0
            
            # Round up to nearest multiple if close
            rounded = round(forecast_value / multiple) * multiple
            
            # If rounded down significantly, round up to ensure availability
            if rounded < forecast_value * 0.9 and forecast_value >= min_order:
                rounded = (int(forecast_value / multiple) + 1) * multiple
            
            return max(int(rounded), min_order if forecast_value >= min_order * 0.5 else 0)
            
        except Exception as e:
            logger.error(f"Error rounding to purchase multiple: {str(e)}")
            return int(round(forecast_value))
    
    def _calculate_confidence(self, sales_data: pd.DataFrame, forecast_values: List[float]) -> float:
        """Calculate confidence score for the forecast"""
        try:
            if sales_data.empty or len(forecast_values) == 0:
                return 0.0
            
            recent_sales = sales_data['Sales'].values[-14:]
            
            # Factors affecting confidence
            data_consistency = 1.0 - (np.std(recent_sales) / (np.mean(recent_sales) + 1))  # Lower std = higher confidence
            data_recency = min(1.0, len(sales_data) / 30)  # More data = higher confidence
            pattern_stability = 1.0 - abs(np.mean(recent_sales[:7]) - np.mean(recent_sales[7:])) / (np.mean(recent_sales) + 1)
            
            confidence = (data_consistency * 0.4 + data_recency * 0.3 + pattern_stability * 0.3)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def get_item_list(self) -> List[Tuple[str, str, str]]:
        """Return a list of unique items with store_id, item_id, and description"""
        try:
            if self.sales_data is not None:
                items = self.sales_data[['store_id', 'item', 'Item_Description']].drop_duplicates()
                return [(str(row['store_id']), str(row['item']), row['Item_Description']) 
                        for _, row in items.iterrows()]
            return []
        except Exception as e:
            logger.error(f"Error in get_item_list: {str(e)}")
            return []

if __name__ == "__main__":
    try:
        # Initialize forecaster
        forecaster = DemandForecaster(data_dir="data")
        
        # Get list of items
        items = forecaster.get_item_list()
        logger.info(f"Found {len(items)} unique store-item combinations")
        
        # Test forecasting with first item
        if items:
            store_id, item_id, desc = items[0]
            logger.info(f"Testing forecast for: Store {store_id}, Item {item_id} ({desc})")
            
            # Generate 1-week and 2-week forecasts
            result = forecaster.generate_forecast(store_id, item_id, forecast_weeks=[1, 2])
            
            if "error" not in result:
                logger.info(f"Forecast results: {result['forecasts']}")
            else:
                logger.error(f"Forecast error: {result['error']}")
        
        logger.info("Demand forecaster testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")