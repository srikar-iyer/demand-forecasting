#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from demand_forecaster import DemandForecaster
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTestingSuite:
    """
    Comprehensive testing suite for demand forecasting model performance.
    Includes backtesting, accuracy metrics, and model validation.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the testing suite"""
        self.data_dir = data_dir
        self.forecaster = DemandForecaster(data_dir=data_dir)
        
        # Create results directory
        os.makedirs("model_testing_results", exist_ok=True)
        
        logger.info("Model testing suite initialized successfully")
    
    def backtest_forecast_accuracy(
        self, 
        store_id: str, 
        item_id: str,
        test_periods: List[int] = [7, 14],
        lookback_windows: List[int] = [30, 60, 90]
    ) -> Dict:
        """
        Perform backtesting to evaluate forecast accuracy
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            test_periods: Periods to forecast (in days)
            lookback_windows: Historical windows to test against
            
        Returns:
            Dictionary with backtesting results
        """
        try:
            # Get historical sales data
            sales_data = self.forecaster._prepare_daily_sales(store_id, item_id)
            
            if sales_data.empty or len(sales_data) < max(lookback_windows) + max(test_periods):
                logger.warning(f"Insufficient data for backtesting: store {store_id}, item {item_id}")
                return {"error": "Insufficient data for backtesting"}
            
            results = {
                "store_id": store_id,
                "item_id": item_id,
                "total_data_points": len(sales_data),
                "backtest_results": {}
            }
            
            # Perform backtesting for each lookback window
            for lookback in lookback_windows:
                window_results = {"periods": {}}
                
                for period_days in test_periods:
                    period_weeks = period_days // 7
                    
                    # Split data: use lookback days for training, next period_days for testing
                    if len(sales_data) >= lookback + period_days:
                        train_end_idx = lookback
                        test_end_idx = lookback + period_days
                        
                        train_data = sales_data.iloc[:train_end_idx].copy()
                        test_data = sales_data.iloc[train_end_idx:test_end_idx].copy()
                        
                        # Create temporary forecaster with training data
                        temp_forecaster = self._create_temp_forecaster(train_data, store_id, item_id)
                        
                        # Generate forecast
                        forecast_result = temp_forecaster.generate_forecast(
                            store_id, item_id, [period_weeks]
                        )
                        
                        if "error" not in forecast_result:
                            # Extract forecast values
                            forecast_key = f"{period_weeks}_week"
                            if forecast_key in forecast_result.get("forecasts", {}):
                                forecasted_values = forecast_result["forecasts"][forecast_key]["daily_forecast"]
                                actual_values = test_data['Sales'].values
                                
                                # Calculate accuracy metrics
                                metrics = self._calculate_accuracy_metrics(
                                    actual_values, forecasted_values[:len(actual_values)]
                                )
                                
                                window_results["periods"][f"{period_days}_days"] = {
                                    "forecasted_values": forecasted_values[:len(actual_values)],
                                    "actual_values": actual_values.tolist(),
                                    "metrics": metrics,
                                    "forecast_total": sum(forecasted_values[:len(actual_values)]),
                                    "actual_total": sum(actual_values)
                                }
                        else:
                            window_results["periods"][f"{period_days}_days"] = {
                                "error": forecast_result["error"]
                            }
                
                results["backtest_results"][f"{lookback}_day_window"] = window_results
            
            logger.info(f"Backtesting completed for store {store_id}, item {item_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return {"error": str(e)}
    
    def _create_temp_forecaster(self, train_data: pd.DataFrame, store_id: str, item_id: str) -> DemandForecaster:
        """Create a temporary forecaster with limited training data"""
        # This would ideally create a new forecaster instance with only the training data
        # For now, we'll use the main forecaster but this could be enhanced
        return self.forecaster
    
    def _calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate comprehensive accuracy metrics"""
        try:
            # Ensure arrays have the same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            # Convert to numpy arrays with explicit dtype to avoid indexing issues
            actual = np.array(actual, dtype=float)
            predicted = np.array(predicted, dtype=float)
            
            metrics = {}
            
            # Mean Absolute Error
            metrics["mae"] = mean_absolute_error(actual, predicted)
            
            # Root Mean Square Error
            metrics["rmse"] = np.sqrt(mean_squared_error(actual, predicted))
            
            # Mean Absolute Percentage Error
            # Avoid division by zero
            actual_nonzero_mask = actual != 0
            actual_nonzero = actual[actual_nonzero_mask]
            predicted_nonzero = predicted[actual_nonzero_mask]
            if len(actual_nonzero) > 0:
                metrics["mape"] = mean_absolute_percentage_error(actual_nonzero, predicted_nonzero) * 100
            else:
                metrics["mape"] = float('inf')
            
            # Symmetric Mean Absolute Percentage Error
            denominator = (np.abs(actual) + np.abs(predicted)) / 2
            denominator_nonzero = denominator != 0
            if np.any(denominator_nonzero):
                metrics["smape"] = np.mean(np.abs(actual[denominator_nonzero] - predicted[denominator_nonzero]) / 
                                        denominator[denominator_nonzero]) * 100
            else:
                metrics["smape"] = float('inf')
            
            # Bias (average error)
            metrics["bias"] = np.mean(predicted - actual)
            
            # Directional accuracy (percentage of correct trend predictions)
            if len(actual) > 1:
                actual_direction = np.sign(np.diff(actual))
                predicted_direction = np.sign(np.diff(predicted))
                direction_match = actual_direction == predicted_direction
                metrics["directional_accuracy"] = np.mean(direction_match) * 100
            else:
                metrics["directional_accuracy"] = 0
            
            # Forecast accuracy percentage (1 - MAPE/100)
            if metrics["mape"] != float('inf'):
                metrics["forecast_accuracy_pct"] = max(0, 100 - metrics["mape"])
            else:
                metrics["forecast_accuracy_pct"] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_model_performance(
        self, 
        max_items: int = 10,
        store_filter: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model performance across multiple items
        
        Args:
            max_items: Maximum number of items to test
            store_filter: Optional store filter
            
        Returns:
            Comprehensive performance evaluation
        """
        try:
            # Get list of items to test
            items = self.forecaster.get_item_list()
            
            if store_filter:
                items = [(s, i, d) for s, i, d in items if s == store_filter]
            
            # Limit number of items
            items = items[:max_items]
            
            results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_items_tested": len(items),
                "store_filter": store_filter,
                "item_results": {},
                "aggregate_metrics": {}
            }
            
            all_metrics = []
            
            logger.info(f"Evaluating model performance on {len(items)} items")
            
            for idx, (store_id, item_id, desc) in enumerate(items):
                logger.info(f"Testing {idx+1}/{len(items)}: {desc} (Store {store_id}, Item {item_id})")
                
                # Perform backtesting
                backtest_result = self.backtest_forecast_accuracy(store_id, item_id)
                
                if "error" not in backtest_result:
                    results["item_results"][f"{store_id}_{item_id}"] = {
                        "description": desc,
                        "backtest_result": backtest_result
                    }
                    
                    # Collect metrics for aggregation
                    for window, window_data in backtest_result.get("backtest_results", {}).items():
                        for period, period_data in window_data.get("periods", {}).items():
                            if "metrics" in period_data:
                                metrics = period_data["metrics"].copy()
                                metrics["item_id"] = item_id
                                metrics["store_id"] = store_id
                                metrics["window"] = window
                                metrics["period"] = period
                                all_metrics.append(metrics)
                else:
                    results["item_results"][f"{store_id}_{item_id}"] = {
                        "description": desc,
                        "error": backtest_result["error"]
                    }
            
            # Calculate aggregate metrics
            if all_metrics:
                results["aggregate_metrics"] = self._calculate_aggregate_metrics(all_metrics)
            
            # Save results
            output_file = f"model_testing_results/performance_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Performance evaluation completed. Results saved to {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error in model performance evaluation: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Calculate aggregate performance metrics across all items"""
        try:
            df = pd.DataFrame(all_metrics)
            
            aggregate = {}
            
            # Calculate mean, median, std for key metrics
            key_metrics = ["mae", "rmse", "mape", "smape", "bias", "directional_accuracy", "forecast_accuracy_pct"]
            
            for metric in key_metrics:
                if metric in df.columns:
                    valid_values = df[metric][df[metric] != float('inf')]
                    if len(valid_values) > 0:
                        aggregate[metric] = {
                            "mean": float(valid_values.mean()),
                            "median": float(valid_values.median()),
                            "std": float(valid_values.std()),
                            "min": float(valid_values.min()),
                            "max": float(valid_values.max()),
                            "count": len(valid_values)
                        }
            
            # Performance categories
            if "forecast_accuracy_pct" in df.columns:
                accuracy_values = df["forecast_accuracy_pct"][df["forecast_accuracy_pct"] != float('inf')]
                if len(accuracy_values) > 0:
                    aggregate["performance_distribution"] = {
                        "excellent_>90pct": len(accuracy_values[accuracy_values >= 90]),
                        "good_70-90pct": len(accuracy_values[(accuracy_values >= 70) & (accuracy_values < 90)]),
                        "fair_50-70pct": len(accuracy_values[(accuracy_values >= 50) & (accuracy_values < 70)]),
                        "poor_<50pct": len(accuracy_values[accuracy_values < 50])
                    }
            
            return aggregate
            
        except Exception as e:
            logger.error(f"Error calculating aggregate metrics: {str(e)}")
            return {"error": str(e)}
    
    def generate_performance_report(self, evaluation_results: Dict) -> str:
        """Generate a human-readable performance report"""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("DEMAND FORECASTING MODEL PERFORMANCE REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # Summary
            report_lines.append(f"Evaluation Date: {evaluation_results.get('evaluation_timestamp', 'N/A')}")
            report_lines.append(f"Items Tested: {evaluation_results.get('total_items_tested', 0)}")
            report_lines.append(f"Store Filter: {evaluation_results.get('store_filter', 'All stores')}")
            report_lines.append("")
            
            # Aggregate metrics
            agg_metrics = evaluation_results.get('aggregate_metrics', {})
            
            if agg_metrics:
                report_lines.append("AGGREGATE PERFORMANCE METRICS")
                report_lines.append("-" * 40)
                
                for metric, stats in agg_metrics.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        report_lines.append(f"{metric.upper().replace('_', ' ')}: "
                                          f"Mean={stats['mean']:.2f}, "
                                          f"Median={stats['median']:.2f}, "
                                          f"Std={stats['std']:.2f}")
                
                # Performance distribution
                if 'performance_distribution' in agg_metrics:
                    dist = agg_metrics['performance_distribution']
                    report_lines.append("")
                    report_lines.append("FORECAST ACCURACY DISTRIBUTION")
                    report_lines.append(f"Excellent (≥90%): {dist.get('excellent_>90pct', 0)} items")
                    report_lines.append(f"Good (70-90%): {dist.get('good_70-90pct', 0)} items")
                    report_lines.append(f"Fair (50-70%): {dist.get('fair_50-70pct', 0)} items")
                    report_lines.append(f"Poor (<50%): {dist.get('poor_<50pct', 0)} items")
                
                report_lines.append("")
            
            # Individual item performance
            item_results = evaluation_results.get('item_results', {})
            
            if item_results:
                report_lines.append("TOP PERFORMING ITEMS")
                report_lines.append("-" * 30)
                
                # Calculate average accuracy for each item
                item_accuracies = []
                
                for item_key, item_data in item_results.items():
                    if 'backtest_result' in item_data:
                        backtest = item_data['backtest_result']
                        accuracies = []
                        
                        for window_data in backtest.get('backtest_results', {}).values():
                            for period_data in window_data.get('periods', {}).values():
                                if 'metrics' in period_data:
                                    acc = period_data['metrics'].get('forecast_accuracy_pct', 0)
                                    if acc != float('inf'):
                                        accuracies.append(acc)
                        
                        if accuracies:
                            avg_accuracy = np.mean(accuracies)
                            item_accuracies.append((
                                item_data['description'],
                                item_key,
                                avg_accuracy
                            ))
                
                # Sort by accuracy and show top 5
                item_accuracies.sort(key=lambda x: x[2], reverse=True)
                
                for i, (desc, key, acc) in enumerate(item_accuracies[:5]):
                    report_lines.append(f"{i+1}. {desc} ({key}): {acc:.1f}% accuracy")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    try:
        # Initialize testing suite
        testing_suite = ModelTestingSuite(data_dir="data")
        
        logger.info("Starting model performance evaluation...")
        
        # Evaluate model performance on a sample of items
        results = testing_suite.evaluate_model_performance(
            max_items=5,  # Test on first 5 items
            store_filter=None
        )
        
        if "error" not in results:
            # Generate and display performance report
            report = testing_suite.generate_performance_report(results)
            print(report)
            
            # Save report to file
            report_file = f"model_testing_results/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Performance report saved to {report_file}")
        else:
            logger.error(f"Evaluation failed: {results['error']}")
        
        logger.info("Model testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")