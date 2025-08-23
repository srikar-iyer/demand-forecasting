#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from demand_forecaster import DemandForecaster
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forecast_explainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForecastExplainer:
    """
    Provide explanations for demand forecast predictions using feature importance
    and trend analysis instead of SHAP (for simplicity and interpretability)
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the forecast explainer"""
        self.data_dir = data_dir
        self.forecaster = DemandForecaster(data_dir=data_dir)
        
        logger.info("Forecast explainer initialized successfully")
    
    def generate_forecast_explanation(
        self, 
        store_id: str, 
        item_id: str,
        forecast_weeks: List[int] = [1, 2]
    ) -> Dict:
        """
        Generate detailed explanations for forecast predictions
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            forecast_weeks: Weeks to forecast and explain
            
        Returns:
            Dictionary with detailed explanations
        """
        try:
            # Generate forecast first
            forecast_result = self.forecaster.generate_forecast(store_id, item_id, forecast_weeks)
            
            if "error" in forecast_result:
                return {"error": forecast_result["error"]}
            
            # Get sales data for analysis
            sales_data = self.forecaster._prepare_daily_sales(store_id, item_id)
            
            if sales_data.empty:
                return {"error": "No sales data available for explanation"}
            
            # Analyze data patterns
            pattern_analysis = self._analyze_data_patterns(sales_data)
            
            # Generate explanations for each forecast period
            explanations = {}
            
            for weeks in forecast_weeks:
                period_key = f"{weeks}_week"
                if period_key in forecast_result.get("forecasts", {}):
                    forecast_info = forecast_result["forecasts"][period_key]
                    
                    explanation = self._explain_forecast_period(
                        sales_data,
                        forecast_info,
                        pattern_analysis,
                        weeks
                    )
                    
                    explanations[period_key] = explanation
            
            return {
                "store_id": store_id,
                "item_id": item_id,
                "pattern_analysis": pattern_analysis,
                "forecast_explanations": explanations,
                "methodology_summary": self._get_methodology_summary()
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast explanation: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_data_patterns(self, sales_data: pd.DataFrame) -> Dict:
        """Analyze historical sales patterns for explanation context"""
        try:
            sales_values = sales_data['Sales'].values
            
            analysis = {
                "total_data_points": len(sales_values),
                "sales_statistics": {},
                "trend_analysis": {},
                "volatility_analysis": {},
                "pattern_characteristics": {}
            }
            
            # Sales statistics
            analysis["sales_statistics"] = {
                "mean": float(np.mean(sales_values)),
                "median": float(np.median(sales_values)),
                "std": float(np.std(sales_values)),
                "min": float(np.min(sales_values)),
                "max": float(np.max(sales_values)),
                "zero_days": int((sales_values == 0).sum()),
                "zero_percentage": float((sales_values == 0).mean() * 100)
            }
            
            # Recent vs historical comparison
            if len(sales_values) >= 14:
                recent_sales = sales_values[-7:]
                prev_week_sales = sales_values[-14:-7] if len(sales_values) >= 14 else sales_values[-7:]
                historical_avg = np.mean(sales_values[:-7]) if len(sales_values) > 7 else np.mean(sales_values)
                
                analysis["trend_analysis"] = {
                    "recent_week_avg": float(np.mean(recent_sales)),
                    "previous_week_avg": float(np.mean(prev_week_sales)),
                    "historical_avg": float(historical_avg),
                    "week_over_week_change": float(np.mean(recent_sales) - np.mean(prev_week_sales)),
                    "recent_vs_historical": float(np.mean(recent_sales) - historical_avg),
                    "trend_direction": "increasing" if np.mean(recent_sales) > np.mean(prev_week_sales) else "decreasing"
                }
            
            # Volatility analysis
            if len(sales_values) >= 7:
                recent_volatility = np.std(sales_values[-7:])
                overall_volatility = np.std(sales_values)
                
                analysis["volatility_analysis"] = {
                    "recent_volatility": float(recent_volatility),
                    "overall_volatility": float(overall_volatility),
                    "volatility_ratio": float(recent_volatility / (overall_volatility + 0.001)),
                    "stability_assessment": "stable" if recent_volatility < overall_volatility * 0.8 else "volatile"
                }
            
            # Pattern characteristics
            analysis["pattern_characteristics"] = {
                "sparsity_level": "high" if analysis["sales_statistics"]["zero_percentage"] > 50 else 
                                 "medium" if analysis["sales_statistics"]["zero_percentage"] > 20 else "low",
                "sales_consistency": "consistent" if analysis["sales_statistics"]["std"] < analysis["sales_statistics"]["mean"] else "inconsistent",
                "demand_level": "high" if analysis["sales_statistics"]["mean"] > 10 else 
                               "medium" if analysis["sales_statistics"]["mean"] > 2 else "low"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing data patterns: {str(e)}")
            return {"error": str(e)}
    
    def _explain_forecast_period(
        self, 
        sales_data: pd.DataFrame,
        forecast_info: Dict,
        pattern_analysis: Dict,
        weeks: int
    ) -> Dict:
        """Generate detailed explanation for a specific forecast period"""
        try:
            explanation = {
                "period_weeks": weeks,
                "predicted_total": forecast_info.get("total_predicted", 0),
                "practical_total": forecast_info.get("total_practical", 0),
                "confidence": forecast_info.get("confidence", 0),
                "key_factors": [],
                "reasoning_breakdown": {},
                "risk_assessment": {}
            }
            
            # Extract key factors that influenced the forecast
            key_factors = []
            
            # Factor 1: Recent demand trend
            trend_analysis = pattern_analysis.get("trend_analysis", {})
            if trend_analysis:
                recent_avg = trend_analysis.get("recent_week_avg", 0)
                historical_avg = trend_analysis.get("historical_avg", 0)
                trend_direction = trend_analysis.get("trend_direction", "stable")
                
                if abs(recent_avg - historical_avg) > historical_avg * 0.2:
                    if recent_avg > historical_avg:
                        factor = f"Recent demand increase: current week averaging {recent_avg:.1f} units vs historical {historical_avg:.1f}"
                    else:
                        factor = f"Recent demand decrease: current week averaging {recent_avg:.1f} units vs historical {historical_avg:.1f}"
                    key_factors.append(factor)
                else:
                    key_factors.append(f"Stable demand pattern: recent average {recent_avg:.1f} units aligns with historical trends")
            
            # Factor 2: Data volatility impact
            volatility = pattern_analysis.get("volatility_analysis", {})
            if volatility:
                stability = volatility.get("stability_assessment", "unknown")
                if stability == "volatile":
                    key_factors.append("High recent volatility detected - forecast adjusted with additional uncertainty")
                else:
                    key_factors.append("Stable sales pattern - forecast based on consistent historical behavior")
            
            # Factor 3: Sparsity considerations
            characteristics = pattern_analysis.get("pattern_characteristics", {})
            sparsity = characteristics.get("sparsity_level", "unknown")
            if sparsity == "high":
                zero_pct = pattern_analysis.get("sales_statistics", {}).get("zero_percentage", 0)
                key_factors.append(f"High sparsity pattern ({zero_pct:.0f}% zero sales days) - forecast maintains sparse characteristics")
            elif sparsity == "medium":
                key_factors.append("Moderate sparsity in sales - forecast accounts for intermittent demand pattern")
            
            # Factor 4: Seasonal or cyclical patterns
            # This could be enhanced with actual seasonality detection
            key_factors.append("Local trajectory weighting applied - recent 1-2 weeks heavily weighted in forecast")
            
            explanation["key_factors"] = key_factors
            
            # Reasoning breakdown with percentages
            explanation["reasoning_breakdown"] = {
                "recent_trend_weight": 40,  # Recent 1-2 weeks
                "historical_average_weight": 25,  # Longer term average
                "pattern_preservation_weight": 20,  # Maintaining sparsity/consistency
                "constraint_adjustment_weight": 15,  # 30% deviation limits
                "description": "Forecast combines recent trends (40%), historical patterns (25%), pattern preservation (20%), and constraint adjustments (15%)"
            }
            
            # Risk assessment
            confidence = explanation["confidence"]
            
            if confidence >= 0.8:
                risk_level = "Low"
                risk_explanation = "High confidence due to consistent data patterns and sufficient historical information"
            elif confidence >= 0.6:
                risk_level = "Medium" 
                risk_explanation = "Moderate confidence with some uncertainty from recent volatility or data limitations"
            else:
                risk_level = "High"
                risk_explanation = "Lower confidence due to volatile patterns, sparse data, or limited historical information"
            
            explanation["risk_assessment"] = {
                "risk_level": risk_level,
                "confidence_score": confidence,
                "explanation": risk_explanation,
                "recommendation": self._generate_recommendation(risk_level, explanation["predicted_total"], explanation["practical_total"])
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining forecast period: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendation(self, risk_level: str, predicted: float, practical: int) -> str:
        """Generate actionable recommendations based on forecast and risk"""
        recommendations = []
        
        if risk_level == "High":
            recommendations.append("Consider ordering conservatively due to forecast uncertainty")
            recommendations.append("Monitor actual sales closely for pattern changes")
        elif risk_level == "Medium":
            recommendations.append("Standard ordering approach with regular monitoring")
        else:
            recommendations.append("Forecast is reliable for planning purposes")
        
        if practical > predicted * 1.5:
            recommendations.append(f"Practical order ({practical}) significantly exceeds prediction ({predicted:.1f}) due to minimum order requirements")
        elif practical < predicted * 0.7:
            recommendations.append(f"Practical order ({practical}) is below prediction ({predicted:.1f}) - consider if sufficient")
        
        return " | ".join(recommendations)
    
    def _get_methodology_summary(self) -> Dict:
        """Return summary of forecasting methodology for explanations"""
        return {
            "primary_method": "Exponential Smoothing with Local Trajectory Weighting",
            "key_principles": [
                "Recent 1-2 weeks weighted more heavily than older data",
                "Maximum 30% deviation from recent averages enforced", 
                "Pattern preservation (sparse → sparse, seasonal → seasonal)",
                "Trend dampening towards historical mean",
                "Purchase quantity optimization for practical ordering"
            ],
            "constraint_details": {
                "max_deviation": "30% from recent average sales",
                "local_weighting_period": "14 days (heavily weighted)",
                "minimum_data_requirement": "14 days of historical sales",
                "trend_dampening": "Applied to prevent excessive extrapolation"
            }
        }
    
    def create_explanation_visualization(
        self, 
        store_id: str, 
        item_id: str,
        explanation_result: Dict
    ) -> go.Figure:
        """Create visualization showing explanation factors"""
        try:
            # Get pattern analysis
            pattern_analysis = explanation_result.get("pattern_analysis", {})
            explanations = explanation_result.get("forecast_explanations", {})
            
            # Create subplots for different explanation aspects
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Sales Pattern Analysis", 
                    "Forecast Reasoning Breakdown",
                    "Risk Assessment by Period",
                    "Key Influencing Factors"
                ),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "table"}]]
            )
            
            # 1. Sales pattern analysis (bar chart)
            stats = pattern_analysis.get("sales_statistics", {})
            trend = pattern_analysis.get("trend_analysis", {})
            
            if stats and trend:
                categories = ["Recent Week", "Previous Week", "Historical Avg", "Overall Max"]
                values = [
                    trend.get("recent_week_avg", 0),
                    trend.get("previous_week_avg", 0), 
                    trend.get("historical_avg", 0),
                    stats.get("max", 0)
                ]
                
                fig.add_trace(
                    go.Bar(x=categories, y=values, name="Sales Levels"),
                    row=1, col=1
                )
            
            # 2. Reasoning breakdown (pie chart)
            if explanations:
                first_explanation = list(explanations.values())[0]
                reasoning = first_explanation.get("reasoning_breakdown", {})
                
                if reasoning:
                    labels = ["Recent Trend", "Historical Avg", "Pattern Preservation", "Constraint Adj"]
                    values = [
                        reasoning.get("recent_trend_weight", 0),
                        reasoning.get("historical_average_weight", 0),
                        reasoning.get("pattern_preservation_weight", 0),
                        reasoning.get("constraint_adjustment_weight", 0)
                    ]
                    
                    fig.add_trace(
                        go.Pie(labels=labels, values=values, name="Reasoning"),
                        row=1, col=2
                    )
            
            # 3. Risk assessment (scatter plot)
            periods = []
            confidences = []
            predictions = []
            risk_colors = []
            
            for period, explanation in explanations.items():
                periods.append(period.replace('_', ' ').title())
                confidences.append(explanation.get("confidence", 0))
                predictions.append(explanation.get("predicted_total", 0))
                
                risk = explanation.get("risk_assessment", {}).get("risk_level", "Medium")
                risk_colors.append("green" if risk == "Low" else "orange" if risk == "Medium" else "red")
            
            if periods:
                fig.add_trace(
                    go.Scatter(
                        x=periods,
                        y=confidences,
                        mode='markers',
                        marker=dict(
                            size=[p*5 + 10 for p in predictions],  # Size based on prediction
                            color=risk_colors,
                            opacity=0.7
                        ),
                        name="Risk vs Confidence"
                    ),
                    row=2, col=1
                )
            
            # 4. Key factors table
            if explanations:
                first_explanation = list(explanations.values())[0]
                factors = first_explanation.get("key_factors", [])
                
                if factors:
                    fig.add_trace(
                        go.Table(
                            header=dict(values=["Key Influencing Factors"]),
                            cells=dict(values=[factors[:4]])  # Show top 4 factors
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title=f"Forecast Explanation Analysis - Store {store_id}, Item {item_id}",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating explanation visualization: {str(e)}")
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(title="Error creating explanation visualization")
            return fig

if __name__ == "__main__":
    try:
        # Initialize explainer
        explainer = ForecastExplainer(data_dir="data")
        
        # Get list of items
        items = explainer.forecaster.get_item_list()
        logger.info(f"Found {len(items)} unique store-item combinations")
        
        # Test explanation with first item
        if items:
            store_id, item_id, desc = items[0]
            logger.info(f"Generating explanation for: Store {store_id}, Item {item_id} ({desc})")
            
            # Generate explanation
            explanation = explainer.generate_forecast_explanation(store_id, item_id, [1, 2])
            
            if "error" not in explanation:
                # Print key insights
                print("\n" + "="*80)
                print("FORECAST EXPLANATION SUMMARY")
                print("="*80)
                
                for period, exp in explanation.get("forecast_explanations", {}).items():
                    print(f"\n{period.replace('_', ' ').title()} Forecast:")
                    print(f"  Predicted: {exp.get('predicted_total', 0):.2f} units")
                    print(f"  Practical: {exp.get('practical_total', 0)} units")
                    print(f"  Confidence: {exp.get('confidence', 0):.0%}")
                    print(f"  Risk Level: {exp.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
                    
                    print("\n  Key Factors:")
                    for factor in exp.get("key_factors", [])[:3]:
                        print(f"    • {factor}")
                    
                    print(f"\n  Recommendation: {exp.get('risk_assessment', {}).get('recommendation', 'N/A')}")
                
                print("\n" + "="*80)
                
                # Create explanation visualization
                fig = explainer.create_explanation_visualization(store_id, item_id, explanation)
                
                # Save visualization
                output_file = f"output/explanation_{store_id}_{item_id}.html"
                fig.write_html(output_file)
                logger.info(f"Explanation visualization saved to {output_file}")
            else:
                logger.error(f"Explanation error: {explanation['error']}")
        
        logger.info("Forecast explanation testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")