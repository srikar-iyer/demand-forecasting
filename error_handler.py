#!/usr/bin/env python3
import os
import logging
import traceback
from functools import wraps
from typing import Callable, Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing/error_handling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Class for handling errors in the visualization and analysis pipeline"""
    
    @staticmethod
    def handle_data_errors(func: Callable) -> Callable:
        """
        Decorator for handling data loading and processing errors
        
        Args:
            func: The function to wrap with error handling
            
        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.error(f"Data file not found: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                return None
            except pd.errors.EmptyDataError as e:
                logger.error(f"Empty data file: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                return None
            except pd.errors.ParserError as e:
                logger.error(f"Data parsing error: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                return None
        return wrapper
    
    @staticmethod
    def handle_visualization_errors(func: Callable) -> Callable:
        """
        Decorator for handling visualization errors
        
        Args:
            func: The function to wrap with error handling
            
        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                logger.error(f"Visualization value error: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return a fallback visualization
                return ErrorHandler.create_error_plot(
                    error_message=str(e),
                    title=f"Error in {func.__name__}",
                    subtitle="Invalid data provided for visualization"
                )
            except KeyError as e:
                logger.error(f"Visualization key error: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return a fallback visualization
                return ErrorHandler.create_error_plot(
                    error_message=str(e),
                    title=f"Error in {func.__name__}",
                    subtitle="Missing data fields required for visualization"
                )
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return a fallback visualization
                return ErrorHandler.create_error_plot(
                    error_message=str(e),
                    title=f"Error in {func.__name__}",
                    subtitle="An unexpected error occurred"
                )
        return wrapper
    
    @staticmethod
    def handle_seasonality_errors(func: Callable) -> Callable:
        """
        Decorator for handling seasonality detection and processing errors
        
        Args:
            func: The function to wrap with error handling
            
        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                logger.error(f"Seasonality value error: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return a default response indicating no seasonality
                return {"has_seasonality": False, "error": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return a default response indicating no seasonality
                return {"has_seasonality": False, "error": str(e)}
        return wrapper
    
    @staticmethod
    def handle_model_comparison_errors(func: Callable) -> Callable:
        """
        Decorator for handling model comparison errors
        
        Args:
            func: The function to wrap with error handling
            
        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError as e:
                logger.error(f"Model comparison key error: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return an empty dict or a fallback visualization based on context
                if 'output_file' in kwargs:
                    return ErrorHandler.create_error_plot(
                        error_message=str(e),
                        title=f"Error in {func.__name__}",
                        subtitle="Missing data fields required for comparison"
                    )
                return {}
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                logger.info(f"Traceback: {traceback.format_exc()}")
                # Return an empty dict or a fallback visualization based on context
                if 'output_file' in kwargs:
                    return ErrorHandler.create_error_plot(
                        error_message=str(e),
                        title=f"Error in {func.__name__}",
                        subtitle="An unexpected error occurred"
                    )
                return {}
        return wrapper
    
    @staticmethod
    def create_error_plot(
        error_message: str,
        title: str = "Error",
        subtitle: str = "An error occurred",
        output_file: Optional[str] = None
    ) -> go.Figure:
        """
        Create a fallback visualization for error cases
        
        Args:
            error_message: The error message to display
            title: Title for the error plot
            subtitle: Subtitle with error category
            output_file: If provided, save the figure to this file
            
        Returns:
            Plotly figure object with error information
        """
        try:
            # Create a simple figure
            fig = go.Figure()
            
            # Add an annotation with the error message
            fig.add_annotation(
                text=f"{subtitle}:<br>{error_message}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                ),
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                height=400
            )
            
            # Save the figure if output file is provided
            if output_file:
                try:
                    os.makedirs("testing/output", exist_ok=True)
                    fig.write_html(f"testing/output/error_{output_file}.html")
                    fig.write_image(f"testing/output/error_{output_file}.png")
                    logger.info(f"Error figure saved to testing/output/error_{output_file}.html and .png")
                except Exception as e:
                    logger.error(f"Error saving error figure: {str(e)}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in create_error_plot: {str(e)}")
            # Create an extremely simple figure as last resort
            fig = go.Figure()
            fig.update_layout(
                title="Visualization Error",
                annotations=[
                    dict(
                        text="Error creating visualization",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="red")
                    )
                ]
            )
            return fig
    
    @staticmethod
    def validate_data(
        data: Union[pd.DataFrame, None],
        required_columns: List[str],
        min_rows: int = 1
    ) -> Tuple[bool, str]:
        """
        Validate that a DataFrame meets requirements
        
        Args:
            data: DataFrame to validate
            required_columns: List of columns that must be present
            min_rows: Minimum number of rows required
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if data is None:
            return False, "Data is None"
        
        if not isinstance(data, pd.DataFrame):
            return False, f"Expected DataFrame, got {type(data)}"
        
        if len(data) < min_rows:
            return False, f"DataFrame has {len(data)} rows, minimum required is {min_rows}"
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        return True, "Data is valid"
    
    @staticmethod
    def safe_file_path(base_path: str, filename: str) -> str:
        """
        Create a safe file path, ensuring the directory exists
        
        Args:
            base_path: Base directory path
            filename: Filename to append
            
        Returns:
            Safe file path
        """
        try:
            # Ensure base directory exists
            os.makedirs(base_path, exist_ok=True)
            
            # Clean filename of problematic characters
            safe_name = filename.replace('/', '_').replace('\\', '_')
            
            return os.path.join(base_path, safe_name)
        except Exception as e:
            logger.error(f"Error creating safe file path: {str(e)}")
            return os.path.join("testing/output", "error_output.html")

class ErrorHandlingDemo:
    """Demo class to show how to use the error handlers"""
    
    @ErrorHandler.handle_data_errors
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Demo data loading function with error handling"""
        return pd.read_csv(file_path)
    
    @ErrorHandler.handle_visualization_errors
    def create_visualization(self, data: pd.DataFrame, x_column: str, y_column: str) -> go.Figure:
        """Demo visualization function with error handling"""
        # This will raise a KeyError if columns don't exist
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='lines'))
        return fig
    
    @ErrorHandler.handle_seasonality_errors
    def detect_seasonality(self, data: pd.DataFrame, value_column: str) -> Dict:
        """Demo seasonality detection with error handling"""
        # This will raise a ValueError if data is invalid
        if len(data) < 10:
            raise ValueError("Insufficient data for seasonality detection")
        
        # Dummy result
        return {"has_seasonality": True, "period": 7}

if __name__ == "__main__":
    # Demo of error handling
    demo = ErrorHandlingDemo()
    
    # Test data loading error handling
    result = demo.load_data("nonexistent_file.csv")
    print(f"Load data result: {result}")
    
    # Test visualization error handling
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 15, 25, 30]
    })
    fig = demo.create_visualization(data, 'x', 'missing_column')
    print(f"Visualization result: {type(fig)}")
    
    # Test seasonality error handling
    seasonality = demo.detect_seasonality(data[:3], 'y')
    print(f"Seasonality result: {seasonality}")
    
    # Test error plot creation
    error_fig = ErrorHandler.create_error_plot(
        "Example error message",
        title="Error Demo",
        subtitle="This is a demonstration of the error handling system",
        output_file="error_demo"
    )
    print(f"Error figure created: {type(error_fig)}")