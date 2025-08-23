#!/usr/bin/env python3
import os
import logging
import pandas as pd
import argparse
from typing import List, Tuple, Dict, Optional, Any
from item_data_visualizer import ItemDataVisualizer
from seasonality_analyzer import ItemSeasonalityAnalyzer
from model_comparator import ModelComparator
from error_handler import ErrorHandler
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VisualizationOrchestrator:
    """Orchestrates the visualization process for items"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the orchestrator with data directory path"""
        self.data_dir = data_dir
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Initialize components
        try:
            self.visualizer = ItemDataVisualizer(data_dir=data_dir)
            self.seasonality_analyzer = ItemSeasonalityAnalyzer(data_dir=data_dir)
            self.model_comparator = ModelComparator(data_dir=data_dir)
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def get_item_list(self) -> List[Tuple[str, str, str]]:
        """Return a list of unique items with store_id, item_id and description"""
        try:
            return self.visualizer.get_item_list()
        except Exception as e:
            logger.error(f"Error getting item list: {str(e)}")
            return []
    
    @ErrorHandler.handle_visualization_errors
    def process_item(
        self, 
        store_id: str, 
        item_id: str,
        generate_sales_viz: bool = True,
        generate_seasonality_viz: bool = True,
        generate_sales_analysis: bool = True,
        generate_inventory_analysis: bool = True
    ) -> Dict[str, Any]:
        """Process a single item to generate all visualizations"""
        results = {}
        
        try:
            # Get item description
            items = self.visualizer.get_item_list()
            item_desc = next((desc for s, i, desc in items if s == store_id and i == item_id), f"Item {item_id}")
            
            logger.info(f"Processing {item_desc} (Store {store_id}, Item {item_id})")
            
            # Generate sales and purchase visualization
            if generate_sales_viz:
                try:
                    sales_viz = self.visualizer.visualize_item_sales_purchases(
                        store_id,
                        item_id,
                        output_file=f"sales_purchases_{store_id}_{item_id}"
                    )
                    results['sales_visualization'] = sales_viz is not None
                except Exception as e:
                    logger.error(f"Error generating sales visualization: {str(e)}")
                    results['sales_visualization'] = False
            
            # Generate seasonality visualization
            if generate_seasonality_viz:
                try:
                    # First detect seasonality
                    seasonality = self.seasonality_analyzer.detect_seasonality(store_id, item_id)
                    results['seasonality'] = seasonality
                    
                    # If seasonality detected, create visualization
                    if seasonality.get('has_seasonality', False):
                        seasonality_viz = self.seasonality_analyzer.visualize_seasonality(
                            store_id,
                            item_id,
                            output_file=f"seasonality_{store_id}_{item_id}"
                        )
                        results['seasonality_visualization'] = seasonality_viz is not None
                    else:
                        results['seasonality_visualization'] = False
                except Exception as e:
                    logger.error(f"Error generating seasonality visualization: {str(e)}")
                    results['seasonality_visualization'] = False
            
            # Generate sales analysis with time periods
            if generate_sales_analysis:
                try:
                    # Generate 3-month sales analysis
                    sales_analysis_3m = self.model_comparator.create_sales_analysis(
                        store_id,
                        item_id,
                        period="3_months",
                        with_seasonality=True,
                        output_file=f"sales_analysis_3m_{store_id}_{item_id}"
                    )
                    results['sales_analysis_3m'] = sales_analysis_3m is not None
                    
                    # Generate all-time sales analysis
                    sales_analysis_all = self.model_comparator.create_sales_analysis(
                        store_id,
                        item_id,
                        period="all",
                        with_seasonality=True,
                        output_file=f"sales_analysis_all_{store_id}_{item_id}"
                    )
                    results['sales_analysis_all'] = sales_analysis_all is not None
                except Exception as e:
                    logger.error(f"Error generating sales analysis: {str(e)}")
                    results['sales_analysis'] = False
            
            # Generate inventory analysis
            if generate_inventory_analysis:
                try:
                    inventory_analysis = self.model_comparator.create_inventory_analysis(
                        store_id,
                        item_id,
                        lookback_days=30,
                        output_file=f"inventory_{store_id}_{item_id}"
                    )
                    results['inventory_analysis'] = inventory_analysis is not None
                except Exception as e:
                    logger.error(f"Error generating inventory analysis: {str(e)}")
                    results['inventory_analysis'] = False
            
            logger.info(f"Completed processing for {item_desc}")
            
        except Exception as e:
            logger.error(f"Error processing item {store_id}_{item_id}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    @ErrorHandler.handle_visualization_errors
    def process_all_items(
        self,
        limit: Optional[int] = None,
        store_filter: Optional[str] = None,
        item_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process all items to generate visualizations"""
        results = {}
        
        try:
            # Get all items
            items = self.get_item_list()
            logger.info(f"Found {len(items)} unique store-item combinations")
            
            # Apply filters
            if store_filter:
                items = [item for item in items if item[0] == store_filter]
                logger.info(f"Filtered to {len(items)} items for store {store_filter}")
                
            if item_filter:
                items = [item for item in items if item[1] == item_filter]
                logger.info(f"Filtered to {len(items)} items with item ID {item_filter}")
            
            # Apply limit
            if limit and limit > 0:
                items = items[:limit]
                logger.info(f"Limited to first {limit} items")
            
            # Process each item
            for i, (store_id, item_id, desc) in enumerate(items):
                logger.info(f"Processing {i+1}/{len(items)}: Store {store_id}, Item {item_id} ({desc})")
                
                item_results = self.process_item(store_id, item_id)
                results[f"{store_id}_{item_id}"] = item_results
            
            logger.info(f"Completed processing {len(items)} items")
            
        except Exception as e:
            logger.error(f"Error processing items: {str(e)}")
            results['error'] = str(e)
        
        return results

    @ErrorHandler.handle_visualization_errors
    def generate_summary_report(self, results: Dict[str, Any], output_file: str = "summary_report.html") -> None:
        """Generate a summary report of all visualizations"""
        try:
            # Import visualization libraries here to avoid circular imports
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create output HTML file
            with open(f"output/{output_file}", "w") as f:
                # Write HTML header
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Sales and Forecast Visualization Summary</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #2c3e50; }
                        h2 { color: #3498db; }
                        .item-section { margin-bottom: 30px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                        .visualization-group { display: flex; flex-wrap: wrap; gap: 20px; }
                        .visualization-card { margin-bottom: 20px; border: 1px solid #eee; border-radius: 5px; padding: 10px; }
                        .status-success { color: green; }
                        .status-failure { color: red; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        .metrics-table { margin-top: 10px; margin-bottom: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Sales and Visualization Summary</h1>
                """)
                
                # Write summary of results
                total_items = len(results)
                if 'error' in results:
                    total_items -= 1
                
                f.write(f"<p>Generated visualizations for {total_items} items</p>")
                
                # Write timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"<p>Generated on: {timestamp}</p>")
                
                # Write details for each item
                for key, item_results in results.items():
                    if key == 'error':
                        continue
                    
                    # Extract store and item IDs
                    store_id, item_id = key.split('_')
                    
                    # Get item description
                    items = self.get_item_list()
                    item_desc = next((desc for s, i, desc in items if s == store_id and i == item_id), f"Item {item_id}")
                    
                    f.write(f"""
                    <div class="item-section">
                        <h2>{item_desc} (Store {store_id}, Item {item_id})</h2>
                    """)
                    
                    # Write status for each visualization type
                    f.write("<h3>Generated Visualizations:</h3><ul>")
                    
                    if 'sales_visualization' in item_results:
                        status = "success" if item_results['sales_visualization'] else "failure"
                        f.write(f'<li>Sales and Purchases: <span class="status-{status}">{status.upper()}</span>')
                        if status == "success":
                            f.write(f' - <a href="sales_purchases_{store_id}_{item_id}.html" target="_blank">View</a>')
                        f.write('</li>')
                    
                    if 'seasonality_visualization' in item_results:
                        status = "success" if item_results['seasonality_visualization'] else "failure"
                        has_seasonality = item_results.get('seasonality', {}).get('has_seasonality', False)
                        if has_seasonality:
                            period = item_results.get('seasonality', {}).get('best_period', 'unknown')
                            f.write(f'<li>Seasonality Analysis ({period}-day pattern): <span class="status-{status}">{status.upper()}</span>')
                            if status == "success":
                                f.write(f' - <a href="seasonality_{store_id}_{item_id}.html" target="_blank">View</a>')
                            f.write('</li>')
                        else:
                            f.write(f'<li>Seasonality Analysis: No significant seasonality detected</li>')
                    
                    if 'sales_analysis_3m' in item_results:
                        status = "success" if item_results['sales_analysis_3m'] else "failure"
                        f.write(f'<li>3-Month Sales Analysis: <span class="status-{status}">{status.upper()}</span>')
                        if status == "success":
                            f.write(f' - <a href="sales_analysis_3m_{store_id}_{item_id}.html" target="_blank">View</a>')
                        f.write('</li>')
                    
                    if 'sales_analysis_all' in item_results:
                        status = "success" if item_results['sales_analysis_all'] else "failure"
                        f.write(f'<li>All-Time Sales Analysis: <span class="status-{status}">{status.upper()}</span>')
                        if status == "success":
                            f.write(f' - <a href="sales_analysis_all_{store_id}_{item_id}.html" target="_blank">View</a>')
                        f.write('</li>')
                    
                    if 'inventory_analysis' in item_results:
                        status = "success" if item_results['inventory_analysis'] else "failure"
                        f.write(f'<li>Inventory Analysis: <span class="status-{status}">{status.upper()}</span>')
                        if status == "success":
                            f.write(f' - <a href="inventory_{store_id}_{item_id}.html" target="_blank">View</a>')
                        f.write('</li>')
                    
                    f.write("</ul>")
                    
                    # Add link to Dash app
                    f.write(f"""
                    <p>View interactive visualizations for this item in the 
                    <a href="/dash/" target="_blank">Interactive Dashboard</a>.</p>
                    """)
                    
                    # Close item section
                    f.write("</div>")
                
                # Write HTML footer
                f.write("""
                </body>
                </html>
                """)
                
            logger.info(f"Summary report generated at output/{output_file}")
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")

def main():
    """Main entry point for the visualization tool"""
    parser = argparse.ArgumentParser(description='Generate visualizations for sales and forecast data')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--limit', type=int, default=5, help='Limit number of items to process')
    parser.add_argument('--store', type=str, help='Filter by store ID')
    parser.add_argument('--item', type=str, help='Filter by item ID')
    parser.add_argument('--output', type=str, default='summary_report.html', help='Output summary report filename')
    parser.add_argument('--run-app', action='store_true', help='Run the interactive dashboard after generating visualizations')
    
    args = parser.parse_args()
    
    try:
        # Initialize the orchestrator
        orchestrator = VisualizationOrchestrator(data_dir=args.data_dir)
        
        # Process items
        results = orchestrator.process_all_items(
            limit=args.limit,
            store_filter=args.store,
            item_filter=args.item
        )
        
        # Generate summary report
        orchestrator.generate_summary_report(results, output_file=args.output)
        
        logger.info("Visualization process completed successfully")
        print(f"Visualizations generated successfully. See output/{args.output} for summary.")
        
        # Run the interactive dashboard if requested
        if args.run_app:
            print("Starting interactive dashboard...")
            import subprocess
            subprocess.Popen(["python", "app.py"])
            print("Dashboard available at http://localhost:5000/")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())