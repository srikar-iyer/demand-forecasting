import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gradio as gr

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(days=365):
    """Generate smaller sample dataset for demo purposes"""
    
    # Define products
    products = ['Potatoes', 'Vegetables', 'Baby Products', 'Fruit', 'Milk']
    
    # Generate dates for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize dataframe
    data = []
    
    # Define holidays and events (simplified)
    holidays = {
        '01-01': 'New Year',
        '02-14': 'Valentine', 
        '07-04': 'Independence Day',
        '11-25': 'Thanksgiving',
        '12-25': 'Christmas'
    }
    
    # Generate data for each product
    for product in products:
        # Base demand parameters
        if product == 'Potatoes':
            base_demand = 500
            seasonal_amplitude = 150
            weekly_pattern = [1.0, 0.9, 0.8, 0.8, 1.1, 1.3, 1.4]  # Higher on weekends
            holiday_factor = 1.7
            price_range = (0.8, 1.5)
            
        elif product == 'Vegetables':
            base_demand = 800
            seasonal_amplitude = 200
            weekly_pattern = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3]
            holiday_factor = 1.5
            price_range = (1.0, 2.0)
            
        elif product == 'Baby Products':
            base_demand = 300
            seasonal_amplitude = 50
            weekly_pattern = [0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.0]
            holiday_factor = 1.1
            price_range = (5.0, 20.0)
            
        elif product == 'Fruit':
            base_demand = 600
            seasonal_amplitude = 250
            weekly_pattern = [1.0, 0.9, 0.8, 0.9, 1.2, 1.4, 1.3]
            holiday_factor = 1.4
            price_range = (1.5, 4.0)
            
        else:  # Milk
            base_demand = 450
            seasonal_amplitude = 50
            weekly_pattern = [1.1, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3]
            holiday_factor = 1.3
            price_range = (2.5, 4.5)
        
        # Generate daily sales for this product
        for i, date in enumerate(date_range):
            # Basic seasonality (annual)
            day_of_year = date.dayofyear
            seasonality = np.sin(day_of_year / 365 * 2 * np.pi) * seasonal_amplitude
            
            # Weekly pattern
            day_of_week = date.weekday()
            weekly_factor = weekly_pattern[day_of_week]
            
            # Holiday effect
            holiday_effect = 1.0
            month_day = f"{date.month:02d}-{date.day:02d}"
            if month_day in holidays:
                holiday_effect = holiday_factor
                holiday_name = holidays[month_day]
            else:
                holiday_name = None
            
            # Weather effect (simulate some random weather events)
            if np.random.random() < 0.03:  # 3% chance of adverse weather
                weather = np.random.choice(['Heavy Rain', 'Snow', 'Storm'])
                weather_effect = 0.7  # Reduce sales during bad weather
            else:
                weather = 'Normal'
                weather_effect = 1.0
                
                # Seasonal weather patterns
                if date.month in [12, 1, 2]:  # Winter
                    if product == 'Fruit':
                        weather_effect *= 0.9
                elif date.month in [6, 7, 8]:  # Summer
                    if product == 'Fruit':
                        weather_effect *= 1.2
            
            # Promotional effect
            if np.random.random() < 0.1:  # 10% chance of promotion
                promotion = True
                promotion_effect = np.random.uniform(1.2, 1.6)
            else:
                promotion = False
                promotion_effect = 1.0
            
            # Price calculation with some randomness
            base_price = np.random.uniform(price_range[0], price_range[1])
            if promotion:
                price = base_price * 0.8  # 20% discount
            else:
                price = base_price
            
            # Calculate final demand
            demand = (base_demand + seasonality) * weekly_factor * holiday_effect * weather_effect * promotion_effect
            
            # Add noise
            noise = np.random.normal(0, demand * 0.1)  # 10% noise
            final_sales = max(0, int(demand + noise))
            
            # Lead time in days (how far in advance orders were placed)
            lead_time = int(np.random.normal(7, 2))
            lead_time = max(1, lead_time)  # Minimum 1 day
            
            # Record the data
            data.append({
                'Date': date,
                'Product': product,
                'Sales': final_sales,
                'Price': price,
                'Promotion': promotion,
                'Weather': weather,
                'Lead_Time': lead_time,
                'Day_Of_Week': date.dayofweek,
                'Month': date.month,
                'Year': date.year,
                'Is_Holiday': 1 if month_day in holidays else 0,
                'Holiday_Name': holiday_name
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def prepare_features(df):
    """Prepare features for the forecasting model"""
    
    # Create copy to avoid modifying the original
    df_features = df.copy()
    
    # Convert Date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
        df_features['Date'] = pd.to_datetime(df_features['Date'])
    
    # One-hot encode product
    product_dummies = pd.get_dummies(df_features['Product'], prefix='Product')
    df_features = pd.concat([df_features, product_dummies], axis=1)
    
    # One-hot encode weather
    weather_dummies = pd.get_dummies(df_features['Weather'], prefix='Weather')
    df_features = pd.concat([df_features, weather_dummies], axis=1)
    
    # Cyclical encoding for day of week, month
    df_features['Day_Sin'] = np.sin(df_features['Day_Of_Week'] * (2 * np.pi / 7))
    df_features['Day_Cos'] = np.cos(df_features['Day_Of_Week'] * (2 * np.pi / 7))
    df_features['Month_Sin'] = np.sin(df_features['Month'] * (2 * np.pi / 12))
    df_features['Month_Cos'] = np.cos(df_features['Month'] * (2 * np.pi / 12))
    
    # Add lag features (sales from previous days) - simplified for the standalone version
    for product in df['Product'].unique():
        product_data = df[df['Product'] == product].sort_values('Date')
        
        # Create 7 and 14 day lags
        for lag in [1, 7]:
            lag_col = f'Sales_Lag_{lag}'
            product_data[lag_col] = product_data['Sales'].shift(lag)
            
        # Calculate rolling averages
        for window in [7]:
            avg_col = f'Sales_Avg_{window}'
            product_data[avg_col] = product_data['Sales'].rolling(window=window).mean()
            
        # Update in the main dataframe
        for col in product_data.columns:
            if col.startswith('Sales_Lag_') or col.startswith('Sales_Avg_'):
                df_features.loc[df_features['Product'] == product, col] = product_data[col].values
    
    # Fill NaN values for lag features
    for col in df_features.columns:
        if col.startswith('Sales_Lag_') or col.startswith('Sales_Avg_'):
            df_features[col] = df_features[col].fillna(df_features.groupby('Product')['Sales'].transform('mean'))
    
    # Features to use in the model
    feature_cols = [
        'Price', 'Promotion', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos',
        'Is_Holiday', 'Lead_Time'
    ]
    
    # Add product and weather dummy columns
    feature_cols.extend([col for col in df_features.columns if col.startswith('Product_')])
    feature_cols.extend([col for col in df_features.columns if col.startswith('Weather_')])
    
    # Add lag features
    feature_cols.extend([col for col in df_features.columns if col.startswith('Sales_Lag_')])
    feature_cols.extend([col for col in df_features.columns if col.startswith('Sales_Avg_')])
    
    return df_features, feature_cols

def train_forecast_model(df):
    """Train a forecasting model using the provided data"""
    
    # Prepare features
    df_features, feature_cols = prepare_features(df)
    
    # Define target
    target = 'Sales'
    
    # Split into features and target
    X = df_features[feature_cols].fillna(0)  # Fill any remaining NaNs
    y = df_features[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train a simpler Random Forest model for the standalone version
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced for speed
        max_depth=10,     # Reduced for speed
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print(f"Model trained with {len(feature_cols)} features")
    
    return model, df_features, feature_cols

def create_gradio_ui(model, df, feature_cols):
    """Create a Gradio UI for the retail forecasting model"""
    
    # Get unique products
    products = sorted(df['Product'].unique())
    
    # Get prepared dataset
    df_features, _ = prepare_features(df)
    
    # Create the prediction function
    def predict_quantity(product, price_adjustment, promotion, holiday, weather, lead_time):
        # Get a row for this product as a template
        product_data = df_features[df_features['Product'] == product].sort_values('Date', ascending=False)
        if len(product_data) == 0:
            return "Product not found in the dataset", None
        
        # Use the most recent row as a template
        sample_row = product_data.iloc[0].copy()
        
        # Calculate adjusted price based on percentage adjustment
        original_price = sample_row['Price']
        adjusted_price = original_price * (1 + price_adjustment / 100)
        
        # Update values based on inputs
        sample_row['Price'] = adjusted_price
        sample_row['Promotion'] = 1 if promotion else 0
        sample_row['Is_Holiday'] = 1 if holiday else 0
        sample_row['Lead_Time'] = lead_time
        
        # Update weather columns
        for col in feature_cols:
            if col.startswith('Weather_'):
                weather_type = col.replace('Weather_', '')
                sample_row[col] = 1 if weather_type == weather else 0
        
        # Extract only the features needed for prediction
        test_features = pd.DataFrame([sample_row[feature_cols]])
        
        # Make prediction
        predicted_sales = int(model.predict(test_features)[0])
        recommended_order = int(predicted_sales * 1.1)  # Add 10% safety stock
        
        # Create a chart showing factors influencing the decision
        plt.figure(figsize=(10, 6))
        
        # Create a bar chart showing the prediction and factors
        factors = {
            'Base Prediction': predicted_sales,
            'With Safety Stock': recommended_order
        }
        
        # Add comparison scenarios
        comparison_scenarios = []
        
        # No promotion scenario
        if promotion:
            no_promo_row = sample_row.copy()
            no_promo_row['Promotion'] = 0
            no_promo_features = pd.DataFrame([no_promo_row[feature_cols]])
            factors['Without Promotion'] = int(model.predict(no_promo_features)[0])
            comparison_scenarios.append('Without Promotion')
        
        # No holiday scenario
        if holiday:
            no_holiday_row = sample_row.copy()
            no_holiday_row['Is_Holiday'] = 0
            no_holiday_features = pd.DataFrame([no_holiday_row[feature_cols]])
            factors['Without Holiday'] = int(model.predict(no_holiday_features)[0])
            comparison_scenarios.append('Without Holiday')
        
        # Normal weather scenario
        if weather != 'Normal':
            normal_weather_row = sample_row.copy()
            for col in feature_cols:
                if col.startswith('Weather_'):
                    weather_type = col.replace('Weather_', '')
                    normal_weather_row[col] = 1 if weather_type == 'Normal' else 0
            normal_weather_features = pd.DataFrame([normal_weather_row[feature_cols]])
            factors['With Normal Weather'] = int(model.predict(normal_weather_features)[0])
            comparison_scenarios.append('With Normal Weather')
        
        # Regular price scenario
        if abs(price_adjustment) > 1:
            reg_price_row = sample_row.copy()
            reg_price_row['Price'] = original_price
            reg_price_features = pd.DataFrame([reg_price_row[feature_cols]])
            factors['With Regular Price'] = int(model.predict(reg_price_features)[0])
            comparison_scenarios.append('With Regular Price')
        
        # Plot the factors
        plt.figure(figsize=(12, 6))
        bars = plt.bar(factors.keys(), factors.values(), color=['blue', 'green'] + ['gray'] * len(comparison_scenarios))
        
        # Highlight the recommended order
        bars[1].set_color('green')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f'Predicted Sales for {product} with Decision Factors')
        plt.ylabel('Quantity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add condition information as text
        condition_text = f"Price: {'↓' if price_adjustment < 0 else '↑' if price_adjustment > 0 else '='} ({adjusted_price:.2f})\n"
        condition_text += f"Promotion: {'Yes' if promotion else 'No'}\n"
        condition_text += f"Holiday: {'Yes' if holiday else 'No'}\n"
        condition_text += f"Weather: {weather}\n"
        condition_text += f"Lead Time: {lead_time} days"
        
        plt.figtext(0.15, 0.01, condition_text, ha='left')
        
        # Add recommendation as text
        recommendation = f"Recommended Order: {recommended_order} units\n"
        recommendation += f"(Includes 10% safety stock above predicted sales of {predicted_sales})"
        
        plt.figtext(0.65, 0.01, recommendation, ha='right', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save the figure to static/images directory
        plt.savefig('static/images/prediction_chart.png')
        
        # Return both the recommendation text and the chart
        explanation = (
            f"# Prediction Results for {product}\n\n"
            f"## Recommended Order: {recommended_order} units\n\n"
            f"### Factors:\n"
            f"- Predicted Sales: {predicted_sales} units\n"
            f"- Safety Stock: {recommended_order - predicted_sales} units (10%)\n\n"
            f"### Conditions:\n"
            f"- Price: ${adjusted_price:.2f}"
            f"{' (' + str(price_adjustment) + '%)' if price_adjustment != 0 else ''}\n"
            f"- Promotion: {'Yes' if promotion else 'No'}\n"
            f"- Holiday: {'Yes' if holiday else 'No'}\n"
            f"- Weather: {weather}\n"
            f"- Lead Time: {lead_time} days\n\n"
        )
        
        return explanation, 'static/images/prediction_chart.png'
    
    # Create the interface
    demo = gr.Interface(
        fn=predict_quantity,
        inputs=[
            gr.Dropdown(products, value=products[0], label="Select Product"),
            gr.Slider(-20, 20, value=0, step=1, label="Price Adjustment (%)"),
            gr.Checkbox(label="Promotional Period"),
            gr.Checkbox(label="Holiday Period"),
            gr.Dropdown(["Normal", "Heavy Rain", "Snow", "Storm"], value="Normal", label="Weather Condition"),
            gr.Slider(1, 30, value=7, step=1, label="Lead Time (days)")
        ],
        outputs=[
            gr.Markdown(label="Prediction Results"),
            gr.Image(label="Decision Factors Chart")
        ],
        title="Retail Product Order Quantity Prediction",
        description="Predict optimal order quantities for retail products based on various factors."
    )
    
    return demo

def main():
    """Main function to run the standalone forecasting UI"""
    
    print("Generating sample retail data...")
    # Generate a smaller dataset for the standalone version (365 days instead of 730)
    df = generate_sample_data(days=365)
    
    print("Training forecasting model...")
    # Train a simpler model for the standalone version
    model, df_features, feature_cols = train_forecast_model(df)
    
    print("Launching Gradio UI...")
    demo = create_gradio_ui(model, df, feature_cols)
    #demo.launch()
    demo.launch(server_name="0.0.0.0",server_port=7860, share=False)

if __name__ == "__main__":
    main()