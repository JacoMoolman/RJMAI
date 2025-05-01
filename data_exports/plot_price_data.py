import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the current directory and file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
normalized_data_path = os.path.join(current_dir, 'GBPUSD_normalized_data.csv')
price_levels_path = os.path.join(current_dir, 'GBPUSD_price_levels.csv')

# Read the CSV files
try:
    normalized_data = pd.read_csv(normalized_data_path)
    price_levels = pd.read_csv(price_levels_path)
    
    print("Successfully loaded data:")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Normalized data columns: {normalized_data.columns.tolist()}")
    print(f"Price levels shape: {price_levels.shape}")
    print(f"Price levels columns: {price_levels.columns.tolist()}")
    
    # Extract the close price from GBPUSD_normalized_data.csv
    if 'close' in normalized_data.columns:
        close_prices = normalized_data['close']
    else:
        print("Warning: 'close' column not found in normalized data. Available columns:", normalized_data.columns.tolist())
        # Try to find a column that might contain close prices
        potential_close_columns = [col for col in normalized_data.columns if 'close' in col.lower()]
        if potential_close_columns:
            print(f"Using '{potential_close_columns[0]}' as close price")
            close_prices = normalized_data[potential_close_columns[0]]
        else:
            raise ValueError("Could not find 'close' price column in the normalized data")
    
    # Extract the normalized_price from GBPUSD_price_levels.csv
    if 'normalized_price' in price_levels.columns:
        normalized_prices = price_levels['normalized_price']
    else:
        print("Warning: 'normalized_price' column not found in price levels data. Available columns:", price_levels.columns.tolist())
        # Try to find a column that might contain normalized prices
        potential_normalized_columns = [col for col in price_levels.columns if 'price' in col.lower()]
        if potential_normalized_columns:
            print(f"Using '{potential_normalized_columns[0]}' as normalized price")
            normalized_prices = price_levels[potential_normalized_columns[0]]
        else:
            raise ValueError("Could not find 'normalized_price' column in the price levels data")
    
    # Check if timeframe column exists
    if 'timeframe' in normalized_data.columns:
        # Identify unique timeframes
        unique_timeframes = normalized_data['timeframe'].unique()
        print(f"Found {len(unique_timeframes)} unique timeframes: {unique_timeframes}")
        
        # Create separate plots for each timeframe
        for timeframe in unique_timeframes:
            # Filter data for the current timeframe
            timeframe_data = normalized_data[normalized_data['timeframe'] == timeframe]
            timeframe_close = timeframe_data['close']
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot close prices for this timeframe
            plt.plot(timeframe_close, color='blue', label=f'Close Price (Timeframe {timeframe})')
            
            # Add price levels as horizontal lines
            for price in normalized_prices:
                plt.axhline(y=price, color='red', linestyle='-', alpha=0.5)
            plt.plot([], [], color='red', label='Normalized Price Levels')  # For legend
            
            plt.title(f'GBPUSD Close Price (Timeframe {timeframe}) and Normalized Price Levels')
            plt.xlabel('Data Points')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
        
        # Also create a combined plot with all timeframes
        plt.figure(figsize=(14, 8))
        
        for timeframe in unique_timeframes:
            timeframe_data = normalized_data[normalized_data['timeframe'] == timeframe]
            timeframe_close = timeframe_data['close']
            plt.plot(range(len(timeframe_close)), timeframe_close, label=f'Timeframe {timeframe}')
        
        # Add price levels as horizontal lines in the combined plot
        for price in normalized_prices:
            plt.axhline(y=price, color='red', linestyle='-', alpha=0.5)
        plt.plot([], [], color='red', label='Normalized Price Levels')  # For legend
        
        plt.title('GBPUSD Close Price Comparison Across All Timeframes with Price Levels')
        plt.xlabel('Data Points')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Show the plots
        plt.show()
        
    else:
        print("Warning: 'timeframe' column not found in normalized data. Creating a single plot instead.")
        
        # Create a single plot (original functionality)
        plt.figure(figsize=(12, 6))
        
        # Plot close prices from normalized data
        plt.plot(close_prices, color='blue', label='Close Price')
        
        # If normalized_prices is a list of values without a time component, 
        # we'll plot them as horizontal lines across the graph
        if len(normalized_prices) < len(close_prices):
            for price in normalized_prices:
                plt.axhline(y=price, color='red', linestyle='-', alpha=0.5)
            plt.plot([], [], color='red', label='Normalized Price Levels')  # For legend
        else:
            # If both datasets have the same length, plot them together
            plt.plot(normalized_prices, color='red', label='Normalized Price')
        
        plt.title('GBPUSD Close Price and Normalized Price Levels')
        plt.xlabel('Data Points')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
