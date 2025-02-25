# Configuration
NUM_BARS = 5  # Number of previous bars to fetch
CUSTOM_DATE = "2025-02-20 12:30:00"  # Format: YYYY-MM-DD HH:MM:SS

# List of forex pairs to analyze
FOREX_PAIRS = [
    "EURUSD",  # Euro/US Dollar
    "GBPUSD",  # British Pound/US Dollar
    "USDJPY",  # US Dollar/Japanese Yen
    "USDCHF",  # US Dollar/Swiss Franc
    "AUDUSD"   # Australian Dollar/US Dollar
]

# List of timeframes to analyze (using Yahoo Finance valid intervals)
TIMEFRAMES = [
    "5m",    # 5 minutes
    "15m",   # 15 minutes
    "30m",   # 30 minutes
    "1h",    # 1 hour
    "1d"     # Daily
]

from get_data import get_forex_data
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import pytz

def display_forex_data(df):
    """Display forex data in a vertical format with timestamps"""
    for column in df.columns:
        if not column.startswith('timestamp_'):  # Skip the timestamp columns
            timeframe = column.split('_')[1]  # Extract timeframe from column name
            bar_num = column.split('_')[2]    # Extract bar number
            timestamp = df[f'timestamp_{timeframe}_{bar_num}'].values[0]
            print(f"{timestamp}::{column}: {df[column].values[0]:.6f}")

def main():
    try:
        start_date = datetime.strptime(CUSTOM_DATE, "%Y-%m-%d %H:%M:%S")
        # Make start_date timezone-aware by setting it to UTC
        start_date = pytz.UTC.localize(start_date)
        print(f"Using custom date: {start_date}")
    except ValueError:
        print("Invalid date format. Please use 'YYYY-MM-DD HH:MM:SS' format")
        return
    
    # First, load all data into memory
    all_data = {}
    for forex_pair in FOREX_PAIRS:
        print("############################################################################")
        all_data[forex_pair] = {}
        for timeframe in TIMEFRAMES:
            data = get_forex_data(forex_pair, timeframe, NUM_BARS, start_date)
            
            if len(data["data"]) == 0:
                print(f"Error for {forex_pair} {timeframe}: No data available")
                continue
                
            all_data[forex_pair][timeframe] = data
            
            # Display the first and last bar times
            if data["data"]:
                start_time = data["data"][0]["time"]
                end_time = data["data"][-1]["time"]
                print(f"Data range: {start_time} to {end_time}")
    
    # Now process each currency pair and combine all timeframes
    dataframes = {}
    for forex_pair in FOREX_PAIRS:
        print(f"Processing {forex_pair}")
        # Create a dictionary to store all data for this currency pair
        df_data = {}
        
        # Process each timeframe
        for timeframe in TIMEFRAMES:
            if timeframe not in all_data[forex_pair]:
                continue
                
            data = all_data[forex_pair][timeframe]
            
            for i in range(1, NUM_BARS + 1):
                # Use the actual timestamp from the data
                df_data[f'timestamp_{timeframe}_{i}'] = [data["data"][i-1]["time"]]
                df_data[f"Open_{timeframe}_{i}"] = [data["data"][i-1]["open"]]
                df_data[f"High_{timeframe}_{i}"] = [data["data"][i-1]["high"]]
                df_data[f"Low_{timeframe}_{i}"] = [data["data"][i-1]["low"]]
                df_data[f"Close_{timeframe}_{i}"] = [data["data"][i-1]["close"]]
        
        # Create a single DataFrame for this currency pair with all timeframes
        dataframes[forex_pair] = pd.DataFrame(df_data)
        
        print(f"\n{forex_pair} Combined Timeframe Data:")
        print("-" * 40)
        display_forex_data(dataframes[forex_pair])

if __name__ == "__main__":
    main()
