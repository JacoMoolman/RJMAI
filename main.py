# Configuration
NUM_BARS = 5  # Number of previous bars to fetch
CUSTOM_DATE = "2024-02-01 12:12:00"  # Format: YYYY-MM-DD HH:MM:SS

# List of forex pairs to analyze
FOREX_PAIRS = [
    "EURUSD",  # Euro/US Dollar
    # "GBPUSD",  # British Pound/US Dollar
    # "USDJPY",  # US Dollar/Japanese Yen
    # "USDCHF",  # US Dollar/Swiss Franc
    # "AUDUSD"   # Australian Dollar/US Dollar
]

# List of timeframes to analyze (using Yahoo Finance valid intervals)
TIMEFRAMES = [
    "5m",    # 5 minutes
    "15m",   # 15 minutes
    "30m",   # 30 minutes
    # "1h",    # 1 hour
    # "1d"     # Daily
]

from get_data import get_forex_data
import pandas as pd
from datetime import datetime
from typing import Optional
import pytz

def display_forex_data(df):
    """Display forex data in a vertical format"""
    for column in df.columns:
        print(f"{column}: {df[column].values[0]:.6f}")

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
        all_data[forex_pair] = {}
        for timeframe in TIMEFRAMES:
            print(f"\nFetching last {NUM_BARS} {timeframe} bars for {forex_pair}...")
            data = get_forex_data(forex_pair, timeframe, NUM_BARS, start_date)
            
            if "error" in data:
                print(f"Error for {forex_pair} {timeframe}: {data['error']}")
                continue
                
            all_data[forex_pair][timeframe] = data
            print(f"Time Range: {data['StartTime']} to {data['EndTime']}")
    
    # Now process each currency pair and timeframe separately
    dataframes = {}
    for forex_pair in FOREX_PAIRS:
        dataframes[forex_pair] = {}
        for timeframe in TIMEFRAMES:
            if timeframe not in all_data[forex_pair]:
                continue
                
            data = all_data[forex_pair][timeframe]
            df_data = {}
            for i in range(1, NUM_BARS + 1):
                df_data[f"Open_{timeframe}_{i}"] = [data[f"Open{i}"]]
                df_data[f"High_{timeframe}_{i}"] = [data[f"High{i}"]]
                df_data[f"Low_{timeframe}_{i}"] = [data[f"Low{i}"]]
                df_data[f"Close_{timeframe}_{i}"] = [data[f"Close{i}"]]
            
            # Store the DataFrame in our dictionary
            dataframes[forex_pair][timeframe] = pd.DataFrame(df_data)
            
            print(f"\n{forex_pair} {timeframe} Data:")
            print("-" * 40)
            display_forex_data(dataframes[forex_pair][timeframe])

if __name__ == "__main__":
    main()
