'''
FOREX DATA ANALYSIS AND AGGREGATION TOOL

DESCRIPTION:
This script functions as a comprehensive forex market analysis tool that retrieves, processes, and displays
historical price data for multiple currency pairs across different timeframes. It's designed to provide
a consolidated view of market conditions at a specific point in time.

FUNCTIONALITY:
1. Retrieves historical OHLC (Open, High, Low, Close) data for configurable currency pairs
2. Supports multiple timeframes simultaneously (5m, 15m, 30m, 1h, 1d)
3. Processes a specified number of historical bars for each currency pair and timeframe
4. Combines data from all timeframes into a single DataFrame for each currency pair
5. For each OHLC entry, calculates the weekday (0-6, with Sunday=0) from the timestamp
6. Creates a comprehensive dataset with both price data and temporal features
7. Displays the data in a formatted, easy-to-read layout with both OHLC and weekday information
8. Can target a specific historical date/time for analysis

CONFIGURATION OPTIONS:
- NUM_BARS: Controls how many historical bars to retrieve for each timeframe
- CUSTOM_DATE: Specifies the target date/time for analysis (format: YYYY-MM-DD HH:MM:SS)
- FOREX_PAIRS: List of currency pairs to analyze (e.g., EURUSD, GBPUSD)
- TIMEFRAMES: List of timeframes to analyze (e.g., 5m, 15m, 30m, 1h, 1d)

DATA STRUCTURE:
- Each currency pair has a dataframe containing entries for all timeframes
- For each OHLC value (Open, High, Low, Close), a corresponding weekday value is calculated
- This creates a 1:1 relationship between price data and weekday information
- The resulting dataset is twice the size of the original, with half being OHLC data and half being weekday data

DEPENDENCIES:
- get_data.py: Contains the function to fetch forex data and calculate weekday values
- pandas: For data manipulation and organization
- datetime: For handling dates and times
- pytz: For timezone management

USAGE:
Simply run the script with appropriate configuration values to get a multi-timeframe
snapshot of the specified forex pairs at the specified date and time, including weekday information.
'''

# Configuration
NUM_BARS = 5  # Number of previous bars to fetch
CUSTOM_DATE = "2024-02-20 12:30:00"  # Format: YYYY-MM-DD HH:MM:SS

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

from get_data import get_forex_data, get_weekday
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
        
        # Step 1: Create a dictionary for each original OHLC entry
        original_entries = []
        total_entries = 0
        
        for timeframe in TIMEFRAMES:
            if timeframe not in all_data[forex_pair]:
                continue
                
            data = all_data[forex_pair][timeframe]
            
            for i in range(1, NUM_BARS + 1):
                timestamp = data["data"][i-1]["time"]
                
                # Create entries for Open
                open_entry = {
                    "timestamp": timestamp,
                    "column_name": f"Open_{timeframe}_{i}",
                    "value": data["data"][i-1]["open"]
                }
                original_entries.append(open_entry)
                total_entries += 1
                
                # Create entries for High
                high_entry = {
                    "timestamp": timestamp,
                    "column_name": f"High_{timeframe}_{i}",
                    "value": data["data"][i-1]["high"]
                }
                original_entries.append(high_entry)
                total_entries += 1
                
                # Create entries for Low
                low_entry = {
                    "timestamp": timestamp,
                    "column_name": f"Low_{timeframe}_{i}",
                    "value": data["data"][i-1]["low"]
                }
                original_entries.append(low_entry)
                total_entries += 1
                
                # Create entries for Close
                close_entry = {
                    "timestamp": timestamp,
                    "column_name": f"Close_{timeframe}_{i}",
                    "value": data["data"][i-1]["close"]
                }
                original_entries.append(close_entry)
                total_entries += 1
        
        # Step 2: Create a duplicate set of entries with weekday values
        weekday_entries = []
        
        for entry in original_entries:
            timestamp = entry["timestamp"]
            column_parts = entry["column_name"].split('_')
            timeframe = column_parts[1]
            bar_num = column_parts[2]
            
            # Parse the timestamp to calculate weekday
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            weekday = (dt.weekday() + 1) % 7  # Convert to 0-6 where 0 is Sunday
            
            weekday_entry = {
                "timestamp": timestamp,
                "column_name": f"weekday_{timeframe}_{bar_num}",
                "value": weekday
            }
            weekday_entries.append(weekday_entry)
        
        # Step 3: Combine all entries
        all_entries = original_entries + weekday_entries
        
        # Step 4: Print all entries in the correct format
        print(f"\n{forex_pair} Combined Timeframe Data:")
        print("-" * 40)
        
        # First print all original OHLC entries
        for entry in original_entries:
            print(f"{entry['timestamp']}::{entry['column_name']}: {entry['value']:.6f}")
        
        # Then print all weekday entries
        for entry in weekday_entries:
            print(f"{entry['timestamp']}::{entry['column_name']}: {int(entry['value'])}")
        
        # Store all entries in the dataframe
        dataframes[forex_pair] = pd.DataFrame(all_entries)
        
if __name__ == "__main__":
    main()
