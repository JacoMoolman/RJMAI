# MT5 Download Bars Script
# This script downloads the latest 10 bars from different timeframes for EURUSD

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

# Set pandas display options to show all rows
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.width', None)     # Auto-detect terminal width
pd.set_option('display.max_columns', None)  # Show all columns

def download_bars(symbol, timeframe, count=10):
    """
    Download specified number of bars for a given symbol and timeframe
    """
    # Initialize connection to MetaTrader 5
    if not mt5.initialize():
        print(f"Initialize failed: {mt5.last_error()}")
        return None
    
    # Prepare timezone for UTC
    timezone = pytz.timezone("UTC")
    
    # Get the latest bars
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if bars is None or len(bars) == 0:
        print(f"Failed to get data for {symbol} on timeframe {timeframe}")
        return None
    
    # Convert to pandas dataframe for easier handling
    df = pd.DataFrame(bars)
    
    # Convert timestamp to datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    return df

def main():
    # Connect to MT5
    if not mt5.initialize():
        print(f"MT5 initialization failed. Error: {mt5.last_error()}")
        return
    
    print(f"MetaTrader5 package version: {mt5.__version__}")
    terminal_info = mt5.terminal_info()
    print(f"Connected to: {terminal_info.name}, build: {terminal_info.build}")
    
    # Symbol to download
    symbol = "EURUSD"
    
    # Timeframes to download (as MT5 constants)
    timeframes = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    # Download 10 bars for each timeframe
    for name, tf in timeframes.items():
        print(f"\nDownloading {name} timeframe data for {symbol}...")
        bars_data = download_bars(symbol, tf, 100)
        
        if bars_data is not None:
            print(f"Downloaded {len(bars_data)} bars from {name} timeframe")
            print(bars_data[['time', 'open', 'high', 'low', 'close']])
        else:
            print(f"Failed to download data for {symbol} on {name} timeframe")
    
    # Shutdown connection to MT5
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed")

if __name__ == "__main__":
    main()