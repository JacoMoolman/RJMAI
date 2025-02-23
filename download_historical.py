import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import os

def download_forex_history(symbol: str, start_date: str, end_date: str, timeframe: str = "1h") -> None:
    """
    Download forex historical data and save to CSV
    
    Args:
        symbol (str): The forex pair symbol (e.g., 'EURUSD')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        timeframe (str): Data timeframe (1m, 5m, 15m, 30m, 1h, 1d)
    """
    # Create Yahoo Finance ticker symbol
    ticker_symbol = f"{symbol[:3]}{symbol[3:]}=X"
    
    # Get data
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, end=end_date, interval=timeframe)
    
    # Select only required columns
    df = df[['Open', 'High', 'Low', 'Close']]
    
    # Format filename
    filename = f"{symbol}_{timeframe}.csv"
    filepath = os.path.join("CSVDUMP", filename)
    
    # Save to CSV
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    print(f"Downloaded {len(df)} records")

def main():
    # Example usage
    symbol = "EURUSD"  # Currency pair
    start_date = "2025-01-01"  # Start date
    end_date = "2025-02-21"    # End date
    timeframe = "5m"           # Timeframe (1m, 5m, 15m, 30m, 1h, 1d)
    
    download_forex_history(symbol, start_date, end_date, timeframe)

if __name__ == "__main__":
    main()
