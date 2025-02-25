"""
Data Management Process:
1. On initialization:
   - Clear any existing cache from memory
   - Load ALL CSV files into memory cache
   - CSV files are sorted by date/time to ensure proper ordering

2. When requesting data:
   - Check if requested date range exists in memory cache
   - If data exists in cache and covers the time range, use it
   - If not, fetch from yfinance, then:
     a) Merge with existing data
     b) Sort by date/time
     c) Update both memory cache and CSV file
     d) Return requested range

This ensures we always:
- Start fresh with each run
- Have all CSV data loaded
- Keep CSV files sorted and up to date
- Only fetch from yfinance when absolutely necessary
"""

import yfinance as yf
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import pytz
import os
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

# Suppress yfinance warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class ForexDataCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ForexDataCache, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize cache and load all CSV files"""
        self.data_cache = {}  # Clear any existing cache
        self.csv_dir = r"J:\CSVDUMPS"
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Load all CSV files into cache
        if os.path.exists(self.csv_dir):
            print(f"Looking for CSV files in: {self.csv_dir}")
            csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} CSV files: {csv_files}")
            for file in csv_files:
                if file.endswith('.csv'):
                    parts = file[:-4].split('_')
                    if len(parts) == 2:
                        symbol, timeframe = parts
                        # Normalize timeframe to match TIMEFRAMES in main.py
                        # Add more detailed debug output
                        print(f"Processing file {file} with symbol={symbol}, raw timeframe={timeframe}")
                        
                        # Normalize timeframes
                        timeframe = timeframe.lower()
                        if timeframe == '60m' or timeframe == 'h1':
                            timeframe = '1h'
                        elif timeframe == 'd1':
                            timeframe = '1d'
                        # Other timeframes (5m, 15m, 30m) should already match
                        
                        print(f"Loading {symbol} {timeframe} from {file}")
                        self.load_from_csv(symbol, timeframe)
                    else:
                        print(f"Skipping file with unexpected format: {file}")
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        # Normalize timeframe format
        timeframe = timeframe.lower()
        if timeframe == '60m' or timeframe == 'h1':
            timeframe = '1h'
        elif timeframe == 'd1':
            timeframe = '1d'
        return f"{symbol}_{timeframe}"
    
    def _get_csv_path(self, symbol: str, timeframe: str) -> str:
        cache_key = self._get_cache_key(symbol, timeframe)
        return os.path.join(self.csv_dir, f"{cache_key}.csv")
    
    def load_from_csv(self, symbol: str, timeframe: str) -> bool:
        """Load data from CSV file into cache"""
        cache_key = self._get_cache_key(symbol, timeframe)
        csv_path = self._get_csv_path(symbol, timeframe)
        print(f"Checking for CSV file: {csv_path}")
        if os.path.exists(csv_path):
            try:
                # Read CSV and parse dates in the exact format from the file
                df = pd.read_csv(csv_path)
                print(f"CSV loaded. First few rows: {df.head(2)}")
                
                # Convert the date string to datetime
                df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
                df.set_index('Date', inplace=True)
                df.index = df.index.tz_localize('UTC')
                df = df.sort_index()
                self.data_cache[cache_key] = df
                print(f"Successfully loaded {symbol} {timeframe} into cache with {len(df)} rows")
                return True
            except Exception as e:
                logging.error(f"Error loading CSV for {symbol} {timeframe}: {e}")
                print(f"Error loading CSV for {symbol} {timeframe}: {e}")
        else:
            print(f"CSV file not found: {csv_path}")
        return False
    
    def save_to_csv(self, symbol: str, timeframe: str):
        """Save data from cache to CSV file, ensuring it's sorted by date"""
        cache_key = self._get_cache_key(symbol, timeframe)
        if cache_key in self.data_cache:
            csv_path = self._get_csv_path(symbol, timeframe)
            df = self.data_cache[cache_key].copy()
            
            # Drop unwanted columns from yfinance
            columns_to_drop = ['Volume', 'Dividends', 'Stock Splits', 'Adj Close']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            # Sort by date and format the index
            df = df.sort_index()
            df.index = df.index.strftime('%Y.%m.%d %H:%M')
            
            # Round all numeric columns to 5 decimal places
            df = df.round(5)
            
            # Save with 'Date' as the index name
            df.to_csv(csv_path, float_format='%.5f', index_label='Date')
    
    def get_data(self, symbol: str, timeframe: str, num_bars: int, start_date: datetime = None) -> pd.DataFrame:
        """Get data from cache or fetch from yfinance"""
        # Ensure start_date has a timezone
        if start_date is None:
            start_date = datetime.now(pytz.UTC)
        elif start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=pytz.UTC)
            
        # Calculate the time range we need
        if timeframe.lower() == '1d' or timeframe.lower() == 'd1':
            # For daily data, we need to be more lenient with the date range
            # Use days instead of more precise times for comparing daily data
            delta = timedelta(days=num_bars + 3)  # Add extra days to ensure we have enough data
            
            # For daily data, we want to truncate the time part to only consider the date
            # This avoids issues with partial days being considered different days
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe.lower() == '1h' or timeframe.lower() == '60m' or timeframe.lower() == 'h1':
            delta = timedelta(hours=num_bars + 2)  # Add extra hours to ensure we have enough data
        elif timeframe.lower() == '30m' or timeframe.lower() == 'm30':
            delta = timedelta(minutes=30 * (num_bars + 1))
        elif timeframe.lower() == '15m' or timeframe.lower() == 'm15':
            delta = timedelta(minutes=15 * (num_bars + 1))
        elif timeframe.lower() == '5m' or timeframe.lower() == 'm5':
            delta = timedelta(minutes=5 * (num_bars + 1))
        else:
            # Default to 1 hour
            delta = timedelta(hours=num_bars + 2)
            
        cache_key = self._get_cache_key(symbol, timeframe)
        print(f"Attempting to get data for {symbol} {timeframe} with cache_key: {cache_key}")
        print(f"Looking for data from {start_date - delta} to {start_date}")
        
        # Check if we have valid data in cache
        if cache_key in self.data_cache:
            print(f"Found {cache_key} in cache with {len(self.data_cache[cache_key])} entries")
            data = self.data_cache[cache_key]
            if not data.empty:
                # Special handling for daily data
                if timeframe.lower() == '1d' or timeframe.lower() == 'd1':
                    # For daily data, we don't care about the time part, only the date
                    # Get all data up to our end date, ignoring time
                    data_dates = data.index.date
                    start_date_date = start_date.date()
                    data_at_time = data[data_dates <= start_date_date]
                    
                    print(f"For daily data, using date comparison instead of datetime")
                    print(f"Found {len(data_at_time)} entries up to {start_date.date()}")
                    
                    if not data_at_time.empty:
                        latest_date = max(data_dates)
                        earliest_needed_date = (start_date - delta).date()
                        print(f"Latest date in cache: {latest_date}")
                        print(f"Earliest date needed: {earliest_needed_date}")
                        
                        # Check if we have enough data in our range
                        # For daily data, we'll count the number of unique dates in the required range
                        dates_in_range = [d for d in data_dates if d >= earliest_needed_date and d <= start_date_date]
                        print(f"Found {len(dates_in_range)} dates in the required date range")
                        
                        if len(dates_in_range) >= num_bars:
                            data_in_range = data_at_time.tail(num_bars)
                            print(f"{Fore.GREEN}Using cached data for {symbol} {timeframe}{Style.RESET_ALL}")
                            return data_in_range
                else:
                    # Standard handling for other timeframes
                    # Get the data up to our end time
                    data_at_time = data[data.index <= start_date]
                    print(f"Found {len(data_at_time)} entries up to {start_date}")
                    if not data_at_time.empty:
                        latest_time = data_at_time.index.max()
                        earliest_needed = start_date - delta
                        print(f"Latest time in cache: {latest_time}")
                        print(f"Earliest time needed: {earliest_needed}")
                        
                        # Check if we have enough data in our range
                        data_in_range = data_at_time[data_at_time.index >= earliest_needed]
                        print(f"Found {len(data_in_range)} entries in the required date range")
                        if len(data_in_range) >= num_bars:
                            print(f"{Fore.GREEN}Using cached data for {symbol} {timeframe}{Style.RESET_ALL}")
                            return data_in_range.tail(num_bars)
                        else:
                            print(f"Not enough data in cache: {len(data_in_range)} < {num_bars} required")
                    else:
                        print(f"No data found in cache up to {start_date}")
            else:
                print(f"Cache for {cache_key} is empty")
        else:
            print(f"No cache entry found for {cache_key}")
        
        # If we get here, we need new data
        print(f"{Fore.RED}Fetching {symbol} {timeframe} data from yfinance...{Style.RESET_ALL}")
        yf_interval = get_timeframe_interval(timeframe)
        ticker = yf.Ticker(f"{symbol}=X")
        
        # Add buffer to ensure we get enough data
        buffer_multiplier = 3
        fetch_start = start_date - (delta * buffer_multiplier)
        
        try:
            hist = ticker.history(interval=yf_interval, start=fetch_start, end=start_date)
            if not hist.empty:
                # Convert timezone to UTC if needed
                if hist.index.tz is None:
                    hist.index = pd.to_datetime(hist.index).tz_localize('UTC')
                elif hist.index.tz != pytz.UTC:
                    hist.index = hist.index.tz_convert('UTC')
                
                # Merge with existing data if we have any
                if cache_key in self.data_cache and not self.data_cache[cache_key].empty:
                    hist = pd.concat([self.data_cache[cache_key], hist])
                    hist = hist[~hist.index.duplicated(keep='last')]  # Remove duplicates
                    hist = hist.sort_index()  # Sort by date
                
                self.data_cache[cache_key] = hist
                self.save_to_csv(symbol, timeframe)  # Save to cache for future use
                return hist[hist.index <= start_date].tail(num_bars)
            else:
                return pd.DataFrame()  # Return empty DataFrame if no data
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

def get_timeframe_interval(timeframe: str) -> str:
    """Convert timeframe to yfinance interval format"""
    # Yahoo Finance valid intervals: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    timeframe = timeframe.lower()  # Convert to lowercase to match Yahoo Finance format
    
    # Handle special cases
    if timeframe == '1h' or timeframe == '60m' or timeframe == 'h1':
        return '60m'
    elif timeframe == '1d' or timeframe == 'd1':
        return '1d'
    elif timeframe == '5m' or timeframe == 'm5':
        return '5m'
    elif timeframe == '15m' or timeframe == 'm15':
        return '15m'
    elif timeframe == '30m' or timeframe == 'm30':
        return '30m'
    # Other timeframes (5m, 15m, 30m) already match Yahoo Finance format
    return timeframe

def get_forex_data(symbol: str, timeframe: str = "M5", num_bars: int = 1, start_date: datetime = None) -> Dict[str, Any]:
    """
    Fetch forex data for a given currency pair using cached data and yfinance as fallback
    
    Args:
        symbol (str): The forex pair symbol (e.g., 'EURUSD')
        timeframe (str): Timeframe for the data (e.g., 'M5', 'M15', 'H1', 'D1')
        num_bars (int): Number of previous bars to fetch
        start_date (datetime, optional): Custom start date. If None, uses current time
        
    Returns:
        Dict containing the forex data including multiple bars of price data
    """
    # Convert MT5 timeframes to yfinance format if needed
    if timeframe.upper().startswith('M'):
        # Convert 'M1', 'M5', 'M15', 'M30' to '1m', '5m', '15m', '30m'
        timeframe = timeframe[1:] + 'm'
    elif timeframe.upper() == 'H1':
        timeframe = '1h'
    elif timeframe.upper() == 'D1':
        timeframe = '1d'
    
    # Normalize symbol format (remove / if present)
    symbol = symbol.replace('/', '')
    
    print(f"Fetching last {num_bars} {timeframe} bars for {symbol}...")
    
    # Get the data from cache or yfinance
    data_cache = ForexDataCache()
    df = data_cache.get_data(symbol, timeframe, num_bars, start_date)
    
    if df.empty:
        print(f"No data available for {symbol} {timeframe}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": []
        }
    
    # Format the time range for display
    start_time = df.index.min().strftime('%Y-%m-%d %H:%M')
    end_time = df.index.max().strftime('%Y-%m-%d %H:%M')
    print(f"Time Range: {start_time} to {end_time}")
    
    # Convert the data to the desired format
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": []
    }
    
    for idx, row in df.iterrows():
        bar_data = {
            "time": idx.strftime('%Y-%m-%d %H:%M:%S'),
            "open": row['Open'],
            "high": row['High'],
            "low": row['Low'],
            "close": row['Close']
        }
        result["data"].append(bar_data)
    
    return result
