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
        self.csv_dir = "CSVDUMP"
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Load all CSV files into cache
        if os.path.exists(self.csv_dir):
            for file in os.listdir(self.csv_dir):
                if file.endswith('.csv'):
                    symbol, timeframe = file[:-4].split('_')
                    self.load_from_csv(symbol, timeframe)
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"
    
    def _get_csv_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self.csv_dir, f"{symbol}_{timeframe}.csv")
    
    def load_from_csv(self, symbol: str, timeframe: str) -> bool:
        """Load data from CSV file into cache"""
        cache_key = self._get_cache_key(symbol, timeframe)
        csv_path = self._get_csv_path(symbol, timeframe)
        if os.path.exists(csv_path):
            try:
                # Read CSV and parse dates in the exact format from the file
                df = pd.read_csv(csv_path)
                # Convert the date string to datetime
                df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
                df.set_index('Date', inplace=True)
                df.index = df.index.tz_localize('UTC')
                df = df.sort_index()
                self.data_cache[cache_key] = df
                return True
            except Exception as e:
                logging.error(f"Error loading CSV: {e}")
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
        cache_key = self._get_cache_key(symbol, timeframe)
        
        if start_date is None:
            start_date = datetime.now(pytz.UTC)
        elif not start_date.tzinfo:
            start_date = pytz.UTC.localize(start_date)
        
        # Calculate how far back we need data
        if timeframe.lower().endswith('m'):
            delta = timedelta(minutes=int(timeframe[:-1]) * num_bars)
        elif timeframe.lower().endswith('h'):
            delta = timedelta(hours=int(timeframe[:-1]) * num_bars)
        else:  # daily
            delta = timedelta(days=num_bars)
        
        required_start = start_date - delta
        
        # Check if we have valid data in cache
        if cache_key in self.data_cache:
            data = self.data_cache[cache_key]
            if not data.empty:
                # Get the data up to our end time
                data_at_time = data[data.index <= start_date]
                if not data_at_time.empty:
                    latest_time = data_at_time.index.max()
                    earliest_needed = start_date - delta
                    
                    # Check if we have enough data in our range
                    data_in_range = data_at_time[data_at_time.index >= earliest_needed]
                    if len(data_in_range) >= num_bars:
                        print(f"Using cached data for {symbol} {timeframe}")
                        return data_in_range.tail(num_bars)
        
        # If we get here, we need new data
        print(f"Fetching {symbol} {timeframe} data from yfinance...")
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
    if timeframe == '1d':
        # For daily data, we need to adjust the period and interval
        return '1d'
    elif timeframe == '1h':
        return '60m'
    
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
    try:
        if start_date and not start_date.tzinfo:
            start_date = pytz.UTC.localize(start_date)
        
        cache = ForexDataCache()
        hist = cache.get_data(symbol, timeframe, num_bars, start_date)
        
        if hist.empty:
            return {"error": "No data available for the specified symbol"}
        
        end_date = start_date if start_date else datetime.now(pytz.UTC)
        last_n_bars = hist[:end_date].tail(num_bars).iloc[::-1]  # Reverse so newest is last
        
        if len(last_n_bars) < num_bars:
            return {"error": f"Only {len(last_n_bars)} bars available"}
        
        result = {
            "From": symbol[:3],
            "To": symbol[3:],
            "Timeframe": timeframe,
        }
        
        # Add numbered price data with timestamps
        for i, (idx, row) in enumerate(last_n_bars.iterrows(), 1):
            # Convert index to datetime if it's not already
            if isinstance(idx, pd.Timestamp):
                bar_time = idx.strftime("%Y-%m-%d %H:%M")
            else:
                # Handle string or other index types
                bar_time = pd.to_datetime(str(idx)).strftime("%Y-%m-%d %H:%M")
            
            result[f"Time{i}"] = bar_time
            result[f"Open{i}"] = row['Open']
            result[f"High{i}"] = row['High']
            result[f"Low{i}"] = row['Low']
            result[f"Close{i}"] = row['Close']
        
        # Set start and end times from the actual data
        result["StartTime"] = result[f"Time{num_bars}"]  # Oldest bar
        result["EndTime"] = result["Time1"]  # Newest bar
        
        return result
        
    except Exception as e:
        return {"error": str(e)}
