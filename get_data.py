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
        self.data_cache = {}
        self.csv_dir = "CSVDUMP"
        os.makedirs(self.csv_dir, exist_ok=True)
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"
    
    def _get_csv_path(self, symbol: str, timeframe: str) -> str:
        # Convert timeframe format to match existing files (H1 -> 1h)
        if timeframe.startswith('H'):
            timeframe = f"{timeframe[1:]}h"
        elif timeframe.startswith('M'):
            timeframe = f"{timeframe[1:]}m"
        return os.path.join(self.csv_dir, f"{symbol}_{timeframe}.csv")
    
    def load_from_csv(self, symbol: str, timeframe: str) -> bool:
        """Load data from CSV file into cache"""
        cache_key = self._get_cache_key(symbol, timeframe)
        csv_path = self._get_csv_path(symbol, timeframe)
        if os.path.exists(csv_path):
            try:
                # Read CSV without index column name
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True, header=0)
                df.index = pd.to_datetime(df.index, utc=True)  # Ensure UTC timezone
                self.data_cache[cache_key] = df
                return True
            except Exception as e:
                logging.error(f"Error loading CSV: {e}")
        return False
    
    def save_to_csv(self, symbol: str, timeframe: str):
        """Save data from cache to CSV file"""
        cache_key = self._get_cache_key(symbol, timeframe)
        if cache_key in self.data_cache:
            csv_path = self._get_csv_path(symbol, timeframe)
            # Format the index dates and decimal numbers before saving
            df = self.data_cache[cache_key].copy()
            # Drop unwanted columns
            columns_to_drop = ['Volume', 'Dividends', 'Stock Splits', 'Adj Close']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            df.index = df.index.strftime('%Y.%m.%d %H:%M')
            # Round all numeric columns to 5 decimal places
            df = df.round(5)
            # Save with 'Date' as the index name
            df.to_csv(csv_path, float_format='%.5f', index_label='Date')
    
    def get_data(self, symbol: str, timeframe: str, num_bars: int, start_date: datetime) -> pd.DataFrame:
        cache_key = self._get_cache_key(symbol, timeframe)
        
        # Load from CSV if not in memory
        if cache_key not in self.data_cache:
            self.load_from_csv(symbol, timeframe)
        
        # Get cached data
        cached_data = self.data_cache.get(cache_key)
        
        # Calculate the required date range
        end_date = start_date if start_date else datetime.now(pytz.UTC)
        if not end_date.tzinfo:
            end_date = pytz.UTC.localize(end_date)
        
        # Parse the timeframe value and unit
        if timeframe.endswith('m'):  # Minutes
            minutes = int(timeframe[:-1]) * num_bars
            required_start = end_date - timedelta(minutes=minutes)
        elif timeframe.endswith('h'):  # Hours
            hours = int(timeframe[:-1]) * num_bars
            required_start = end_date - timedelta(hours=hours)
        elif timeframe.endswith('d'):  # Days
            days = int(timeframe[:-1]) * num_bars
            required_start = end_date - timedelta(days=days)
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
            
        # Ensure required_start is timezone-aware
        if not required_start.tzinfo:
            required_start = pytz.UTC.localize(required_start)
        
        # Check if we need to fetch new data
        need_update = True
        if cached_data is not None:
            cached_range = cached_data[required_start:end_date]
            if len(cached_range) >= num_bars:
                # Make sure we have all the required bars with data
                last_n_bars = cached_range.tail(num_bars)
                if not last_n_bars.empty and not last_n_bars.isnull().values.any():
                    need_update = False
                    print(f"Using cached data for {symbol} {timeframe}")
        
        # Fetch new data if needed
        if need_update:
            print(f"Fetching {symbol} {timeframe} data from yfinance...")
            ticker_symbol = f"{symbol[:3]}{symbol[3:]}=X"
            interval = get_timeframe_interval(timeframe)
            
            ticker = yf.Ticker(ticker_symbol)
            
            # For daily data, we need to fetch more data to ensure we get enough valid bars
            if timeframe.lower() == '1d':
                # Fetch more days to account for market holidays/weekends
                fetch_start = required_start - timedelta(days=num_bars * 3)
                new_data = ticker.history(start=fetch_start, end=end_date, interval=interval)
            else:
                new_data = ticker.history(start=required_start, end=end_date, interval=interval)
            
            # Ensure we have a DatetimeIndex
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index)
            
            if cached_data is not None:
                # Ensure cached data has DatetimeIndex
                if not isinstance(cached_data.index, pd.DatetimeIndex):
                    cached_data.index = pd.to_datetime(cached_data.index)
                
                # Only concatenate non-empty DataFrames
                dfs_to_concat = []
                if not cached_data.empty:
                    dfs_to_concat.append(cached_data)
                if not new_data.empty:
                    dfs_to_concat.append(new_data)
                
                if len(dfs_to_concat) > 0:
                    combined_data = pd.concat(dfs_to_concat).fillna(0)
                    # Remove duplicates and sort
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data.sort_index(inplace=True)
                else:
                    # Create empty DataFrame with proper structure
                    combined_data = pd.DataFrame(columns=new_data.columns, index=pd.DatetimeIndex([])).fillna(0)
            else:
                combined_data = new_data.fillna(0)
            
            # For daily data, make sure we have business days only
            if timeframe.lower() == '1d':
                combined_data = combined_data.asfreq('B', method='ffill')  # Business days only, forward fill missing values
            
            self.data_cache[cache_key] = combined_data
            self.save_to_csv(symbol, timeframe)
            
        return self.data_cache[cache_key]

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
            # Get the actual timestamp from the data
            bar_time = idx.strftime("%Y-%m-%d %H:%M")
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
