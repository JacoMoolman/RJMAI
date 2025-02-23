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
    
    def load_from_csv(self, symbol: str, timeframe: str):
        cache_key = self._get_cache_key(symbol, timeframe)
        csv_path = self._get_csv_path(symbol, timeframe)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index)
                self.data_cache[cache_key] = df
                return True
            except Exception as e:
                logging.error(f"Error loading CSV: {e}")
        return False
    
    def save_to_csv(self, symbol: str, timeframe: str):
        cache_key = self._get_cache_key(symbol, timeframe)
        if cache_key in self.data_cache:
            csv_path = self._get_csv_path(symbol, timeframe)
            self.data_cache[cache_key].to_csv(csv_path)
    
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
        
        if timeframe.startswith('M'):
            minutes = int(timeframe[1:]) * (num_bars + 1)
            required_start = end_date - timedelta(minutes=minutes)
        elif timeframe.startswith('H'):
            hours = int(timeframe[1:]) * (num_bars + 1)
            required_start = end_date - timedelta(hours=hours)
        else:  # Daily
            required_start = end_date - timedelta(days=(num_bars + 1))
        
        # Check if we need to fetch new data
        need_update = True
        if cached_data is not None:
            cached_range = cached_data[required_start:end_date]
            if len(cached_range) >= num_bars:
                need_update = False
        
        # Fetch new data if needed
        if need_update:
            ticker_symbol = f"{symbol[:3]}{symbol[3:]}=X"
            interval = get_timeframe_interval(timeframe)
            
            ticker = yf.Ticker(ticker_symbol)
            new_data = ticker.history(start=required_start, end=end_date, interval=interval)
            
            if cached_data is not None:
                # Merge new data with cached data
                combined_data = pd.concat([cached_data, new_data])
                # Remove duplicates and sort
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
            else:
                combined_data = new_data
            
            self.data_cache[cache_key] = combined_data
            self.save_to_csv(symbol, timeframe)
            
        return self.data_cache[cache_key]

def get_timeframe_interval(timeframe: str) -> str:
    """Convert timeframe to yfinance interval format"""
    timeframe = timeframe.upper()
    if timeframe.startswith('M'):
        return f"{timeframe[1:]}m"  # M5 -> 5m
    elif timeframe.startswith('H'):
        return f"{timeframe[1:]}h"  # H1 -> 1h
    elif timeframe == 'D1':
        return '1d'
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

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
            "StartTime": last_n_bars.index[0].strftime("%Y-%m-%d %H:%M"),
            "EndTime": last_n_bars.index[-1].strftime("%Y-%m-%d %H:%M"),
        }
        
        # Add numbered price data
        for i, (idx, row) in enumerate(last_n_bars.iterrows(), 1):
            result[f"Open{i}"] = row['Open']
            result[f"High{i}"] = row['High']
            result[f"Low{i}"] = row['Low']
            result[f"Close{i}"] = row['Close']
        
        return result
        
    except Exception as e:
        return {"error": str(e)}
