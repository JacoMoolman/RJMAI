"""
Data Management Process:
1. On initialization:
   - Clear any existing cache from memory
   - Load ALL CSV files from J:\\CSVDUMPS into memory cache
   - Normalize timeframe formats during loading (e.g., 'd1' -> '1d', 'h1' -> '1h')
   - CSV files are sorted by date/time to ensure proper ordering

2. When requesting data:
   - First normalize timeframe format for consistency (e.g., 'd1'/'1d', 'h1'/'1h')
   - Generate a cache key based on symbol and normalized timeframe
   - Check if requested date range exists in memory cache
   - For daily data ('1d'/'d1'):
     a) Ignore time parts and only compare dates
     b) Look for data between earliest needed date and the requested date
     c) Ensure there are enough unique dates in the required range
   - For other timeframes (5m, 15m, 30m, 1h):
     a) Compare exact datetimes including time parts
     b) Ensure there are enough entries within the required time range
   - If sufficient data exists in cache, return it with GREEN status message
   - If not, fetch from yfinance with RED status message, then:
     a) Add buffer time to ensure enough data is fetched
     b) Convert timezone to UTC if needed
     c) Merge with existing data if available
     d) Remove duplicates and sort by date/time
     e) Update both memory cache and CSV file in J:\\CSVDUMPS
     f) Return requested range

3. Date and Time Processing:
   - Provides weekday calculation functionality (get_weekday function)
   - Converts date strings to datetime objects
   - Determines the day of week (0-6, with Sunday as 0)
   - Supports calculations for data analysis and feature engineering

This ensures we always:
- Start fresh with each run
- Have all CSV data loaded
- Handle different timeframe formats consistently
- Process daily data correctly by date rather than datetime
- Keep CSV files sorted and up to date
- Only fetch from yfinance when absolutely necessary
- Can determine weekday information for temporal analysis
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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

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

def get_weekday(date_string):
    """
    Get the weekday (0-6) for a given date string, where Sunday is 0.
    
    Args:
        date_string (str): Date string in format 'YYYY-MM-DD HH:MM:SS' or datetime object
        
    Returns:
        int: Weekday number (0 for Sunday, 1 for Monday, ..., 6 for Saturday)
    """
    if isinstance(date_string, str):
        # Parse the date string to a datetime object
        date_obj = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    else:
        # Already a datetime object
        date_obj = date_string
    
    # Get weekday (0 is Monday in Python's datetime, so we need to adjust)
    weekday = date_obj.weekday()
    
    # Adjust to make Sunday 0 (Python's weekday is 0 for Monday, 6 for Sunday)
    # So we add 1 and take modulo 7 to make Sunday 0
    adjusted_weekday = (weekday + 1) % 7
    
    return adjusted_weekday

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

def visualize_forex_data(dataframes):
    """
    Create bar chart visualizations for each forex pair.
    
    Args:
        dataframes (dict): Dictionary containing dataframes for each forex pair
    """
    for forex_pair, df in dataframes.items():
        # Group by timeframe
        timeframes = set()
        for column_name in df['column_name'].unique():
            if not column_name.startswith('weekday_'):
                timeframe = column_name.split('_')[1]  # Extract timeframe
                timeframes.add(timeframe)
        
        timeframes = sorted(list(timeframes))
        num_timeframes = len(timeframes)
        
        # Create a single figure for this forex pair
        fig = plt.figure(figsize=(14, 10))  # More square aspect ratio
        fig.suptitle(f"{forex_pair} - Price Data Visualization", fontsize=14)
        
        # Calculate grid dimensions
        if num_timeframes <= 5:
            # For 5 timeframes: 3 on top row, 2 on bottom row
            nrows = 2
            ncols = 3
        else:
            # For more timeframes, use a more square layout
            nrows = (num_timeframes + 2) // 3  # Ceiling division to get number of rows
            ncols = min(3, num_timeframes)     # Maximum 3 columns
        
        # Create GridSpec with proper spacing
        gs = GridSpec(nrows, ncols, figure=fig)
        gs.update(hspace=0.3, wspace=0.3)  # Adjust spacing between subplots
        
        # For each timeframe, create a subplot
        for i, timeframe in enumerate(timeframes):
            row = i // ncols  # Integer division to get row index
            col = i % ncols   # Modulo to get column index
            
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"{timeframe}", fontsize=12, pad=4)
            
            # Get all bar numbers for this timeframe
            bar_numbers = set()
            for column_name in df['column_name'].unique():
                if not column_name.startswith('weekday_') and f"_{timeframe}_" in column_name:
                    bar_number = int(column_name.split('_')[2])  # Extract bar number
                    bar_numbers.add(bar_number)
            
            bar_numbers = sorted(list(bar_numbers))
            
            # For each bar, plot OHLC as a candlestick-like bar chart
            x_positions = []
            opens = []
            highs = []
            lows = []
            closes = []
            timestamps = []
            
            # Sort bar numbers to ensure correct ordering
            bar_numbers = sorted(list(bar_numbers))
            
            # Process each bar and collect the data
            for bar_num in bar_numbers:
                # Filter for this specific bar
                bar_data = df[df['column_name'].isin([
                    f"Open_{timeframe}_{bar_num}",
                    f"High_{timeframe}_{bar_num}",
                    f"Low_{timeframe}_{bar_num}",
                    f"Close_{timeframe}_{bar_num}"
                ])]
                
                if not bar_data.empty:
                    # Extract timestamp from any of the rows (they should all be the same for a single bar)
                    timestamp = bar_data['timestamp'].iloc[0]
                    timestamps.append(timestamp)
                    
                    # Extract OHLC values
                    o_row = bar_data[bar_data['column_name'] == f"Open_{timeframe}_{bar_num}"]
                    h_row = bar_data[bar_data['column_name'] == f"High_{timeframe}_{bar_num}"]
                    l_row = bar_data[bar_data['column_name'] == f"Low_{timeframe}_{bar_num}"]
                    c_row = bar_data[bar_data['column_name'] == f"Close_{timeframe}_{bar_num}"]
                    
                    if not (o_row.empty or h_row.empty or l_row.empty or c_row.empty):
                        opens.append(o_row['value'].iloc[0])
                        highs.append(h_row['value'].iloc[0])
                        lows.append(l_row['value'].iloc[0])
                        closes.append(c_row['value'].iloc[0])
            
            # Plot the data if we have any
            if opens:  # Check if we have data
                # Create SEQUENTIAL x-positions with NO GAPS
                x_positions = list(range(len(opens)))  # This ensures bars are adjacent with no spaces
                
                # Plot high-low range with sequential positions
                ax.vlines(x_positions, lows, highs, color='black', linewidth=0.7)
                
                # Make bars touch each other with ABSOLUTELY NO gaps
                width = 1.0  # Width of 1.0 ensures bars fully touch with no gap
                
                # Plot open-close bars with colors
                for j, (xpos, open_val, close_val) in enumerate(zip(x_positions, opens, closes)):
                    if close_val >= open_val:
                        color = 'green'  # Bullish bar
                    else:
                        color = 'red'    # Bearish bar
                    
                    # Draw rectangle for open-close range
                    ax.bar(xpos, close_val - open_val, bottom=open_val, color=color, width=width, alpha=0.8)
                
                # Set x-axis labels - extremely minimal
                ax.set_xticks(x_positions)
                
                # Show drastically fewer labels for readability
                # If we have many bars, show only a few time labels
                if len(timestamps) > 7:
                    num_labels = min(7, len(timestamps))
                    indices = [int(i * len(timestamps)/num_labels) for i in range(num_labels)]
                    visible_ticks = [x_positions[i] for i in indices]
                    visible_labels = []
                    for i in indices:
                        t = timestamps[i]
                        time_part = t.split(' ')[1]
                        hour_min = ":".join(time_part.split(':')[:2])  # Just HH:MM
                        visible_labels.append(hour_min)
                else:
                    # If we have few bars, show all labels
                    visible_ticks = x_positions
                    visible_labels = []
                    for t in timestamps:
                        time_part = t.split(' ')[1]
                        hour_min = ":".join(time_part.split(':')[:2])  # Just HH:MM
                        visible_labels.append(hour_min)
                
                ax.set_xticks(visible_ticks)
                ax.set_xticklabels(visible_labels, rotation=0, fontsize=6)
                
                # Make tick labels smaller
                ax.tick_params(axis='y', labelsize=6)
                
                # Grid for better readability - lighter grid
                ax.grid(True, alpha=0.2, linestyle=':')
                
                # Set axis labels - minimal and compact
                ax.set_xlabel('Time', fontsize=8, labelpad=2)
                ax.set_ylabel('Price', fontsize=8, labelpad=2)
                
                # Adjust y-axis to have a reasonable range
                y_range = max(highs) - min(lows)
                y_padding = y_range * 0.1  # 10% padding
                ax.set_ylim([min(lows) - y_padding, max(highs) + y_padding])
        
        # Adjust layout without using tight_layout (to avoid warnings)
        fig.subplots_adjust(top=0.95, hspace=0.35, wspace=0.25, left=0.05, right=0.98, bottom=0.05)
        
        # Display the figure non-blocking
        plt.figure(fig.number)
        plt.show(block=False)
    
    # Keep the windows open until manually closed
    plt.show()

def main():
    """
    Main function to test visualization independently.
    Fetches forex data and visualizes it.
    """
    from main import FOREX_PAIRS, TIMEFRAMES, NUM_BARS, CUSTOM_DATE
    import pandas as pd
    from datetime import datetime
    import pytz
    
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
        print(f"Processing {forex_pair}")
        all_data[forex_pair] = {}
        for timeframe in TIMEFRAMES:
            data = get_forex_data(forex_pair, timeframe, NUM_BARS, start_date)
            
            if len(data["data"]) == 0:
                print(f"Error for {forex_pair} {timeframe}: No data available")
                continue
                
            all_data[forex_pair][timeframe] = data
    
    # Process each currency pair and combine all timeframes
    dataframes = {}
    for forex_pair in FOREX_PAIRS:
        # Create entries
        original_entries = []
        
        for timeframe in TIMEFRAMES:
            if timeframe not in all_data[forex_pair]:
                continue
                
            data = all_data[forex_pair][timeframe]
            
            for i in range(1, NUM_BARS + 1):
                if i-1 < len(data["data"]):  # Check if this bar exists
                    timestamp = data["data"][i-1]["time"]
                    
                    # Create entries for Open, High, Low, Close
                    for value_type in ["open", "high", "low", "close"]:
                        entry = {
                            "timestamp": timestamp,
                            "column_name": f"{value_type.capitalize()}_{timeframe}_{i}",
                            "value": data["data"][i-1][value_type]
                        }
                        original_entries.append(entry)
        
        # Calculate weekday entries
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
        
        # Combine all entries
        all_entries = original_entries + weekday_entries
        
        # Store in dataframe
        dataframes[forex_pair] = pd.DataFrame(all_entries)
        
        # Print sample data
        print(f"\n{forex_pair} Sample Data:")
        print("-" * 40)
        
        # Print first few entries
        for i, row in dataframes[forex_pair].head(10).iterrows():
            timestamp = row['timestamp']
            column_name = row['column_name']
            value = row['value']
            
            if column_name.startswith('weekday_'):
                print(f"{timestamp}::{column_name}: {int(value)}")
            else:
                print(f"{timestamp}::{column_name}: {value:.6f}")
    
    # Visualize the data
    print("\nGenerating visualizations for each forex pair...")
    visualize_forex_data(dataframes)

if __name__ == "__main__":
    main()
