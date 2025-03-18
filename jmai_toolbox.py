"""
JMAI Toolbox - A collection of utility functions for RJMAI project

This module provides reusable functions that can be imported and used by other scripts
in the RJMAI project.
"""

import datetime
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib import patches
import mplfinance as mpf


def get_day_of_week_num(date_input: Union[str, datetime.date, datetime.datetime], 
                       format_str: str = "%Y-%m-%d") -> int:
    """
    Calculate the day of the week as a numerical value (1-7, with 1 being Monday)
    
    Args:
        date_input: Date in string format or as datetime/date object
        format_str: Format string for parsing date strings (default: "%Y-%m-%d")
        
    Returns:
        int: The day of the week as a number (1=Monday, 2=Tuesday, ..., 7=Sunday)
        
    Raises:
        ValueError: If the date_input cannot be parsed
    """
    try:
        # Convert string to datetime if needed
        if isinstance(date_input, str):
            date_obj = datetime.datetime.strptime(date_input, format_str).date()
        elif isinstance(date_input, datetime.datetime):
            date_obj = date_input.date()
        elif isinstance(date_input, datetime.date):
            date_obj = date_input
        else:
            raise TypeError("date_input must be a string, datetime, or date object")
        
        # Get day of week as a number (isoweekday: 1=Monday, 7=Sunday)
        day_of_week_num = date_obj.isoweekday()
        return day_of_week_num
        
    except Exception as e:
        raise


def load_currency_pairs(currency_pairs, timeframes, pickle_dir):
    print(f"Loading currency pairs from pickle files...")
    start_time = time.time()
    
    # Create empty dictionary to store DataFrames
    dfs = {}
    loaded_count = 0
    
    # Check if pickle directory exists
    if not os.path.exists(pickle_dir):
        print(f"Error: Pickle directory {pickle_dir} does not exist!")
        print("Please run convert_to_pickle.py first to create pickle files.")
        return dfs
    
    # Process each currency pair and timeframe combination
    for pair in currency_pairs:
        for timeframe in timeframes:
            # Create the full pair_timeframe string (e.g., "EURUSD_H1")
            pair_timeframe = f"{pair}_{timeframe}"
            
            # Full path to pickle file
            pickle_file = os.path.join(pickle_dir, f"{pair_timeframe}.pkl")
            
            # Check if pickle file exists
            if os.path.exists(pickle_file):
                try:
                    # Load from pickle file
                    df = pd.read_pickle(pickle_file)
                    
                    # Store DataFrame in dictionary
                    dfs[pair_timeframe] = df
                    loaded_count += 1
                    print(f"Loaded {pair_timeframe} - {len(df)} rows")
                except Exception as e:
                    print(f"Error loading {pair_timeframe}: {e}")
            else:
                print(f"Skipped {pair_timeframe} - pickle file not found")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Loaded {loaded_count} DataFrames in {elapsed_time:.2f} seconds")
    
    return dfs


def display_currency_pairs(display_dfs, rows=None):
    if rows:
        print(f"\nDisplaying first {rows} rows of each display DataFrame:")
    else:
        print(f"\nDisplaying all rows of each display DataFrame:")
    
    # Check if any DataFrames were loaded
    if not display_dfs:
        print("No display DataFrames to display!")
        return
    
    # Display each DataFrame
    for pair_timeframe, df in display_dfs.items():
        print(f"\n{pair_timeframe}:")
        if rows:
            print(df.head(rows))
        else:
            print(df)
        # print("-" * 80)


def graph_display_dataframes(display_dfs, fig=None):
    """
    Graph the display dataframes with 3 charts at the top and 3 at the bottom.
    
    Args:
        display_dfs: Dictionary of display dataframes to graph
        fig: Optional matplotlib figure to plot on. If None, a new figure is created.
    """
    print("\nGenerating forex bar charts...")
    
    # Check if any DataFrames were loaded
    if not display_dfs:
        print("No display DataFrames to graph!")
        return
    
    # Get the keys from the display_dfs dictionary
    keys = list(display_dfs.keys())
    
    # Only process up to 6 charts
    chart_keys = keys[:6]
    num_charts = len(chart_keys)
    
    # Set dark mode style
    plt.style.use('dark_background')
    
    # Define subplot layout - 2 rows, 3 columns
    if fig is None:
        fig = plt.figure(figsize=(18, 12))
    
    # Plot each chart in its own subplot
    for i, key in enumerate(chart_keys):
        df = display_dfs[key]
        
        # Calculate row and column position
        row = i // 3
        col = i % 3
        
        # Create subplot
        ax = fig.add_subplot(2, 3, i+1)
        
        # Format data for plotting
        # Ensure the DataFrame has the required OHLC columns
        required_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in required_cols):
            # Create OHLC candlestick chart
            if 'time' in df.columns:
                x = range(len(df))
                
                # Draw candlesticks
                width = 0.6
                width2 = width / 2
                
                up = df[df.close >= df.open]
                down = df[df.close < df.open]
                
                # Draw up candles - bright green for dark mode
                if not up.empty:
                    ax.bar(x=up.index, height=up.close-up.open, bottom=up.open, 
                           width=width, color='#00ff00', alpha=0.7)
                    ax.bar(x=up.index, height=up.high-up.close, bottom=up.close, 
                           width=width2, color='#00ff00', alpha=0.7)
                    ax.bar(x=up.index, height=up.open-up.low, bottom=up.low, 
                           width=width2, color='#00ff00', alpha=0.7)
                
                # Draw down candles - bright red for dark mode
                if not down.empty:
                    ax.bar(x=down.index, height=down.open-down.close, bottom=down.close, 
                           width=width, color='#ff3333', alpha=0.7)
                    ax.bar(x=down.index, height=down.high-down.open, bottom=down.open, 
                           width=width2, color='#ff3333', alpha=0.7)
                    ax.bar(x=down.index, height=down.close-down.low, bottom=down.low, 
                           width=width2, color='#ff3333', alpha=0.7)
                
                # Set x-axis ticks to show dates
                if 'time' in df.columns:
                    # Only show start and end dates instead of all dates
                    if len(df) > 0:
                        start_date = df['time'].iloc[0]
                        end_date = df['time'].iloc[-1]
                        
                        # Remove x-axis ticks completely
                        ax.set_xticks([])
                        
                        # Add a text label at bottom center showing date range
                        date_range_text = f"{start_date} to {end_date}"
                        ax.annotate(date_range_text, 
                                   xy=(0.5, -0.12), 
                                   xycoords='axes fraction',
                                   ha='center',
                                   fontsize=10,
                                   color='#cccccc')
                    
                    # Hide other x-tick labels to avoid clutter
                    ax.tick_params(axis='x', pad=8)
                
                # Calculate and set appropriate y-axis limits with a small margin
                y_min = df['low'].min()
                y_max = df['high'].max()
                y_range = y_max - y_min
                margin = y_range * 0.05  # 5% margin
                ax.set_ylim(y_min - margin, y_max + margin)
                
            else:
                # Just plot a basic chart if we don't have time data
                ax.plot(df.close, label='Close', color='#33ccff')
                
            # Set title and labels
            ax.set_title(key, color='white')
            ax.set_ylabel('Price', color='white')
            ax.tick_params(axis='y', colors='#cccccc')
            ax.grid(True, color='#555555', linestyle='-', linewidth=0.5, alpha=0.7)
            
        else:
            ax.text(0.5, 0.5, f"Missing OHLC data for {key}", 
                    ha='center', va='center', fontsize=12, color='white')
    
    # Adjust layout with more space for x-axis labels and no space for title
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(bottom=0.15)
    
    # Show the plot
    plt.show()


def filter_dataframes_before_date(dfs, start_date, num_bars):
    """
    Filter dataframes to get data BEFORE a specified start date.
    
    Args:
        dfs: Dictionary of dataframes to filter
        start_date: The cutoff date (will include data on or before this date)
        num_bars: Number of bars to get before start_date
        
    Returns:
        Dictionary of filtered dataframes
    """
    print(f"\nFiltering dataframes to get {num_bars} bars before {start_date}...")
    
    # Create a deep copy to avoid modifying the original
    filtered_dfs = {}
    
    # Convert start_date to pandas datetime if it's a string
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Process each dataframe
    for key, df in dfs.items():
        # Get the data before start_date
        before_start = df[df['time'] <= start_date].sort_values('time', ascending=False)
        
        # Take the last num_bars before (and including) start_date
        selected_data = before_start.head(num_bars).sort_values('time')
        
        # Update the dataframe
        filtered_dfs[key] = selected_data.reset_index(drop=True)
        
        # Make sure we have at least some data
        if len(filtered_dfs[key]) < num_bars:
            print(f"Warning: {key} has only {len(filtered_dfs[key])} rows before {start_date}, which is less than the requested {num_bars}")
    
    return filtered_dfs
