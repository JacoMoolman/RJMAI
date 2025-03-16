"""
JMAI Toolbox - A collection of utility functions for RJMAI project

This module provides reusable functions that can be imported and used by other scripts
in the RJMAI project.
"""

import datetime
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import time


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
    """
    Load currency pairs from pickle files.
    
    Args:
        currency_pairs: List of currency pairs to load
        timeframes: List of timeframes to load
        pickle_dir: Directory containing pickle files
        
    Returns:
        Dictionary of DataFrames containing forex data
    """
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


def create_display_dataframes(dfs, num_bars=20):
    """
    Create display dataframes containing the specified number of bars for each currency pair.
    
    Args:
        dfs: Dictionary of DataFrames containing forex data
        num_bars: Number of bars to include in each display DataFrame
        
    Returns:
        Dictionary of display DataFrames
    """
    print(f"\nCreating display dataframes with {num_bars} bars each...")
    
    # Create empty dictionary to store display DataFrames
    display_dfs = {}
    
    # Check if any DataFrames were loaded
    if not dfs:
        print("No DataFrames to process!")
        return display_dfs
    
    # Create display DataFrame for each currency pair
    for pair_timeframe, df in dfs.items():
        # Copy the first num_bars rows to the display DataFrame
        display_dfs[pair_timeframe] = df.head(num_bars).copy()
        print(f"Created display DataFrame for {pair_timeframe} with {len(display_dfs[pair_timeframe])} bars")
    
    return display_dfs


def display_currency_pairs(display_dfs, rows=None):
    """
    Display the content of each display DataFrame.
    
    Args:
        display_dfs: Dictionary of display DataFrames
        rows: Optional parameter to limit the number of rows to display (if None, displays all rows)
    """
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
        print("-" * 80)


def plot_forex_data(dfs, currency_pairs, timeframes, num_bars=20):
    """
    Plot forex data as candlestick charts for each currency pair and timeframe.
    
    Args:
        dfs: Dictionary of DataFrames containing forex data
        currency_pairs: List of currency pairs to plot
        timeframes: List of timeframes to plot
        num_bars: Number of bars to plot for each timeframe
    """
    print(f"\nPlotting forex data with {num_bars} bars per timeframe...")
    
    # Check if any DataFrames were loaded
    if not dfs:
        print("No DataFrames to plot!")
        return
    
    # Create a figure for each currency pair
    for pair in currency_pairs:
        # Check if we have data for this pair
        pair_data = {tf: dfs.get(f"{pair}_{tf}") for tf in timeframes if f"{pair}_{tf}" in dfs}
        
        if not pair_data:
            print(f"No data available for {pair}, skipping...")
            continue
        
        # Calculate rows and columns for the grid layout
        n_timeframes = len(pair_data)
        if n_timeframes <= 3:
            n_rows, n_cols = 1, n_timeframes
        else:
            n_rows, n_cols = 2, 3
        
        # Create a figure with subplots in a grid layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharex=False)
        fig.suptitle(f"{pair} - Forex Data", fontsize=16)
        
        # Flatten axes array for easy indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each timeframe
        for i, (timeframe, df) in enumerate(pair_data.items()):
            if df is None or df.empty or i >= len(axes):
                continue
                
            # Get the data for plotting (limit to num_bars)
            plot_data = df.head(num_bars).copy()
            
            # Create candlestick chart
            ax = axes[i]
            
            # Plot candlesticks
            width = 0.6  # width of candlesticks
            indices = range(len(plot_data))
            
            for j in indices:
                # Get OHLC values
                open_price = plot_data.iloc[j]['open']
                high_price = plot_data.iloc[j]['high']
                low_price = plot_data.iloc[j]['low']
                close_price = plot_data.iloc[j]['close']
                
                # Determine color based on price movement
                color = 'green' if close_price >= open_price else 'red'
                
                # Plot the high-low line (wick)
                ax.plot([j, j], [low_price, high_price], color='black', linewidth=1)
                
                # Plot the body of the candlestick
                if close_price >= open_price:
                    # Bullish candle (green)
                    rect = patches.Rectangle((j-width/2, open_price), width, close_price-open_price, 
                                            facecolor='green', edgecolor='black', linewidth=1)
                else:
                    # Bearish candle (red)
                    rect = patches.Rectangle((j-width/2, close_price), width, open_price-close_price, 
                                            facecolor='red', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            # Set labels and title
            ax.set_title(f"{pair}_{timeframe}")
            ax.set_ylabel("Price")
            
            # Remove x-axis labels (dates)
            ax.set_xticks([])
            ax.set_xticklabels([])
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits with a bit of padding
            if len(plot_data) > 0:
                y_min = plot_data['low'].min()
                y_max = plot_data['high'].max()
                padding = (y_max - y_min) * 0.05  # 5% padding
                ax.set_ylim(y_min - padding, y_max + padding)
        
        # Hide any unused subplots
        for i in range(len(pair_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Adjust to make room for the suptitle
    
    # Show all plots
    plt.show()
