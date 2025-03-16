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
