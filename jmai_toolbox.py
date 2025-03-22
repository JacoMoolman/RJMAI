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


def graph_display_dataframes(display_dfs, figs=None, current_date=None):
    """
    Graph the display dataframes with one window per currency pair.
    Each window will contain multiple charts for the different timeframes of that currency pair.
    
    Args:
        display_dfs: Dictionary of display dataframes to graph
        figs: Optional dictionary of existing figures to plot on. If None, new figures are created.
        current_date: Optional current date to display in the figure title
    """
    print("\nGenerating forex bar charts...")
    
    # Check if any DataFrames were loaded
    if not display_dfs:
        print("No display DataFrames to graph!")
        return {}
    
    # Group the keys by currency pair
    pair_to_timeframes = {}
    for key in display_dfs.keys():
        # Split the key into currency pair and timeframe (e.g., "EURUSD_H1" -> "EURUSD" and "H1")
        parts = key.split('_')
        if len(parts) >= 2:
            currency_pair = parts[0]
            timeframe = '_'.join(parts[1:])  # Handle case where timeframe might contain underscores
            
            if currency_pair not in pair_to_timeframes:
                pair_to_timeframes[currency_pair] = []
            
            pair_to_timeframes[currency_pair].append(key)
    
    # Create or use dictionary to store figures
    if figs is None:
        figs = {}
    
    # Set dark mode style
    plt.style.use('dark_background')
    
    # For each currency pair, create or update a figure with subplots for each timeframe
    for currency_pair, timeframe_keys in pair_to_timeframes.items():
        # Sort timeframe keys to ensure consistent layout
        timeframe_keys.sort()
        
        # Calculate the optimal subplot layout based on number of timeframes
        num_timeframes = len(timeframe_keys)
        if num_timeframes <= 3:
            rows, cols = 1, num_timeframes
        elif num_timeframes <= 6:
            rows, cols = 2, 3
        elif num_timeframes <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = (num_timeframes + 3) // 4, 4  # Make sure there are enough slots
        
        # Create or clear figure for this currency pair
        if currency_pair not in figs:
            figs[currency_pair] = plt.figure(figsize=(cols*4, rows*3))
            figs[currency_pair].canvas.manager.set_window_title(f"{currency_pair} Charts")
        else:
            figs[currency_pair].clear()
        
        # Set the figure title with current date if provided
        if current_date:
            current_date_str = current_date.strftime('%Y-%m-%d %H:%M') if hasattr(current_date, 'strftime') else str(current_date)
            figs[currency_pair].suptitle(f"{currency_pair} - Data as of {current_date_str}", fontsize=14, color='white')
        
        # Create subplots for each timeframe
        for i, key in enumerate(timeframe_keys):
            if i >= rows * cols:
                print(f"Warning: Not enough subplot space for {key}, skipping.")
                continue
            
            df = display_dfs[key]
            
            # Create subplot
            ax = figs[currency_pair].add_subplot(rows, cols, i+1)
            
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
                                       fontsize=8,
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
                    
                # Extract timeframe from key for the title
                timeframe = key.split('_')[1] if '_' in key else key
                
                # Set title and labels
                ax.set_title(timeframe, color='white')
                ax.set_ylabel('Price', color='white')
                ax.tick_params(axis='y', colors='#cccccc')
                ax.grid(True, color='#555555', linestyle='-', linewidth=0.5, alpha=0.7)
                
            else:
                ax.text(0.5, 0.5, f"Missing OHLC data for {key}", 
                        ha='center', va='center', fontsize=12, color='white')
        
        # Adjust layout with more space for x-axis labels
        figs[currency_pair].tight_layout(pad=2.5)
        figs[currency_pair].subplots_adjust(bottom=0.15)
        
        # Draw the figure and process events
        figs[currency_pair].canvas.draw()
        figs[currency_pair].canvas.flush_events()
    
    # Return the dictionary of figures
    return figs


def filter_dataframes_before_date(dfs, start_date, num_bars):
    """
    Filter dataframes to get data BEFORE a specified start date, ensuring higher timeframes
    don't reveal future information about lower timeframes.
    
    Args:
        dfs: Dictionary of dataframes to filter
        start_date: The cutoff date (only data that is fully completed by this date will be included)
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
        if df.empty:
            filtered_dfs[key] = df.copy()
            continue
            
        # Extract timeframe from the key (e.g., 'AUDUSD_M5' -> 'M5')
        timeframe = key.split('_')[-1] if '_' in key else None
        
        # Calculate the adjusted cutoff time based on timeframe to prevent future information leakage
        cutoff_time = start_date
        
        if timeframe:
            # Calculate the end time of the bar that would contain the start_date
            if timeframe == 'M1':
                # For M1, the bar ends at the next minute
                bar_duration = pd.Timedelta(minutes=1)
            elif timeframe == 'M5':
                bar_duration = pd.Timedelta(minutes=5)
            elif timeframe == 'M15':
                bar_duration = pd.Timedelta(minutes=15)
            elif timeframe == 'M30':
                bar_duration = pd.Timedelta(minutes=30)
            elif timeframe == 'H1':
                bar_duration = pd.Timedelta(hours=1)
            elif timeframe == 'H4':
                bar_duration = pd.Timedelta(hours=4)
            elif timeframe == 'D1':
                bar_duration = pd.Timedelta(days=1)
            else:
                # Default to no adjustment
                bar_duration = pd.Timedelta(minutes=0)
            
            # Calculate when the current bar started
            if timeframe.startswith('M'):  # Minute-based timeframes
                minutes = int(timeframe[1:]) if len(timeframe) > 1 else 1
                # Calculate the start of the current bar
                bar_start = pd.Timestamp(
                    year=start_date.year, 
                    month=start_date.month, 
                    day=start_date.day,
                    hour=start_date.hour, 
                    minute=(start_date.minute // minutes) * minutes
                )
                # Only use bars that are fully completed
                cutoff_time = bar_start
            elif timeframe.startswith('H'):  # Hour-based timeframes
                hours = int(timeframe[1:]) if len(timeframe) > 1 else 1
                # Calculate the start of the current bar
                bar_start = pd.Timestamp(
                    year=start_date.year, 
                    month=start_date.month, 
                    day=start_date.day,
                    hour=(start_date.hour // hours) * hours, 
                    minute=0
                )
                # Only use bars that are fully completed
                cutoff_time = bar_start
            elif timeframe == 'D1':  # Daily timeframe
                bar_start = pd.Timestamp(
                    year=start_date.year, 
                    month=start_date.month, 
                    day=start_date.day,
                    hour=0, 
                    minute=0
                )
                # Only use bars that are fully completed
                cutoff_time = bar_start
        
        # Get the data strictly before the adjusted cutoff time
        # This ensures we only show bars that are fully completed as of the current time
        before_cutoff = df[df['time'] < cutoff_time].sort_values('time', ascending=False)
        
        # Take the last num_bars before the cutoff
        selected_data = before_cutoff.head(num_bars).sort_values('time')
        
        # Update the dataframe
        filtered_dfs[key] = selected_data.reset_index(drop=True)
        
        # Make sure we have at least some data
        if len(filtered_dfs[key]) < num_bars:
            print(f"Warning: {key} has only {len(filtered_dfs[key])} rows before {cutoff_time}, which is less than the requested {num_bars}")
    
    return filtered_dfs


def normalize_dataframes_separately(display_dfs):
    """
    Normalize each timeframe of each currency pair separately.
    
    This function takes the display_dataframes dictionary and normalizes
    the OHLC values and tick_volume of each timeframe for each currency pair independently.
    Spread values are kept as is.
    
    Args:
        display_dfs: Dictionary of dataframes to normalize, where keys are in 
                    the format 'CURRENCY_TIMEFRAME' (e.g., 'EURUSD_M5')
    
    Returns:
        Dictionary of normalized dataframes with the same structure as the input
    """
    print("\nNormalizing each timeframe of each currency pair separately...")
    
    # Create a deep copy to avoid modifying the original
    normalized_dfs = {}
    
    # Process each dataframe separately
    for key, df in display_dfs.items():
        if df.empty:
            normalized_dfs[key] = df.copy()
            continue
        
        # Create a copy of the dataframe
        normalized_df = df.copy()
        
        # Normalize the OHLC columns for this specific currency pair and timeframe
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Get min and max values for OHLC columns in this specific dataframe
            min_val = min(
                df['open'].min(),
                df['high'].min(),
                df['low'].min(),
                df['close'].min()
            )
            
            max_val = max(
                df['open'].max(),
                df['high'].max(),
                df['low'].max(),
                df['close'].max()
            )
            
            # Avoid division by zero
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1
                
            # Normalize OHLC columns to range [0, 1]
            normalized_df['open'] = (df['open'] - min_val) / range_val
            normalized_df['high'] = (df['high'] - min_val) / range_val
            normalized_df['low'] = (df['low'] - min_val) / range_val
            normalized_df['close'] = (df['close'] - min_val) / range_val
        
        # Normalize tick_volume if it exists
        if 'tick_volume' in df.columns and not df['tick_volume'].empty:
            min_tick = df['tick_volume'].min()
            max_tick = df['tick_volume'].max()
            
            # Avoid division by zero
            range_tick = max_tick - min_tick
            if range_tick == 0:
                range_tick = 1
                
            # Normalize tick_volume to range [0, 1]
            normalized_df['tick_volume'] = (df['tick_volume'] - min_tick) / range_tick
        
        # Note: Spread is intentionally not normalized and kept as is
            
        # Store the normalized dataframe
        normalized_dfs[key] = normalized_df
    
    return normalized_dfs


def create_flat_dataframes(display_dfs):
    """
    Create flattened 1D dataframes from the display dataframes.
    
    This function takes the display_dataframes dictionary and creates a flat representation
    with one dataframe per currency pair. Each dataframe contains all timeframes with columns
    prefixed by the timeframe name and numbered according to the row index
    (e.g., M1_open_0, M1_close_0, M1_open_1, M1_close_1, etc.)
    
    Args:
        display_dfs: Dictionary of display dataframes where keys are in 
                    the format 'CURRENCY_TIMEFRAME' (e.g., 'EURUSD_M5')
    
    Returns:
        Dictionary of flattened dataframes, one per currency pair
    """
    # Check if any DataFrames were loaded
    if not display_dfs:
        print("No display DataFrames to flatten!")
        return {}
    
    # Group the keys by currency pair
    pair_to_timeframes = {}
    for key in display_dfs.keys():
        # Split the key into currency pair and timeframe (e.g., "EURUSD_H1" -> "EURUSD" and "H1")
        parts = key.split('_')
        if len(parts) >= 2:
            currency_pair = parts[0]
            timeframe = '_'.join(parts[1:])  # Handle case where timeframe might contain underscores
            
            if currency_pair not in pair_to_timeframes:
                pair_to_timeframes[currency_pair] = []
            
            pair_to_timeframes[currency_pair].append((timeframe, key))
    
    # Create a dictionary to store the flat dataframes
    flat_dfs = {}
    
    # Process each currency pair
    for currency_pair, timeframe_keys in pair_to_timeframes.items():
        # Create a dictionary to hold the flattened data
        flat_data = {}
        
        # Process each timeframe for this currency pair
        for timeframe, full_key in timeframe_keys:
            df = display_dfs[full_key]
            
            # Extract the key columns we want to keep
            columns_to_keep = ['open', 'high', 'low', 'close', 'tick_volume', 'spread']
            
            # Add each column to the flat data with timeframe prefix and row index
            for idx, row in df.iterrows():
                for col in columns_to_keep:
                    if col in df.columns:
                        # Use the row index to number each entry
                        flat_data[f"{timeframe}_{col}_{idx}"] = row[col] if not pd.isna(row[col]) else None
        
        # Create a dataframe from the flattened data
        flat_df = pd.DataFrame([flat_data])
        
        # Store in the result dictionary
        flat_dfs[currency_pair] = flat_df
    
    return flat_dfs


def display_flat_dataframes(flat_dfs):
    """
    Display the flattened dataframes in the console.
    
    Args:
        flat_dfs: Dictionary of flattened dataframes, one per currency pair
    """
    print("\nDisplaying flattened dataframes:")
    
    # Check if any DataFrames were created
    if not flat_dfs:
        print("No flattened DataFrames to display!")
        return
    
    # Display each DataFrame
    for currency_pair, df in flat_dfs.items():
        print(f"\n{currency_pair} Flat DataFrame:")
        
        # Transpose the dataframe for cleaner display in console
        # This will show each indicator as a row instead of a column
        transposed = df.T
        transposed.columns = ['Value']
        print(transposed)
