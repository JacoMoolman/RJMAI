import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from jmai_toolbox import get_day_of_week_num, load_currency_pairs, display_currency_pairs, graph_display_dataframes, filter_dataframes_before_date, normalize_dataframes_separately, create_flat_dataframes, display_flat_dataframes, add_day_of_week_to_dataframes, setup_matplotlib_visualization
from datetime import datetime, timedelta

# Path to pickle files directory
PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picklefiles")

# List of all timeframes (uncomment/comment to include/exclude)
TIMEFRAMES = [
    'D1',
    'H1',
    'H4',
    'M1',
    'M5',
    'M30'
]

def get_dataframe(currency_pair, start_date, num_bars_to_plot):
    """
    Get data for a single currency pair and return it as a flat dataframe.
    
    Args:
        currency_pair (str): The currency pair to retrieve (e.g., 'EURUSD')
        start_date (str): Start date from where the display dataframes should start getting data
        num_bars_to_plot (int): Number of bars to plot for each timeframe
    
    Returns:
        pandas.DataFrame: Flat dataframe containing the requested currency pair data
    """
    # Convert to list for compatibility with load_currency_pairs function
    currency_pairs = [currency_pair]
    
    # Load the specified currency pair
    dataframes = load_currency_pairs(currency_pairs, TIMEFRAMES, PICKLE_DIR)
    
    # Convert start_date to datetime
    current_date = pd.to_datetime(start_date)
    
    # Format current date as string for display
    current_date_str = current_date.strftime('%Y-%m-%d %H:%M')
    # print(f"\nProcessing date: {current_date_str}")
    
    # Find the last date in the dataframes to know when to stop the loop
    # last_date = datetime.min
    # for key, df in dataframes.items():
    #     if not df.empty and 'time' in df.columns:
    #         current_last_date = df['time'].max()
    #         if current_last_date > last_date:
    #             last_date = current_last_date

    # print(f"Animation will run from {start_date} to {last_date}")
    
    #EXTRACT DATES
    display_dataframes = filter_dataframes_before_date(dataframes, current_date, num_bars_to_plot)
    
    #NORMALIZE
    display_dataframes = normalize_dataframes_separately(display_dataframes)
    
    #CREATE FLAT DATAFRAMES
    display_dataframes_flat = create_flat_dataframes(display_dataframes)
    
    #DISPLAY IN CONSOLE
    # display_currency_pairs(display_dataframes)
    
    #DISPLAY FLAT DATAFRAMES IN CONSOLE
    # display_flat_dataframes(display_dataframes_flat)
    
    #GRAPH
    # figures = graph_display_dataframes(display_dataframes, figures, current_date)
    
    #PAUSE
    # plt.pause(5.5)
    
    #PLUS ONE MINUTE
    # current_date += timedelta(minutes=1)
    
    # Return the flat dataframe for the requested currency pair
    if currency_pair in display_dataframes_flat:
        return display_dataframes_flat[currency_pair]
    else:
        print(f"Warning: No data found for {currency_pair}")
        return pd.DataFrame()  # Return empty dataframe if no data found
