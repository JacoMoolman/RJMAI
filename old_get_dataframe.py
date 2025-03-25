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

    # Convert to list for compatibility with load_currency_pairs function
    currency_pairs = [currency_pair]
    
    # Load the specified currency pair
    dataframes = load_currency_pairs(currency_pairs, TIMEFRAMES, PICKLE_DIR)
    
    # Convert start_date to datetime
    current_date = pd.to_datetime(start_date)
    
    # Format current date as string for display
    current_date_str = current_date.strftime('%Y-%m-%d %H:%M')
    # print(f"\nProcessing date: {current_date_str}")
    

    
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
