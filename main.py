import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from jmai_toolbox import get_day_of_week_num, create_display_dataframes, load_currency_pairs, display_currency_pairs, graph_display_dataframes, filter_dataframes_before_date

# Start date from where the display dataframes should start getting data
START_DATE = '2025-02-01 00:00'

# Number of bars to plot for each timeframe
NUM_BARS_TO_PLOT = 100

# Dictionary to store all DataFrames
dataframes = {}

# Dictionary to store display DataFrames
display_dataframes = {}

# Path to pickle files directory
PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picklefiles")

# List of all currency pairs (uncomment/comment to include/exclude)
CURRENCY_PAIRS = [
    'AUDUSD',
    # 'EURUSD',
    # 'GBPUSD',
    # 'USDCAD',
    # 'USDCHF',
    # 'USDJPY'
]

# List of all timeframes (uncomment/comment to include/exclude)
TIMEFRAMES = [
    'D1',
    'H1',
    'H4',
    'M1',
    'M5',
    'M30'
]


def add_day_of_week_to_dataframes(dfs):
    for key in dfs:
        dfs[key]['day_of_week'] = dfs[key]['time'].apply(get_day_of_week_num)
    return dfs



#####MAIN#####
# Load all currency pairs
dataframes = load_currency_pairs(CURRENCY_PAIRS, TIMEFRAMES, PICKLE_DIR)

# Filter dataframes to get data BEFORE START_DATE for the specified number of bars
dataframes_filtered = filter_dataframes_before_date(dataframes, START_DATE, NUM_BARS_TO_PLOT)

# Create display dataframes with the filtered data
display_dataframes = create_display_dataframes(dataframes_filtered, NUM_BARS_TO_PLOT)

# Add day of week to all dataframes
# dataframes = add_day_of_week_to_dataframes(display_dataframes)

# Display the display DataFrames
display_currency_pairs(display_dataframes, rows=NUM_BARS_TO_PLOT)

# Graph the display dataframes with 3 charts at top and 3 at bottom
graph_display_dataframes(display_dataframes)
#####MAIN#####