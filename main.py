import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from jmai_toolbox import get_day_of_week_num, create_display_dataframes, load_currency_pairs, display_currency_pairs

# Number of bars to plot for each timeframe
NUM_BARS_TO_PLOT = 5

# Dictionary to store all DataFrames
dataframes = {}

# Dictionary to store display DataFrames
display_dataframes = {}

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

# Path to pickle files directory
PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picklefiles")

def add_day_of_week_to_dataframes(dfs):
    for key in dfs:
        dfs[key]['day_of_week'] = dfs[key]['time'].apply(get_day_of_week_num)
    return dfs



#####MAIN#####
# Load all currency pairs
dataframes = load_currency_pairs(CURRENCY_PAIRS, TIMEFRAMES, PICKLE_DIR)

# Create display dataframes with the specified number of bars
display_dataframes = create_display_dataframes(dataframes, NUM_BARS_TO_PLOT)

# Add day of week to all dataframes
# dataframes = add_day_of_week_to_dataframes(display_dataframes)

# Display the display DataFrames
display_currency_pairs(display_dataframes, rows=NUM_BARS_TO_PLOT)
#####MAIN#####