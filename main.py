import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from jmai_toolbox import get_day_of_week_num, create_display_dataframes, load_currency_pairs, display_currency_pairs, graph_display_dataframes

# Start date from where the display dataframes should start getting data
START_DATE = '2024-02-01 00:00'

# Number of bars to plot for each timeframe
NUM_BARS_TO_PLOT = 100

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

# Filter dataframes to only include data from START_DATE onwards
for key, df in dataframes.items():
    # Convert START_DATE to pandas datetime
    start_date = pd.to_datetime(START_DATE)
    # Filter the dataframe to only include rows from start_date onwards
    dataframes[key] = df[df['time'] >= start_date].reset_index(drop=True)
    # Make sure we have at least NUM_BARS_TO_PLOT rows
    if len(dataframes[key]) < NUM_BARS_TO_PLOT:
        print(f"Warning: {key} has only {len(dataframes[key])} rows from {START_DATE}, which is less than the requested {NUM_BARS_TO_PLOT}")

# Create display dataframes with the specified number of bars
display_dataframes = create_display_dataframes(dataframes, NUM_BARS_TO_PLOT)

# Print debug information about the data we're about to plot
for key, df in display_dataframes.items():
    print(f"\nData to be plotted for {key}:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Min/Max values - Open: {df['open'].min():.5f}/{df['open'].max():.5f}, Close: {df['close'].min():.5f}/{df['close'].max():.5f}")

# Add day of week to all dataframes
# dataframes = add_day_of_week_to_dataframes(display_dataframes)

# Display the display DataFrames
display_currency_pairs(display_dataframes, rows=NUM_BARS_TO_PLOT)

# Graph the display dataframes with 3 charts at top and 3 at bottom
graph_display_dataframes(display_dataframes)
#####MAIN#####