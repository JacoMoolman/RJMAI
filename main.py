import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from jmai_toolbox import get_day_of_week_num, load_currency_pairs, display_currency_pairs, graph_display_dataframes, filter_dataframes_before_date, normalize_dataframes_separately, create_flat_dataframes, display_flat_dataframes
from datetime import datetime, timedelta

# Start date from where the display dataframes should start getting data
START_DATE = '2024-01-08 00:00'

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
    'EURUSD',
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

# Find the last date in the dataframes to know when to stop the loop
last_date = datetime.min
for key, df in dataframes.items():
    if not df.empty and 'time' in df.columns:
        current_last_date = df['time'].max()
        if current_last_date > last_date:
            last_date = current_last_date

print(f"Animation will run from {START_DATE} to {last_date}")

# Convert START_DATE to datetime
current_date = pd.to_datetime(START_DATE)

# Set the style to dark background
plt.style.use('dark_background')

# Enable interactive mode for real-time updates
plt.ion()

# Dictionary to store figures for each currency pair
figures = {}

#LOOP FROM HERE
try:
    while current_date <= last_date:
        # Format current date as string for display
        current_date_str = current_date.strftime('%Y-%m-%d %H:%M')
        print(f"\nProcessing date: {current_date_str}")
        
        #EXTRACT DATES
        display_dataframes = filter_dataframes_before_date(dataframes, current_date, NUM_BARS_TO_PLOT)
        
        #NORMALIZE
        display_dataframes = normalize_dataframes_separately(display_dataframes)
        
        #CREATE FLAT DATAFRAMES
        display_dataframes_flat = create_flat_dataframes(display_dataframes)
        
        #DISPLAY IN CONSOLE
        display_currency_pairs(display_dataframes)
        
        #DISPLAY FLAT DATAFRAMES IN CONSOLE
        display_flat_dataframes(display_dataframes_flat)
        
        #GRAPH
        # figures = graph_display_dataframes(display_dataframes, figures, current_date)
        
        #PAUSE
        plt.pause(5.5)
        
        #PLUS ONE MINUTE
        current_date += timedelta(minutes=1)
        
except KeyboardInterrupt:
    print("\nAnimation stopped")
except Exception as e:
    print(f"\nAnimation error: {e}")
finally:
    # Turn off interactive mode when done
    plt.ioff()
    
    # Inform user the program is ending
    print("\nAnimation ended. Exiting program.")

### END LOOP
#####MAIN#####