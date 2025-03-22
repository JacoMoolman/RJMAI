import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from jmai_toolbox import get_day_of_week_num, load_currency_pairs, display_currency_pairs, graph_display_dataframes, filter_dataframes_before_date
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

# Create the figure 
fig = plt.figure(figsize=(18, 12))

#LOOP FROM HERE
try:
    while current_date <= last_date:
        # Format current date as string for display
        current_date_str = current_date.strftime('%Y-%m-%d %H:%M')
        print(f"\nProcessing date: {current_date_str}")
        
        # Filter dataframes to get data BEFORE current_date for the specified number of bars
        display_dataframes = filter_dataframes_before_date(dataframes, current_date, NUM_BARS_TO_PLOT)
        
        # Clear the figure for the new frame
        plt.clf()
        
        # Display the current date on the figure
        plt.suptitle(f"Data as of {current_date_str}", fontsize=16, color='white')
        
        # Graph the display dataframes with 3 charts at top and 3 at bottom
        graph_display_dataframes(display_dataframes, fig)
        
        # Force a draw of the figure
        plt.draw()
        
        # Pause to allow the GUI to update and to control animation speed
        plt.pause(0.5)
        
        # Increment the date by 1 minute
        current_date += timedelta(minutes=1)
        
        # Process any pending GUI events
        fig.canvas.flush_events()
        
except KeyboardInterrupt:
    print("\nAnimation stopped by user")
except Exception as e:
    print(f"\nAnimation error: {e}")
finally:
    # Turn off interactive mode when done
    plt.ioff()
    
    # Keep the window open after the animation completes
    print("\nAnimation complete. Close the plot window to exit.")
    plt.show(block=True)

### END LOOP
#####MAIN#####