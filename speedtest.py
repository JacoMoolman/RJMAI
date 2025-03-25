from jmai_toolbox import load_currency_pairs,filter_dataframes_before_date,normalize_dataframes_separately,create_flat_dataframes
import os
from datetime import datetime, timedelta
from time import time

PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picklefiles")

num_bars_to_plot=100
START_DATE = '2024-01-01 00:00'

current_date = datetime.strptime('2025-03-25', '%Y-%m-%d')

TIMEFRAMES = [
    'D1',
    'H1',
    'H4',
    'M1',
    'M5',
    'M30'
]

currency_pairs = [
    'AUDUSD', 
    'EURUSD', 
    'GBPUSD', 
    'USDCAD', 
    'USDCHF', 
    'USDJPY'
]

#LOADING PKL FILES
print("Loading PKL files...")
dataframes = load_currency_pairs(currency_pairs, TIMEFRAMES, PICKLE_DIR)

# Test loop
start_time = time()
while True:
    loop_start_time = time()
    
    # EXTRACT DATES
    print('Get display frames')
    start_extract = time()
    display_dataframes = filter_dataframes_before_date(dataframes, datetime.strptime(START_DATE, '%Y-%m-%d %H:%M'), num_bars_to_plot)
    print(f'Extract time: {time() - start_extract:.2f}s')
    
    # NORMALIZE
    print('Normalize')
    start_normalize = time()
    display_dataframes = normalize_dataframes_separately(display_dataframes)
    print(f'Normalize time: {time() - start_normalize:.2f}s')
    
    # CREATE FLAT DATAFRAMES
    print('Create FLAT DF')
    start_flat = time()
    display_dataframes_flat = create_flat_dataframes(display_dataframes)
    print(f'Create flat time: {time() - start_flat:.2f}s')
    
    # Print timing
    print(f'Loop time: {time() - loop_start_time:.2f}s')
    
    # Print out display_dataframes_flat
    for currency_pair, flat_df in display_dataframes_flat.items():
        print(f"\nFlat DataFrame for {currency_pair}:")
        print(flat_df)
    
    # Update START_DATE
    START_DATE = (datetime.strptime(START_DATE, '%Y-%m-%d %H:%M') + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M')
    
    # Break if START_DATE reaches current_date
    if datetime.strptime(START_DATE, '%Y-%m-%d %H:%M') >= current_date:
        break

print(f'Total time: {time() - start_time:.2f}s')