import os
import pandas as pd
import time

# List of all currency pairs (uncomment/comment to include/exclude)
CURRENCY_PAIRS = [
    'AUDUSD',
    'EURUSD',
    'GBPUSD',
    'USDCAD',
    'USDCHF',
    'USDJPY'
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

# Dictionary to store all DataFrames
dataframes = {}

# Path to pickle files directory
PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picklefiles")

def load_currency_pairs(pickle_dir=PICKLE_DIR):
    """
    Load all currency pair data into DataFrames from pickle files
    
    Args:
        pickle_dir (str): Directory containing the pickle files
        
    Returns:
        dict: Dictionary of DataFrames with currency pair names as keys
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
    for pair in CURRENCY_PAIRS:
        for timeframe in TIMEFRAMES:
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

def display_currency_pairs(dfs, rows=2):
    """
    Display the first few rows of each DataFrame
    
    Args:
        dfs (dict): Dictionary of DataFrames
        rows (int): Number of rows to display
    """
    print(f"\nDisplaying first {rows} rows of each DataFrame:")
    
    # Check if any DataFrames were loaded
    if not dfs:
        print("No DataFrames to display!")
        return
    
    # Display each DataFrame
    for pair_timeframe, df in dfs.items():
        print(f"\n{pair_timeframe}:")
        print(df.head(rows))
        print("-" * 80)

# Load all currency pairs
dataframes = load_currency_pairs()

# Display the first 2 rows of each DataFrame
display_currency_pairs(dataframes, rows=2)