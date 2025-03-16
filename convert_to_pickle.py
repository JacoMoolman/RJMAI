import os
import pandas as pd
import glob
from tqdm import tqdm
import time

# Create directory for pickle files if it doesn't exist
PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picklefiles")
os.makedirs(PICKLE_DIR, exist_ok=True)

# List of all currency pairs
CURRENCY_PAIRS = [
    'AUDUSD',
    'EURUSD',
    'GBPUSD',
    'USDCAD',
    'USDCHF',
    'USDJPY'
]

# List of all timeframes
TIMEFRAMES = [
    'D1',
    'H1',
    'H4',
    'M1',
    'M5',
    'M30'
]

def convert_csv_to_pickle(csv_dir="M:\\CSV_PROCCESSED"):
    """
    Convert all CSV files to pickle files for faster loading
    
    Args:
        csv_dir (str): Directory containing the CSV files
        
    Returns:
        int: Number of files successfully converted
    """
    print(f"Converting CSV files from {csv_dir} to pickle files in {PICKLE_DIR}")
    
    # Count total files to process
    total_files = len(CURRENCY_PAIRS) * len(TIMEFRAMES)
    
    # Check if directory exists
    if not os.path.exists(csv_dir):
        print(f"Error: Directory {csv_dir} does not exist!")
        return 0
    
    # Create progress bar
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    print(f"Processing {total_files} potential files...")
    
    # Process each currency pair and timeframe combination
    for pair in tqdm(CURRENCY_PAIRS, desc="Currency Pairs"):
        for timeframe in TIMEFRAMES:
            # Create the full pair_timeframe string (e.g., "EURUSD_H1")
            pair_timeframe = f"{pair}_{timeframe}"
            
            # Full path to CSV file
            csv_file = os.path.join(csv_dir, f"{pair_timeframe}.csv")
            
            # Full path to pickle file (destination)
            pickle_file = os.path.join(PICKLE_DIR, f"{pair_timeframe}.pkl")
            
            # Check if file exists
            if os.path.exists(csv_file):
                try:
                    # Load CSV into DataFrame
                    df = pd.read_csv(csv_file, 
                                    parse_dates=['time'],
                                    names=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread'],
                                    header=0)
                    
                    # Save as pickle file
                    df.to_pickle(pickle_file)
                    
                    # Increment counter
                    success_count += 1
                except Exception as e:
                    print(f"Error converting {pair_timeframe}: {e}")
                    error_count += 1
            else:
                print(f"File not found: {csv_file}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\nConversion Summary:")
    print(f"- Successfully converted: {success_count} files")
    print(f"- Errors: {error_count} files")
    print(f"- Total time: {elapsed_time:.2f} seconds")
    
    return success_count

if __name__ == "__main__":
    convert_csv_to_pickle()
    print("\nDone! You can now use the pickle files for faster data loading.")
