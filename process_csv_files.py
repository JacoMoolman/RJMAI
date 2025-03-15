## PROCESSES RAW CAS FILES FROM MT5

import pandas as pd
import os
import glob

# Path to directory containing CSV files
csv_dir = r'J:\\CSVDUMPS'

# Set output directory to M:\CSV_PROCCESSED
processed_dir = r'M:\CSV_PROCCESSED'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    print(f"Created directory: {processed_dir}")

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# Track statistics
files_processed = 0
files_skipped = 0
skipped_files = []

print(f"Found {len(csv_files)} CSV files to check")

# Process each CSV file individually
for file in csv_files:
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(file))[0]
    
    # Check if output file already exists
    output_file = os.path.join(processed_dir, f"{base_filename}.csv")
    if os.path.exists(output_file):
        print(f"Skipping {base_filename}: already processed")
        files_skipped += 1
        skipped_files.append(base_filename)
        continue
    
    # Clear visual indicator for current file being processed
    print("\n" + "="*60)
    print(f"PROCESSING FILE: {base_filename}")
    print("="*60 + "\n")
    
    # Read the file with UTF-16LE encoding without parsing dates yet
    df = pd.read_csv(file, encoding='utf-16le')
    
    # Display original data before processing
    print(f"Original data from {base_filename}:")
    print(df.head(5))
    print("\n")
    
    # Parse the time column while preserving the original format
    df['time'] = pd.to_datetime(df['time'], format="%Y.%m.%d %H:%M")
    
    # Convert back to the original format for output
    df['time'] = df['time'].dt.strftime("%Y.%m.%d %H:%M")
    
    # Ensure numeric columns maintain exact decimal precision
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # Convert to string to preserve exact decimal places
            df[col] = df[col].astype(str)
    
    # Save the processed file without index
    df.to_csv(output_file, index=False)
    files_processed += 1
    
    print(f"Processed {base_filename}, saved to: {output_file}")
    
    # Display the top 10 rows of processed file
    print(f"\nProcessed data (top 10 rows of {base_filename}):")
    print(df.head(10))
    print("\n" + "-"*80 + "\n")

# Print summary
print("\nSummary:")
print(f"Total files found: {len(csv_files)}")
print(f"Files processed: {files_processed}")
print(f"Files skipped (already processed): {files_skipped}")
if skipped_files:
    print(f"Skipped files: {', '.join(skipped_files)}")