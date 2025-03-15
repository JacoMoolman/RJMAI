"""
CSV Processor - Script that uses the jmai_toolbox

This script reads all CSV files from M:\CSV_PROCCESSED, creates dataframes,
splits date and time fields, and calculates day of week.
"""

import os
import pandas as pd
import jmai_toolbox

# Global variables
csv_dir_path = r'M:\CSV_PROCCESSED'
dataframes = {}

def check_directory():
    """
    Check if the CSV directory exists
    """
    if not os.path.exists(csv_dir_path):
        print(f"Error: Directory not found at {csv_dir_path}")
        return False
    return True

def get_csv_files():
    """
    Get all CSV files in the directory
    """
    csv_files = [f for f in os.listdir(csv_dir_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir_path}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to process")
    return csv_files

def process_csv_file(csv_file):
    """
    Process a single CSV file into a dataframe
    """
    file_path = os.path.join(csv_dir_path, csv_file)
    print(f"\nProcessing file: {csv_file}")
    
    # Read the CSV file into a dataframe
    df = pd.read_csv(file_path)
    
    # Split the date and time into separate columns
    df[['date', 'time']] = df['time'].str.split(' ', expand=True)
    
    # Convert date format from YYYY.MM.DD to YYYY-MM-DD for processing
    df['formatted_date'] = df['date'].str.replace('.', '-')
    
    # Calculate day of week for each row
    df['day_of_week'] = df['formatted_date'].apply(jmai_toolbox.get_day_of_week_num)
    
    # Remove the temporary formatted_date column
    df = df.drop('formatted_date', axis=1)
    
    # Store the dataframe with the same name as the file (without extension)
    df_name = os.path.splitext(csv_file)[0]
    dataframes[df_name] = df
    
    print(f"Created dataframe '{df_name}' with {len(df)} rows")

def process_dataframes():
    """
    Process each CSV file into a dataframe
    """
    if not check_directory():
        return
    
    csv_files = get_csv_files()
    if not csv_files:
        return
    
    try:
        # Process each CSV file
        for csv_file in csv_files:
            process_csv_file(csv_file)
        
        print("\nDataframes processed successfully.")
        return True
        
    except Exception as e:
        print(f"Error processing dataframes: {e}")
        return False

def print_dataframe_heads():
    """
    Print the head of each dataframe
    """
    if not dataframes:
        print("No dataframes to display. Run process_dataframes() first.")
        return
    
    print("\n\n===== DATAFRAME HEADS =====")
    for df_name, df in dataframes.items():
        print(f"\nDataframe: {df_name}")
        print(df.head(10))

# Steps that can be hashed/unhashed
process_dataframes()
print_dataframe_heads()
