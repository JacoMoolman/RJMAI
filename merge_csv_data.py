'''
CSV DATA MERGING AND SYNCHRONIZATION TOOL

DESCRIPTION:
This script is designed to merge and synchronize financial market data from MetaTrader (MQL5) CSV files
into an existing dataset in the CSVDUMPS directory. It handles various file encoding issues and ensures
proper data integration without data loss or corruption.

FUNCTIONALITY:
1. Automatically maps MQL5 timeframe notation to CSVDUMPS notation (e.g., "1H" to "1h")
2. Reads and processes MQL5 CSV files with robust encoding detection and binary file handling
3. Extracts valid OHLC (Open, High, Low, Close) data with date/time information
4. Preserves existing data in CSVDUMPS files while adding new data from MQL5
5. Removes duplicate entries based on the date/time column
6. Sorts all entries chronologically by date/time
7. Maintains the header structure of the destination files

KEY COMPONENTS:
- detect_encoding(): Attempts to detect the encoding of files by examining byte patterns
- read_binary_file(): Robust function to extract CSV data from potentially binary files
- process_files(): Main function that handles the entire workflow

MAPPINGS:
- "1H" (MQL5) → "1h" (CSVDUMPS) [1-hour timeframe]
- "5M" (MQL5) → "5m" (CSVDUMPS) [5-minute timeframe]
- "15M" (MQL5) → "15m" (CSVDUMPS) [15-minute timeframe]
- "30M" (MQL5) → "30m" (CSVDUMPS) [30-minute timeframe]
- "D1" (MQL5) → "1d" (CSVDUMPS) [1-day timeframe]

SOURCE AND DESTINATION:
- Source: C:\\Users\\NVG\\AppData\\Roaming\\MetaQuotes\\Tester\\...\\MQL5\\Files\\
- Destination: J:\\CSVDUMPS\\

USAGE:
Run this script to synchronize data whenever new MQL5 CSV files become available.
The script handles all the necessary file operations and data merging automatically.
'''

import os
import pandas as pd
import glob
import csv
import codecs
import re

# Define source and destination directories
mql5_dir = r"C:\Users\NVG\AppData\Roaming\MetaQuotes\Tester\010E047102812FC0C18890992854220E\Agent-127.0.0.1-3000\MQL5\Files"
csvdumps_dir = r"J:\CSVDUMPS"

# Define mapping between MQL5 file naming and CSVDUMPS naming
file_mapping = {
    "1H": "1h",
    "5M": "5m",
    "15M": "15m",
    "30M": "30m",
    "D1": "1d"
}

def detect_encoding(file_path):
    """Try to detect the encoding of a file by reading the first few bytes"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(4096)  # Read first 4096 bytes
    
    # Try to detect BOM (Byte Order Mark)
    if raw_data.startswith(codecs.BOM_UTF8):
        return 'utf-8-sig'  # UTF-8 with BOM
    elif raw_data.startswith(codecs.BOM_UTF16_LE):
        return 'utf-16-le'  # UTF-16 Little Endian
    elif raw_data.startswith(codecs.BOM_UTF16_BE):
        return 'utf-16-be'  # UTF-16 Big Endian
    
    # If no BOM is found, print the first few bytes for debugging
    print(f"First few bytes (hex): {raw_data[:20].hex()}")
    
    # Try a few common encodings
    for enc in ['utf-8', 'latin1', 'cp1252', 'utf-16']:
        try:
            raw_data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    
    # Default to binary as a last resort
    return 'binary'

def read_binary_file(file_path):
    """Read a binary file and try to extract CSV-like data"""
    with open(file_path, 'rb') as f:
        binary_data = f.read()
    
    # Try different approaches
    data_lines = []
    
    # First try to decode as various encodings
    for encoding in ['utf-8', 'latin1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']:
        try:
            text = binary_data.decode(encoding, errors='ignore')
            # Look for date patterns
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line has a date pattern and commas
                if re.search(r'20\d{2}\.\d{2}\.\d{2}', line) and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 5:  # Date, Open, High, Low, Close
                        data_lines.append(line)
            
            if data_lines:
                print(f"Successfully decoded with {encoding}, found {len(data_lines)} lines")
                return data_lines
        except Exception as e:
            continue
    
    # If no encoding worked, try binary approach
    # Look for byte patterns that might indicate dates (20xx.xx.xx)
    # This is a bit hacky but might work for binary formats
    text = binary_data.decode('latin1', errors='ignore')
    date_pattern = r'20\d{2}\.\d{2}\.\d{2}'
    matches = re.finditer(date_pattern, text)
    
    for match in matches:
        start_idx = match.start()
        # Find the next newline or end of string
        end_idx = text.find('\n', start_idx)
        if end_idx == -1:
            end_idx = len(text)
            
        line = text[start_idx:end_idx]
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 5:  # Ensure we have date and OHLC
                data_lines.append(line)
    
    print(f"Binary approach found {len(data_lines)} lines")
    return data_lines

def process_files():
    # Get all CSV files in MQL5 directory
    mql5_files = glob.glob(os.path.join(mql5_dir, "*.csv"))
    
    for mql5_file in mql5_files:
        # Extract the base filename (e.g., "EURUSD_1H.csv")
        base_filename = os.path.basename(mql5_file)
        
        # Extract currency pair and timeframe
        parts = os.path.splitext(base_filename)[0].split('_')
        if len(parts) != 2:
            print(f"Skipping file with unexpected format: {base_filename}")
            continue
            
        currency_pair, timeframe = parts
        
        # Map MQL5 timeframe to CSVDUMPS timeframe
        if timeframe not in file_mapping:
            print(f"No mapping found for timeframe: {timeframe}")
            continue
            
        csvdumps_timeframe = file_mapping[timeframe]
        
        # Construct the corresponding CSVDUMPS filename
        csvdumps_filename = f"{currency_pair}_{csvdumps_timeframe}.csv"
        csvdumps_filepath = os.path.join(csvdumps_dir, csvdumps_filename)
        
        # Check if the destination file exists
        if not os.path.exists(csvdumps_filepath):
            print(f"Destination file does not exist: {csvdumps_filepath}")
            continue
            
        print(f"Processing: {base_filename} -> {csvdumps_filename}")
        
        try:
            # Read MQL5 file as binary first to detect format
            mql5_data = read_binary_file(mql5_file)
            
            if not mql5_data:
                print(f"No valid data found in {base_filename}")
                continue
            
            print(f"Found {len(mql5_data)} valid lines in {base_filename}")
            
            # Read CSVDUMPS file line by line
            csvdumps_data = []
            with open(csvdumps_filepath, 'r') as f:
                header = f.readline().strip()  # Save the header
                for line in f:
                    line = line.strip()
                    if line:
                        csvdumps_data.append(line)
            
            # Combine data (add MQL5 data to CSVDUMPS data)
            # Create a dictionary with date as the key to remove duplicates
            all_data = {}
            
            # Process header separately
            if header.lower().startswith("date,open,high,low,close"):
                header_parts = header.split(',')
                
            # Add MQL5 data to the dictionary
            for line in mql5_data:
                parts = line.split(',')
                date = parts[0]  # Date is the first column
                all_data[date] = line
            
            # Add CSVDUMPS data to the dictionary
            # Only overwrite if the entry doesn't already exist (prioritize MQL5)
            for line in csvdumps_data:
                parts = line.split(',')
                if len(parts) >= 1:
                    date = parts[0]  # Date is the first column
                    if date not in all_data:
                        all_data[date] = line
            
            # Sort the data by date
            sorted_dates = sorted(all_data.keys())
            sorted_data = [all_data[date] for date in sorted_dates]
            
            # Write back to CSVDUMPS file
            with open(csvdumps_filepath, 'w', newline='') as f:
                f.write(header + '\n')  # Write the header first
                for line in sorted_data:
                    f.write(line + '\n')
            
            print(f"Successfully merged and sorted data for {csvdumps_filename}")
            
        except Exception as e:
            print(f"Error processing {base_filename}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    process_files()
    print("Data merging completed!")
