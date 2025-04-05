from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import datetime
import random
import math
import numpy as np

app = Flask(__name__)

cache = {}

def normalize_data(timeframes_data, timeframe_map):
    """
    Process each timeframe's data to create normalized dataframes with time features.
    
    Args:
        timeframes_data (dict): Dictionary of timeframes and their corresponding bars data
        timeframe_map (dict): Mapping of timeframe strings to normalized values
        
    Returns:
        list: List of processed dataframes for each timeframe
    """
    dfs_list = []
    
    # Process each timeframe
    for timeframe, bars in timeframes_data.items():
        if timeframe in timeframe_map:
            # Convert the bars to a DataFrame
            df = pd.DataFrame(bars)
            
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M:%S')
            
            # Add day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
            # Pandas weekday is 0=Monday, so we need to adjust to make Sunday=0
            raw_day_of_week = (df['time'].dt.weekday + 1) % 7
            
            # Normalize day of week between 0 and 1
            df['day_of_week'] = raw_day_of_week / 6.0  # Divide by (7-1) to normalize
            
            # Extract time components (remove seconds, split hour and minute)
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            
            # Normalize hour and minute between 0 and 1
            df['hour'] = df['hour'] / 23.0  # Divide by (24-1) to normalize
            df['minute'] = df['minute'] / 59.0  # Divide by (60-1) to normalize
            
            # Extract date components (drop year, split month and day)
            df['day'] = df['time'].dt.day
            
            # Normalize day (of month) between 0 and 1
            # Using 31 as the maximum possible day to normalize
            df['day'] = df['day'] / 31.0
            
            # Drop the original time column and spread column
            df = df.drop(['time', 'spread'], axis=1)
            
            # Add the timeframe column and convert to numerical
            df['timeframe'] = timeframe
            df['timeframe'] = df['timeframe'].map(timeframe_map)
            
            # Add to list for concatenation
            dfs_list.append(df)
    
    return dfs_list

@app.route('/test', methods=['POST'])
def test_endpoint():
    print("\n===== NEW REQUEST RECEIVED =====")
    
    try:
        # Handle JSON data from MQL5
        if request.is_json:
            data = request.json
        else:
            # Try to parse raw data as JSON, fixing the timestamp format issue
            raw_data = request.data.decode('utf-8')
            fixed_data = re.sub(r'("time": )(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})', r'\1"\2"', raw_data)
            
            # Parse the fixed JSON data
            data = json.loads(fixed_data)
        
        # Extract symbol, balance, and PnL first
        symbol = data.get('symbol', 'UNKNOWN') # Use .get for safety
        account_balance = data.get('account_balance', 0.0) # Default to 0.0 if missing
        open_trade_pnl = data.get('open_trade_pnl', 0.0)   # Default to 0.0 if missing

        # Display balance and PnL
        print(f"\n----- Account Info ----- ")
        print(f"Symbol: {symbol}")
        print(f"Account Balance: {account_balance:.2f}")
        print(f"Open Trade PnL: {open_trade_pnl:.2f}")
        print(f"-------------------------")

        # Process and display forex data using pandas
        if 'data' in data:
            timeframes_data = data['data']
            
            print(f"\n===== RECEIVED DATA FOR SYMBOL: {symbol} =====")
            
            # List to store DataFrames for each timeframe for concatenation
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            # Timeframe mapping to normalized values between 0 and 1
            timeframe_map = {
                'M5': 0.0,     # 5-minute - Now the minimum value
                'M30': 0.25,   # 30-minute - Adjusted scaling
                'H1': 0.5,     # 1-hour - Adjusted scaling
                'H4': 0.75,    # 4-hour - Adjusted scaling
                'D1': 1.0      # Daily (maximum)
            }
            
            # Process timeframe data using the new function
            dfs_list = normalize_data(timeframes_data, timeframe_map)
            
                
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            
            # Concatenate all dataframes into a single dataframe
            if dfs_list:
                combined_df = pd.concat(dfs_list, ignore_index=True)

                # --- START NORMALIZATION BLOCK ---
                # Normalize price and volume columns
                cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
                
                # Normalize across the entire dataframe, not by timeframe
                min_vals = combined_df[cols_to_normalize].min()
                max_vals = combined_df[cols_to_normalize].max()
                range_vals = max_vals - min_vals

                # Handle columns where min == max (range is 0) to avoid division by zero
                for col in cols_to_normalize:
                    if range_vals[col] == 0:
                        combined_df[col] = 0.0
                    else:
                        combined_df[col] = (combined_df[col] - min_vals[col]) / range_vals[col]
                
                # No longer need to normalize by timeframe and recombine
                normalized_df = combined_df
                # --- END NORMALIZATION BLOCK ---
                
                # Cache the processed dataframe
                cache[symbol] = normalized_df
                
                # Display the combined DataFrame with truncation (showing ... in the middle)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)  # Show all rows
                pd.set_option('display.width', 1000)
                
                # Print full normalized dataframe
                print(normalized_df)
                
                # Print total number of rows for verification
                print(f"\nTotal number of rows in combined dataframe: {len(normalized_df)}")
                print(f"Rows per timeframe:")
                for tf in timeframe_map.keys():
                    count = len([x for x in timeframes_data.keys() if x == tf])
                    if count > 0:
                        print(f"  {tf}: {len(timeframes_data[tf])} bars")
                
            print("\n===== DATA PROCESSING COMPLETE =====")
            
            # Generate random trade instruction (B=Buy, S=Sell, H=Hold, C=Close)
            trade_instruction = random.choice(["B", "S", "H", "C"])
            
            print(f"\n===== SENDING TRADE INSTRUCTION: {trade_instruction} =====")                        
            return jsonify({"response": f"INSTRUCTION: {trade_instruction}"})
        else:
            print("Invalid data format - missing 'symbol' or 'data' fields")
            return jsonify({"response": "Error: Invalid data format"})
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == '__main__':
    print("\n=================================")
    print("API test server running on http://localhost:5000")
    print("Send a POST request to http://localhost:5000/test with data")
    print("Waiting for MQL5 data...")
    print("=================================\n")
    app.run(debug=True)