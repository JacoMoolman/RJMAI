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
            dfs_list = []
            
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            # Timeframe mapping to normalized values between 0 and 1
            timeframe_map = {
                'M1': 0.0,    # 1-minute (minimum)
                'M5': 0.2,    # 5-minute
                'M30': 0.4,   # 30-minute
                'H1': 0.6,    # 1-hour
                'H4': 0.8,    # 4-hour
                'D1': 1.0     # Daily (maximum)
            }
            
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
                    df['month'] = df['time'].dt.month
                    df['day'] = df['time'].dt.day
                    
                    # Normalize day (of month) between 0 and 1
                    # Using 31 as the maximum possible day to normalize
                    df['day'] = df['day'] / 31.0
                    
                    # Create cyclical features for month
                    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
                    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
                    
                    # Drop the original time and month columns
                    df = df.drop(['time', 'month'], axis=1)
                    
                    # Add the timeframe column and convert to numerical
                    df['timeframe'] = timeframe
                    df['timeframe'] = df['timeframe'].map(timeframe_map)
                    
                    # Add to list for concatenation
                    dfs_list.append(df)
                
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            ##### PROCESS TIME #######
            
            # Concatenate all dataframes into a single dataframe
            if dfs_list:
                combined_df = pd.concat(dfs_list, ignore_index=True)

                # --- START NORMALIZATION BLOCK ---
                cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
                normalized_dfs = []
                grouped = combined_df.groupby('timeframe')

                for name, group in grouped:
                    group_copy = group.copy()
                    min_vals = group_copy[cols_to_normalize].min()
                    max_vals = group_copy[cols_to_normalize].max()
                    range_vals = max_vals - min_vals

                    # Handle columns where min == max (range is 0) to avoid division by zero
                    # Set normalized value to 0 in this case (or could use 0.5)
                    for col in cols_to_normalize:
                        if range_vals[col] == 0:
                            group_copy[col] = 0.0
                        else:
                            group_copy[col] = (group_copy[col] - min_vals[col]) / range_vals[col]
                
                    normalized_dfs.append(group_copy)
            
                # Recombine the normalized groups
                if normalized_dfs:
                     combined_df = pd.concat(normalized_dfs).sort_index() # Sort index to maintain original order if needed
                # --- END NORMALIZATION BLOCK ---
                
                # Cache the processed dataframe
                cache[symbol] = combined_df
                
                # Display the combined DataFrame with truncation (showing ... in the middle)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', 10)  # Show limited rows with ... in the middle
                pd.set_option('display.width', 1000)
                print(combined_df)
                       
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