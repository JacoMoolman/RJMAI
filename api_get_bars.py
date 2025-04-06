from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import datetime
import random
import math
import numpy as np
from jmaitoolbox import normalize_data, detect_support_resistance, cluster_price_levels_with_strength

app = Flask(__name__)

# Global configuration variables
cache = {}
MAX_SR_LEVELS = 20  # Maximum number of support/resistance levels to track

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
            
            # Timeframe mapping to normalized values between 0 and 1
            timeframe_map = {
                'M1': 0.0,     # 1-minute - Now the minimum value
                'M5': 0.2,     # 5-minute - Adjusted scaling
                'M30': 0.4,    # 30-minute - Adjusted scaling
                'H1': 0.6,     # 1-hour - Adjusted scaling
                'H4': 0.8,     # 4-hour - Adjusted scaling
                'D1': 1.0      # Daily (maximum)
            }
            
            # Get normalized dataframes for each timeframe
            dfs_list = normalize_data(timeframes_data, timeframe_map)
                            
            ##### PROCESS TIME #######

            
            # Concatenate all dataframes into a single dataframe
            if dfs_list:
                # --- MODIFIED NORMALIZATION BLOCK ---
                # First normalize each dataframe individually by timeframe
                normalized_dfs = []
                
                for df in dfs_list:
                    # Get the current timeframe for this dataframe
                    tf = df['timeframe'].iloc[0]  # All rows have the same timeframe value
                    
                    # Normalize price and volume columns for this timeframe
                    cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
                    
                    # Normalize within this timeframe's dataframe
                    tf_min_vals = df[cols_to_normalize].min()
                    tf_max_vals = df[cols_to_normalize].max()
                    tf_range_vals = tf_max_vals - tf_min_vals
                    
                    # Create a copy to avoid modifying the original
                    normalized_df = df.copy()
                    
                    # Handle columns where min == max (range is 0) to avoid division by zero
                    for col in cols_to_normalize:
                        if tf_range_vals[col] == 0:
                            normalized_df[col] = 0.0
                        else:
                            normalized_df[col] = (df[col] - tf_min_vals[col]) / tf_range_vals[col]
                    
                    normalized_dfs.append(normalized_df)
                
                # Now concatenate the normalized dataframes
                normalized_df = pd.concat(normalized_dfs, ignore_index=True)
                # --- END MODIFIED NORMALIZATION BLOCK ---
                
                # Detect support and resistance levels
                levels_df = detect_support_resistance(normalized_df)
                
                # Cache the processed dataframes
                cache[symbol] = {
                    'price_data': normalized_df,
                    'levels': levels_df
                }
                
                # Display the combined DataFrame with truncation (showing ... in the middle)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)  # Show all rows
                pd.set_option('display.width', 1000)
                
                # Print full normalized dataframe
                print(cache[symbol]['price_data'])
                
                # Print total number of rows for verification
                print(f"\nTotal number of rows in combined dataframe: {len(cache[symbol]['price_data'])}")
                print(f"Rows per timeframe:")
                for tf in timeframe_map.keys():
                    count = len([x for x in timeframes_data.keys() if x == tf])
                    if count > 0:
                        print(f"  {tf}: {len(timeframes_data[tf])} bars")
                
                # Print support and resistance levels
                print("\nSupport and Resistance Levels:")
                print(cache[symbol]['levels'])
                
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