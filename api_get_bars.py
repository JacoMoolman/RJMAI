from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import datetime
import random
import math
import numpy as np
import os
import random
import logging
from datetime import datetime
from jmaitoolbox import normalize_data, cluster_price_levels_with_strength, export_df_to_csv, identify_price_levels, calculate_sl_tp, add_difference_columns, create_spread_dataframe

# Disable Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global trade tracking variables
current_trade_open = False
previous_pnl = 0.0
last_trade_action = None
trade_count = 0

# Configuration
EXPORT_TO_CSV = True  # Set to False to disable CSV exports
SHOWDF = True  # Set to True to display dataframe information in console
CSV_OUTPUT_DIR = "data_exports"  # Directory to store CSV files

# Default lot size
DEFAULT_LOT_SIZE = 0.01

# Create output directory if it doesn't exist
if EXPORT_TO_CSV and not os.path.exists(CSV_OUTPUT_DIR):
    os.makedirs(CSV_OUTPUT_DIR)

app = Flask(__name__)

# Global configuration variables
cache = {}
MAX_SR_LEVELS = 20  # Maximum number of support/resistance levels to track

@app.route('/test', methods=['POST'])
def test_endpoint():
    print("\n===== NEW REQUEST RECEIVED =====")
    
    # Access global variables
    global current_trade_open, previous_pnl, last_trade_action, trade_count
    
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
        
        # Check for error message from MQL
        if 'error' in data:
            error_msg = data['error']
            print(f"\n----- ERROR RECEIVED FROM MQL -----")
            print(f"Error: {error_msg}")
            print(f"-------------------------")
            # Return acknowledgment of error
            return jsonify({"response": f"Error received: {error_msg}"})
        
        # Extract symbol and account information
        symbol = data.get('symbol', 'UNKNOWN')
        account_balance = data.get('account_balance', 0.0)
        open_trade_pnl = data.get('open_trade_pnl', 0.0)
        position_exists = data.get('position_exists', False)

        # Display account info
        print(f"\n----- Account Info ----- ")
        print(f"Symbol: {symbol}")
        print(f"Account Balance: {account_balance:.2f}")
        print(f"Open Trade PnL: {open_trade_pnl:.2f}")
        print(f"Position Exists: {position_exists}")
        
        # Under the new communication flow, we only receive data when 
        # a trade is closed (position_exists = False)
        if not position_exists:
            print(f"\n===== TRADE CLOSED OR NO ACTIVE TRADE =====")
            
            # If there was a trade previously that is now closed
            if current_trade_open:
                trade_count += 1
                profit = previous_pnl
                print(f"Final P/L: {profit:.2f}")
                print(f"Total Completed Trades: {trade_count}")
            
            # Update our internal state
            current_trade_open = False
            
            # Process the received market data
            instruction = ""
            if 'data' in data and data['data']:
                print(f"\n----- Processing Market Data -----")
                
                timeframes_data = data['data']
                timeframe_keys = list(timeframes_data.keys())
                print(f"Timeframes received: {', '.join(timeframe_keys)}")
                
                # Process the data and determine trading instruction
                if timeframes_data and len(timeframes_data) > 0:
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
                    
                    # Concatenate all dataframes into a single dataframe
                    normalized_df = pd.concat(dfs_list, ignore_index=True)
                    
                    # Add difference columns
                    normalized_df = add_difference_columns(normalized_df)
                    
                    # Create spread dataframe with bid/ask prices and effective spread
                    spread_df = create_spread_dataframe(timeframes_data)
                    
                    # Display spread dataframe information if enabled
                    if SHOWDF and not spread_df.empty:
                        print("\n----- SPREAD ANALYSIS DATAFRAME -----")
                        print(f"Shape: {spread_df.shape}")
                        print(f"Columns: {spread_df.columns.tolist()}")
                        print("\nFirst few rows:")
                        print(spread_df.head())
                        print("\n------------------------------")
                    
                    # Export spread dataframe to CSV
                    if not spread_df.empty:
                        spread_csv_path = f"{CSV_OUTPUT_DIR}/{symbol}_spread_analysis.csv"
                        # Format the spread with full decimal representation
                        spread_df['spread'] = spread_df['spread'].apply(lambda x: f"{x:.6f}")
                        spread_df.to_csv(spread_csv_path, index=False)
                        if SHOWDF:
                            print(f"Exported spread analysis to {spread_csv_path}")
                    
                    # Display complete normalized dataframe information AFTER all processing
                    if SHOWDF:
                        print("\n----- COMPLETE NORMALIZED DATAFRAME -----")
                        print(f"Shape: {normalized_df.shape}")  # Shows number of rows and columns
                        print(f"Columns: {normalized_df.columns.tolist()}")
                        print("\nFirst few rows:")
                        pd.set_option('display.max_columns', None)  # Show all columns
                        pd.set_option('display.width', 1000)        # Wider display
                        print(normalized_df.head())
                        print("\n------------------------------")
                    
                    # Export normalized dataframe to CSV
                    if not normalized_df.empty:
                        normalized_csv_path = f"{CSV_OUTPUT_DIR}/{symbol}_normalized_data.csv"
                        normalized_df.to_csv(normalized_csv_path, index=False)
                        if SHOWDF:
                            print(f"Exported normalized data to {normalized_csv_path}")
                    
                    # Get frequency-based price levels from raw data
                    price_levels_df = identify_price_levels(timeframes_data, normalized_df)
                    
                    # Display price levels
                    if SHOWDF:
                        print("\n----- PRICE LEVELS BASED ON FREQUENCY -----")
                        print(f"Shape: {price_levels_df.shape}")
                        print("Columns: timeframe, price_level, normalized_price, frequency, strength")
                        print("\nPrice Levels:")
                        print(price_levels_df)
                        print("\n------------------------------")
                    
                    # Export price levels to CSV - no need for one-hot encoding anymore as it's handled in jmaitoolbox.py
                    if not price_levels_df.empty:
                        price_levels_csv_path = f"{CSV_OUTPUT_DIR}/{symbol}_price_levels.csv"
                        price_levels_df.to_csv(price_levels_csv_path, index=False)
                        if SHOWDF:
                            print(f"Exported price levels to {price_levels_csv_path}")
                    
                    # Finally, determine trading instruction
                    # This is a simplified placeholder - replace with your actual logic

                    # For demonstration, randomly choose an instruction
                    choices = ["BUY", "SELL", "HOLD"]
                    weights = [0.4, 0.4, 0.2]  # 40% buy, 40% sell, 20% hold
                    instruction = random.choices(choices, weights=weights, k=1)[0]
                    
                    # Calculate SL/TP if it's a BUY or SELL instruction
                    sl_tp_part = ""
                    if instruction in ["BUY", "SELL"]:
                        # Get the current price for the symbol
                        current_price = float(timeframes_data[timeframe_keys[0]][0]['close'])  # Use the most recent close price from the first timeframe
                        
                        # Define percentage-based SL/TP values (these will be replaced by AI in future)
                        sl_percent = 2.0  
                        tp_percent = 2.0 
                        
                        # Convert percentage to normalized values (0-1 range with 6 decimal places)
                        # The division by 100 converts percentage to decimal, then scale as needed
                        ai_normalized_sl = round(sl_percent / 100, 6)
                        ai_normalized_tp = round(tp_percent / 100, 6)
                        
                        # Calculate SL/TP values
                        norm_sl, norm_tp, actual_sl, actual_tp = calculate_sl_tp(
                            normalized_df, instruction, current_price,
                            ai_normalized_sl=ai_normalized_sl,
                            ai_normalized_tp=ai_normalized_tp
                        )
                        
                        # Format SL/TP part of the instruction
                        # Format: "BUY:0.01:1.25648:1.35792" (Action:Lots:SL:TP)
                        sl_tp_part = f":{DEFAULT_LOT_SIZE}:{actual_sl:.5f}:{actual_tp:.5f}"
                        
                        print(f"Normalized SL: {norm_sl:.6f}")
                        print(f"Normalized TP: {norm_tp:.6f}")
                        print(f"Actual SL: {actual_sl:.5f}")
                        print(f"Actual TP: {actual_tp:.5f}")
                    
                    # Complete instruction with SL/TP if applicable
                    final_instruction = instruction + (sl_tp_part if instruction in ["BUY", "SELL"] else "")
                    
                    print(f"\n----- TRADING INSTRUCTION -----")
                    print(f"Instruction: {final_instruction}")
                    
                    # Remember this action for next time a trade completes
                    last_trade_action = instruction
                    
                    # If instruction is to open a new trade, update our state
                    if instruction in ["BUY", "SELL"]:
                        current_trade_open = True
            
            # Return the trading instruction
            return jsonify({"response": f"Data processed successfully. INSTRUCTION: {final_instruction}"})
        else:
            # This should not happen with the new communication flow
            # MQL should never send data when a position exists
            print(f"\n----- UNEXPECTED: Received data while position exists -----")
            return jsonify({"response": "Unexpected data received - position exists"})
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == '__main__':
    print("\n=================================")
    print("API test server running on http://localhost:5000")
    print("Send a POST request to http://localhost:5000/test with data")
    print("Waiting for MQL5 data...")
    print("=================================\n")
    app.run(debug=True, host='localhost', port=5000, use_reloader=False)
