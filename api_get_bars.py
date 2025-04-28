from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import datetime
import random
import math
import numpy as np
import os
import logging
from datetime import datetime
from jmaitoolbox import normalize_data, detect_support_resistance, cluster_price_levels_with_strength

# Disable Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global trade tracking variables
current_trade_open = False
previous_pnl = 0.0
last_trade_action = None
trade_count = 0

# Configuration
EXPORT_TO_CSV = False  # Set to False to disable CSV exports
CSV_OUTPUT_DIR = "data_exports"  # Directory to store CSV files

# Create output directory if it doesn't exist
if EXPORT_TO_CSV and not os.path.exists(CSV_OUTPUT_DIR):
    os.makedirs(CSV_OUTPUT_DIR)

# Function to export DataFrame to CSV
def export_df_to_csv(df, stage_name, symbol="UNKNOWN"):
    """
    Export a DataFrame to CSV file
    
    Args:
        df: DataFrame to export
        stage_name: Name of the processing stage (for filename)
        symbol: Trading symbol
    """
    if not EXPORT_TO_CSV:
        return
        
    # Use a single file per stage that gets overwritten
    filename = f"{CSV_OUTPUT_DIR}/{symbol}_{stage_name}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Export to CSV, overwriting previous file
    df.to_csv(filename, index=True)
    print(f"Exported {stage_name} DataFrame to {filename}")

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
                    
                    # --- ADD DIFFERENCE COLUMNS ---
                    # Calculate differences between OHLC values
                    normalized_df['diff_open_high'] = normalized_df['open'] - normalized_df['high']
                    normalized_df['diff_open_low'] = normalized_df['open'] - normalized_df['low']
                    normalized_df['diff_open_close'] = normalized_df['open'] - normalized_df['close']

                    normalized_df['diff_high_open'] = normalized_df['high'] - normalized_df['open']
                    normalized_df['diff_high_low'] = normalized_df['high'] - normalized_df['low']
                    normalized_df['diff_high_close'] = normalized_df['high'] - normalized_df['close']

                    normalized_df['diff_low_open'] = normalized_df['low'] - normalized_df['open']
                    normalized_df['diff_low_high'] = normalized_df['low'] - normalized_df['high']
                    normalized_df['diff_low_close'] = normalized_df['low'] - normalized_df['close']

                    normalized_df['diff_close_open'] = normalized_df['close'] - normalized_df['open']
                    normalized_df['diff_close_high'] = normalized_df['close'] - normalized_df['high']
                    normalized_df['diff_close_low'] = normalized_df['close'] - normalized_df['low']
                    
                    # Add indicator crossovers and differences (where applicable)
                    if all(col in normalized_df.columns for col in ['ma20', 'ma50']):
                        normalized_df['ma20_50_diff'] = normalized_df['ma20'] - normalized_df['ma50']
                        
                    if all(col in normalized_df.columns for col in ['ma50', 'ma100']):
                        normalized_df['ma50_100_diff'] = normalized_df['ma50'] - normalized_df['ma100']
                        
                    if all(col in normalized_df.columns for col in ['ma20', 'ma100']):
                        normalized_df['ma20_100_diff'] = normalized_df['ma20'] - normalized_df['ma100']
                    
                    if all(col in normalized_df.columns for col in ['macd_main', 'macd_signal']):
                        normalized_df['macd_crossover'] = normalized_df['macd_main'] - normalized_df['macd_signal']
                    
                    if all(col in normalized_df.columns for col in ['stoch_k', 'stoch_d']):
                        normalized_df['stoch_crossover'] = normalized_df['stoch_k'] - normalized_df['stoch_d']
                    
                    if all(col in normalized_df.columns for col in ['plus_di', 'minus_di']):
                        normalized_df['di_crossover'] = normalized_df['plus_di'] - normalized_df['minus_di']
                    
                    # Bollinger Band position
                    if all(col in normalized_df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                        normalized_df['bb_position'] = (normalized_df['close'] - normalized_df['bb_lower']) / (normalized_df['bb_upper'] - normalized_df['bb_lower'])
                        # Handle division by zero
                        normalized_df['bb_position'] = normalized_df['bb_position'].fillna(0.5)
                        # Normalize to 0-1 range
                        normalized_df['bb_position'] = normalized_df['bb_position'].clip(0, 1)
                    
                    # Ichimoku cloud status
                    if all(col in normalized_df.columns for col in ['close', 'ichimoku_senkou_a', 'ichimoku_senkou_b']):
                        # Above cloud = 1, Below cloud = 0, In cloud = 0.5
                        normalized_df['cloud_position'] = ((normalized_df['close'] > normalized_df['ichimoku_senkou_a']) & 
                                                        (normalized_df['close'] > normalized_df['ichimoku_senkou_b'])).astype(float)
                        
                        in_cloud = ((normalized_df['close'] >= normalized_df['ichimoku_senkou_a']) & 
                                   (normalized_df['close'] <= normalized_df['ichimoku_senkou_b'])) | \
                                  ((normalized_df['close'] <= normalized_df['ichimoku_senkou_a']) & 
                                   (normalized_df['close'] >= normalized_df['ichimoku_senkou_b']))
                        
                        normalized_df.loc[in_cloud, 'cloud_position'] = 0.5
                    
                    # Normalize crossover signals to 0-1 range
                    if 'macd_crossover' in normalized_df.columns:
                        # Scale and shift MACD crossover to 0-1 range
                        min_macd = normalized_df['macd_crossover'].min()
                        max_macd = normalized_df['macd_crossover'].max()
                        if max_macd != min_macd:  # Prevent division by zero
                            normalized_df['macd_crossover'] = (normalized_df['macd_crossover'] - min_macd) / (max_macd - min_macd)
                    
                    if 'stoch_crossover' in normalized_df.columns:
                        # Scale and shift Stochastic crossover to 0-1 range
                        min_stoch = normalized_df['stoch_crossover'].min()
                        max_stoch = normalized_df['stoch_crossover'].max()
                        if max_stoch != min_stoch:  # Prevent division by zero
                            normalized_df['stoch_crossover'] = (normalized_df['stoch_crossover'] - min_stoch) / (max_stoch - min_stoch)
                    
                    if 'di_crossover' in normalized_df.columns:
                        # Scale and shift DI crossover to 0-1 range
                        min_di = normalized_df['di_crossover'].min()
                        max_di = normalized_df['di_crossover'].max()
                        if max_di != min_di:  # Prevent division by zero
                            normalized_df['di_crossover'] = (normalized_df['di_crossover'] - min_di) / (max_di - min_di)
                    
                    # Normalize MA difference values to 0-1 range
                    for ma_diff in ['ma20_50_diff', 'ma50_100_diff', 'ma20_100_diff']:
                        if ma_diff in normalized_df.columns:
                            min_val = normalized_df[ma_diff].min()
                            max_val = normalized_df[ma_diff].max()
                            if max_val != min_val:  # Prevent division by zero
                                normalized_df[ma_diff] = (normalized_df[ma_diff] - min_val) / (max_val - min_val)
                    
                    # Finally, determine trading instruction
                    # This is a simplified placeholder - replace with your actual logic
                    import random
                    # For demonstration, randomly choose an instruction
                    choices = ["BUY", "SELL", "HOLD"]
                    weights = [0.4, 0.4, 0.2]  # 40% buy, 40% sell, 20% hold
                    instruction = random.choices(choices, weights=weights, k=1)[0]
                    
                    print(f"\n----- TRADING INSTRUCTION -----")
                    print(f"Instruction: {instruction}")
                    
                    # Remember this action for next time a trade completes
                    last_trade_action = instruction
                    
                    # If instruction is to open a new trade, update our state
                    if instruction in ["BUY", "SELL"]:
                        current_trade_open = True
            
            # Return the trading instruction
            return jsonify({"response": f"Data processed successfully. INSTRUCTION: {instruction}"})
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