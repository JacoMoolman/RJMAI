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
from jmaitoolbox import normalize_data, cluster_price_levels_with_strength

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
CSV_OUTPUT_DIR = "data_exports"  # Directory to store CSV files

# Default lot size
DEFAULT_LOT_SIZE = 0.01

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

def identify_price_levels(timeframes_data, normalized_df=None):
    """
    Identify important price levels based on frequency of occurrence per timeframe.
    
    Args:
        timeframes_data: Raw price data from all timeframes
        normalized_df: The normalized dataframe to match normalization with
        
    Returns:
        DataFrame: DataFrame containing the most frequently occurring price levels per timeframe
    """
    # Get normalization values - if we have a normalized dataframe, use its min/max
    all_open_prices = []
    all_high_prices = []
    all_low_prices = []
    all_close_prices = []
    
    for timeframe, bars in timeframes_data.items():
        for bar in bars:
            all_open_prices.append(float(bar['open']))
            all_high_prices.append(float(bar['high']))
            all_low_prices.append(float(bar['low']))
            all_close_prices.append(float(bar['close']))
    
    # Find global min/max across all price columns
    min_price = min(
        min(all_open_prices),
        min(all_high_prices),
        min(all_low_prices),
        min(all_close_prices)
    )
    max_price = max(
        max(all_open_prices),
        max(all_high_prices),
        max(all_low_prices),
        max(all_close_prices)
    )
    
    price_range = max_price - min_price
    
    # Round prices to handle very close values
    precision = 5  # Adjust based on your instrument's typical precision
    
    # First, gather all price levels and their count across ALL timeframes
    from collections import Counter
    all_prices_counter = Counter()
    
    for timeframe, bars in timeframes_data.items():
        timeframe_prices = [round(float(bar['close']), precision) for bar in bars]
        # Update the global counter with this timeframe's prices
        all_prices_counter.update(timeframe_prices)
    
    # Create a DataFrame to store all timeframe price levels
    all_timeframe_levels = pd.DataFrame()
    
    # Process each timeframe separately
    for timeframe, bars in timeframes_data.items():
        # Extract close prices from this timeframe
        timeframe_prices = [round(float(bar['close']), precision) for bar in bars]
        
        # Create a set of unique prices in this timeframe
        unique_timeframe_prices = set(timeframe_prices)
        
        # Create a dataframe of price levels for this timeframe
        price_data = []
        for price in unique_timeframe_prices:
            # Get the global frequency (from all timeframes)
            global_frequency = all_prices_counter[price]
            price_data.append({
                'price_level': price,
                'frequency': global_frequency
            })
        
        # Convert to DataFrame
        price_levels = pd.DataFrame(price_data)
        
        # If no prices found for this timeframe, continue to next timeframe
        if price_levels.empty:
            continue
            
        # Sort by frequency (highest first)
        price_levels = price_levels.sort_values('frequency', ascending=False)
        
        # Take top 10 prices
        levels_per_timeframe = 10
        price_levels = price_levels.head(levels_per_timeframe).reset_index(drop=True)
        
        # Calculate normalized frequency
        max_freq = price_levels['frequency'].max() if not price_levels.empty else 1
        price_levels['strength'] = (price_levels['frequency'] / max_freq).round(6)
        
        # Add normalized price values using EXACTLY the same min/max as the main dataframe
        price_levels['normalized_price'] = ((price_levels['price_level'] - min_price) / price_range).round(6)
        
        # Add timeframe column
        price_levels['timeframe'] = timeframe
        
        # Append to the all timeframes DataFrame
        all_timeframe_levels = pd.concat([all_timeframe_levels, price_levels])
    
    # Reorder columns for better readability
    all_timeframe_levels = all_timeframe_levels[['timeframe', 'price_level', 'normalized_price', 'frequency', 'strength']]
    
    # Reset index to have a continuous index across all timeframes
    all_timeframe_levels = all_timeframe_levels.reset_index(drop=True)
    
    return all_timeframe_levels

def calculate_sl_tp(normalized_df, action, current_price, ai_normalized_sl, ai_normalized_tp):
    """
    Calculate Stop Loss and Take Profit values based on normalized data.
    
    Args:
        normalized_df: DataFrame with normalized market data
        action: Trading action ('BUY' or 'SELL')
        current_price: Current market price
        ai_normalized_sl: AI-selected normalized SL value (0-1 range)
        ai_normalized_tp: AI-selected normalized TP value (0-1 range)
        
    Returns:
        tuple: (normalized_sl, normalized_tp, actual_sl, actual_tp)
    """
    # These normalized values (0-1) are what the AI would select
    normalized_sl = round(ai_normalized_sl, 6)  # Ensure 6 decimal places
    normalized_tp = round(ai_normalized_tp, 6)  # Ensure 6 decimal places
    
    # Convert normalized SL/TP back to actual price values
    # First, find the actual price range in the current symbol
    price_data = normalized_df[['timeframe', 'open', 'high', 'low', 'close']]
    actual_min = float('inf')
    actual_max = float('-inf')
    
    # Find the original price range before normalization
    for timeframe_group in price_data.groupby('timeframe'):
        tf_data = timeframe_group[1]
        actual_min = min(actual_min, tf_data['low'].min())
        actual_max = max(actual_max, tf_data['high'].max())
    
    actual_range = actual_max - actual_min
    
    # Convert to actual price values based on the action
    if action == "BUY":
        actual_sl = current_price - (normalized_sl * actual_range / 10)
        actual_tp = current_price + (normalized_tp * actual_range / 10)
    else:  # SELL
        actual_sl = current_price + (normalized_sl * actual_range / 10)
        actual_tp = current_price - (normalized_tp * actual_range / 10)
    
    # Format to 5 digits after decimal for actual SL/TP
    actual_sl = round(actual_sl, 5)
    actual_tp = round(actual_tp, 5)
    
    return normalized_sl, normalized_tp, actual_sl, actual_tp

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
                    normalized_df['diff_open_high'] = (normalized_df['open'] - normalized_df['high']).round(6)
                    normalized_df['diff_open_low'] = (normalized_df['open'] - normalized_df['low']).round(6)
                    normalized_df['diff_open_close'] = (normalized_df['open'] - normalized_df['close']).round(6)

                    normalized_df['diff_high_open'] = (normalized_df['high'] - normalized_df['open']).round(6)
                    normalized_df['diff_high_low'] = (normalized_df['high'] - normalized_df['low']).round(6)
                    normalized_df['diff_high_close'] = (normalized_df['high'] - normalized_df['close']).round(6)

                    normalized_df['diff_low_open'] = (normalized_df['low'] - normalized_df['open']).round(6)
                    normalized_df['diff_low_high'] = (normalized_df['low'] - normalized_df['high']).round(6)
                    normalized_df['diff_low_close'] = (normalized_df['low'] - normalized_df['close']).round(6)

                    normalized_df['diff_close_open'] = (normalized_df['close'] - normalized_df['open']).round(6)
                    normalized_df['diff_close_high'] = (normalized_df['close'] - normalized_df['high']).round(6)
                    normalized_df['diff_close_low'] = (normalized_df['close'] - normalized_df['low']).round(6)
                    
                    # Add indicator crossovers and differences (where applicable)
                    if all(col in normalized_df.columns for col in ['ma20', 'ma50']):
                        normalized_df['ma20_50_diff'] = (normalized_df['ma20'] - normalized_df['ma50']).round(6)
                        
                    if all(col in normalized_df.columns for col in ['ma50', 'ma100']):
                        normalized_df['ma50_100_diff'] = (normalized_df['ma50'] - normalized_df['ma100']).round(6)
                        
                    if all(col in normalized_df.columns for col in ['ma20', 'ma100']):
                        normalized_df['ma20_100_diff'] = (normalized_df['ma20'] - normalized_df['ma100']).round(6)
                    
                    if all(col in normalized_df.columns for col in ['macd_main', 'macd_signal']):
                        normalized_df['macd_crossover'] = (normalized_df['macd_main'] - normalized_df['macd_signal']).round(6)
                    
                    if all(col in normalized_df.columns for col in ['stoch_k', 'stoch_d']):
                        normalized_df['stoch_crossover'] = (normalized_df['stoch_k'] - normalized_df['stoch_d']).round(6)
                    
                    if all(col in normalized_df.columns for col in ['plus_di', 'minus_di']):
                        normalized_df['di_crossover'] = (normalized_df['plus_di'] - normalized_df['minus_di']).round(6)
                    
                    # Bollinger Band position
                    if all(col in normalized_df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                        normalized_df['bb_position'] = ((normalized_df['close'] - normalized_df['bb_lower']) / (normalized_df['bb_upper'] - normalized_df['bb_lower'])).round(6)
                        # Handle division by zero
                        normalized_df['bb_position'] = normalized_df['bb_position'].fillna(0.5)
                        # Normalize to 0-1 range
                        normalized_df['bb_position'] = normalized_df['bb_position'].clip(0, 1)
                    
                    # Ichimoku cloud status
                    if all(col in normalized_df.columns for col in ['close', 'ichimoku_senkou_a', 'ichimoku_senkou_b']):
                        # Above cloud = 1, Below cloud = 0, In cloud = 0.5
                        normalized_df['cloud_position'] = ((normalized_df['close'] > normalized_df['ichimoku_senkou_a']) & 
                                                        (normalized_df['close'] > normalized_df['ichimoku_senkou_b'])).astype(float).round(6)
                        
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
                            normalized_df['macd_crossover'] = ((normalized_df['macd_crossover'] - min_macd) / (max_macd - min_macd)).round(6)
                    
                    if 'stoch_crossover' in normalized_df.columns:
                        # Scale and shift Stochastic crossover to 0-1 range
                        min_stoch = normalized_df['stoch_crossover'].min()
                        max_stoch = normalized_df['stoch_crossover'].max()
                        if max_stoch != min_stoch:  # Prevent division by zero
                            normalized_df['stoch_crossover'] = ((normalized_df['stoch_crossover'] - min_stoch) / (max_stoch - min_stoch)).round(6)
                    
                    if 'di_crossover' in normalized_df.columns:
                        # Scale and shift DI crossover to 0-1 range
                        min_di = normalized_df['di_crossover'].min()
                        max_di = normalized_df['di_crossover'].max()
                        if max_di != min_di:  # Prevent division by zero
                            normalized_df['di_crossover'] = ((normalized_df['di_crossover'] - min_di) / (max_di - min_di)).round(6)
                    
                    # Normalize MA difference values to 0-1 range
                    for ma_diff in ['ma20_50_diff', 'ma50_100_diff', 'ma20_100_diff']:
                        if ma_diff in normalized_df.columns:
                            min_val = normalized_df[ma_diff].min()
                            max_val = normalized_df[ma_diff].max()
                            if max_val != min_val:  # Prevent division by zero
                                normalized_df[ma_diff] = ((normalized_df[ma_diff] - min_val) / (max_val - min_val)).round(6)
                    
                    # Display complete normalized dataframe information AFTER all processing
                    print("\n----- COMPLETE NORMALIZED DATAFRAME -----")
                    print(f"Shape: {normalized_df.shape}")  # Shows number of rows and columns
                    print(f"Columns: {normalized_df.columns.tolist()}")
                    print("\nFirst few rows:")
                    pd.set_option('display.max_columns', None)  # Show all columns
                    pd.set_option('display.width', 1000)        # Wider display
                    print(normalized_df.head())
                    print("\n------------------------------")
                    
                    # Export normalized dataframe to CSV
                    export_df_to_csv(normalized_df, "normalized_data", symbol)
                    
                    # Get frequency-based price levels from raw data
                    price_levels_df = identify_price_levels(timeframes_data, normalized_df)
                    
                    # Display price levels
                    print("\n----- PRICE LEVELS BASED ON FREQUENCY -----")
                    print(f"Shape: {price_levels_df.shape}")
                    print("Columns: timeframe, price_level, normalized_price, frequency, strength")
                    print("\nPrice Levels:")
                    print(price_levels_df)
                    print("\n------------------------------")
                    
                    # Export price levels to CSV
                    price_levels_df = price_levels_df[['timeframe', 'price_level', 'normalized_price', 'frequency', 'strength']]
                    export_df_to_csv(price_levels_df, "price_levels", symbol)
                    
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
