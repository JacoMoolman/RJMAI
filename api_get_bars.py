from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import datetime
import random
import math
import numpy as np
import os
from datetime import datetime
from jmaitoolbox import normalize_data, detect_support_resistance, cluster_price_levels_with_strength

# Configuration
EXPORT_TO_CSV = True  # Set to False to disable CSV exports
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
            
            # Export raw dataframes before additional normalization
            for i, df in enumerate(dfs_list):
                tf = df['timeframe'].iloc[0]
                export_df_to_csv(df, f"stage1_raw_timeframe_{tf}", symbol)
                            
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
                    
                    # Add technical indicators to normalization if they exist
                    technical_indicators = [
                        'ma20', 'ma50', 'ma100',                                # Moving Averages
                        'bb_upper', 'bb_middle', 'bb_lower',                    # Bollinger Bands
                        'macd_main', 'macd_signal', 'macd_hist',                # MACD
                        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b'  # Ichimoku
                    ]
                    
                    # Add technical indicators that exist to the list for normalization
                    for indicator in technical_indicators:
                        if indicator in df.columns:
                            cols_to_normalize.append(indicator)
                    
                    # Oscillators (already between 0-100 or similar ranges, so normalize separately)
                    oscillator_indicators = {
                        'rsi': (0, 100),
                        'stoch_k': (0, 100),
                        'stoch_d': (0, 100),
                        'adx': (0, 100), 
                        'plus_di': (0, 100),
                        'minus_di': (0, 100)
                    }
                    
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
                    
                    # Normalize oscillators (which typically have fixed ranges)
                    for indicator, (min_val, max_val) in oscillator_indicators.items():
                        if indicator in df.columns:
                            # Normalize between 0 and 1 using the fixed range
                            normalized_df[indicator] = (df[indicator] - min_val) / (max_val - min_val)
                    
                    # Export individual normalized timeframe data
                    export_df_to_csv(normalized_df, f"stage2_normalized_timeframe_{tf}", symbol)
                    
                    normalized_dfs.append(normalized_df)
                
                # Now concatenate the normalized dataframes
                normalized_df = pd.concat(normalized_dfs, ignore_index=True)
                
                # Export concatenated data before adding difference columns
                export_df_to_csv(normalized_df, "stage3_concatenated", symbol)

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
                
                # --- END ADD DIFFERENCE COLUMNS ---
                
                # Export the data after adding all derived features
                export_df_to_csv(normalized_df, "stage4_with_derived_features", symbol)
                
                # Detect support and resistance levels
                levels_df = detect_support_resistance(normalized_df)
                
                # Cache the processed dataframes
                cache[symbol] = {
                    'price_data': normalized_df,
                    'levels': levels_df
                }
                
                # Display the combined DataFrame with truncation (showing ... in the middle)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', 10)  # Show all rows
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
                
                # Export final data
                export_df_to_csv(normalized_df, "stage5_final_normalized_data", symbol)
                export_df_to_csv(levels_df, "stage6_support_resistance_levels", symbol)
                
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