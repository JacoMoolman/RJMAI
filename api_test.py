from flask import Flask, request, jsonify
import json
import re
import pandas as pd
import datetime

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test_endpoint():
    print("\n===== NEW REQUEST RECEIVED =====")
    
    try:
        # Handle JSON data from MQL5
        if request.is_json:
            data = request.json
        else:
            # Try to parse raw data as JSON, fixing the timestamp format issue
            try:
                raw_data = request.data.decode('utf-8')
                print("Raw data received (first 100 characters):")
                print(raw_data[:100] + "...")
                
                # Fix timestamp format in the raw JSON string (convert 2025.03.25 00:00:00 to "2025.03.25 00:00:00")
                # This regex looks for time patterns that aren't properly quoted
                fixed_data = re.sub(r'("time": )(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})', r'\1"\2"', raw_data)
                
                # Parse the fixed JSON data
                data = json.loads(fixed_data)
            except Exception as e:
                print(f"Error parsing data: {str(e)}")
                print(f"Error at position {e.pos if hasattr(e, 'pos') else 'unknown'}")
                return jsonify({"response": "Error: Could not parse data"})
        
        # Process and display forex data using pandas
        if 'symbol' in data and 'data' in data:
            symbol = data['symbol']
            timeframes_data = data['data']
            
            print(f"\n===== RECEIVED DATA FOR SYMBOL: {symbol} =====")
            
            # Create a dictionary to store DataFrames for each timeframe
            dataframes = {}
            
            # Process each timeframe
            for timeframe, bars in timeframes_data.items():
                print(f"\n----- Timeframe: {timeframe} - {len(bars)} bars -----")
                
                # Convert the bars to a DataFrame
                df = pd.DataFrame(bars)
                
                # Convert timestamp to datetime
                df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M:%S')
                
                # Add the timeframe column
                df['timeframe'] = timeframe
                
                # Store the DataFrame
                dataframes[timeframe] = df
                
                # Display the DataFrame
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                print(df)
                print(f"\nSummary Statistics for {timeframe}:")
                print(df[['open', 'high', 'low', 'close']].describe())
            
            # Summarize the data
            summary_data = []
            for tf, df in dataframes.items():
                row = {
                    'Timeframe': tf,
                    'Bars': len(df),
                    'Start': df['time'].min(),
                    'End': df['time'].max(),
                    'Avg_Range': (df['high'] - df['low']).mean()
                }
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            print("\n===== SUMMARY =====")
            print(summary_df)
            
            print("\n===== DATA PROCESSING COMPLETE =====")
            return jsonify({"response": f"GOT IT! Successfully received and processed data for {symbol} with {len(timeframes_data)} timeframes"})
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