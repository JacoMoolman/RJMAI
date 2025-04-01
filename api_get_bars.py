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
            raw_data = request.data.decode('utf-8')
            fixed_data = re.sub(r'("time": )(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})', r'\1"\2"', raw_data)
            
            # Parse the fixed JSON data
            data = json.loads(fixed_data)
        
        # Process and display forex data using pandas
        if 'symbol' in data and 'data' in data:
            symbol = data['symbol']
            timeframes_data = data['data']
            
            print(f"\n===== RECEIVED DATA FOR SYMBOL: {symbol} =====")
            
            # List to store DataFrames for each timeframe for concatenation
            dfs_list = []
            
            # Process each timeframe
            for timeframe, bars in timeframes_data.items():
                # Convert the bars to a DataFrame
                df = pd.DataFrame(bars)
                
                # Convert timestamp to datetime
                df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M:%S')
                
                # Add the timeframe column
                df['timeframe'] = timeframe
                
                # Add to list for concatenation
                dfs_list.append(df)
            
            # Concatenate all dataframes into a single dataframe
            if dfs_list:
                combined_df = pd.concat(dfs_list, ignore_index=True)
                
                # Display the combined DataFrame
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                print(combined_df)
                       
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