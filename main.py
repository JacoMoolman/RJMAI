# Configuration
NUM_BARS = 5  # Number of previous bars to fetch

from get_data import get_forex_data
import pandas as pd

def main():
    # Example: Get EUR/USD forex data
    forex_pair = "EURUSD"
    timeframe = "M5"
    
    print(f"Fetching last {NUM_BARS} {timeframe} bars for {forex_pair}...")
    
    # Get data for multiple bars
    data = get_forex_data(forex_pair, timeframe, NUM_BARS)
    
    if "error" in data:
        print(f"Error: {data['error']}")
    else:
        print(f"\nTime Range: {data['StartTime']} to {data['EndTime']}")
        
        # Create flat DataFrame with timeframe in column names
        df_data = {}
        for i in range(1, NUM_BARS + 1):
            df_data[f"Open_{timeframe}_{i}"] = [data[f"Open{i}"]]
            df_data[f"High_{timeframe}_{i}"] = [data[f"High{i}"]]
            df_data[f"Low_{timeframe}_{i}"] = [data[f"Low{i}"]]
            df_data[f"Close_{timeframe}_{i}"] = [data[f"Close{i}"]]
            
        df = pd.DataFrame(df_data)
        print("\nForex Data:")
        print("-" * 40)
        print(df)

if __name__ == "__main__":
    main()
