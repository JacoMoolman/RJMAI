###THIS GETS DATA FROM MT5


import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz

def download_forex_data(currency_pair="USDZAR", timeframe=mt5.TIMEFRAME_H1, days_back=7, start_date=None, end_date=None):
    """
    Download forex data for a specified currency pair and time range.
    
    Args:
        currency_pair (str): The currency pair to download (e.g., "USDZAR", "EURUSD")
        timeframe (int): MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)
        days_back (int): Number of days to look back if start_date and end_date are not provided
        start_date (str, optional): Start date in format 'YYYY-MM-DD'
        end_date (str, optional): End date in format 'YYYY-MM-DD'
    
    Returns:
        DataFrame: Pandas DataFrame with the downloaded data or None if failed
    """
    # Connect to MetaTrader 5
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return None
    
    # Set the timezone to UTC
    timezone = pytz.timezone("Etc/UTC")
    
    # Calculate the start and end time based on input parameters
    if start_date and end_date:
        # Convert string dates to datetime objects
        utc_from = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone)
        utc_to = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone)
    else:
        # Use days_back parameter
        utc_to = datetime.now(timezone)
        utc_from = utc_to - timedelta(days=days_back)
    
    # Get forex data
    print(f"Downloading {currency_pair} data from {utc_from} to {utc_to}")
    forex_data = mt5.copy_rates_range(currency_pair, timeframe, utc_from, utc_to)
    
    if forex_data is None:
        print(f"Failed to download {currency_pair} data, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(forex_data)
    
    # Convert time in seconds into datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Save to CSV
    csv_file = f"{currency_pair}_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")
    
    # Shutdown connection to MetaTrader 5
    mt5.shutdown()
    
    return df

if __name__ == "__main__":
    # Check if MetaTrader 5 package is installed
    try:
        import MetaTrader5
    except ImportError:
        print("MetaTrader5 package is not installed. Please install it using:")
        print("pip install MetaTrader5")
        exit(1)
    
    # Configuration parameters
    CURRENCY_PAIR = "USDZAR"  # Change this to any currency pair available in your MT5 terminal
    TIMEFRAME = mt5.TIMEFRAME_H1  # Options: TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_H1, TIMEFRAME_D1, etc.
    
    # Option 1: Specify number of days to look back
    # DAYS_BACK = 7
    
    # Option 2: Specify exact date range (comment out if using DAYS_BACK)
    START_DATE = "2023-01-01"  # Format: YYYY-MM-DD
    END_DATE = "2023-01-31"    # Format: YYYY-MM-DD
    
    # Download data using the configuration above
    data = download_forex_data(
        currency_pair=CURRENCY_PAIR,
        timeframe=TIMEFRAME,
        # days_back parameter is no longer needed since we're using start_date and end_date
        # days_back=DAYS_BACK,
        # Uncomment to use specific date range instead of days_back
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    if data is not None:
        print("Data preview:")
        print(data.head())
