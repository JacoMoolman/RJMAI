import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import time # For a small delay if needed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def add_frequency_heatmap_background(ax, df_ohlc, num_levels=50, cmap_name='RdYlGn_r', alpha=0.35):
    """
    Adds a frequency-based heatmap as the background of the price chart.
    - Green: Low frequency of price interaction
    - Yellow: Medium frequency
    - Red: High frequency
    """
    if df_ohlc.empty:
        return

    lowest_low = df_ohlc['Low'].min()
    highest_high = df_ohlc['High'].max()

    if pd.isna(lowest_low) or pd.isna(highest_high) or lowest_low >= highest_high:
        print("Cannot determine price range for heatmap or range is invalid.")
        return

    # price_levels_coords defines the bottom of each band
    price_levels_coords = np.linspace(lowest_low, highest_high, num_levels, endpoint=False)
    band_height = (highest_high - lowest_low) / num_levels
    level_frequency_counts = np.zeros(num_levels)

    # Calculate frequency: count how many bars intersected each price_level_coord
    for i in range(num_levels):
        current_level_price = price_levels_coords[i]
        count = 0
        for _, row in df_ohlc.iterrows():
            # Check if the bar's Low-High range straddles the current_level_price line
            if row['Low'] <= current_level_price < row['High']:
                count += 1
        level_frequency_counts[i] = count
    
    min_count = level_frequency_counts.min()
    max_count = level_frequency_counts.max()

    if max_count == min_count: # Avoid division by zero if all counts are the same
        # Assign a neutral middle color (e.g., yellow) to all bands
        normalized_counts = np.full(num_levels, 0.5)
    else:
        normalized_counts = (level_frequency_counts - min_count) / (max_count - min_count)

    cmap = plt.get_cmap(cmap_name) # RdYlGn_r: Red=High, Yellow=Mid, Green=Low
    colors_for_bands = cmap(normalized_counts)

    for i in range(num_levels):
        y_bottom = price_levels_coords[i]
        y_top = y_bottom + band_height
        # zorder=0.5 places heatmap bands behind candles (zorder=1.0) but above default background/grid (zorder < 0.9)
        ax.axhspan(y_bottom, y_top, color=colors_for_bands[i], alpha=alpha, zorder=0.5, ec='none')

    # Ensure y-axis limits accommodate the heatmap bands fully
    ax.set_ylim(lowest_low, highest_high)

def fetch_and_plot_m5_data(symbol="EURUSD", num_heatmap_levels=50):
    """
    Connects to MetaTrader 5, fetches M5 data for the last month for the given symbol,
    and plots it as a candlestick chart with a frequency heatmap background.
    """
    print(f"Attempting to initialize MetaTrader 5 connection...")
    # Attempt to initialize with a timeout and retries
    initialized = False
    for i in range(3): # Try 3 times
        if mt5.initialize():
            initialized = True
            print("MetaTrader 5 initialized successfully.")
            break
        else:
            print(f"initialize() failed, error code = {mt5.last_error()}. Retrying in 5 seconds... (Attempt {i+1}/3)")
            time.sleep(5)
            
    if not initialized:
        print("Failed to initialize MetaTrader 5 after multiple attempts.")
        return

    print(f"Checking symbol: {symbol}")
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found in MetaTrader 5. Error code: {mt5.last_error()}")
        mt5.shutdown()
        return

    if not symbol_info.visible:
        print(f"Symbol {symbol} is not visible, attempting to enable it.")
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}, True) failed. Error code: {mt5.last_error()}")
            mt5.shutdown()
            return
        print(f"Symbol {symbol} enabled successfully.")
        time.sleep(1) # Give a moment for the terminal to update

    # Define timeframe
    timeframe = mt5.TIMEFRAME_M5

    # Calculate date range for the last month (approx 30 days)
    utc_to = datetime.utcnow()
    utc_from = utc_to - timedelta(days=30)

    print(f"Fetching {symbol} M5 data from {utc_from.strftime('%Y-%m-%d %H:%M')} UTC to {utc_to.strftime('%Y-%m-%d %H:%M')} UTC.")

    # Get rates
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

    # Shutdown MT5 connection
    print("Shutting down MetaTrader 5 connection.")
    mt5.shutdown()

    if rates is None:
        print(f"No data received for {symbol}. mt5.copy_rates_range() returned None. Error code: {mt5.last_error()}")
        return
    
    if len(rates) == 0:
        print(f"No rates returned for {symbol} in the specified range.")
        return

    print(f"Successfully fetched {len(rates)} bars for {symbol}.")

    # Convert to pandas DataFrame
    rates_df = pd.DataFrame(rates)
    # Convert time in seconds into datetime format (MT5 times are usually UTC)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    rates_df.set_index('time', inplace=True)
    
    # Rename columns for mplfinance compatibility (Open, High, Low, Close, Volume)
    rates_df.rename(columns={
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'tick_volume': 'Volume' # or 'real_volume' if available and preferred
    }, inplace=True)
    
    if rates_df.empty:
        print("Dataframe is empty after processing. Cannot plot.")
        return

    print(f"Data for {symbol} M5 (first 5 bars):")
    print(rates_df.head())

    # Plotting the candlestick chart
    print(f"Plotting candlestick chart with frequency heatmap for {symbol}...")
    try:
        fig, axes = mpf.plot(rates_df, type='candle', style='yahoo',
                 title=f'{symbol} M5 Candlestick Chart - Last 30 Days',
                 ylabel='Price',
                 volume=False, # Volume subplot removed
                 figratio=(16,8), # Aspect ratio of the figure
                 tight_layout=True,
                 returnfig=True) # Needed to get axes for heatmap
        
        if axes and len(axes) > 0:
            price_ax = axes[0] # The main panel where candles are plotted
            add_frequency_heatmap_background(price_ax, rates_df[['Low', 'High']], num_levels=num_heatmap_levels)
        
        plt.show() # Ensure the plot is displayed
        
        print("Plot displayed. Close the plot window to continue.")
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- IMPORTANT --- 
    # Ensure your MetaTrader 5 terminal is running and you are logged into your account.
    # The script will attempt to connect to it.
    
    # You can change the default trading symbol here:
    target_symbol = "EURUSD" 
    heatmap_levels_count = 500 # Number of horizontal bands for the frequency heatmap
    
    print(f"--- Starting M5 Data Fetch and Plot for {target_symbol} with {heatmap_levels_count} heatmap levels ---")
    print("Ensure your MetaTrader 5 terminal is running and logged in.")
    
    fetch_and_plot_m5_data(symbol=target_symbol, num_heatmap_levels=heatmap_levels_count)
    
    print(f"--- Script finished ---")
