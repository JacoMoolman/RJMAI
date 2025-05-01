import pandas as pd
import numpy as np
import os

# Maximum number of support/resistance levels to track
MAX_SR_LEVELS = 20

def normalize_data(timeframes_data, timeframe_map):
    """
    Process each timeframe's data to create normalized dataframes with time features.
    All price data is normalized across all timeframes as a whole, not per timeframe.
    
    Args:
        timeframes_data (dict): Dictionary of timeframes and their corresponding bars data
        timeframe_map (dict): Mapping of timeframe strings to normalized values
        
    Returns:
        list: List of processed dataframes for each timeframe
    """
    dfs_list = []
    all_price_data = []
    
    # First pass: Create dataframes for each timeframe and gather all price data
    temp_dfs = {}
    for timeframe, bars in timeframes_data.items():
        if timeframe in timeframe_map:
            # Convert the bars to a DataFrame
            df = pd.DataFrame(bars)
            
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M:%S')
            
            # Add day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
            # Pandas weekday is 0=Monday, so we need to adjust to make Sunday=0
            raw_day_of_week = (df['time'].dt.weekday + 1) % 7
            
            # Normalize day of week between 0 and 1
            df['day_of_week'] = (raw_day_of_week / 6.0).round(6)  # Divide by (7-1) to normalize
            
            # Extract time components (remove seconds, split hour and minute)
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            
            # cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).round(6)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).round(6)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60).round(6)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60).round(6)
            
            # Drop the raw hour and minute columns
            df = df.drop(['hour', 'minute'], axis=1)
            
            # Replace day-of-month with day-of-year normalization
            df['day_of_year'] = (df['time'].dt.dayofyear / 365.0).round(6)
            
            # Convert all numeric columns to float to avoid type issues
            for col in df.columns:
                if col != 'time' and col != 'spread' and col != 'timeframe':
                    df[col] = df[col].astype(float)
            
            # Store the dataframe for later processing
            temp_dfs[timeframe] = df
            
            # Collect price data for global normalization
            price_columns = ['open', 'high', 'low', 'close']
            price_data = df[price_columns].copy()
            all_price_data.append(price_data)
    
    # Calculate global min and max across ALL timeframes for price data
    combined_price_data = pd.concat(all_price_data, ignore_index=True)
    global_min_price = combined_price_data.min().min()  # Global min across all timeframes and price columns
    global_max_price = combined_price_data.max().max()  # Global max across all timeframes and price columns
    global_price_range = global_max_price - global_min_price
    
    # Gather indicator data for global normalization
    indicator_groups = {
        'moving_averages': ['ma20', 'ma50', 'ma100'],
        'bollinger': ['bb_upper', 'bb_middle', 'bb_lower'],
        'stochastic': ['stoch_k', 'stoch_d'],
        'macd': ['macd_main', 'macd_signal', 'macd_hist'],
        'directional': ['adx', 'plus_di', 'minus_di'],
        'ichimoku': ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b']
    }
    
    # Create dictionaries to store global min/max for each indicator
    global_min_max = {}
    
    # Collect volume data for global normalization
    all_volume = []
    
    # Identify which indicators exist in the data and gather their values
    for timeframe, df in temp_dfs.items():
        # Check volume
        if 'volume' in df.columns:
            all_volume.append(df['volume'])
        
        # Check indicators
        for group_name, indicators in indicator_groups.items():
            existing_indicators = [ind for ind in indicators if ind in df.columns]
            
            if existing_indicators:
                for indicator in existing_indicators:
                    if indicator not in global_min_max:
                        global_min_max[indicator] = {'data': []}
                    
                    global_min_max[indicator]['data'].append(df[[indicator]])
    
    # Calculate global min/max for each indicator
    for indicator, indicator_info in global_min_max.items():
        if indicator_info['data']:
            combined_data = pd.concat(indicator_info['data'], ignore_index=True)
            indicator_info['min'] = combined_data[indicator].min()
            indicator_info['max'] = combined_data[indicator].max()
    
    # Calculate global min/max for volume
    global_min_vol = None
    global_max_vol = None
    if all_volume:
        combined_volume = pd.concat(all_volume, ignore_index=True)
        global_min_vol = combined_volume.min()
        global_max_vol = combined_volume.max()
    
    # Second pass: Apply global normalization to each dataframe
    for timeframe, df in temp_dfs.items():
        # Normalize price columns using global min/max
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = ((df[col] - global_min_price) / global_price_range).round(6)
        
        # Normalize volume using global min/max if it exists
        if 'volume' in df.columns and global_min_vol is not None and global_max_vol > global_min_vol:
            df['volume'] = ((df['volume'] - global_min_vol) / (global_max_vol - global_min_vol)).round(6)
        
        # Normalize technical indicators individually using their own global min/max
        for indicator, indicator_info in global_min_max.items():
            if indicator in df.columns and indicator_info.get('max', 0) > indicator_info.get('min', 0):
                df[indicator] = ((df[indicator] - indicator_info['min']) / 
                                (indicator_info['max'] - indicator_info['min'])).round(6)
        
        # Handle RSI separately as it already has a natural 0-100 range
        if 'rsi' in df.columns:
            df['rsi'] = (df['rsi'] / 100.0).round(6)
        
        # Drop the original time column and spread column
        df = df.drop(['time', 'spread'], axis=1)
        
        # Add the timeframe column and convert to numerical
        df['timeframe'] = timeframe
        df['timeframe'] = df['timeframe'].map(timeframe_map).round(6)
        
        # Add to list for concatenation
        dfs_list.append(df)
    
    return dfs_list

def cluster_price_levels_with_strength(price_points, threshold, touch_counts):
    """
    Group similar price levels and return the average of each cluster along with its strength.
    
    Args:
        price_points: List of price points to cluster
        threshold: Minimum distance between clusters
        touch_counts: Dictionary mapping rounded price levels to their touch counts
        
    Returns:
        list: List of tuples (level, strength) for each clustered price level
    """
    if not price_points:
        return []
    
    price_points = sorted(price_points)
    clusters = []
    current_cluster = [price_points[0]]
    
    for price in price_points[1:]:
        if abs(price - current_cluster[0]) <= threshold:
            current_cluster.append(price)
        else:
            # Calculate average level and total strength for this cluster
            avg_level = sum(current_cluster) / len(current_cluster)
            
            # Calculate strength based on number of touches
            # First, find all touch counts that are close to this cluster
            total_touches = 0
            for level in current_cluster:
                rounded = round(level, 3)
                if rounded in touch_counts:
                    total_touches += touch_counts[rounded]
            
            # Calculate raw strength (will be normalized later)
            strength = total_touches
            
            clusters.append((avg_level, strength))
            current_cluster = [price]
    
    # Add the last cluster
    if current_cluster:
        avg_level = sum(current_cluster) / len(current_cluster)
        
        # Calculate strength for last cluster
        total_touches = 0
        for level in current_cluster:
            rounded = round(level, 3)
            if rounded in touch_counts:
                total_touches += touch_counts[rounded]
        
        strength = total_touches
        clusters.append((avg_level, strength))
    
    return clusters

def export_df_to_csv(df, stage_name, symbol="UNKNOWN", csv_output_dir="data/csv_output", export_to_csv=True):
    """
    Export a DataFrame to CSV file
    
    Args:
        df: DataFrame to export
        stage_name: Name of the processing stage (for filename)
        symbol: Trading symbol
        csv_output_dir: Directory to save CSV files to
        export_to_csv: Flag to determine if export should occur
    """
    if not export_to_csv:
        return
        
    # Use a single file per stage that gets overwritten
    filename = f"{csv_output_dir}/{symbol}_{stage_name}.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Export to CSV, overwriting previous file
    df.to_csv(filename, index=True)
    print(f"Exported {stage_name} DataFrame to {filename}")

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

def add_difference_columns(normalized_df):
    """
    Add difference and relationship columns to a normalized dataframe
    
    Args:
        normalized_df: DataFrame with normalized market data
        
    Returns:
        DataFrame: DataFrame with added difference columns
    """
    # Calculate differences between OHLC values - use absolute values to avoid negatives
    normalized_df['diff_open_high'] = np.abs(normalized_df['open'] - normalized_df['high']).round(6)
    normalized_df['diff_open_low'] = np.abs(normalized_df['open'] - normalized_df['low']).round(6)
    normalized_df['diff_open_close'] = np.abs(normalized_df['open'] - normalized_df['close']).round(6)
    normalized_df['diff_high_low'] = np.abs(normalized_df['high'] - normalized_df['low']).round(6)
    normalized_df['diff_high_close'] = np.abs(normalized_df['high'] - normalized_df['close']).round(6)
    normalized_df['diff_low_close'] = np.abs(normalized_df['low'] - normalized_df['close']).round(6)

    
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
        
    return normalized_df
