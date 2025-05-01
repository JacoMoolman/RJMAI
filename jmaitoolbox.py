import pandas as pd

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
            
            # Normalize hour and minute between 0 and 1
            df['hour'] = (df['hour'] / 23.0).round(6)  # Divide by (24-1) to normalize
            df['minute'] = (df['minute'] / 59.0).round(6)  # Divide by (60-1) to normalize
            
            # Extract date components (drop year, split month and day)
            df['day'] = df['time'].dt.day
            
            # Normalize day (of month) between 0 and 1
            # Using 31 as the maximum possible day to normalize
            df['day'] = (df['day'] / 31.0).round(6)
            
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
    
    # Create dictionaries to store global min/max for each indicator group
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
                if group_name not in global_min_max:
                    global_min_max[group_name] = {'data': [], 'indicators': existing_indicators}
                
                global_min_max[group_name]['data'].append(df[existing_indicators])
    
    # Calculate global min/max for each indicator group
    for group_name, group_info in global_min_max.items():
        if group_info['data']:
            combined_data = pd.concat(group_info['data'], ignore_index=True)
            group_info['min'] = combined_data.min().min()
            group_info['max'] = combined_data.max().max()
    
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
        
        # Normalize technical indicators by groups using global min/max
        for group_name, group_info in global_min_max.items():
            existing_indicators = [ind for ind in group_info['indicators'] if ind in df.columns]
            
            if existing_indicators and group_info.get('max', 0) > group_info.get('min', 0):
                for indicator in existing_indicators:
                    df[indicator] = ((df[indicator] - group_info['min']) / (group_info['max'] - group_info['min'])).round(6)
        
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
