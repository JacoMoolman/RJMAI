import pandas as pd

# Maximum number of support/resistance levels to track
MAX_SR_LEVELS = 20

def normalize_data(timeframes_data, timeframe_map):
    """
    Process each timeframe's data to create normalized dataframes with time features.
    
    Args:
        timeframes_data (dict): Dictionary of timeframes and their corresponding bars data
        timeframe_map (dict): Mapping of timeframe strings to normalized values
        
    Returns:
        list: List of processed dataframes for each timeframe
    """
    dfs_list = []
    
    # Process each timeframe
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
            df['day_of_week'] = raw_day_of_week / 6.0  # Divide by (7-1) to normalize
            
            # Extract time components (remove seconds, split hour and minute)
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            
            # Normalize hour and minute between 0 and 1
            df['hour'] = df['hour'] / 23.0  # Divide by (24-1) to normalize
            df['minute'] = df['minute'] / 59.0  # Divide by (60-1) to normalize
            
            # Extract date components (drop year, split month and day)
            df['day'] = df['time'].dt.day
            
            # Normalize day (of month) between 0 and 1
            # Using 31 as the maximum possible day to normalize
            df['day'] = df['day'] / 31.0
            
            # Check for technical indicators and ensure they exist
            # List of all expected technical indicators
            indicator_columns = [
                'ma20', 'ma50', 'ma100',                                # Moving Averages
                'bb_upper', 'bb_middle', 'bb_lower',                    # Bollinger Bands
                'rsi',                                                  # RSI
                'stoch_k', 'stoch_d',                                   # Stochastic
                'macd_main', 'macd_signal', 'macd_hist',                # MACD
                'adx', 'plus_di', 'minus_di',                           # ADX
                'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b'  # Ichimoku
            ]
            
            # Identify which indicators are available in this data
            available_indicators = [col for col in indicator_columns if col in df.columns]
            
            # Drop the original time column and spread column
            df = df.drop(['time', 'spread'], axis=1)
            
            # Add the timeframe column and convert to numerical
            df['timeframe'] = timeframe
            df['timeframe'] = df['timeframe'].map(timeframe_map)
            
            # Add to list for concatenation
            dfs_list.append(df)
    
    return dfs_list

def detect_support_resistance(df, window_size=5, threshold=0.01):
    """
    Detect support and resistance levels in normalized price data.
    
    Args:
        df: Normalized DataFrame with OHLC data
        window_size: Number of bars to look at for local extrema
        threshold: Minimum distance between levels
        
    Returns:
        DataFrame: DataFrame containing the top support and resistance levels
    """
    # Find local maxima for resistance and local minima for support
    resistance_points = []
    support_points = []
    resistance_touches = {}  # Dictionary to count touches of resistance levels
    support_touches = {}     # Dictionary to count touches of support levels
    
    # Get the sorted high and low values from the full normalized dataframe
    highs = df['high'].values
    lows = df['low'].values
    
    # Find local maxima and minima across the entire dataset
    for i in range(window_size, len(df) - window_size):
        # Check if this point is a local maximum
        if highs[i] == max(highs[i-window_size:i+window_size+1]):
            resistance_points.append(highs[i])
            # Count this as a resistance touch
            r_level = round(highs[i], 3)  # Round to group similar levels
            resistance_touches[r_level] = resistance_touches.get(r_level, 0) + 1
            
        # Check if this point is a local minimum
        if lows[i] == min(lows[i-window_size:i+window_size+1]):
            support_points.append(lows[i])
            # Count this as a support touch
            s_level = round(lows[i], 3)  # Round to group similar levels
            support_touches[s_level] = support_touches.get(s_level, 0) + 1
    
    # Cluster similar levels
    resistance_clusters = cluster_price_levels_with_strength(resistance_points, threshold, resistance_touches)
    support_clusters = cluster_price_levels_with_strength(support_points, threshold, support_touches)
    
    # Find the maximum strength across all clusters for normalization
    all_strengths = [strength for _, strength in support_clusters + resistance_clusters]
    max_strength = max(all_strengths) if all_strengths else 1.0
    
    # Normalize strengths to be between 0 and 1
    normalized_support_clusters = [(level, strength/max_strength) for level, strength in support_clusters]
    normalized_resistance_clusters = [(level, strength/max_strength) for level, strength in resistance_clusters]
    
    # Limit to MAX_SR_LEVELS/2 for each type (support and resistance)
    max_per_type = MAX_SR_LEVELS // 2
    
    # Sort by strength (descending) to get strongest levels
    normalized_support_clusters = sorted(normalized_support_clusters, key=lambda x: x[1], reverse=True)[:max_per_type]
    normalized_resistance_clusters = sorted(normalized_resistance_clusters, key=lambda x: x[1], reverse=True)[:max_per_type]
    
    # Create a dataframe with the levels
    levels_df = pd.DataFrame({
        'level_type': [0]*len(normalized_support_clusters) + [1]*len(normalized_resistance_clusters),  # Binary: 0=support, 1=resistance
        'price_level': [level for level, strength in normalized_support_clusters + normalized_resistance_clusters],
        'strength': [strength for level, strength in normalized_support_clusters + normalized_resistance_clusters]
    })
    
    return levels_df

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
            
            # Calculate raw strength (will be normalized later in detect_support_resistance)
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
