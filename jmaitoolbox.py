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
            
            # NORMALIZE PRICE DATA
            # Convert all numeric columns to float to avoid type issues
            for col in df.columns:
                if col != 'time' and col != 'spread' and col != 'timeframe':
                    df[col] = df[col].astype(float)
            
            # Find global min and max across all price columns for unified normalization
            price_columns = ['open', 'high', 'low', 'close']
            min_price = df[price_columns].min().min()  # Global min across all price columns
            max_price = df[price_columns].max().max()  # Global max across all price columns
            price_range = max_price - min_price
            
            # Normalize all price columns using the same min/max to preserve relationships
            for col in price_columns:
                df[col] = ((df[col] - min_price) / price_range).round(6)
            
            # Volume often needs special handling due to its distribution
            if 'volume' in df.columns:
                min_vol = df['volume'].min()
                max_vol = df['volume'].max()
                if max_vol > min_vol:  # Prevent division by zero
                    df['volume'] = ((df['volume'] - min_vol) / (max_vol - min_vol)).round(6)
            
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
            
            # Group similar indicators for consistent normalization
            indicator_groups = {
                'moving_averages': ['ma20', 'ma50', 'ma100'],
                'bollinger': ['bb_upper', 'bb_middle', 'bb_lower'],
                'stochastic': ['stoch_k', 'stoch_d'],
                'macd': ['macd_main', 'macd_signal', 'macd_hist'],
                'directional': ['adx', 'plus_di', 'minus_di'],
                'ichimoku': ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b']
            }
            
            # Normalize technical indicators by groups
            for group_name, indicators in indicator_groups.items():
                # Filter to only include indicators that exist in the dataframe
                existing_indicators = [ind for ind in indicators if ind in df.columns]
                
                if existing_indicators:
                    # Find min and max across all indicators in this group
                    group_min = df[existing_indicators].min().min()
                    group_max = df[existing_indicators].max().max()
                    
                    if group_max > group_min:  # Prevent division by zero
                        for indicator in existing_indicators:
                            df[indicator] = ((df[indicator] - group_min) / (group_max - group_min)).round(6)
            
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
