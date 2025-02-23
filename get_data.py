import yfinance as yf
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import pytz

# Suppress yfinance warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

def get_timeframe_interval(timeframe: str) -> str:
    """Convert timeframe to yfinance interval format"""
    timeframe = timeframe.upper()
    if timeframe.startswith('M'):
        return f"{timeframe[1:]}m"  # M5 -> 5m
    elif timeframe.startswith('H'):
        return f"{timeframe[1:]}h"  # H1 -> 1h
    elif timeframe == 'D1':
        return '1d'
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

def get_forex_data(symbol: str, timeframe: str = "M5", num_bars: int = 1, start_date: datetime = None) -> Dict[str, Any]:
    """
    Fetch forex data for a given currency pair using yfinance with specified timeframe
    
    Args:
        symbol (str): The forex pair symbol (e.g., 'EURUSD')
        timeframe (str): Timeframe for the data (e.g., 'M5', 'M15', 'H1', 'D1')
        num_bars (int): Number of previous bars to fetch
        start_date (datetime, optional): Custom start date. If None, uses current time
        
    Returns:
        Dict containing the forex data including multiple bars of price data
    """
    try:
        # Extract currency pairs
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        # Create the Yahoo Finance ticker symbol
        ticker_symbol = f"{base_currency}{quote_currency}=X"
        
        # Convert timeframe to yfinance interval
        interval = get_timeframe_interval(timeframe)
        
        # Calculate the end date (which will be our start_date or current time)
        end_date = start_date if start_date else datetime.now()
        
        # Calculate start period based on num_bars and timeframe
        if timeframe.startswith('M'):
            minutes = int(timeframe[1:]) * (num_bars + 1)  # Add 1 for safe buffer
            start_period = end_date - timedelta(minutes=minutes)
        elif timeframe.startswith('H'):
            hours = int(timeframe[1:]) * (num_bars + 1)
            start_period = end_date - timedelta(hours=hours)
        else:  # Daily
            start_period = end_date - timedelta(days=(num_bars + 1))
            
        # Get data for specified period
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(start=start_period, end=end_date, interval=interval)
        
        if hist.empty:
            return {"error": "No data available for the specified symbol"}
            
        # Get the last n bars
        last_n_bars = hist.tail(num_bars).iloc[::-1]  # Reverse so newest is last
        if len(last_n_bars) < num_bars:
            return {"error": f"Only {len(last_n_bars)} bars available"}
            
        result = {
            "From": base_currency,
            "To": quote_currency,
            "Timeframe": timeframe,
            "StartTime": last_n_bars.index[0].strftime("%Y-%m-%d %H:%M"),
            "EndTime": last_n_bars.index[-1].strftime("%Y-%m-%d %H:%M"),
        }
        
        # Add numbered price data
        for i, (idx, row) in enumerate(last_n_bars.iterrows(), 1):
            result[f"Open{i}"] = row['Open']
            result[f"High{i}"] = row['High']
            result[f"Low{i}"] = row['Low']
            result[f"Close{i}"] = row['Close']
        
        return result
        
    except Exception as e:
        return {"error": str(e)}
