"""
JMAI Toolbox - A collection of utility functions for RJMAI project

This module provides reusable functions that can be imported and used by other scripts
in the RJMAI project.
"""

import datetime
from typing import Union


# Date and time utilities
def get_day_of_week(date_input: Union[str, datetime.date, datetime.datetime], 
                   format_str: str = "%Y-%m-%d") -> str:
    """
    Calculate the day of the week based on a date input
    
    Args:
        date_input: Date in string format or as datetime/date object
        format_str: Format string for parsing date strings (default: "%Y-%m-%d")
        
    Returns:
        str: The day of the week (e.g., "Monday", "Tuesday", etc.)
        
    Raises:
        ValueError: If the date_input cannot be parsed
    """
    try:
        # Convert string to datetime if needed
        if isinstance(date_input, str):
            date_obj = datetime.datetime.strptime(date_input, format_str).date()
        elif isinstance(date_input, datetime.datetime):
            date_obj = date_input.date()
        elif isinstance(date_input, datetime.date):
            date_obj = date_input
        else:
            raise TypeError("date_input must be a string, datetime, or date object")
        
        # Get day of week as a string
        day_of_week = date_obj.strftime("%A")
        return day_of_week
        
    except Exception as e:
        raise

def get_day_of_week_num(date_input: Union[str, datetime.date, datetime.datetime], 
                       format_str: str = "%Y-%m-%d") -> int:
    """
    Calculate the day of the week as a numerical value (1-7, with 1 being Monday)
    
    Args:
        date_input: Date in string format or as datetime/date object
        format_str: Format string for parsing date strings (default: "%Y-%m-%d")
        
    Returns:
        int: The day of the week as a number (1=Monday, 2=Tuesday, ..., 7=Sunday)
        
    Raises:
        ValueError: If the date_input cannot be parsed
    """
    try:
        # Convert string to datetime if needed
        if isinstance(date_input, str):
            date_obj = datetime.datetime.strptime(date_input, format_str).date()
        elif isinstance(date_input, datetime.datetime):
            date_obj = date_input.date()
        elif isinstance(date_input, datetime.date):
            date_obj = date_input
        else:
            raise TypeError("date_input must be a string, datetime, or date object")
        
        # Get day of week as a number (isoweekday: 1=Monday, 7=Sunday)
        day_of_week_num = date_obj.isoweekday()
        return day_of_week_num
        
    except Exception as e:
        raise
