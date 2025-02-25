import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import os
import sys
from datetime import datetime
import pytz
from io import StringIO

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Loading test_main.py")

import main
from main import display_forex_data, main as main_function


class TestDisplayForexData(unittest.TestCase):
    """Test the display_forex_data function"""
    
    @patch('builtins.print')
    def test_display_forex_data(self, mock_print):
        """Test that forex data is displayed correctly"""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'timestamp_5m_1': ['2025-02-20 10:00'],
            'Open_5m_1': [1.1],
            'High_5m_1': [1.15],
            'Low_5m_1': [1.05],
            'Close_5m_1': [1.12],
            'timestamp_15m_1': ['2025-02-20 09:45'],
            'Open_15m_1': [1.2],
            'High_15m_1': [1.25],
            'Low_15m_1': [1.15],
            'Close_15m_1': [1.22]
        })
        
        # Call the function
        display_forex_data(test_df)
        
        # Check that print was called with the correct arguments
        expected_calls = [
            call('2025-02-20 10:00::Open_5m_1: 1.100000'),
            call('2025-02-20 10:00::High_5m_1: 1.150000'),
            call('2025-02-20 10:00::Low_5m_1: 1.050000'),
            call('2025-02-20 10:00::Close_5m_1: 1.120000'),
            call('2025-02-20 09:45::Open_15m_1: 1.200000'),
            call('2025-02-20 09:45::High_15m_1: 1.250000'),
            call('2025-02-20 09:45::Low_15m_1: 1.150000'),
            call('2025-02-20 09:45::Close_15m_1: 1.220000')
        ]
        
        # Verify all expected calls were made
        for expected in expected_calls:
            mock_print.assert_has_calls([expected], any_order=True)


class TestMainFunction(unittest.TestCase):
    """Test the main function"""
    
    @patch('main.get_forex_data')
    @patch('builtins.print')
    def test_main_with_valid_date(self, mock_print, mock_get_forex_data):
        """Test the main function with a valid date"""
        # Mock the get_forex_data function to return test data
        sample_data = {
            'StartTime': '2025-02-20 09:00',
            'EndTime': '2025-02-20 10:00',
            'Time1': '2025-02-20 10:00',
            'Open1': 1.1,
            'High1': 1.15,
            'Low1': 1.05,
            'Close1': 1.12,
            'Time2': '2025-02-20 09:55',
            'Open2': 1.09,
            'High2': 1.14,
            'Low2': 1.04,
            'Close2': 1.11,
            'Time3': '2025-02-20 09:50',
            'Open3': 1.08,
            'High3': 1.13,
            'Low3': 1.03,
            'Close3': 1.10,
            'Time4': '2025-02-20 09:45',
            'Open4': 1.07,
            'High4': 1.12,
            'Low4': 1.02,
            'Close4': 1.09,
            'Time5': '2025-02-20 09:40',
            'Open5': 1.06,
            'High5': 1.11,
            'Low5': 1.01,
            'Close5': 1.08
        }
        mock_get_forex_data.return_value = sample_data
        
        # Set a custom date in the main module
        original_date = main.CUSTOM_DATE
        main.CUSTOM_DATE = "2025-02-20 10:30:00"
        
        # Execute main with mocked dependencies
        with patch('main.display_forex_data') as mock_display:
            main_function()
            
            # Verify get_forex_data was called for each forex pair and timeframe
            expected_calls = []
            for forex_pair in main.FOREX_PAIRS:
                for timeframe in main.TIMEFRAMES:
                    expected_calls.append(call(forex_pair, timeframe, main.NUM_BARS, 
                                              datetime.strptime(main.CUSTOM_DATE, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)))
            
            mock_get_forex_data.assert_has_calls(expected_calls, any_order=True)
            
            # Verify display_forex_data was called
            self.assertTrue(mock_display.called)
        
        # Restore the original date
        main.CUSTOM_DATE = original_date
    
    @patch('builtins.print')
    def test_main_with_invalid_date(self, mock_print):
        """Test the main function with an invalid date"""
        # Set an invalid date
        original_date = main.CUSTOM_DATE
        main.CUSTOM_DATE = "invalid date format"
        
        # Execute main
        main_function()
        
        # Verify error message was printed
        mock_print.assert_any_call("Invalid date format. Please use 'YYYY-MM-DD HH:MM:SS' format")
        
        # Restore the original date
        main.CUSTOM_DATE = original_date


if __name__ == '__main__':
    unittest.main()
