import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import os
import sys
from datetime import datetime
import pytz
from io import StringIO

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from get_data import get_timeframe_interval, get_forex_data, ForexDataCache


class TestGetTimeframeInterval(unittest.TestCase):
    """Test the get_timeframe_interval function"""
    
    def test_timeframe_conversion(self):
        """Test that timeframes are correctly converted to yfinance format"""
        self.assertEqual(get_timeframe_interval('5m'), '5m')
        self.assertEqual(get_timeframe_interval('15m'), '15m')
        self.assertEqual(get_timeframe_interval('30m'), '30m')
        self.assertEqual(get_timeframe_interval('1h'), '60m')
        self.assertEqual(get_timeframe_interval('1d'), '1d')
        
    def test_case_insensitivity(self):
        """Test that the function handles case insensitivity"""
        self.assertEqual(get_timeframe_interval('5M'), '5m')
        self.assertEqual(get_timeframe_interval('1H'), '60m')
        self.assertEqual(get_timeframe_interval('1D'), '1d')


class TestForexDataCache(unittest.TestCase):
    """Test the ForexDataCache class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Reset the singleton instance before each test
        ForexDataCache._instance = None
        self.patcher = patch('os.path.exists', return_value=True)
        self.mock_exists = self.patcher.start()
        
        # Mock os.listdir to return no files
        self.listdir_patcher = patch('os.listdir', return_value=[])
        self.mock_listdir = self.listdir_patcher.start()
        
        # Mock os.makedirs
        self.makedirs_patcher = patch('os.makedirs')
        self.mock_makedirs = self.makedirs_patcher.start()
        
    def tearDown(self):
        """Tear down test fixtures"""
        self.patcher.stop()
        self.listdir_patcher.stop()
        self.makedirs_patcher.stop()
    
    def test_singleton_pattern(self):
        """Test that ForexDataCache is a singleton"""
        cache1 = ForexDataCache()
        cache2 = ForexDataCache()
        self.assertIs(cache1, cache2)
    
    @patch('pandas.read_csv')
    def test_load_from_csv(self, mock_read_csv):
        """Test loading data from CSV"""
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'Open': [1.1, 1.2],
            'High': [1.15, 1.25],
            'Low': [1.05, 1.15],
            'Close': [1.12, 1.22]
        })
        mock_df.index = pd.DatetimeIndex(['2025-02-20 10:00', '2025-02-20 10:05'])
        mock_read_csv.return_value = pd.DataFrame({
            'Date': ['2025.02.20 10:00', '2025.02.20 10:05'],
            'Open': [1.1, 1.2],
            'High': [1.15, 1.25],
            'Low': [1.05, 1.15],
            'Close': [1.12, 1.22]
        })
        
        cache = ForexDataCache()
        result = cache.load_from_csv('EURUSD', '5m')
        
        self.assertTrue(result)
        self.assertTrue('EURUSD_5m' in cache.data_cache)
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_to_csv(self, mock_to_csv):
        """Test saving data to CSV"""
        cache = ForexDataCache()
        
        # Create test data
        test_df = pd.DataFrame({
            'Open': [1.1, 1.2],
            'High': [1.15, 1.25],
            'Low': [1.05, 1.15],
            'Close': [1.12, 1.22]
        })
        test_df.index = pd.DatetimeIndex(['2025-02-20 10:00', '2025-02-20 10:05']).tz_localize('UTC')
        
        # Add to cache
        cache.data_cache['EURUSD_5m'] = test_df
        
        # Test save_to_csv
        cache.save_to_csv('EURUSD', '5m')
        
        # Verify to_csv was called
        mock_to_csv.assert_called_once()
    
    @patch.object(ForexDataCache, 'get_data')
    def test_get_forex_data(self, mock_get_data):
        """Test the get_forex_data function"""
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'Open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'High': [1.15, 1.25, 1.35, 1.45, 1.55],
            'Low': [1.05, 1.15, 1.25, 1.35, 1.45],
            'Close': [1.12, 1.22, 1.32, 1.42, 1.52]
        })
        mock_df.index = pd.DatetimeIndex([
            '2025-02-20 09:40', 
            '2025-02-20 09:45', 
            '2025-02-20 09:50',
            '2025-02-20 09:55', 
            '2025-02-20 10:00'
        ]).tz_localize('UTC')
        
        mock_get_data.return_value = mock_df
        
        # Test with a fixed start date
        start_date = datetime(2025, 2, 20, 10, 0, 0, tzinfo=pytz.UTC)
        result = get_forex_data('EURUSD', '5m', 5, start_date)
        
        # Verify the result structure
        self.assertEqual(result['From'], 'EUR')
        self.assertEqual(result['To'], 'USD')
        self.assertEqual(result['Timeframe'], '5m')
        
        # Check that we have 5 bars of data
        for i in range(1, 6):
            self.assertIn(f'Time{i}', result)
            self.assertIn(f'Open{i}', result)
            self.assertIn(f'High{i}', result)
            self.assertIn(f'Low{i}', result)
            self.assertIn(f'Close{i}', result)
        
        # Test error handling
        mock_get_data.return_value = pd.DataFrame()  # Empty DataFrame
        result = get_forex_data('EURUSD', '5m', 5, start_date)
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
