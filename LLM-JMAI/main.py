#!/usr/bin/env python
# Simple main script with direct calls to MT5 and GPT-3

import os
import sys
from datetime import datetime

# Import the custom modules
try:
    from gpt3_deep_research import deep_research
    from mt5_download_bars import main as mt5_main
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Run MT5 download
print("Running MT5 data download...")
mt5_main()

# # Run GPT-3 analysis
# print("\nRunning GPT-3 deep research...")
# research_query = "Do an analysis of the EURUSD currency pair and based on that provide advice if one should BUY, SELL, or DO NOTHING."
# result = deep_research(research_query)
# print(f"\nGPT-3 RESULT:\n{result}")
