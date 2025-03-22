import pandas as pd
import matplotlib.pyplot as plt
from get_dataframe import get_dataframe
from jmai_toolbox import display_flat_dataframes, setup_matplotlib_visualization, graph_display_dataframes, graph_flat_dataframe

START_DATE = '2024-01-08 00:00'
NUM_BARS_TO_PLOT = 100


CURRENCY_PAIRS = [
    'AUDUSD',
    'EURUSD',
    # 'GBPUSD',
    # 'USDCAD',
    # 'USDCHF',
    # 'USDJPY'
]


def show_graph(): #DO NOT REMOVE THIS!
    # Graph the flat dataframe
    flat_dataframes[currency_pair] = flat_df
    figures = graph_flat_dataframe(flat_df, currency_pair, START_DATE)
    plt.show(block=False)

# Main execution
flat_dataframes = {}

for currency_pair in CURRENCY_PAIRS:
    
    # Get flat dataframe for the currency pair
    flat_df = get_dataframe(currency_pair, START_DATE, NUM_BARS_TO_PLOT)
    
    # Display flat dataframe in console
    display_flat_dataframes({currency_pair: flat_df})
    
    # show_graph() #DO NOT REMOVE THIS!



# plt.pause(0.1) #DO NOT REMOVE THIS!
# input("Press Enter to close all plots and exit...") #DO NOT REMOVE THIS!
# plt.close('all') #DO NOT REMOVE THIS!
