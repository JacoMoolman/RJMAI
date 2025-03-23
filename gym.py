import pandas as pd
import matplotlib.pyplot as plt
from get_dataframe import get_dataframe
from jmai_toolbox import display_flat_dataframes, graph_flat_dataframe

START_DATE = '2024-01-08 00:00'
NUM_BARS_TO_PLOT = 100


CURRENCY_PAIRS = [
    'AUDUSD',
    # 'EURUSD',
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
figures = {}

current_date = pd.to_datetime(START_DATE)
running = True

try:
    while running:
        # Advance time by 1 minute
        current_date += pd.Timedelta(minutes=1)
        print(f"\nAdvancing time to: {current_date}")
        
        # Update data for each currency pair
        for currency_pair in CURRENCY_PAIRS:
            # Update the flat dataframe with new time
            flat_df = get_dataframe(currency_pair, current_date, NUM_BARS_TO_PLOT)
            flat_dataframes[currency_pair] = flat_df
            
            # IMPORTANT: Pass existing figures dictionary to reuse the same windows
            graph_flat_dataframe(flat_df, currency_pair, current_date, figures)
        
        # Redraw all figures and handle events
        for fig in figures.values():
            fig.canvas.draw()
        plt.pause(0.5)  # Small pause to allow GUI updates
        
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    plt.close('all')  #DO NOT REMOVE THIS!
