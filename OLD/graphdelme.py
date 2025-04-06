import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the file path
file_path = 'e:\\Projects\\RJMAI\\delme.txt'

try:
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by lines
    lines = content.strip().split('\n')
    
    # Find the section with price data
    price_data_start = -1
    for i, line in enumerate(lines):
        if 'open' in line and 'high' in line and 'low' in line and 'close' in line:
            price_data_start = i + 1
            break
    
    if price_data_start == -1:
        raise ValueError("Could not find price data in the file")
    
    # Extract close values for plotting
    close_values = []
    for line in lines[price_data_start:]:
        parts = line.strip().split()
        if len(parts) >= 5:  # Ensure we have enough elements
            try:
                close_value = float(parts[4])  # Extract close value (index 4)
                close_values.append(close_value)
            except (ValueError, IndexError):
                continue
    
    # HARDCODED PRICE LEVELS FROM THE USER'S DATA - ONLY THE 20 UNIQUE PRICE LEVELS
    # Format: (level_type, price_level)
    # level_type 0 = support (green), level_type 1 = resistance (red)
    price_levels = [
        # Support levels (0)
        (0, 0.003586),  # 1
        (0, 0.039993),  # 2 
        (0, 0.054719),  # 3
        (0, 0.142068),  # 4
        (0, 0.204317),  # 5
        (0, 0.252761),  # 6
        (0, 0.277443),  # 7
        (0, 0.284137),  # 8
        (0, 0.309488),  # 9
        (0, 0.361780),  # 10

        # Resistance levels (1)
        (1, 0.061925),  # 11
        (1, 0.207439),  # 12
        (1, 0.213612),  # 13
        (1, 0.238323),  # 14
        (1, 0.257962),  # 15
        (1, 0.274239),  # 16
        (1, 0.294881),  # 17
        (1, 0.307502),  # 18
        (1, 0.342887),  # 19
        (1, 0.393489)   # 20
    ]
    
    # Create a plot
    plt.figure(figsize=(14, 7))
    
    # Plot the close price
    plt.plot(range(len(close_values)), close_values, 'b-', linewidth=1.5, label='Close Price')
    
    # Plot all price levels as horizontal lines with numbers
    for i, (level_type, price) in enumerate(price_levels, 1):
        # Choose color based on level type
        color = 'g' if level_type == 0 else 'r'
        
        # Plot the horizontal line
        line = plt.axhline(y=price, color=color, linestyle='-', alpha=0.7, linewidth=1)
        
        # Add text label with line number at the left edge of the chart - smaller font size
        plt.text(5, price + 0.0025, f"#{i}", color=color, fontsize=7, fontweight='bold')
    
    # Add title and labels
    plt.title('Close Price with 20 Numbered Support (Green) and Resistance (Red) Levels')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    
    # Create a custom legend
    plt.legend(handles=[
        plt.Line2D([0], [0], color='b', linewidth=1.5, label='Close Price'),
        plt.Line2D([0], [0], color='g', linewidth=1.5, label='Support Levels'),
        plt.Line2D([0], [0], color='r', linewidth=1.5, label='Resistance Levels')
    ])
    
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
