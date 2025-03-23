import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# First set environment variables before importing torch 
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '0'

# Now import torch
import torch
torch.cuda.empty_cache()

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Basic imports
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Tuple
from get_dataframe import get_dataframe

# Variables from gym.py
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

# Constants for the trading environment
MAX_POSITION = 1.0
TRANSACTION_FEE = 0.0001  # 0.01% per transaction

class TradingEnv(gym.Env):
    def __init__(self, currency_pair: str, start_date=START_DATE, visualizer=None):
        super(TradingEnv, self).__init__()
        
        self.currency_pair = currency_pair
        self.current_date = pd.to_datetime(start_date)
        self.initial_date = self.current_date
        
        # Initial data - using the flat_dataframes dictionary pattern
        self.flat_dataframes = {}
        flat_df = get_dataframe(self.currency_pair, self.current_date, NUM_BARS_TO_PLOT)
        self.flat_dataframes[self.currency_pair] = flat_df
        
        # Environment setup
        # State: flattened dataframe (all available features)
        num_features = len(self.flat_dataframes[self.currency_pair].columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
        # Action: 1D continuous space [-1, 1], will be mapped to discrete actions
        # SAC requires a continuous action space (Box)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Trading variables
        self.position = 0.0  # 0=no position, 1=long, -1=short
        self.entry_price = 0
        self.balance = 10000.0  # Initial balance
        self.initial_balance = self.balance
        self.step_counter = 0
        self.total_pnl = 0.0
        
        # Track the trading history
        self.trades = []
        
        print(f"TradingEnv initialized for {currency_pair} with {num_features} features")
        
        self.visualizer = visualizer
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial date
        self.current_date = self.initial_date
        flat_df = get_dataframe(self.currency_pair, self.current_date, NUM_BARS_TO_PLOT)
        self.flat_dataframes[self.currency_pair] = flat_df
        
        # Reset trading variables
        self.position = 0
        self.entry_price = 0
        self.balance = self.initial_balance
        self.trades = []
        self.step_counter = 0
        self.total_pnl = 0.0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Return the flattened dataframe as observation
        return np.array(self.flat_dataframes[self.currency_pair].values.flatten(), dtype=np.float32)
    
    def _get_current_price(self):
        # Get the most recent close price
        # Using timeframe M1 (1 minute)
        if 'M1_close_0' in self.flat_dataframes[self.currency_pair].columns:
            return self.flat_dataframes[self.currency_pair]['M1_close_0'].values[0]
        
        # Fallback to the first 'close' column if M1_close_0 is not available
        for col in self.flat_dataframes[self.currency_pair].columns:
            if 'close_0' in col:
                return self.flat_dataframes[self.currency_pair][col].values[0]
        
        # Default return if no close price is found
        print("Warning: No close price found in dataframe")
        return 1.0  # Default value
    
    def _calculate_reward(self, action):
        current_price = self._get_current_price()
        reward = 0
        
        # Calculate PnL if we have a position
        if self.position != 0:
            price_diff = current_price - self.entry_price
            if self.position == 1:  # Long position
                unrealized_pnl = price_diff
            else:  # Short position
                unrealized_pnl = -price_diff
            
            # Assign reward based on this minute's PnL
            reward = unrealized_pnl
        
        return reward
    
    def _execute_trade(self, action):
        current_price = self._get_current_price()
        
        # Execute the trade based on the action
        if action == 0:  # Do nothing
            pass
        
        elif action == 1:  # Buy
            if self.position == 0:  # Only buy if no position
                self.position = 1
                self.entry_price = current_price
                # Apply transaction fee
                self.balance -= self.balance * TRANSACTION_FEE
                # Record trade
                self.trades.append({
                    'time': self.current_date,
                    'action': 'buy',
                    'price': current_price,
                    'balance': self.balance
                })
            
        elif action == 2:  # Sell
            if self.position == 0:  # Only sell if no position
                self.position = -1
                self.entry_price = current_price
                # Apply transaction fee
                self.balance -= self.balance * TRANSACTION_FEE
                # Record trade
                self.trades.append({
                    'time': self.current_date,
                    'action': 'sell',
                    'price': current_price,
                    'balance': self.balance
                })
            
        elif action == 3:  # Close position
            if self.position != 0:  # Only close if we have a position
                # Calculate profit/loss
                price_diff = current_price - self.entry_price
                if self.position == 1:  # Long position
                    profit = price_diff * MAX_POSITION 
                else:  # Short position
                    profit = -price_diff * MAX_POSITION
                
                # Update balance
                self.balance += profit
                # Apply transaction fee
                self.balance -= self.balance * TRANSACTION_FEE
                
                # Record trade
                self.trades.append({
                    'time': self.current_date,
                    'action': 'close',
                    'price': current_price,
                    'profit': profit,
                    'balance': self.balance
                })
                
                # Reset position
                self.position = 0
                self.entry_price = 0
    
    def _map_continuous_to_discrete(self, continuous_action):
        """
        Map continuous action from SAC (-1 to 1) to discrete trading action:
        -1.0 to -0.5: Sell (2)
        -0.5 to 0.0: Hold (0) 
        0.0 to 0.5: Buy (1)
        0.5 to 1.0: Close (3)
        """
        action = continuous_action[0]  # Get the scalar value from the array
        
        if action < -0.5:
            return 2  # Sell
        elif action < 0.0:
            return 0  # Do nothing/Hold
        elif action < 0.5:
            return 1  # Buy
        else:
            return 3  # Close
    
    def step(self, action):
        self.step_counter += 1
        
        # Print progress every 5 steps
        if self.step_counter % 5 == 0:
            print(f"{self.currency_pair} | Step {self.step_counter} | Position: {self.position} | Balance: {self.balance:.2f}")
        
        # Map continuous action to discrete
        discrete_action = self._map_continuous_to_discrete(action)
        
        # Execute the trade
        self._execute_trade(discrete_action)
        
        # Calculate reward
        reward = self._calculate_reward(discrete_action)
        
        # Advance time by 1 minute
        self.current_date += pd.Timedelta(minutes=1)
        
        # Get new data using the flat_dataframes pattern
        flat_df = get_dataframe(self.currency_pair, self.current_date, NUM_BARS_TO_PLOT)
        self.flat_dataframes[self.currency_pair] = flat_df
        
        # Check if we've reached the end of data
        done = False
        if self.flat_dataframes[self.currency_pair].empty:
            done = True
            # Close any open positions
            if self.position != 0:
                action = 3  # Close position
                self._execute_trade(action)
        
        # Debug output
        if self.step_counter % 10 == 0:
            print(f"{self.currency_pair} | Step {self.step_counter}: Action={discrete_action}, Reward={reward:.4f}, PnL={self.balance:.4f}")
            
        # Update visualization if available
        if self.visualizer is not None:
            self.visualizer.update(self.currency_pair, self.step_counter, self.balance, self.position, discrete_action)
        
        # Return the next observation
        return self._get_observation(), reward, done, False, {}

class ProgressCallback(BaseCallback):
    """
    Callback for tracking training progress.
    """
    def __init__(self, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.ep_rewards = []
        self.ep_lengths = []
    
    def _on_step(self) -> bool:
        # For vectorized environments, dones is an array
        dones = self.locals.get("dones", False)
        
        # Check if any environment is done
        if isinstance(dones, np.ndarray) and dones.any():
            # For vectorized environments, rewards and lengths are arrays
            rewards = self.locals.get("rewards", [0])
            episode_lengths = self.locals.get("episode_lengths", [0])
            
            # Get indices of done environments
            done_indices = np.where(dones)[0]
            
            for idx in done_indices:
                # Log rewards and lengths for done environments
                self.ep_rewards.append(rewards[idx])
                self.ep_lengths.append(episode_lengths[idx])
                
                # Print every single episode for debugging
                print(f"Episode: {len(self.ep_rewards)}, Reward: {rewards[idx]:.2f}, Length: {episode_lengths[idx]}")
            
            # More detailed every 10 episodes
            if len(self.ep_rewards) % 10 == 0:
                avg_reward = sum(self.ep_rewards[-10:]) / min(10, len(self.ep_rewards))
                avg_length = sum(self.ep_lengths[-10:]) / min(10, len(self.ep_lengths))
                print(f"=== SUMMARY === Episode: {len(self.ep_rewards)}, Avg. Reward: {avg_reward:.2f}, Avg. Length: {avg_length:.2f}")
        
        return True

class TrainingVisualizer:
    def __init__(self, currency_pairs):
        self.currency_pairs = currency_pairs
        self.balances = {pair: [] for pair in currency_pairs}
        self.positions = {pair: [] for pair in currency_pairs}
        self.steps = {pair: [] for pair in currency_pairs}
        self.actions = {pair: [] for pair in currency_pairs}
        self.figs = {}
        
        # Create a figure for each currency pair
        for pair in currency_pairs:
            self.figs[pair] = plt.figure(figsize=(10, 6))
            self.figs[pair].suptitle(f"{pair} Trading Performance", fontsize=14)
            plt.ion()  # Enable interactive mode
    
    def update(self, currency_pair, step, balance, position, action):
        # Update data
        self.steps[currency_pair].append(step)
        self.balances[currency_pair].append(balance)
        self.positions[currency_pair].append(position)
        self.actions[currency_pair].append(action)
        
        # Clear figure
        self.figs[currency_pair].clear()
        
        # Create subplots
        ax1 = self.figs[currency_pair].add_subplot(211)  # Balance plot
        ax2 = self.figs[currency_pair].add_subplot(212)  # Position plot
        
        # Plot balance
        ax1.plot(self.steps[currency_pair], self.balances[currency_pair], 'b-')
        ax1.set_ylabel('Balance')
        ax1.set_title(f'Account Balance - Current: {balance:.2f}')
        
        # Plot position and actions
        ax2.plot(self.steps[currency_pair], self.positions[currency_pair], 'g-')
        
        # Color the action points
        action_colors = {0: 'gray', 1: 'green', 2: 'red', 3: 'blue'}  # Hold, Buy, Sell, Close
        for i, a in enumerate(self.actions[currency_pair]):
            # Show all actions including holds (0)
            ax2.scatter(self.steps[currency_pair][i], self.positions[currency_pair][i], 
                       c=action_colors.get(a, 'black'), marker='o', alpha=0.7 if a == 0 else 1.0)
        
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Position')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Short', 'None', 'Long'])
        ax2.set_title('Position and Actions')
        
        # Add legend for actions
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Buy', markersize=8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Sell', markersize=8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Close', markersize=8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Hold', markersize=8)
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        # Update the figure
        self.figs[currency_pair].tight_layout()
        self.figs[currency_pair].canvas.draw()
        self.figs[currency_pair].canvas.flush_events()

def make_env(currency_pair, rank, seed=0, visualizer=None):
    """
    Create a gym environment for trading with the specified currency pair.
    Utility function for multiprocessed env.
    
    :param currency_pair: Currency pair to trade
    :param rank: Instance number for seed
    :param seed: Random seed
    :param visualizer: Optional visualizer for tracking progress
    :return: Environment creator function
    """
    def _init():
        env = TradingEnv(currency_pair, visualizer=visualizer)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init

def train_model(currency_pairs=CURRENCY_PAIRS, total_timesteps=1000000):
    """
    Train SAC models for multiple currency pairs in parallel
    """
    models = {}
    
    # Create visualizer
    visualizer = TrainingVisualizer(currency_pairs)
    
    # Creating environment for each currency pair
    envs = []
    
    # Create parallel environments
    print("Creating parallel environments for currency pairs:", currency_pairs)
    envs = DummyVecEnv([make_env(pair, i, visualizer=visualizer) for i, pair in enumerate(currency_pairs)])
    print(f"Created {len(currency_pairs)} parallel environments")
    
    # Set up policy kwargs
    policy_kwargs = dict(
        net_arch=[256, 256, 256, 256, 256]
    )
    
    # Create single SAC model for all environments
    print("Creating SAC model for parallel training...")
    model = SAC(
        "MlpPolicy", 
        envs, 
        learning_rate=0.0003,
        buffer_size=50000,
        batch_size=128,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device
    )
    
    # Setup callback for progress tracking
    callback = ProgressCallback()
    
    # Train the model - using a large number to let environment determine when to stop
    print(f"Starting parallel training for all currency pairs until end of data...")
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)
    
    # Save the model
    model_path = "models/sac_multi_pair"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def test_model(model, currency_pair, start_date=None):
    """
    Test a trained model on historical data
    """
    if start_date is None:
        # Use a date different from training for testing
        start_date = pd.to_datetime(START_DATE) + pd.Timedelta(days=30)
        start_date = start_date.strftime('%Y-%m-%d %H:%M')
    
    # Create test environment
    env = TradingEnv(currency_pair, start_date=start_date)
    
    # Reset the environment
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print(f"\nTesting model for {currency_pair} starting at {start_date}")
    
    while not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, done, _, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Print progress every 10 steps
        if steps % 10 == 0:
            print(f"Step: {steps}, Total Reward: {total_reward:.4f}, Balance: {env.balance:.2f}")
    
    # Print final results
    print(f"\nTest complete for {currency_pair}")
    print(f"Total steps: {steps}")
    print(f"Final reward: {total_reward:.4f}")
    print(f"Final balance: {env.balance:.2f}")
    print(f"Total trades: {len(env.trades)}")
    
    return env.trades, env.balance

if __name__ == "__main__":
    # Train models
    print("Starting training of SAC models for currency pairs:", CURRENCY_PAIRS)
    model = train_model(CURRENCY_PAIRS)
