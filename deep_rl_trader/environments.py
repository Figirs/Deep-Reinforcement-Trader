
import numpy as np
import pandas as pd
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    """A custom Gym environment for stock trading."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, stock_data_path, initial_balance=10000, lookback_window_size=10):
        super().__init__()

        self.stock_data = pd.read_csv(stock_data_path, index_col=0, parse_dates=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = lookback_window_size
        self.lookback_window_size = lookback_window_size

        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space: current stock prices + balance + shares held
        # For simplicity, let's use the 'Close' price and normalize it later
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(lookback_window_size + 2,), dtype=np.float32)

    def _get_observation(self):
        current_prices = self.stock_data["Close"].iloc[self.current_step - self.lookback_window_size : self.current_step].values
        obs = np.append(current_prices, [self.balance, self.shares_held])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.lookback_window_size
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        current_price = self.stock_data["Close"].iloc[self.current_step]
        reward = 0
        done = False

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                reward = 0.1 # Small reward for taking action
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                reward = 0.1 # Small reward for taking action

        # Move to next day
        self.current_step += 1

        # Calculate reward based on portfolio value change (simplified)
        # A more complex reward function would consider profit/loss, risk, etc.
        portfolio_value = self.balance + (self.shares_held * current_price)
        # reward += (portfolio_value - self.initial_balance) / self.initial_balance * 100 # Percentage change

        # End episode if out of data
        if self.current_step >= len(self.stock_data) - 1:
            done = True
            # Final reward based on total portfolio value
            reward = (portfolio_value - self.initial_balance) / self.initial_balance * 100

        observation = self._get_observation()
        info = {"balance": self.balance, "shares_held": self.shares_held, "current_price": current_price}

        return observation, reward, done, info

    def render(self):
        # Implement visualization of trading activity if needed
        pass

    def close(self):
        pass


if __name__ == '__main__':
    # Create a dummy CSV for testing
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = np.random.rand(100) * 100 + 50 # Prices between 50 and 150
    dummy_data = pd.DataFrame({"Close": prices}, index=dates)
    dummy_data.to_csv("dummy_stock_data.csv")

    env = StockTradingEnv(stock_data_path="dummy_stock_data.csv")
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    for _ in range(5):
        action = env.action_space.sample() # Random action
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            break

    env.close()
    print("Environment closed.")

