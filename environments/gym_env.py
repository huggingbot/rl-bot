from os.path import join
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.spaces import Space, Box
from pandas import DataFrame
from stable_baselines3.common.vec_env import DummyVecEnv

from pydantic_models.models import State, Reward, Done, Info, IDateKey, IBalanceKeys, IOhlcvKeys, IIndicatorKeys, \
    IStats, IMemory, Action

DATE_KEY: list[str] = list(IDateKey().dict().keys())
BALANCE_KEYS: list[str] = list(IBalanceKeys().dict().keys())
OHLCV_KEYS: list[str] = list(IOhlcvKeys().dict().keys())
INDICATOR_KEYS: list[str] = list(IIndicatorKeys().dict().keys())

SHARED_KEYS = OHLCV_KEYS + INDICATOR_KEYS

DATAFRAME_KEYS: list[str] = DATE_KEY + SHARED_KEYS
STATE_KEYS: list[str] = BALANCE_KEYS + SHARED_KEYS

DEFAULT_ENV_PARAMS = {
    'df': None,
    'window_length': 14,
    'initial_capital': 1e4,
    'transaction_cost_pct': 0.00115,
    'reward_scale': 1,
    'print_verbosity': 10,
    'memory_dir': '',
    'action_qty': 500
}


class ValidationError(Exception):
    """raise this when there's a validation error"""
    pass


class CryptoTradingEnv(gym.Env):
    """A crypto trading environments using OpenAI gym"""
    observation_space: Space = None
    action_space: Space = None

    def __init__(self, df: DataFrame, window_length: int, initial_capital: float, transaction_cost_pct: float,
                 action_qty: int, reward_scale: float, print_verbosity: int, memory_dir: str):
        self._df = df
        self._window_length = window_length
        self._initial_capital = initial_capital

        self._current_df_row = window_length - 1
        self._balance = IBalanceKeys(capital=self._initial_capital, token=0)
        CryptoTradingEnv._validate_dataframe_and_state(self._df, self._get_state(), self._window_length)

        self._transaction_cost_pct = transaction_cost_pct
        self._action_qty = action_qty
        self._reward_scale = reward_scale
        self._print_verbosity = print_verbosity

        self.action_space = Box(low=-1, high=1, shape=(1,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self._get_state().shape)

        self._stats = IStats(episode=0, transaction_costs=0, trades=0)
        date = self._get_current_date_from_df()
        self._memory = IMemory(date=[date], action=[], asset=[self._initial_capital], reward=[], save_dir=memory_dir)

    def _get_current_date_from_df(self):
        col_index = self._df.columns.get_loc(DATE_KEY[0])
        return self._df.iloc[self._current_df_row, col_index]

    @staticmethod
    def _validate_dataframe_and_state(df: DataFrame, state: State, window_length: int):
        for idx, col in enumerate(df.columns):
            if col != DATAFRAME_KEYS[idx]:
                raise ValidationError(f'Dataframe columns must exactly match {DATAFRAME_KEYS}')
        if np.unique(DATAFRAME_KEYS).size != len(DATAFRAME_KEYS):
            raise ValidationError(f'Dataframe columns must be unique: {DATAFRAME_KEYS}')
        if np.unique(STATE_KEYS).size != len(STATE_KEYS):
            raise ValidationError(f'State columns must be unique: {STATE_KEYS}')
        if not len(state) == len(STATE_KEYS) + len(SHARED_KEYS) * (window_length - 1):
            raise ValidationError(f'State shape must be: ({window_length}, {len(STATE_KEYS)})')

    def _sell_token(self, action: Action) -> None:
        ohlcv = self._get_state_ohlcv()

        sell_token_count = min(int(-action * self._action_qty), int(self._balance.token))
        sell_token_value = sell_token_count * ohlcv.close
        sell_token_value_after_transaction_cost = sell_token_value * (1 - self._transaction_cost_pct)

        self._balance.capital += sell_token_value_after_transaction_cost
        self._balance.token -= sell_token_count

        self._stats.transaction_costs += sell_token_value * self._transaction_cost_pct
        self._stats.trades += 1 if sell_token_count else 0

    def _buy_token(self, action: Action) -> None:
        ohlcv = self._get_state_ohlcv()

        buy_token_count = min(int(action * self._action_qty),
                              int((self._balance.capital / (1 + self._transaction_cost_pct)) // ohlcv.close))
        buy_token_value = buy_token_count * ohlcv.close
        buy_token_value_after_transaction_cost = buy_token_value * (1 + self._transaction_cost_pct)

        self._balance.capital -= buy_token_value_after_transaction_cost
        self._balance.token += buy_token_count

        self._stats.transaction_costs += buy_token_value_after_transaction_cost * self._transaction_cost_pct
        self._stats.trades += 1 if buy_token_count else 0

    def step(self, action: Action) -> (State, Reward, Done, Info):
        start_asset = self._get_state_asset()

        if action > 0:
            self._buy_token(action)
        elif action < 0:
            self._sell_token(action)

        self._current_df_row += 1

        date = self._get_current_date_from_df()
        self._memory.date.append(date)
        self._memory.action.append(action)

        end_asset = self._get_state_asset()
        self._memory.asset.append(end_asset)

        reward = end_asset - start_asset
        self._memory.reward.append(reward)
        scaled_reward = reward * self._reward_scale

        done = self._current_df_row >= len(self._df.index.unique()) - 1
        if done:
            if self._stats.episode % self._print_verbosity == 0:
                print(f'current index: {self._current_df_row}, episode: {self._stats.episode}')
                print(f'start asset: {self._memory.asset[0]:0.2f}')
                print(f'end asset: {end_asset:0.2f}')
                print(f'total reward: {end_asset - self._memory.asset[0]:0.2f}')
                print(f'total transaction cost: {self._stats.transaction_costs:0.2f}')
                print(f'total trades: {self._stats.trades}')
                print('=================================')
            self._save_action_memory()
            self._save_asset_memory()
            self._save_reward_memory()

        return self._get_state(), scaled_reward, done, {}

    def reset(self) -> State:
        self._current_df_row = self._window_length - 1
        self._balance = IBalanceKeys(capital=self._initial_capital, token=0)

        episode = self._stats.episode + 1
        self._stats = IStats(episode=episode, transaction_costs=0, trades=0)
        date = self._get_current_date_from_df()
        self._memory = IMemory(date=[date], action=[], asset=[self._initial_capital], reward=[],
                               save_dir=self._memory.save_dir)

        return self._get_state()

    def render(self, mode='human', close=False) -> None:
        pass

    def _get_state(self) -> State:
        state = list(self._balance.dict().values())
        for row_index in range(self._window_length):
            state_row = self._build_state_row_from_shared_keys(self._current_df_row - row_index)
            state = np.hstack((state, state_row))
        return State(state)

    def _build_state_row_from_shared_keys(self, row_index: int) -> list[float]:
        state_row: list[float] = []
        for key in STATE_KEYS:
            if key in SHARED_KEYS:
                col_index = self._df.columns.get_loc(key)
                cell: float = self._df.iloc[row_index, col_index]
                state_row.append(cell)
        return state_row

    def _get_state_ohlcv(self) -> IOhlcvKeys:
        return IOhlcvKeys(**self._get_partial_state_row(OHLCV_KEYS))

    def _get_state_asset(self):
        ohlcv = self._get_state_ohlcv()
        return self._balance.capital + (self._balance.token * ohlcv.close)

    def _get_partial_state_row(self, keys: list[str]) -> dict[str, float]:
        partial_row_dict = {}
        for key in keys:
            key_index = STATE_KEYS.index(key)
            cell = self._get_state()[key_index]
            partial_row_dict[key] = cell
        return partial_row_dict

    def _save_action_memory(self) -> pd.DataFrame:
        action_df = pd.DataFrame({'date': self._memory.date[:-1], 'action': self._memory.action})
        if not self._memory.save_dir:
            return action_df
        directory = join(self._memory.save_dir, 'action')
        Path(directory).mkdir(parents=True, exist_ok=True)
        action_df.to_csv(f'{join(directory, str(self._stats.episode))}.csv')
        scalar_date = list(range(len(self._memory.date[:-1])))
        scalar_action = [a[0] for a in self._memory.action]
        self._plot('scatter', directory, scalar_date, scalar_action, s=[1 for _ in scalar_action])
        return action_df

    def _save_asset_memory(self) -> pd.DataFrame:
        asset_df = pd.DataFrame({'date': self._memory.date, 'asset': self._memory.asset})
        asset_df['pct_change'] = asset_df['asset'].pct_change(1)
        if not self._memory.save_dir:
            return asset_df
        directory = join(self._memory.save_dir, 'asset')
        Path(directory).mkdir(parents=True, exist_ok=True)
        asset_df.to_csv(f'{join(directory, str(self._stats.episode))}.csv')
        self._plot('plot', directory, self._memory.asset)
        return asset_df

    def _save_reward_memory(self) -> pd.DataFrame:
        reward_df = pd.DataFrame({'date': self._memory.date[:-1], 'reward': self._memory.reward})
        if not self._memory.save_dir:
            return reward_df
        directory = join(self._memory.save_dir, 'reward')
        Path(directory).mkdir(parents=True, exist_ok=True)
        reward_df.to_csv(f'{join(directory, str(self._stats.episode))}.csv')
        self._plot('plot', directory, self._memory.reward)
        return reward_df

    def _plot(self, plt_type: str, save_dir: str, memory_x: list, memory_y: list = None, **kwargs):
        args = [memory_x]
        if memory_y is not None:
            args = [memory_x, memory_y]
        plt.__dict__[plt_type](*args, **kwargs)
        plt.savefig(f'{join(save_dir, str(self._stats.episode))}.png')
        plt.close()

    def get_sb_env(self):
        env = DummyVecEnv([lambda: self])
        state = env.reset()
        return env, state
