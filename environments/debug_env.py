from os.path import join

import pandas as pd
from stable_baselines3.common.env_checker import check_env

from environments.gym_env_discrete import CryptoTradingEnv, DEFAULT_ENV_PARAMS
from settings import ASSET_DIR
from utils.model_utils import load_csv_from_df

DF_FILE_PATH = join(ASSET_DIR, '01Jan21-00꞉00_to_01Nov21-00꞉00.csv')
DF_START_DATE = pd.Timestamp(2021, 10, 31, 23, 0).tz_localize('utc')
DF_END_DATE = pd.Timestamp(2021, 11, 1, 0, 0).tz_localize('utc')
df = load_csv_from_df(DF_FILE_PATH, DF_START_DATE, DF_END_DATE)

ENV_PARAMS = {**DEFAULT_ENV_PARAMS, 'df': df}

gym_env = CryptoTradingEnv(**ENV_PARAMS)
check_env(gym_env)

obs = gym_env.reset()

n_steps = 100
for _ in range(n_steps):
    action = gym_env.action_space.sample()
    obs, reward, done, info = gym_env.step(action)
