from datetime import datetime
from os.path import join

import pandas as pd
from finrl.drl_agents.stablebaselines3.models import MODELS

from environments.gym_env import CryptoTradingEnv, DEFAULT_ENV_PARAMS
from settings import MODEL_SAVE_DIR, ASSET_DIR, MODEL_RESULT_DIR
from utils.model_utils import load_csv_from_df

MODEL = 'ppo'
MODEL_CLASS = MODELS[MODEL]
MODEL_NAME = f'{MODEL}_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
TIMESTEPS = 100000

DF_FILE_PATH = join(ASSET_DIR, '01Jan21-00꞉00_to_01Nov21-00꞉00.csv')
DF_START_DATE = pd.Timestamp(2021, 10, 25, 0, 0).tz_localize('utc')
DF_END_DATE = pd.Timestamp(2021, 11, 1, 0, 0).tz_localize('utc')
df = load_csv_from_df(DF_FILE_PATH, DF_START_DATE, DF_END_DATE)

ENV_PARAMS = {
    **DEFAULT_ENV_PARAMS,
    'df': df,
    'memory_dir': join(MODEL_RESULT_DIR, MODEL_NAME)
}


def main():
    model = MODEL_CLASS.load(MODEL_SAVE_DIR)

    gym_env = CryptoTradingEnv(**ENV_PARAMS)
    vectorized_gym_env, _ = gym_env.get_sb_env()
    state = vectorized_gym_env.reset()

    done = False
    while not done:
        action = model.predict(state)[0]
        state, reward, done, _ = vectorized_gym_env.step(action)


if __name__ == '__main__':
    main()
