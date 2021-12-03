from datetime import datetime
from os.path import join
from pathlib import Path

import pandas as pd
from finrl.drl_agents.stablebaselines3.models import DRLAgent

from environments.gym_env_discrete import CryptoTradingEnv, DEFAULT_ENV_PARAMS
from settings import MODEL_SAVE_DIR, MODEL_RESULT_DIR, MODEL_TENSORBOARD_DIR, ASSET_DIR
from utils.model_utils import load_csv_from_df

MODEL = 'ppo'
MODEL_NAME = f'{MODEL}_{datetime.now().strftime("%d%m%Y_%H%M%S")}'
TIMESTEPS = 200000

DF_FILE_PATH = join(ASSET_DIR, '01Jan21-00꞉00_to_01Nov21-00꞉00.csv')
DF_START_DATE = pd.Timestamp(2021, 10, 25, 0, 0).tz_localize('utc')
DF_END_DATE = pd.Timestamp(2021, 11, 1, 0, 0).tz_localize('utc')
df = load_csv_from_df(DF_FILE_PATH, DF_START_DATE, DF_END_DATE)

ENV_PARAMS = {
    **DEFAULT_ENV_PARAMS,
    'df': df,
    'memory_dir': join(MODEL_RESULT_DIR, MODEL_NAME)
}

MODEL_PARAMS = {
    'n_steps': 2048,
    'learning_rate': 2 ** -15,
    'batch_size': 64,
    'gamma': 0.99
}


def setup() -> None:
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_RESULT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_TENSORBOARD_DIR).mkdir(parents=True, exist_ok=True)


def main():
    setup()
    gym_env = CryptoTradingEnv(**ENV_PARAMS)
    vectorized_gym_env, _ = gym_env.get_sb_env()
    agent = DRLAgent(env=vectorized_gym_env)
    model = agent.get_model(MODEL, model_kwargs=MODEL_PARAMS)
    trained_model = agent.train_model(model=model, tb_log_name=MODEL_NAME, total_timesteps=TIMESTEPS)
    trained_model.save(join(MODEL_SAVE_DIR, MODEL_NAME))


if __name__ == '__main__':
    main()
