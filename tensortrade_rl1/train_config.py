from os.path import join

import pandas as pd
from ray.train.trainer import tune
from tensortrade.oms.instruments import Instrument

from settings import ASSET_DIR

MODE = 'test'
ENV_NAME = 'TradingEnv'
EXCHANGE_NAME = 'binance'
EXCHANGE_COMMISSION = 0.00115
OBS_WINDOW_SIZE = 14
REWARD_WINDOW_SIZE = 7
MATIC = Instrument('MATIC', 8, 'Matic')

DF_FILE_PATH = join(ASSET_DIR, '01Sep21-00꞉00_to_01Dec21-00꞉00.csv')

DF_TRAIN_START_DATE = pd.Timestamp(2021, 9, 1, 0, 0).tz_localize('utc')
DF_TRAIN_END_DATE = pd.Timestamp(2021, 11, 17, 0, 0).tz_localize('utc')
DF_EVAL_START_DATE = pd.Timestamp(2021, 11, 17, 0, 0).tz_localize('utc')
DF_EVAL_END_DATE = pd.Timestamp(2021, 11, 24, 0, 0).tz_localize('utc')
DF_TEST_START_DATE = pd.Timestamp(2021, 11, 24, 0, 0).tz_localize('utc')
DF_TEST_END_DATE = pd.Timestamp(2021, 12, 1, 0, 0).tz_localize('utc')

DATE_COLUMN = ['date']
OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
OHLCV_WITH_DATE_COLUMNS = DATE_COLUMN + OHLCV_COLUMNS

RAY_CONFIG = {
    'env': ENV_NAME,
    'env_config': {
        'train_sub_mode': 'training',
        'max_allowed_loss': 0.10 if MODE == 'train' else 1.00,
        'obs_window_size': OBS_WINDOW_SIZE,
        'reward_window_size': REWARD_WINDOW_SIZE,
    },
    'log_level': 'DEBUG',
    'framework': 'torch',
    'ignore_worker_failures': True,
    'num_workers': 1,
    'num_envs_per_worker': 1,
    'num_gpus': 1,
    'clip_rewards': True,
    'lr': 0.0005,
    'model': {
        'use_lstm': True,
        'lstm_cell_size': 256,
    },
    'gamma': 0.99,
    'observation_filter': 'MeanStdFilter',
    # 'evaluation_interval': 1,
    # 'evaluation_config': {
    #     'env_config': {
    #         'train_sub_mode': 'evaluation',
    #         'max_allowed_loss': 1.00,
    #     },
    #     'explore': False,
    # },
}

RAY_STOP = {
    # 'episode_reward_mean': 500
    'training_iteration': 100
}

INDICATORS = ['momentum_rsi', 'volume_mfi', 'momentum_tsi', 'momentum_uo', 'momentum_ao', 'trend_macd_diff',
              'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index',
              'trend_cci', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_aroon_up',
              'trend_aroon_down', 'trend_aroon_ind', 'volatility_bbh', 'volatility_bbl', 'volatility_bbm',
              'volatility_bbhi', 'volatility_bbli', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',
              'volatility_dcl', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_vpt',
              'volume_nvi', 'others_dr', 'others_dlr']

# SimpleProfit, BSH, OHLCV_COLUMNS, Window7, 50
# CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-02_21-02-25\PPO_TradingEnv_1844e_00000_0_2021-12-02_21-02-25\checkpoint_000050\checkpoint-50'

# SimpleProfit, BSH, OHLCV_COLUMNS, Window7, 100
CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-03_08-13-29\PPO_TradingEnv_d77c6_00000_0_2021-12-03_08-13-29\checkpoint_000100\checkpoint-100'

# SimpleProfit, BSH, Diff, OHLCV_COLUMNS, Window7
# CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-11-30_20-28-18\PPO_TradingEnv_ff5e1_00000_0_2021-11-30_20-28-18\checkpoint_000050\checkpoint-50'
