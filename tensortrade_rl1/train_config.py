from os.path import join

import pandas as pd
from tensortrade.oms.instruments import Instrument

from settings import ASSET_DIR

MODE = 'train'
ENV_NAME = 'TradingEnv'
EXCHANGE_NAME = 'binance'
EXCHANGE_COMMISSION = 0.00115
OBS_WINDOW_SIZE = 14
REWARD_WINDOW_SIZE = 14
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
    'lr_schedule': [
        # 1 iteration == 4k timesteps
        [int(0), 5e-6],
        # [int(4e5), 3e-6],
        # [int(8e5), 1e-6],
        # [int(12e5), 9e-7],
        # [int(16e5), 7e-7],
        # [int(20e5), 5e-7],
        # [int(24e5), 3e-7],
        # [int(28e5), 1e-7],
    ],
    'model': {
        'use_lstm': True,
        'lstm_cell_size': 256,
    },
    'gamma': 0.25,
    'observation_filter': 'MeanStdFilter',
    'evaluation_interval': 1e1,
    'evaluation_config': {
        'env_config': {
            'train_sub_mode': 'evaluation',
            'max_allowed_loss': 1.00,
        },
        'explore': False,
    },
}

RAY_STOP = {
    # 'episode_reward_mean': 500
    'training_iteration': 3e3
}

INDICATORS = ['momentum_rsi', 'volume_mfi', 'momentum_tsi', 'momentum_uo', 'momentum_ao', 'trend_macd_diff',
              'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index',
              'trend_cci', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_aroon_up',
              'trend_aroon_down', 'trend_aroon_ind', 'volatility_bbh', 'volatility_bbl', 'volatility_bbm',
              'volatility_bbhi', 'volatility_bbli', 'volatility_kchi', 'volatility_kcli', 'volatility_dch',
              'volatility_dcl', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_vpt',
              'volume_nvi', 'others_dr', 'others_dlr']

# SimpleProfit, BSH, OHLCV_COLUMNS, Window14, iteration3000, lr5e-6, gamma0.25, pct_change, RSI, MFI, MACD
# CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-18_08-56-24\PPO_TradingEnv_52aaf_00000_0_2021-12-18_08-56-24\checkpoint_001000\checkpoint-1000'

# SimpleProfit, BSH, OHLCV_COLUMNS, Window14, iteration3000, lr5e-6, gamma0.25, pct_change, RSI, MFI
# CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-17_20-34-50\PPO_TradingEnv_ba73f_00000_0_2021-12-17_20-34-51\checkpoint_001000\checkpoint-1000'

# SimpleProfit, BSH, OHLCV_COLUMNS, Window14, iteration3000, lr5e-6, gamma0.25, pct_change, RSI
# CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-22_08-28-07\PPO_TradingEnv_08aab_00000_0_2021-12-22_08-28-07\checkpoint_003000\checkpoint-3000'

# SimpleProfit, BSH, OHLCV_COLUMNS, Window7, iteration3000, lr5e-6, gamma0.25, pct_change, RSI, MFI
# CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-24_21-55-17\PPO_TradingEnv_200b6_00000_0_2021-12-24_21-55-17\checkpoint_003000\checkpoint-3000'

# SimpleProfit, BSH, OHLCV_COLUMNS, Window14, iteration3000, lr5e-6, gamma0.25, INDICATORS
CHECKPOINT_PATH = r'C:\Users\weechien\ray_results\PPO_2021-12-27_08-46-43\PPO_TradingEnv_763cf_00000_0_2021-12-27_08-46-43\checkpoint_003000\checkpoint-3000'
