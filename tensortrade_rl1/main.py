import numpy as np
import pandas as pd
import ray
import ta
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensortrade.env import default
from tensortrade.env.default.actions import BSH, SimpleOrders
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger
from tensortrade.env.default.rewards import PBR, SimpleProfit
from tensortrade.env.generic import Renderer
from tensortrade.feed import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from tensortrade_rl1.train_config import EXCHANGE_COMMISSION, EXCHANGE_NAME, DF_FILE_PATH, DF_TRAIN_START_DATE, \
    DF_TRAIN_END_DATE, OHLCV_WITH_DATE_COLUMNS, MATIC, OHLCV_COLUMNS, ENV_NAME, RAY_CONFIG, RAY_STOP, INDICATORS, \
    DF_TEST_START_DATE, DF_TEST_END_DATE, MODE, DF_EVAL_START_DATE, DF_EVAL_END_DATE
from utils.model_utils import load_csv_from_df


def setup_df(config) -> pd.DataFrame:
    start_date = DF_TEST_START_DATE
    end_date = DF_TEST_END_DATE
    if MODE == 'train' and config['train_sub_mode'] == 'training':
        start_date = DF_TRAIN_START_DATE
        end_date = DF_TRAIN_END_DATE
    elif MODE == 'train' and config['train_sub_mode'] == 'evaluation':
        start_date = DF_EVAL_START_DATE
        end_date = DF_EVAL_END_DATE
    return load_csv_from_df(DF_FILE_PATH, start_date, end_date)


def setup_exchange(df: pd.DataFrame) -> (Stream, Exchange):
    price_stream = Stream.source(list(df['close']), dtype='float').rename(f'{USD.symbol}-{MATIC.symbol}')
    options = ExchangeOptions(commission=EXCHANGE_COMMISSION)
    return price_stream, Exchange(EXCHANGE_NAME, service=execute_order, options=options)(price_stream)


def setup_portfolio(exchange: Exchange) -> (Wallet, Wallet, Portfolio):
    cash = Wallet(exchange, 10000 * USD)
    asset = Wallet(exchange, 0 * MATIC)
    return cash, asset, Portfolio(USD, [cash, asset])


def setup_renderer(df: pd.DataFrame) -> tuple[DataFeed, list[Renderer]]:
    renderer_feed = DataFeed([
        Stream.source(list(df['date'])).rename('date'),
        Stream.source(list(df['open']), dtype='float').rename('open'),
        Stream.source(list(df['high']), dtype='float').rename('high'),
        Stream.source(list(df['low']), dtype='float').rename('low'),
        Stream.source(list(df['close']), dtype='float').rename('close'),
        Stream.source(list(df['volume']), dtype='float').rename('volume'),
    ])
    chart_renderer = PlotlyTradingChart(
        display=True,  # show the chart on screen (default)
        height=800,  # affects both displayed and saved file height. None for 100% height.
        save_format="html",  # save the chart to an HTML file
        auto_open_html=True,  # open the saved HTML chart in a new browser tab
    )

    file_logger = FileLogger(
        filename="example.log",  # omit or None for automatic file name
        path="training_logs"  # create a new directory if doesn't exist, None for no directory
    )
    return renderer_feed, [chart_renderer, file_logger]


def setup_feed(df: pd.DataFrame) -> DataFeed:
    ta.add_all_ta_features(df, **{c: c for c in OHLCV_COLUMNS})

    # past_df = self.stationary_df['Close'][:
    #                                       self.current_step + self.min_observations]
    # forecast_model = SARIMAX(df['close'].values)
    # model_fit = forecast_model.fit(method='bfgs', disp=False)
    # forecast = model_fit.get_forecast(steps=self.n_forecasts, alpha=self.arma_alpha)

    df = df[OHLCV_COLUMNS]
    # df = df.pct_change()
    # df = np.log(df)
    # df = df.diff()
    # df = df.diff()
    with NameSpace(EXCHANGE_NAME):
        streams = [Stream.source(list(df[c]), dtype='float').rename(c) for c in df.columns]
    return DataFeed(streams)


def create_env(config):
    df = setup_df(config)
    price_stream, exchange = setup_exchange(df)
    cash, asset, portfolio = setup_portfolio(exchange)

    reward_scheme = SimpleProfit(window_size=config['reward_window_size'])
    action_scheme = BSH(cash=cash, asset=asset)

    renderer_feed, renderer = setup_renderer(df)
    feed = setup_feed(df)
    feed.compile()

    return default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        window_size=config['obs_window_size'],
        renderer_feed=renderer_feed,
        renderer=renderer,
        max_allowed_loss=config['max_allowed_loss']
    )


def train():
    analysis = tune.run(PPOTrainer, config=RAY_CONFIG, stop=RAY_STOP, checkpoint_at_end=True)

    # Get checkpoint
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial('episode_reward_mean', mode='max'),
        metric='episode_reward_mean'
    )
    checkpoint_path = checkpoints[0][0]
    print('checkpoint_path', checkpoint_path)
    ray.shutdown()


def main():
    assert MODE == 'train'
    ray.init()
    register_env(ENV_NAME, create_env)
    train()


if __name__ == '__main__':
    main()
