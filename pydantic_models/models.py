from datetime import datetime
from typing import NewType

import numpy as np
import pandas as pd
from pydantic import BaseModel


class IBaseModel(BaseModel):
    class Config:
        validate_assignment = True


class IDateKey(IBaseModel):
    date: datetime = None


class IBalanceKeys(IBaseModel):
    capital: float = None
    token: float = None


class IOhlcvKeys(IBaseModel):
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    volume: float = None


class IIndicatorKeys(IBaseModel):
    rsi: float = None
    mfi: float = None
    macd: float = None
    vwap: float = None


State = NewType('State', np.ndarray)
Action = NewType('Action', float)
Reward = NewType('Reward', float)
Done = bool
Info = NewType('Info', dict)


class IStats(IBaseModel):
    episode: int
    transaction_costs: float
    trades: int


class IMemory(IBaseModel):
    date: list[pd.Timestamp]
    action: list[Action]
    asset: list[float]
    reward: list[float]
    save_dir: str
