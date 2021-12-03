from datetime import timezone, datetime

import pandas as pd


def load_csv_from_df(path: str, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'],
                     date_parser=lambda i: datetime.strptime(i, '%d-%m-%y %H:%M').replace(tzinfo=timezone.utc))
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] < end_date]
    return df
