"""Time-window aggregate features built on the Time column."""

from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_WINDOW_S = 60
LONG_WINDOW_S = 600


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(df["Time"], unit="s")


def _rolling(series: pd.Series, window_seconds: float) -> pd.Series:
    return series.rolling(f"{int(window_seconds)}s")


def rolling_count(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    s = pd.Series(1.0, index=_time_index(df))
    return pd.Series(_rolling(s, window_seconds).sum().to_numpy(), index=df.index)


def rolling_amount_sum(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    s = pd.Series(df["Amount"].to_numpy(), index=_time_index(df))
    return pd.Series(_rolling(s, window_seconds).sum().to_numpy(), index=df.index)


def time_since_last(df: pd.DataFrame) -> pd.Series:
    return df["Time"].diff().fillna(0.0)


def amount_zscore_rolling(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    s = pd.Series(df["Amount"].to_numpy(), index=_time_index(df))
    window = f"{int(window_seconds)}s"
    mean = s.rolling(window).mean()
    std = s.rolling(window).std().replace(0.0, np.nan)
    z = ((s - mean) / std).fillna(0.0)
    return pd.Series(z.to_numpy(), index=df.index)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"rolling_count_{SHORT_WINDOW_S}s": rolling_count(df, SHORT_WINDOW_S).to_numpy(),
            f"rolling_count_{LONG_WINDOW_S}s": rolling_count(df, LONG_WINDOW_S).to_numpy(),
            f"rolling_amount_sum_{LONG_WINDOW_S}s": rolling_amount_sum(df, LONG_WINDOW_S).to_numpy(),
            "time_since_last": time_since_last(df).to_numpy(),
            f"amount_zscore_{LONG_WINDOW_S}s": amount_zscore_rolling(df, LONG_WINDOW_S).to_numpy(),
        },
        index=df.index,
    )
