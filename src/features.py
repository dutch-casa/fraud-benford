"""Time-window aggregate features built on the Time column."""

from __future__ import annotations

import pandas as pd


def rolling_count(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    """Count of transactions within the last `window_seconds` for each row."""
    raise NotImplementedError


def rolling_amount_sum(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    """Sum of Amount within the last `window_seconds` for each row."""
    raise NotImplementedError


def time_since_last(df: pd.DataFrame) -> pd.Series:
    """Seconds since the previous transaction. NaN for the first row."""
    raise NotImplementedError


def amount_zscore_rolling(df: pd.DataFrame, window_seconds: float) -> pd.Series:
    """(Amount - rolling mean) / rolling std over the last `window_seconds`."""
    raise NotImplementedError


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compose all engineered features into one DataFrame aligned to df's index."""
    raise NotImplementedError
