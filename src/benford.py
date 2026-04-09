"""Benford's Law features and diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BENFORD_PROBS: dict[int, float] = {
    d: math.log10(1 + 1 / d) for d in range(1, 10)
}


def leading_digit(x: float) -> int | None:
    """Leftmost nonzero digit of |x|. Returns None if x is 0, NaN, or inf."""
    raise NotImplementedError


def leading_digit_series(amounts: pd.Series) -> pd.Series:
    """Vectorized leading_digit. NaN where input is 0/NaN/inf."""
    raise NotImplementedError


def empirical_distribution(digits: pd.Series) -> dict[int, float]:
    """Fraction of each digit 1-9 in the series. Missing digits map to 0.0."""
    raise NotImplementedError


def chi_square_distance(empirical: dict[int, float]) -> float:
    """Chi-squared statistic vs. Benford's predicted distribution. Larger = further from Benford."""
    raise NotImplementedError


def benford_features(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    """Return a DataFrame with per-row leading digit + one-hot columns for it."""
    raise NotImplementedError
