"""Benford's Law features and diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DIGITS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)

BENFORD_PROBS: dict[int, float] = {d: math.log10(1 + 1 / d) for d in DIGITS}


def leading_digit(x: float) -> int | None:
    if x is None or not math.isfinite(x) or x == 0:
        return None
    log = math.log10(abs(x))
    return int(10 ** (log - math.floor(log)))


def leading_digit_series(amounts: pd.Series) -> pd.Series:
    vals = amounts.astype(float)
    mask = (vals > 0) & np.isfinite(vals)
    log_vals = np.log10(vals[mask].to_numpy())
    digits = np.floor(10 ** (log_vals - np.floor(log_vals))).astype(int)
    result = pd.Series(np.nan, index=amounts.index, dtype=float)
    result.loc[mask] = digits
    return result


def empirical_distribution(digits: pd.Series) -> dict[int, float]:
    valid = digits.dropna().astype(int)
    if len(valid) == 0:
        return {d: 0.0 for d in DIGITS}
    counts = valid.value_counts()
    total = len(valid)
    return {d: float(counts.get(d, 0)) / total for d in DIGITS}


def chi_square_distance(empirical: dict[int, float]) -> float:
    return sum(
        (empirical.get(d, 0.0) - BENFORD_PROBS[d]) ** 2 / BENFORD_PROBS[d]
        for d in DIGITS
    )


def benford_features(df: pd.DataFrame, amount_col: str = "Amount") -> pd.DataFrame:
    digits = leading_digit_series(df[amount_col]).fillna(0).astype(int)
    return pd.DataFrame(
        {f"ld_{d}": (digits == d).astype(float) for d in DIGITS},
        index=df.index,
    )
