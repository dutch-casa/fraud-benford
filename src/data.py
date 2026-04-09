"""Load the ULB credit card fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492


@dataclass(frozen=True)
class FraudSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def load_raw() -> pd.DataFrame:
    """Download the dataset. Returns 284807x31, sorted by Time."""
    raise NotImplementedError


def time_ordered_split(df: pd.DataFrame, test_fraction: float = 0.2) -> FraudSplit:
    """Split by Time so test is strictly later than train. No random shuffle."""
    raise NotImplementedError
