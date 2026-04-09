"""Load the ULB credit card fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492


@dataclass(frozen=True, slots=True)
class FraudSplit:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FraudSplit3:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATASET_URL)
    if df.shape[0] != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {df.shape[0]}")
    if int(df["Class"].sum()) != EXPECTED_FRAUD:
        raise RuntimeError(f"expected {EXPECTED_FRAUD} fraud rows, got {int(df['Class'].sum())}")
    return df.sort_values("Time").reset_index(drop=True)


def time_ordered_split(df: pd.DataFrame, test_fraction: float = 0.2) -> FraudSplit:
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    if train["Class"].sum() == 0 or test["Class"].sum() == 0:
        raise ValueError("split produces a fraud-less partition")
    return FraudSplit(train=train, test=test)


def time_ordered_three_way_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> FraudSplit3:
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_cut = int(n * (1 - val_fraction - test_fraction))
    val_cut = int(n * (1 - test_fraction))
    train = df.iloc[:train_cut].copy()
    val = df.iloc[train_cut:val_cut].copy()
    test = df.iloc[val_cut:].copy()
    if not all(part["Class"].sum() > 0 for part in (train, val, test)):
        raise ValueError("split produces a fraud-less partition")
    return FraudSplit3(train=train, val=val, test=test)
