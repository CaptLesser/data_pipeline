from __future__ import annotations

from typing import Tuple
import pandas as pd


def split_time(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, embargo_minutes: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = df["timestamp"].sort_values().unique()
    if len(ts) < 10:
        n = len(df)
        i1 = int(n * train_ratio)
        i2 = int(n * (train_ratio + val_ratio))
        return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    t1 = ts[int(len(ts) * train_ratio)]
    t2 = ts[int(len(ts) * (train_ratio + val_ratio))]
    emb = pd.Timedelta(minutes=int(embargo_minutes))
    train = df[df["timestamp"] <= (t1 - emb)]
    val = df[(df["timestamp"] > t1) & (df["timestamp"] <= (t2 - emb))]
    test = df[df["timestamp"] > t2]
    return train, val, test


class PurgedKFold:
    """Time-aware K-Fold with embargo to prevent leakage.

    Splits on unique timestamps in chronological order, yields train/val indices
    with an embargo gap around validation folds.
    """

    def __init__(self, n_splits: int = 5, embargo_minutes: int = 60):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.embargo = int(embargo_minutes)

    def split(self, df: pd.DataFrame):
        ts = pd.Series(df["timestamp"]).sort_values().unique()
        n = len(ts)
        fold_sizes = [n // self.n_splits] * self.n_splits
        for i in range(n % self.n_splits):
            fold_sizes[i] += 1
        indices = []
        start = 0
        for size in fold_sizes:
            end = start + size
            indices.append((start, end))
            start = end
        for (s, e) in indices:
            val_start = ts[s]
            val_end = ts[e - 1]
            emb = pd.Timedelta(minutes=self.embargo)
            # Train portion excludes embargo around validation fold
            train_mask = (df["timestamp"] <= (val_start - emb)) | (df["timestamp"] >= (val_end + emb))
            val_mask = (df["timestamp"] >= val_start) & (df["timestamp"] <= val_end)
            train_idx = df[train_mask].index.to_numpy()
            val_idx = df[val_mask].index.to_numpy()
            yield train_idx, val_idx
