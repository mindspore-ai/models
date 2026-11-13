"""Chronological multivariate dataset and MindSpore input pipeline."""

# Dataset objects intentionally retain split metadata used to configure the
# model after the CSV schema is known.
# pylint: disable=too-many-instance-attributes

import os

import mindspore.dataset as ds
import numpy as np
import pandas as pd

from .time_features import time_features


COLUMN_NAMES = ["batch_x", "batch_y", "batch_x_mark", "batch_y_mark"]


class StandardScaler:
    """Feature-wise standardisation fitted only on the training partition."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, values):
        """Fit feature statistics and return this scaler."""
        self.mean_ = np.mean(values, axis=0, keepdims=True)
        self.scale_ = np.std(values, axis=0, keepdims=True)
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        return self

    def transform(self, values):
        """Standardise an array using fitted feature statistics."""
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values):
        """Restore values to their original feature scale."""
        return values * self.scale_ + self.mean_


class PATDataset:
    """Return full multivariate windows for Multi-Patch Attention."""

    def __init__(self, config, split):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.config = config
        self.split = split
        self.scaler = StandardScaler()
        csv_path = os.path.join(config.root_path, config.data_path)
        self._read_data(csv_path)

    def _read_data(self, csv_path):
        frame = pd.read_csv(csv_path)
        if "date" not in frame.columns:
            raise ValueError(f"CSV must contain a date column: {csv_path}")
        if self.config.target not in frame.columns:
            raise ValueError(
                f"target {self.config.target!r} not found in {csv_path}"
            )

        feature_columns = [
            column
            for column in frame.columns
            if column not in {"date", self.config.target}
        ]
        frame = frame[["date", *feature_columns, self.config.target]]
        if self.config.features == "S":
            self.variable_names = [self.config.target]
        else:
            self.variable_names = [*feature_columns, self.config.target]
        self.target_index = self.variable_names.index(self.config.target)

        values = frame[self.variable_names].apply(
            pd.to_numeric, errors="coerce"
        )
        if values.isna().any().any():
            values = values.interpolate(limit_direction="both").ffill().bfill()
        values = values.to_numpy(dtype=np.float32)

        total = len(frame)
        train_end = int(total * 0.70)
        validation_end = train_end + int(total * 0.20)
        boundaries = {
            "train": (0, train_end),
            "val": (train_end - self.config.seq_len, validation_end),
            "test": (validation_end - self.config.seq_len, total),
        }
        start, end = boundaries[self.split]
        start = max(0, start)
        if self.split == "train" and self.config.percent < 100:
            available = end - start - self.config.seq_len
            end = (
                start
                + self.config.seq_len
                + available * self.config.percent // 100
            )

        self.scaler.fit(values[:train_end])
        values = self.scaler.transform(values).astype(np.float32)
        dates = self._parse_dates(frame["date"])
        stamps = time_features(dates, self.config.freq).transpose(1, 0)
        self.data_x = values[start:end]
        self.data_y = values[start:end]
        self.data_stamp = stamps[start:end].astype(np.float32)
        self.length = (
            len(self.data_x)
            - self.config.seq_len
            - self.config.pred_len
            + 1
        )
        if self.length <= 0:
            raise ValueError(
                f"{self.split} split is too short for seq_len="
                f"{self.config.seq_len} and pred_len={self.config.pred_len}"
            )

    @staticmethod
    def _parse_dates(date_column):
        cleaned = date_column.astype(str).str.strip().str.strip("'\"")
        try:
            dates = pd.to_datetime(cleaned, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            dates = pd.to_datetime(cleaned, errors="coerce")
        if dates.isna().any():
            reparsed = cleaned[dates.isna()].map(
                lambda value: pd.to_datetime(value, errors="coerce")
            )
            dates.loc[dates.isna()] = reparsed
        if dates.isna().any():
            raise ValueError("date column contains unparseable values")
        return pd.DatetimeIndex(dates)

    def __getitem__(self, index):
        sequence_end = index + self.config.seq_len
        target_start = sequence_end - self.config.label_len
        target_end = sequence_end + self.config.pred_len
        return (
            self.data_x[index:sequence_end].astype(np.float32),
            self.data_y[target_start:target_end].astype(np.float32),
            self.data_stamp[index:sequence_end].astype(np.float32),
            self.data_stamp[target_start:target_end].astype(np.float32),
        )

    def __len__(self):
        return self.length


def create_dataset(config, split):
    """Create a source dataset and its batched MindSpore pipeline."""
    source = PATDataset(config, split)
    pipeline = ds.GeneratorDataset(
        source=source,
        column_names=COLUMN_NAMES,
        shuffle=split == "train",
        num_parallel_workers=max(1, config.num_workers),
        python_multiprocessing=False,
    )
    pipeline = pipeline.batch(
        batch_size=config.batch_size,
        drop_remainder=split == "train",
    )
    return source, pipeline
