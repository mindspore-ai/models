"""Calendar features for time-series CSV input."""

import numpy as np
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def _second(index):
    return index.second / 59.0 - 0.5


def _minute(index):
    return index.minute / 59.0 - 0.5


def _hour(index):
    return index.hour / 23.0 - 0.5


def _weekday(index):
    return index.dayofweek / 6.0 - 0.5


def _day(index):
    return (index.day - 1) / 30.0 - 0.5


def _day_of_year(index):
    return (index.dayofyear - 1) / 365.0 - 0.5


def _month(index):
    return (index.month - 1) / 11.0 - 0.5


def _week(index):
    return (index.isocalendar().week.to_numpy() - 1) / 52.0 - 0.5


def time_features(dates, freq="h"):
    """Return normalised calendar features for a pandas DatetimeIndex."""
    feature_map = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [_month],
        offsets.MonthEnd: [_month],
        offsets.Week: [_day, _week],
        offsets.Day: [_weekday, _day, _day_of_year],
        offsets.BusinessDay: [_weekday, _day, _day_of_year],
        offsets.Hour: [_hour, _weekday, _day, _day_of_year],
        offsets.Minute: [_minute, _hour, _weekday, _day, _day_of_year],
        offsets.Second: [
            _second,
            _minute,
            _hour,
            _weekday,
            _day,
            _day_of_year,
        ],
    }
    offset = to_offset(freq)
    for offset_type, functions in feature_map.items():
        if isinstance(offset, offset_type):
            return np.vstack([function(dates) for function in functions])
    raise ValueError(f"unsupported time frequency: {freq}")
