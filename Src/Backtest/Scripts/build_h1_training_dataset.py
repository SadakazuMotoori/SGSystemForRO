import argparse
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.MTSystem.MTManager import MTManager_Initialize
from Framework.Utility.Utility import (
    EnsureParentDirectory,
    FormatJSTDateTime,
    JST,
    ParseJSTDateTime,
    UTC,
)


H1_FEATURE_COLUMNS = [
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "body",
    "range",
    "upper_wick",
    "lower_wick",
    "body_ratio",
    "close_return_1",
    "return_std_5",
    "return_std_10",
    "ma_gap_sma20",
    "ma_gap_sma50",
    "range_mean_5",
    "range_mean_10",
    "range_ratio_5",
    "range_ratio_10",
    "close_position_5",
    "close_position_10",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]
H1_WINDOW_COLUMNS = ["open", "high", "low", "close", *H1_FEATURE_COLUMNS]
MIN_HISTORY_BARS_FOR_FEATURES = 50
MIN_FETCH_LOOKBACK_DAYS = 10
FETCH_LOOKBACK_BUFFER_DAYS = 7
PROGRESS_INTERVAL = 100
DEFAULT_EXTRA_TARGET_HOURS = "6,7,8"
DEFAULT_AUTO_LOOKBACK_DAYS = 365


def parse_args():
    if len(sys.argv) == 1:
        _default_future_hours = 2
        _default_target_hours = sorted(set([_default_future_hours, *parse_hour_list(DEFAULT_EXTRA_TARGET_HOURS)]))
        _default_start_jst, _default_end_jst = build_default_dataset_range(
            _default_target_hours,
            DEFAULT_AUTO_LOOKBACK_DAYS,
        )
        return SimpleNamespace(
            symbol="USDJPY",
            start=FormatJSTDateTime(_default_start_jst),
            end=FormatJSTDateTime(_default_end_jst),
            output="Src/Backtest/Output/datasets/h1_training_dataset.csv",
            future_hours=_default_future_hours,
            extra_target_hours=DEFAULT_EXTRA_TARGET_HOURS,
            sequence_length=32,
            lookback_days=DEFAULT_AUTO_LOOKBACK_DAYS,
            h1_source_csv="",
            verbose=True,
        )

    _parser = argparse.ArgumentParser(description="Build H1 training dataset for SGSystemForRO")
    _parser.add_argument("--symbol", required=True, help="Symbol. Example: USDJPY")
    _parser.add_argument("--start", help="Start datetime in JST. Example: 2026-03-01 00:00:00")
    _parser.add_argument("--end", help="End datetime in JST. Example: 2026-03-10 23:59:59")
    _parser.add_argument("--output", required=True, help="Output CSV path")
    _parser.add_argument(
        "--future-hours",
        type=int,
        default=2,
        help="Primary target horizon in hours",
    )
    _parser.add_argument(
        "--extra-target-hours",
        default=DEFAULT_EXTRA_TARGET_HOURS,
        help="Additional target horizons in hours. Example: 6,7,8",
    )
    _parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Number of H1 bars per record",
    )
    _parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_AUTO_LOOKBACK_DAYS,
        help="Used only when start/end are omitted. The builder will create a latest-available range ending at the latest labelable H1 bar.",
    )
    _parser.add_argument(
        "--h1-source-csv",
        default="",
        help="Optional H1 OHLC CSV path used instead of MT5. Required columns: timestamp/time + open/high/low/close",
    )
    _parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs",
    )

    return _parser.parse_args()


def parse_hour_list(_hour_text):
    if _hour_text is None:
        return []

    _text = str(_hour_text).strip()
    if _text == "":
        return []

    _hour_list = []
    for _token in _text.split(","):
        _token = _token.strip()
        if _token == "":
            continue

        try:
            _hour_value = int(_token)
        except Exception as _error:
            raise RuntimeError(f"Failed to parse hour list token: {_token}") from _error

        if _hour_value <= 0:
            raise RuntimeError("All target horizons must be 1 or greater")

        _hour_list.append(_hour_value)

    return sorted(set(_hour_list))


def floor_to_hour(_dt):
    return _dt.replace(minute=0, second=0, microsecond=0)


def build_default_dataset_range(_target_hours, _lookback_days):
    _lookback_days = int(_lookback_days)
    if _lookback_days <= 0:
        raise RuntimeError("lookback_days must be 1 or greater")

    _now_jst = datetime.now(JST)
    _latest_closed_h1_timestamp = floor_to_hour(_now_jst) - timedelta(hours=1)
    _latest_labelable_h1_timestamp = _latest_closed_h1_timestamp - timedelta(hours=max(_target_hours))
    _default_end_jst = floor_to_hour(_latest_labelable_h1_timestamp)
    _default_start_jst = floor_to_hour(_default_end_jst - timedelta(days=_lookback_days))

    if _default_start_jst >= _default_end_jst:
        raise RuntimeError("Failed to build a valid default date range")

    return _default_start_jst, _default_end_jst


def validate_args(_args):
    _sequence_length = int(_args.sequence_length)
    if _sequence_length <= 1:
        raise RuntimeError("sequence_length must be 2 or greater")

    _future_hours = int(_args.future_hours)
    if _future_hours <= 0:
        raise RuntimeError("future_hours must be 1 or greater")

    _extra_target_hours = parse_hour_list(getattr(_args, "extra_target_hours", ""))
    _target_hours = sorted(set([_future_hours, *_extra_target_hours]))
    _start_text = str(getattr(_args, "start", "") or "").strip()
    _end_text = str(getattr(_args, "end", "") or "").strip()

    if _start_text == "" and _end_text == "":
        _start_jst, _end_jst = build_default_dataset_range(_target_hours, int(_args.lookback_days))
    else:
        if _start_text == "" or _end_text == "":
            raise RuntimeError("start and end must both be provided when using explicit date range")

        _start_jst = ParseJSTDateTime(_start_text)
        _end_jst = ParseJSTDateTime(_end_text)

        if _start_jst is None:
            raise RuntimeError(f"Failed to parse start datetime: {_args.start}")

        if _end_jst is None:
            raise RuntimeError(f"Failed to parse end datetime: {_args.end}")

    if _start_jst > _end_jst:
        raise RuntimeError("start must be earlier than or equal to end")

    return _start_jst, _end_jst, _sequence_length, _future_hours, _target_hours


def _to_utc(_dt_jst):
    if _dt_jst is None:
        return None

    if _dt_jst.tzinfo is None:
        _dt_jst = _dt_jst.replace(tzinfo=JST)

    return _dt_jst.astimezone(UTC)


def fetch_mt5_rates_range(_symbol, _timeframe, _start_jst, _end_jst):
    _start_utc = _to_utc(_start_jst)
    _end_utc = _to_utc(_end_jst)

    _rates = mt5.copy_rates_range(_symbol, _timeframe, _start_utc, _end_utc)

    if _rates is None:
        raise RuntimeError(
            f"MT5 rate fetch failed: symbol={_symbol}, timeframe={_timeframe}, error={mt5.last_error()}"
        )

    if len(_rates) == 0:
        raise RuntimeError(
            f"MT5 returned no rates: symbol={_symbol}, timeframe={_timeframe}, start={_start_jst}, end={_end_jst}"
        )

    return _rates


def convert_rates_to_dataframe(_rates):
    _df = pd.DataFrame(_rates)

    if "time" not in _df.columns:
        raise RuntimeError("MT5 rates do not include a time column")

    _df["timestamp"] = pd.to_datetime(_df["time"], unit="s", utc=True).dt.tz_convert(JST)
    _df = _df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return _df[["timestamp", "time", "open", "high", "low", "close"]].copy()


def calculate_fetch_lookback_days(_sequence_length):
    _required_bars = max(MIN_HISTORY_BARS_FOR_FEATURES, int(_sequence_length))
    _required_days = math.ceil(_required_bars / 24)
    return max(MIN_FETCH_LOOKBACK_DAYS, _required_days + FETCH_LOOKBACK_BUFFER_DAYS)


def load_mt5_history(_symbol, _start_jst, _end_jst, _sequence_length, _future_hours):
    _lookback_days = calculate_fetch_lookback_days(_sequence_length)
    _h1_fetch_start = _start_jst - timedelta(days=_lookback_days)
    _h1_fetch_end = _end_jst + timedelta(hours=int(_future_hours))

    _h1_rates = fetch_mt5_rates_range(
        _symbol,
        mt5.TIMEFRAME_H1,
        _h1_fetch_start,
        _h1_fetch_end,
    )

    return {
        "H1": convert_rates_to_dataframe(_h1_rates),
    }


def normalize_csv_timestamp_series(_timestamp_series):
    _timestamp_series = pd.to_datetime(_timestamp_series, errors="coerce")
    if getattr(_timestamp_series.dt, "tz", None) is None:
        return _timestamp_series.dt.tz_localize(JST)

    return _timestamp_series.dt.tz_convert(JST)


def resolve_csv_column_name(_df, _candidate_names):
    _normalized_map = {
        str(_column_name).strip().lower(): _column_name
        for _column_name in _df.columns
    }

    for _candidate_name in _candidate_names:
        _resolved_name = _normalized_map.get(str(_candidate_name).strip().lower())
        if _resolved_name is not None:
            return _resolved_name

    return None


def load_h1_history_from_csv(_csv_path, _start_jst, _end_jst, _sequence_length, _future_hours):
    _csv_path = Path(_csv_path).resolve()
    if not _csv_path.exists():
        raise RuntimeError(f"H1 source CSV was not found: {_csv_path}")

    _raw_df = pd.read_csv(_csv_path)
    if len(_raw_df) == 0:
        raise RuntimeError(f"H1 source CSV is empty: {_csv_path}")

    _timestamp_column = resolve_csv_column_name(
        _raw_df,
        ["timestamp_jst", "timestamp", "time_jst", "time", "datetime", "date"],
    )
    if _timestamp_column is None:
        raise RuntimeError("Failed to find a timestamp column in H1 source CSV")

    _open_column = resolve_csv_column_name(_raw_df, ["open", "Open"])
    _high_column = resolve_csv_column_name(_raw_df, ["high", "High"])
    _low_column = resolve_csv_column_name(_raw_df, ["low", "Low"])
    _close_column = resolve_csv_column_name(_raw_df, ["close", "Close"])

    if None in (_open_column, _high_column, _low_column, _close_column):
        raise RuntimeError("H1 source CSV must include open/high/low/close columns")

    _lookback_days = calculate_fetch_lookback_days(_sequence_length)
    _fetch_start = _start_jst - timedelta(days=_lookback_days)
    _fetch_end = _end_jst + timedelta(hours=int(_future_hours))

    _history_df = pd.DataFrame(
        {
            "timestamp": normalize_csv_timestamp_series(_raw_df[_timestamp_column]),
            "open": pd.to_numeric(_raw_df[_open_column], errors="coerce"),
            "high": pd.to_numeric(_raw_df[_high_column], errors="coerce"),
            "low": pd.to_numeric(_raw_df[_low_column], errors="coerce"),
            "close": pd.to_numeric(_raw_df[_close_column], errors="coerce"),
        }
    )
    _history_df = _history_df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    _history_df = _history_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    _history_df = _history_df[
        (_history_df["timestamp"] >= _fetch_start) &
        (_history_df["timestamp"] <= _fetch_end)
    ].copy().reset_index(drop=True)

    if len(_history_df) == 0:
        raise RuntimeError(
            f"No H1 rows remained after filtering source CSV: start={FormatJSTDateTime(_fetch_start)}, end={FormatJSTDateTime(_fetch_end)}"
        )

    _history_df["time"] = (
        _history_df["timestamp"]
        .dt.tz_convert(UTC)
        .astype("int64")
        .floordiv(10 ** 9)
        .astype(np.int64)
    )

    return {
        "H1": _history_df[["timestamp", "time", "open", "high", "low", "close"]].copy(),
    }


def load_h1_history(_symbol, _start_jst, _end_jst, _sequence_length, _target_hours, _h1_source_csv):
    _max_future_hours = max(int(_hour_value) for _hour_value in _target_hours)

    if str(_h1_source_csv).strip() != "":
        return load_h1_history_from_csv(
            _h1_source_csv,
            _start_jst,
            _end_jst,
            _sequence_length,
            _max_future_hours,
        )

    return load_mt5_history(
        _symbol,
        _start_jst,
        _end_jst,
        _sequence_length,
        _max_future_hours,
    )


def calc_rsi(_close_series, _period=14):
    _delta = _close_series.diff()
    _gain = _delta.clip(lower=0.0)
    _loss = -_delta.clip(upper=0.0)

    _avg_gain = _gain.rolling(window=_period, min_periods=_period).mean()
    _avg_loss = _loss.rolling(window=_period, min_periods=_period).mean()

    _rs = _avg_gain.div(_avg_loss.where(_avg_loss != 0.0))
    _rsi = 100.0 - (100.0 / (1.0 + _rs))

    _both_zero_mask = (_avg_gain == 0.0) & (_avg_loss == 0.0)
    _up_only_mask = (_avg_gain > 0.0) & (_avg_loss == 0.0)
    _down_only_mask = (_avg_gain == 0.0) & (_avg_loss > 0.0)

    _rsi = _rsi.mask(_both_zero_mask, 50.0)
    _rsi = _rsi.mask(_up_only_mask, 100.0)
    _rsi = _rsi.mask(_down_only_mask, 0.0)

    return _rsi.fillna(50.0)


def calc_macd(_close_series):
    _ema_fast = _close_series.ewm(span=12, adjust=False).mean()
    _ema_slow = _close_series.ewm(span=26, adjust=False).mean()
    _macd = _ema_fast - _ema_slow
    _macd_signal = _macd.ewm(span=9, adjust=False).mean()
    _macd_hist = _macd - _macd_signal

    return _macd, _macd_signal, _macd_hist


def safe_divide(_numerator_series, _denominator_series, _default_value=0.0):
    _safe_denominator = _denominator_series.where(_denominator_series != 0.0)
    return _numerator_series.div(_safe_denominator).fillna(float(_default_value))


def calc_close_position(_close_series, _high_series, _low_series, _window):
    _rolling_high = _high_series.rolling(window=_window, min_periods=_window).max()
    _rolling_low = _low_series.rolling(window=_window, min_periods=_window).min()
    return safe_divide(_close_series - _rolling_low, _rolling_high - _rolling_low, 0.5)


def build_h1_feature_dataframe(_h1_df):
    _feature_df = _h1_df.copy().sort_values("timestamp").reset_index(drop=True)

    _feature_df["sma_20"] = _feature_df["close"].rolling(window=20, min_periods=20).mean()
    _feature_df["sma_50"] = _feature_df["close"].rolling(window=50, min_periods=50).mean()
    _feature_df["ema_20"] = _feature_df["close"].ewm(span=20, adjust=False).mean()
    _feature_df["rsi_14"] = calc_rsi(_feature_df["close"], 14)

    _macd, _macd_signal, _macd_hist = calc_macd(_feature_df["close"])
    _feature_df["macd"] = _macd
    _feature_df["macd_signal"] = _macd_signal
    _feature_df["macd_hist"] = _macd_hist

    _feature_df["body"] = _feature_df["close"] - _feature_df["open"]
    _feature_df["range"] = _feature_df["high"] - _feature_df["low"]
    _feature_df["upper_wick"] = _feature_df["high"] - _feature_df[["open", "close"]].max(axis=1)
    _feature_df["lower_wick"] = _feature_df[["open", "close"]].min(axis=1) - _feature_df["low"]
    _feature_df["body_ratio"] = safe_divide(_feature_df["body"], _feature_df["range"], 0.0)
    _feature_df["close_return_1"] = _feature_df["close"].pct_change().fillna(0.0)
    _feature_df["return_std_5"] = _feature_df["close_return_1"].rolling(window=5, min_periods=5).std()
    _feature_df["return_std_10"] = _feature_df["close_return_1"].rolling(window=10, min_periods=10).std()
    _feature_df["ma_gap_sma20"] = _feature_df["close"] - _feature_df["sma_20"]
    _feature_df["ma_gap_sma50"] = _feature_df["close"] - _feature_df["sma_50"]
    _feature_df["range_mean_5"] = _feature_df["range"].rolling(window=5, min_periods=5).mean()
    _feature_df["range_mean_10"] = _feature_df["range"].rolling(window=10, min_periods=10).mean()
    _feature_df["range_ratio_5"] = safe_divide(_feature_df["range"], _feature_df["range_mean_5"], 1.0)
    _feature_df["range_ratio_10"] = safe_divide(_feature_df["range"], _feature_df["range_mean_10"], 1.0)
    _feature_df["close_position_5"] = calc_close_position(_feature_df["close"], _feature_df["high"], _feature_df["low"], 5)
    _feature_df["close_position_10"] = calc_close_position(_feature_df["close"], _feature_df["high"], _feature_df["low"], 10)

    _hour = _feature_df["timestamp"].dt.hour.astype(float)
    _weekday = _feature_df["timestamp"].dt.dayofweek.astype(float)
    _feature_df["hour_sin"] = np.sin((2.0 * math.pi * _hour) / 24.0)
    _feature_df["hour_cos"] = np.cos((2.0 * math.pi * _hour) / 24.0)
    _feature_df["weekday_sin"] = np.sin((2.0 * math.pi * _weekday) / 7.0)
    _feature_df["weekday_cos"] = np.cos((2.0 * math.pi * _weekday) / 7.0)

    return _feature_df


def build_training_target_frame(_feature_df, _start_jst, _end_jst, _sequence_length, _future_hours, _target_hours):
    _work_df = _feature_df.copy().reset_index(drop=True)
    _work_df["row_index"] = _work_df.index
    _work_df["entry_price"] = _work_df["close"].astype(float)
    _future_price_lookup = _work_df.set_index("timestamp")["close"].to_dict()

    for _target_hour in sorted(set(int(_hour_value) for _hour_value in _target_hours)):
        _future_timestamp_column = f"future_timestamp_t_plus_{_target_hour}"
        _future_price_column = f"future_price_t_plus_{_target_hour}"
        _target_close_column = f"target_close_t_plus_{_target_hour}"
        _target_delta_column = f"target_delta_t_plus_{_target_hour}"

        _future_timestamp_series = _work_df["timestamp"] + pd.to_timedelta(int(_target_hour), unit="h")
        _work_df[_future_timestamp_column] = _future_timestamp_series
        _work_df[_future_price_column] = _future_timestamp_series.map(_future_price_lookup)
        _work_df[_target_close_column] = _work_df[_future_price_column]
        _work_df[_target_delta_column] = _work_df[_future_price_column] - _work_df["entry_price"]

    _work_df["future_timestamp"] = _work_df[f"future_timestamp_t_plus_{int(_future_hours)}"]
    _work_df["future_price"] = _work_df[f"future_price_t_plus_{int(_future_hours)}"]
    _work_df["target_close"] = _work_df[f"target_close_t_plus_{int(_future_hours)}"]
    _work_df["target_delta"] = _work_df[f"target_delta_t_plus_{int(_future_hours)}"]

    _feature_row_ready = _work_df[H1_FEATURE_COLUMNS].notna().all(axis=1)
    _work_df["window_ready"] = (
        _feature_row_ready.astype(int)
        .rolling(window=int(_sequence_length), min_periods=int(_sequence_length))
        .sum()
        .eq(int(_sequence_length))
    )

    _range_df = _work_df[
        (_work_df["timestamp"] >= _start_jst) &
        (_work_df["timestamp"] <= _end_jst)
    ].copy().reset_index(drop=True)

    return _work_df, _range_df


def summarize_target_frame(_target_df):
    _window_skip_count = int((~_target_df["window_ready"]).sum())
    _future_skip_count = int((_target_df["window_ready"] & _target_df["future_price"].isna()).sum())
    _eligible_count = int((_target_df["window_ready"] & _target_df["future_price"].notna()).sum())

    return {
        "target_count": int(len(_target_df)),
        "window_skip_count": _window_skip_count,
        "future_skip_count": _future_skip_count,
        "eligible_count": _eligible_count,
    }


def select_eligible_targets(_target_df):
    return _target_df[
        _target_df["window_ready"] &
        _target_df["future_price"].notna()
    ].copy().reset_index(drop=True)


def build_h1_feature_window(_feature_df, _row_index, _sequence_length):
    _start_index = int(_row_index) - int(_sequence_length) + 1
    if _start_index < 0:
        return None

    _window = _feature_df.iloc[_start_index:int(_row_index) + 1].copy().reset_index(drop=True)

    if len(_window) != int(_sequence_length):
        return None

    if _window[H1_WINDOW_COLUMNS].isnull().any().any():
        return None

    return _window


def build_regression_targets_from_row(_row, _entry_price, _future_hours, _target_hours):
    _target_info = {}

    for _target_hour in sorted(set(int(_hour_value) for _hour_value in _target_hours)):
        _target_close_value = getattr(_row, f"target_close_t_plus_{_target_hour}")
        _target_delta_value = getattr(_row, f"target_delta_t_plus_{_target_hour}")
        _target_info[f"target_close_t_plus_{_target_hour}"] = float(_target_close_value)
        _target_info[f"target_delta_t_plus_{_target_hour}"] = float(_target_delta_value)

    _target_info["target_close"] = float(_target_info[f"target_close_t_plus_{int(_future_hours)}"])
    _target_info["target_delta"] = float(_target_info[f"target_delta_t_plus_{int(_future_hours)}"])

    return _target_info


def calc_h1_momentum(_close_list, _lookback=5):
    if _close_list is None or len(_close_list) <= _lookback:
        return 0.0

    _current_close = _close_list[-1]
    _past_close = _close_list[-1 - _lookback]

    return float(_current_close - _past_close)


def calc_h1_trend_consistency(_close_list):
    if _close_list is None or len(_close_list) < 2:
        return 0.0

    _diff_list = [
        float(_close_list[_index] - _close_list[_index - 1])
        for _index in range(1, len(_close_list))
    ]

    _up_count = sum(1 for _diff in _diff_list if _diff > 0.0)
    _down_count = sum(1 for _diff in _diff_list if _diff < 0.0)
    _dominant_count = max(_up_count, _down_count)

    return float(_dominant_count / len(_diff_list))


def calc_h1_up_ratio(_close_list):
    if _close_list is None or len(_close_list) < 2:
        return 0.0

    _diff_list = [
        float(_close_list[_index] - _close_list[_index - 1])
        for _index in range(1, len(_close_list))
    ]
    _up_count = sum(1 for _diff in _diff_list if _diff > 0.0)
    return float(_up_count / len(_diff_list))


def calc_window_close_position(_close_list, _high_list, _low_list, _lookback):
    if _close_list is None or _high_list is None or _low_list is None:
        return 0.5

    if len(_close_list) == 0 or len(_high_list) == 0 or len(_low_list) == 0:
        return 0.5

    _window_size = min(int(_lookback), len(_close_list))
    _rolling_high = max(float(_value) for _value in _high_list[-_window_size:])
    _rolling_low = min(float(_value) for _value in _low_list[-_window_size:])
    _range = float(_rolling_high - _rolling_low)
    if _range <= 0.0:
        return 0.5

    return float((float(_close_list[-1]) - _rolling_low) / _range)


def build_h1_training_record(
    _timestamp_jst,
    _future_timestamp_jst,
    _symbol,
    _sequence_length,
    _future_hours,
    _target_hours,
    _entry_price,
    _target_info,
    _h1_window_df,
):
    _record = {
        "timestamp_jst": FormatJSTDateTime(_timestamp_jst),
        "symbol": _symbol,
        "sequence_length": int(_sequence_length),
        "future_hours": int(_future_hours),
        "target_hours": ",".join(str(int(_hour_value)) for _hour_value in sorted(set(_target_hours))),
        "entry_price": float(_entry_price),
        "future_timestamp_jst": FormatJSTDateTime(_future_timestamp_jst),
    }
    _record.update(_target_info)

    for _target_hour in sorted(set(int(_hour_value) for _hour_value in _target_hours)):
        _future_timestamp_value = _timestamp_jst + timedelta(hours=int(_target_hour))
        _record[f"future_timestamp_t_plus_{_target_hour}_jst"] = FormatJSTDateTime(_future_timestamp_value)

    for _index, _row in enumerate(_h1_window_df.itertuples(index=False)):
        for _column_name in H1_WINDOW_COLUMNS:
            _record[f"h1_{_column_name}_{_index:02d}"] = float(getattr(_row, _column_name))

    _close_list = [float(_value) for _value in _h1_window_df["close"].tolist()]
    _high_list = [float(_value) for _value in _h1_window_df["high"].tolist()]
    _low_list = [float(_value) for _value in _h1_window_df["low"].tolist()]
    _range_list = [float(_value) for _value in _h1_window_df["range"].tolist()]
    _current_timestamp = _h1_window_df["timestamp"].iloc[-1]
    _record["h1_last_close"] = float(_close_list[-1])
    _record["h1_recent_momentum"] = float(calc_h1_momentum(_close_list, 5))
    _record["h1_trend_consistency"] = float(calc_h1_trend_consistency(_close_list))
    _record["h1_window_range_mean"] = float(_h1_window_df["range"].mean())
    _record["h1_window_range_std"] = float(_h1_window_df["range"].std(ddof=0))
    _record["h1_window_return_std"] = float(_h1_window_df["close_return_1"].std(ddof=0))
    _record["h1_window_up_ratio"] = float(calc_h1_up_ratio(_close_list))
    _record["h1_window_range_ratio_5bar"] = float(
        _range_list[-1] / max(float(np.mean(_range_list[-5:])), 1.0e-8)
    )
    _record["h1_window_range_ratio_10bar"] = float(
        _range_list[-1] / max(float(np.mean(_range_list[-10:])), 1.0e-8)
    )
    _record["h1_window_close_slope_5bar"] = float(calc_h1_momentum(_close_list, 5))
    _record["h1_window_close_slope_10bar"] = float(calc_h1_momentum(_close_list, 10))
    _record["h1_window_close_position_10bar"] = float(
        calc_window_close_position(_close_list, _high_list, _low_list, 10)
    )
    _record["h1_window_close_position_full"] = float(
        calc_window_close_position(_close_list, _high_list, _low_list, len(_close_list))
    )
    _record["h1_current_hour_sin"] = float(np.sin((2.0 * math.pi * float(_current_timestamp.hour)) / 24.0))
    _record["h1_current_hour_cos"] = float(np.cos((2.0 * math.pi * float(_current_timestamp.hour)) / 24.0))
    _record["h1_current_weekday_sin"] = float(np.sin((2.0 * math.pi * float(_current_timestamp.dayofweek)) / 7.0))
    _record["h1_current_weekday_cos"] = float(np.cos((2.0 * math.pi * float(_current_timestamp.dayofweek)) / 7.0))

    return _record


def save_dataset(_output_path, _records):
    EnsureParentDirectory(_output_path)
    pd.DataFrame(_records).to_csv(_output_path, index=False, encoding="utf-8")


def main():
    _args = parse_args()
    _start_jst, _end_jst, _sequence_length, _future_hours, _target_hours = validate_args(_args)

    print("========== H1 Training Dataset Build Start ==========")
    print(f"[INFO] cwd={Path.cwd()}")
    print(f"[INFO] source_csv={Path(_args.h1_source_csv).resolve() if str(_args.h1_source_csv).strip() != '' else ''}")

    if str(_args.h1_source_csv).strip() == "":
        if not MTManager_Initialize():
            raise RuntimeError("Failed to initialize MT5")

    _history = load_h1_history(
        _args.symbol,
        _start_jst,
        _end_jst,
        _sequence_length,
        _target_hours,
        _args.h1_source_csv,
    )
    _feature_df = build_h1_feature_dataframe(_history["H1"])
    _, _target_df = build_training_target_frame(
        _feature_df,
        _start_jst,
        _end_jst,
        _sequence_length,
        _future_hours,
        _target_hours,
    )
    _target_summary = summarize_target_frame(_target_df)
    _eligible_target_df = select_eligible_targets(_target_df)

    print(f"[INFO] symbol={_args.symbol}")
    print(f"[INFO] start={FormatJSTDateTime(_start_jst)}")
    print(f"[INFO] end={FormatJSTDateTime(_end_jst)}")
    print(f"[INFO] sequence_length={_sequence_length}")
    print(f"[INFO] future_hours={_future_hours}")
    print(f"[INFO] target_hours={','.join(str(_hour_value) for _hour_value in _target_hours)}")
    print(f"[INFO] target_count={_target_summary['target_count']}")
    print(f"[INFO] eligible_count={_target_summary['eligible_count']}")
    print(f"[INFO] window_skip_count={_target_summary['window_skip_count']}")
    print(f"[INFO] future_skip_count={_target_summary['future_skip_count']}")

    if _target_summary["target_count"] == 0:
        raise RuntimeError("No H1 target timestamps were found in the requested range")

    if _target_summary["eligible_count"] == 0:
        raise RuntimeError("No eligible records were found after feature and target filtering")

    _records = []
    for _index, _row in enumerate(_eligible_target_df.itertuples(index=False), start=1):
        _h1_window_df = build_h1_feature_window(
            _feature_df,
            int(_row.row_index),
            _sequence_length,
        )

        if _h1_window_df is None:
            raise RuntimeError(f"Failed to build H1 feature window at {FormatJSTDateTime(_row.timestamp)}")

        _target_info = build_regression_targets_from_row(
            _row,
            float(_row.entry_price),
            _future_hours,
            _target_hours,
        )

        _record = build_h1_training_record(
            _timestamp_jst=_row.timestamp,
            _future_timestamp_jst=_row.future_timestamp,
            _symbol=_args.symbol,
            _sequence_length=_sequence_length,
            _future_hours=_future_hours,
            _target_hours=_target_hours,
            _entry_price=float(_row.entry_price),
            _target_info=_target_info,
            _h1_window_df=_h1_window_df,
        )
        _records.append(_record)

        if _args.verbose and (_index % PROGRESS_INTERVAL == 0):
            print(
                f"[INFO] progress={_index}/{len(_eligible_target_df)} "
                f"records={len(_records)}"
            )

    if len(_records) == 0:
        raise RuntimeError("No training records were created")

    _output_path = Path(_args.output).resolve()

    print(f"[INFO] output_dir={_output_path.parent}")
    print(f"[INFO] output_file={_output_path}")

    save_dataset(str(_output_path), _records)

    print(f"[INFO] saved={_output_path}")
    print(f"[INFO] record_count={len(_records)}")
    print(f"[INFO] skip_count={_target_summary['target_count'] - len(_records)}")
    print("[INFO] error_count=0")
    print("========== H1 Training Dataset Build End ==========")


if __name__ == "__main__":
    main()
