import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.Utility.Utility import EnsureParentDirectory


DEFAULT_DATASET_PATH = "Src/Backtest/Output/datasets/h1_training_dataset.csv"
DEFAULT_MODEL_OUTPUT_PATH = "Asset/Models/h1_multi_horizon_patch_mixer.pt"
DEFAULT_METADATA_OUTPUT_PATH = "Asset/Models/h1_multi_horizon_patch_mixer_metadata.json"
DEFAULT_PREDICTION_OUTPUT_PATH = "Asset/Models/h1_multi_horizon_patch_mixer_test_predictions.csv"
DEFAULT_SUMMARY_OUTPUT_PATH = "Asset/Models/h1_multi_horizon_patch_mixer_summary.json"
DEFAULT_HORIZONS = "6,7,8"
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 5.0e-4
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_MIXER_LAYERS = 1
DEFAULT_PATCH_LENGTH = 8
DEFAULT_DROPOUT = 0.20
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 10
DEFAULT_HUBER_BETA = 0.5
DEFAULT_ACTIVE_QUANTILE = 0.65
DEFAULT_ACTIVE_WEIGHT = 0.0
DEFAULT_DIRECTION_LOSS_WEIGHT = 0.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_TARGET_SCALE_LOOKBACK = 12
MIN_TARGET_SCALE = 1.0e-6
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15


def parse_args():
    if len(sys.argv) == 1:
        return SimpleNamespace(
            dataset=DEFAULT_DATASET_PATH,
            horizons=DEFAULT_HORIZONS,
            model_output=DEFAULT_MODEL_OUTPUT_PATH,
            metadata_output=DEFAULT_METADATA_OUTPUT_PATH,
            prediction_output=DEFAULT_PREDICTION_OUTPUT_PATH,
            summary_output=DEFAULT_SUMMARY_OUTPUT_PATH,
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            mixer_layers=DEFAULT_MIXER_LAYERS,
            patch_length=DEFAULT_PATCH_LENGTH,
            dropout=DEFAULT_DROPOUT,
            seed=DEFAULT_SEED,
            patience=DEFAULT_PATIENCE,
            huber_beta=DEFAULT_HUBER_BETA,
            active_quantile=DEFAULT_ACTIVE_QUANTILE,
            active_weight=DEFAULT_ACTIVE_WEIGHT,
            direction_loss_weight=DEFAULT_DIRECTION_LOSS_WEIGHT,
            grad_clip=DEFAULT_GRAD_CLIP,
            target_scale_lookback=DEFAULT_TARGET_SCALE_LOOKBACK,
            verbose=True,
        )

    _parser = argparse.ArgumentParser(description="Train multi-horizon H1 forecaster for SGSystemForRO")
    _parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to h1_training_dataset.csv")
    _parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help="Forecast horizons in hours. Example: 6,7,8")
    _parser.add_argument("--model-output", default=DEFAULT_MODEL_OUTPUT_PATH, help="Path to save trained model")
    _parser.add_argument("--metadata-output", default=DEFAULT_METADATA_OUTPUT_PATH, help="Path to save metadata json")
    _parser.add_argument("--prediction-output", default=DEFAULT_PREDICTION_OUTPUT_PATH, help="Path to save test predictions csv")
    _parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT_PATH, help="Path to save summary json")
    _parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    _parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    _parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    _parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay")
    _parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help="Hidden size")
    _parser.add_argument("--mixer-layers", type=int, default=DEFAULT_MIXER_LAYERS, help="Number of mixer blocks")
    _parser.add_argument("--patch-length", type=int, default=DEFAULT_PATCH_LENGTH, help="Bars per patch")
    _parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout ratio")
    _parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    _parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    _parser.add_argument("--huber-beta", type=float, default=DEFAULT_HUBER_BETA, help="Smooth L1 beta")
    _parser.add_argument(
        "--active-quantile",
        type=float,
        default=DEFAULT_ACTIVE_QUANTILE,
        help="Train-set quantile of abs(delta / recent_range_mean) used for active-case masking",
    )
    _parser.add_argument(
        "--active-weight",
        type=float,
        default=DEFAULT_ACTIVE_WEIGHT,
        help="Extra weight multiplier for strong-move samples",
    )
    _parser.add_argument(
        "--direction-loss-weight",
        type=float,
        default=DEFAULT_DIRECTION_LOSS_WEIGHT,
        help="Loss weight for the auxiliary direction head",
    )
    _parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP, help="Gradient clipping value")
    _parser.add_argument(
        "--target-scale-lookback",
        type=int,
        default=DEFAULT_TARGET_SCALE_LOOKBACK,
        help="Lookback window used to normalize targets by recent H1 range mean",
    )
    _parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    return _parser.parse_args()


def parse_horizon_list(_horizon_text):
    _tokens = [str(_token).strip() for _token in str(_horizon_text).split(",")]
    _horizon_list = []

    for _token in _tokens:
        if _token == "":
            continue

        try:
            _horizon_value = int(_token)
        except Exception as _error:
            raise RuntimeError(f"Failed to parse horizon token: {_token}") from _error

        if _horizon_value <= 0:
            raise RuntimeError("All horizons must be 1 or greater")

        _horizon_list.append(_horizon_value)

    _horizon_list = sorted(set(_horizon_list))
    if len(_horizon_list) == 0:
        raise RuntimeError("At least one horizon must be provided")

    return _horizon_list


def set_seed(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)


def safe_divide_array(_numerator_array, _denominator_array, _default_value):
    _numerator_array = np.asarray(_numerator_array, dtype=np.float32)
    _denominator_array = np.asarray(_denominator_array, dtype=np.float32)
    _result = np.full_like(_numerator_array, float(_default_value), dtype=np.float32)
    _valid_mask = np.abs(_denominator_array) > 1.0e-8
    np.divide(_numerator_array, _denominator_array, out=_result, where=_valid_mask)
    return _result.astype(np.float32)


def load_dataset(_dataset_path):
    _dataset_path = Path(_dataset_path).resolve()
    if not _dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {_dataset_path}")

    _df = pd.read_csv(_dataset_path)
    if len(_df) == 0:
        raise RuntimeError(f"Dataset is empty: {_dataset_path}")

    if "timestamp_jst" not in _df.columns:
        raise RuntimeError("timestamp_jst column was not found")

    _df["timestamp_jst"] = pd.to_datetime(_df["timestamp_jst"], errors="coerce")
    _df = _df.dropna(subset=["timestamp_jst"]).copy()
    _df = _df.sort_values("timestamp_jst").drop_duplicates(subset=["timestamp_jst"]).reset_index(drop=True)

    return _df, _dataset_path


def get_sequence_length(_df):
    if "sequence_length" not in _df.columns:
        raise RuntimeError("sequence_length column was not found")

    _unique_values = sorted(pd.to_numeric(_df["sequence_length"], errors="coerce").dropna().unique().tolist())
    if len(_unique_values) != 1:
        raise RuntimeError(f"sequence_length must be unique, found: {_unique_values}")

    return int(_unique_values[0])


def get_step_matrix(_df, _base_name, _sequence_length, _default_value=None):
    _column_names = [f"{_base_name}_{_index:02d}" for _index in range(_sequence_length)]
    _column_exists = [(_column_name in _df.columns) for _column_name in _column_names]

    if all(_column_exists):
        return _df[_column_names].to_numpy(dtype=np.float32)

    if _default_value is None:
        _missing_columns = [name for name, exists in zip(_column_names, _column_exists) if not exists]
        raise RuntimeError(f"Required feature columns were not found: {_missing_columns[:5]}")

    return np.full((len(_df), _sequence_length), float(_default_value), dtype=np.float32)


def get_split_boundaries(_total_count):
    if _total_count < 32:
        raise RuntimeError("Dataset is too small for time-based split")

    _train_end = int(_total_count * TRAIN_RATIO)
    _valid_end = int(_total_count * (TRAIN_RATIO + VALID_RATIO))

    if _train_end <= 0 or _valid_end <= _train_end or _valid_end >= _total_count:
        raise RuntimeError("Invalid split boundaries were produced")

    return _train_end, _valid_end


def ensure_multihorizon_targets(_df, _horizons):
    _work_df = _df.copy().reset_index(drop=True)

    if "entry_price" not in _work_df.columns:
        raise RuntimeError("entry_price column was not found")

    _work_df["entry_price"] = pd.to_numeric(_work_df["entry_price"], errors="coerce")
    _future_lookup = _work_df.set_index("timestamp_jst")["entry_price"].to_dict()

    for _horizon in _horizons:
        _future_timestamp_column = f"future_timestamp_t_plus_{_horizon}_jst"
        _target_close_column = f"target_close_t_plus_{_horizon}"
        _target_delta_column = f"target_delta_t_plus_{_horizon}"

        if _future_timestamp_column in _work_df.columns:
            _work_df[_future_timestamp_column] = pd.to_datetime(_work_df[_future_timestamp_column], errors="coerce")
        else:
            _work_df[_future_timestamp_column] = _work_df["timestamp_jst"] + pd.to_timedelta(int(_horizon), unit="h")

        _future_close_series = _work_df[_future_timestamp_column].map(_future_lookup)
        if _target_close_column in _work_df.columns:
            _work_df[_target_close_column] = pd.to_numeric(_work_df[_target_close_column], errors="coerce")
            _work_df[_target_close_column] = _work_df[_target_close_column].fillna(_future_close_series)
        else:
            _work_df[_target_close_column] = _future_close_series

        if _target_delta_column in _work_df.columns:
            _work_df[_target_delta_column] = pd.to_numeric(_work_df[_target_delta_column], errors="coerce")
            _target_delta_fill = _work_df[_target_close_column] - _work_df["entry_price"]
            _work_df[_target_delta_column] = _work_df[_target_delta_column].fillna(_target_delta_fill)
        else:
            _work_df[_target_delta_column] = _work_df[_target_close_column] - _work_df["entry_price"]

    _required_target_columns = [f"target_delta_t_plus_{_horizon}" for _horizon in _horizons]
    _work_df = _work_df.dropna(subset=_required_target_columns).copy().reset_index(drop=True)
    return _work_df


def build_time_feature_matrices(_timestamp_series, _sequence_length):
    _hour_columns = []
    _weekday_columns = []

    for _step_index in range(_sequence_length):
        _offset_hours = int(_sequence_length) - 1 - int(_step_index)
        _shifted_series = _timestamp_series - pd.to_timedelta(_offset_hours, unit="h")
        _hour_columns.append(_shifted_series.dt.hour.to_numpy(dtype=np.float32))
        _weekday_columns.append(_shifted_series.dt.weekday.to_numpy(dtype=np.float32))

    _hour_matrix = np.stack(_hour_columns, axis=1).astype(np.float32)
    _weekday_matrix = np.stack(_weekday_columns, axis=1).astype(np.float32)
    _hour_sin = np.sin((2.0 * np.pi * _hour_matrix) / 24.0).astype(np.float32)
    _hour_cos = np.cos((2.0 * np.pi * _hour_matrix) / 24.0).astype(np.float32)
    _weekday_sin = np.sin((2.0 * np.pi * _weekday_matrix) / 7.0).astype(np.float32)
    _weekday_cos = np.cos((2.0 * np.pi * _weekday_matrix) / 7.0).astype(np.float32)

    return _hour_sin, _hour_cos, _weekday_sin, _weekday_cos


def build_close_position_feature(_close_array, _high_array, _low_array, _lookback):
    _lookback = int(min(_lookback, _close_array.shape[1]))
    _window_high = _high_array[:, -_lookback:].max(axis=1)
    _window_low = _low_array[:, -_lookback:].min(axis=1)
    return safe_divide_array(_close_array[:, -1] - _window_low, _window_high - _window_low, 0.5)


def build_trend_consistency(_diff_window):
    _up_count = (_diff_window > 0.0).sum(axis=1)
    _down_count = (_diff_window < 0.0).sum(axis=1)
    _dominant = np.maximum(_up_count, _down_count)
    _denominator = max(_diff_window.shape[1], 1)
    return (_dominant.astype(np.float32) / float(_denominator)).astype(np.float32)


def build_feature_payload(_df, _sequence_length, _horizons, _target_scale_lookback):
    _open_array = get_step_matrix(_df, "h1_open", _sequence_length)
    _high_array = get_step_matrix(_df, "h1_high", _sequence_length)
    _low_array = get_step_matrix(_df, "h1_low", _sequence_length)
    _close_array = get_step_matrix(_df, "h1_close", _sequence_length)
    _rsi_array = get_step_matrix(_df, "h1_rsi_14", _sequence_length, 50.0)
    _macd_array = get_step_matrix(_df, "h1_macd", _sequence_length, 0.0)
    _macd_hist_array = get_step_matrix(_df, "h1_macd_hist", _sequence_length, 0.0)
    _ma_gap20_array = get_step_matrix(_df, "h1_ma_gap_sma20", _sequence_length, 0.0)
    _ma_gap50_array = get_step_matrix(_df, "h1_ma_gap_sma50", _sequence_length, 0.0)

    _range_array = (_high_array - _low_array).astype(np.float32)
    _body_array = (_close_array - _open_array).astype(np.float32)
    _upper_wick_array = (_high_array - np.maximum(_open_array, _close_array)).astype(np.float32)
    _lower_wick_array = (np.minimum(_open_array, _close_array) - _low_array).astype(np.float32)

    _close_prev_array = np.concatenate([_close_array[:, :1], _close_array[:, :-1]], axis=1)
    _close_return_array = safe_divide_array(_close_array - _close_prev_array, _close_prev_array, 0.0)
    _close_return_array[:, 0] = 0.0

    _target_scale_lookback = max(1, min(int(_target_scale_lookback), int(_sequence_length)))
    _recent_scale_array = _range_array[:, -_target_scale_lookback:].mean(axis=1).astype(np.float32)
    _recent_scale_array = np.maximum(_recent_scale_array, np.float32(MIN_TARGET_SCALE))
    _recent_scale_matrix = _recent_scale_array.reshape(-1, 1)

    _last_close_array = _close_array[:, -1:].astype(np.float32)
    _last_close_array = np.where(np.abs(_last_close_array) < 1.0e-8, 1.0, _last_close_array).astype(np.float32)

    _hour_sin_array, _hour_cos_array, _weekday_sin_array, _weekday_cos_array = build_time_feature_matrices(
        _df["timestamp_jst"],
        _sequence_length,
    )

    _sequence_feature_list = [
        safe_divide_array(_open_array - _last_close_array, _last_close_array, 0.0),
        safe_divide_array(_high_array - _last_close_array, _last_close_array, 0.0),
        safe_divide_array(_low_array - _last_close_array, _last_close_array, 0.0),
        safe_divide_array(_close_array - _last_close_array, _last_close_array, 0.0),
        safe_divide_array(_body_array, _recent_scale_matrix, 0.0),
        safe_divide_array(_range_array, _recent_scale_matrix, 0.0),
        safe_divide_array(_upper_wick_array, _recent_scale_matrix, 0.0),
        safe_divide_array(_lower_wick_array, _recent_scale_matrix, 0.0),
        _close_return_array,
        ((_rsi_array / 100.0) - 0.5).astype(np.float32),
        safe_divide_array(_macd_array, _recent_scale_matrix, 0.0),
        safe_divide_array(_macd_hist_array, _recent_scale_matrix, 0.0),
        safe_divide_array(_ma_gap20_array, _recent_scale_matrix, 0.0),
        safe_divide_array(_ma_gap50_array, _recent_scale_matrix, 0.0),
        _hour_sin_array,
        _hour_cos_array,
        _weekday_sin_array,
        _weekday_cos_array,
    ]
    _sequence_feature_names = [
        "rel_open",
        "rel_high",
        "rel_low",
        "rel_close",
        "body_scaled",
        "range_scaled",
        "upper_wick_scaled",
        "lower_wick_scaled",
        "close_return_1",
        "rsi_centered",
        "macd_scaled",
        "macd_hist_scaled",
        "ma_gap_sma20_scaled",
        "ma_gap_sma50_scaled",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    _sequence_x = np.stack(_sequence_feature_list, axis=-1).astype(np.float32)

    _diff_array = np.diff(_close_array, axis=1).astype(np.float32)
    _return_mean_6 = _close_return_array[:, -6:].mean(axis=1)
    _return_mean_12 = _close_return_array[:, -12:].mean(axis=1)
    _return_mean_24 = _close_return_array[:, -24:].mean(axis=1)
    _return_std_6 = _close_return_array[:, -6:].std(axis=1)
    _return_std_12 = _close_return_array[:, -12:].std(axis=1)
    _return_std_24 = _close_return_array[:, -24:].std(axis=1)
    _range_mean_6 = safe_divide_array(_range_array[:, -6:].mean(axis=1), _recent_scale_array, 0.0)
    _range_mean_12 = safe_divide_array(_range_array[:, -12:].mean(axis=1), _recent_scale_array, 0.0)
    _range_mean_24 = safe_divide_array(_range_array[:, -24:].mean(axis=1), _recent_scale_array, 0.0)
    _range_ratio_last_6 = safe_divide_array(_range_array[:, -1], _range_array[:, -6:].mean(axis=1), 1.0)
    _range_ratio_last_12 = safe_divide_array(_range_array[:, -1], _range_array[:, -12:].mean(axis=1), 1.0)
    _range_ratio_last_24 = safe_divide_array(_range_array[:, -1], _range_array[:, -24:].mean(axis=1), 1.0)
    _slope_6 = safe_divide_array(_close_array[:, -1] - _close_array[:, -6], _recent_scale_array, 0.0)
    _slope_12 = safe_divide_array(_close_array[:, -1] - _close_array[:, -12], _recent_scale_array, 0.0)
    _slope_24 = safe_divide_array(_close_array[:, -1] - _close_array[:, -24], _recent_scale_array, 0.0)
    _close_position_12 = build_close_position_feature(_close_array, _high_array, _low_array, 12)
    _close_position_24 = build_close_position_feature(_close_array, _high_array, _low_array, 24)
    _close_position_full = build_close_position_feature(_close_array, _high_array, _low_array, _sequence_length)
    _up_ratio_6 = (_diff_array[:, -5:] > 0.0).mean(axis=1).astype(np.float32)
    _up_ratio_12 = (_diff_array[:, -11:] > 0.0).mean(axis=1).astype(np.float32)
    _up_ratio_24 = (_diff_array[:, -23:] > 0.0).mean(axis=1).astype(np.float32)
    _trend_consistency_12 = build_trend_consistency(_diff_array[:, -11:])
    _trend_consistency_24 = build_trend_consistency(_diff_array[:, -23:])
    _recent_diff_mean = _diff_array[:, -6:].mean(axis=1).astype(np.float32)
    _current_hour = _df["timestamp_jst"].dt.hour.to_numpy(dtype=np.float32)
    _current_weekday = _df["timestamp_jst"].dt.weekday.to_numpy(dtype=np.float32)
    _current_hour_sin = np.sin((2.0 * np.pi * _current_hour) / 24.0).astype(np.float32)
    _current_hour_cos = np.cos((2.0 * np.pi * _current_hour) / 24.0).astype(np.float32)
    _current_weekday_sin = np.sin((2.0 * np.pi * _current_weekday) / 7.0).astype(np.float32)
    _current_weekday_cos = np.cos((2.0 * np.pi * _current_weekday) / 7.0).astype(np.float32)

    _static_feature_list = [
        np.log1p(_recent_scale_array).astype(np.float32),
        _return_mean_6.astype(np.float32),
        _return_mean_12.astype(np.float32),
        _return_mean_24.astype(np.float32),
        _return_std_6.astype(np.float32),
        _return_std_12.astype(np.float32),
        _return_std_24.astype(np.float32),
        _range_mean_6.astype(np.float32),
        _range_mean_12.astype(np.float32),
        _range_mean_24.astype(np.float32),
        _range_ratio_last_6.astype(np.float32),
        _range_ratio_last_12.astype(np.float32),
        _range_ratio_last_24.astype(np.float32),
        _slope_6.astype(np.float32),
        _slope_12.astype(np.float32),
        _slope_24.astype(np.float32),
        _close_position_12.astype(np.float32),
        _close_position_24.astype(np.float32),
        _close_position_full.astype(np.float32),
        _up_ratio_6.astype(np.float32),
        _up_ratio_12.astype(np.float32),
        _up_ratio_24.astype(np.float32),
        _trend_consistency_12.astype(np.float32),
        _trend_consistency_24.astype(np.float32),
        _current_hour_sin.astype(np.float32),
        _current_hour_cos.astype(np.float32),
        _current_weekday_sin.astype(np.float32),
        _current_weekday_cos.astype(np.float32),
    ]
    _static_feature_names = [
        "log_recent_range_mean",
        "return_mean_6",
        "return_mean_12",
        "return_mean_24",
        "return_std_6",
        "return_std_12",
        "return_std_24",
        "range_mean_6",
        "range_mean_12",
        "range_mean_24",
        "range_ratio_last_6",
        "range_ratio_last_12",
        "range_ratio_last_24",
        "slope_6",
        "slope_12",
        "slope_24",
        "close_position_12",
        "close_position_24",
        "close_position_full",
        "up_ratio_6",
        "up_ratio_12",
        "up_ratio_24",
        "trend_consistency_12",
        "trend_consistency_24",
        "current_hour_sin",
        "current_hour_cos",
        "current_weekday_sin",
        "current_weekday_cos",
    ]

    _horizon_array = np.asarray(_horizons, dtype=np.float32).reshape(1, -1)
    _drift_baseline = (_recent_diff_mean.reshape(-1, 1) * _horizon_array).astype(np.float32)
    for _horizon_index, _horizon_value in enumerate(_horizons):
        _static_feature_list.append(safe_divide_array(_drift_baseline[:, _horizon_index], _recent_scale_array, 0.0))
        _static_feature_names.append(f"drift_baseline_scaled_t_plus_{_horizon_value}")

    _static_x = np.stack(_static_feature_list, axis=1).astype(np.float32)
    _target_raw = np.stack(
        [_df[f"target_delta_t_plus_{_horizon}"].to_numpy(dtype=np.float32) for _horizon in _horizons],
        axis=1,
    ).astype(np.float32)

    return {
        "sequence_x": _sequence_x,
        "static_x": _static_x,
        "target_raw": _target_raw,
        "target_scale": _recent_scale_array.astype(np.float32),
        "sequence_feature_names": _sequence_feature_names,
        "static_feature_names": _static_feature_names,
        "close_array": _close_array.astype(np.float32),
        "drift_baseline": _drift_baseline.astype(np.float32),
    }


def fit_feature_scaler(_x_train):
    if _x_train.ndim == 3:
        _flat_array = _x_train.reshape(-1, _x_train.shape[-1])
    elif _x_train.ndim == 2:
        _flat_array = _x_train
    else:
        raise RuntimeError(f"Unsupported feature ndim: {_x_train.ndim}")

    _mean = _flat_array.mean(axis=0)
    _std = _flat_array.std(axis=0)
    _std = np.where(_std <= 1.0e-8, 1.0, _std)
    return _mean.astype(np.float32), _std.astype(np.float32)


def transform_feature_array(_x, _mean, _std):
    if _x.ndim == 3:
        return ((_x - _mean.reshape(1, 1, -1)) / _std.reshape(1, 1, -1)).astype(np.float32)

    if _x.ndim == 2:
        return ((_x - _mean.reshape(1, -1)) / _std.reshape(1, -1)).astype(np.float32)

    raise RuntimeError(f"Unsupported feature ndim: {_x.ndim}")


def build_direction_arrays(_target_raw, _target_scale, _threshold_array):
    _scaled_target = safe_divide_array(_target_raw, _target_scale.reshape(-1, 1), 0.0)
    _direction_y = (_target_raw >= 0.0).astype(np.float32)
    _direction_active = (np.abs(_scaled_target) >= _threshold_array.reshape(1, -1)).astype(np.float32)
    return _direction_y.astype(np.float32), _direction_active.astype(np.float32), _scaled_target.astype(np.float32)


def fit_active_thresholds(_target_raw_train, _target_scale_train, _active_quantile):
    _active_quantile = float(_active_quantile)
    if _active_quantile <= 0.0:
        return np.zeros(_target_raw_train.shape[1], dtype=np.float32)

    if _active_quantile >= 1.0:
        raise RuntimeError("active_quantile must be smaller than 1.0")

    _scaled_target_train = safe_divide_array(
        _target_raw_train,
        _target_scale_train.reshape(-1, 1),
        0.0,
    )
    return np.quantile(np.abs(_scaled_target_train), _active_quantile, axis=0).astype(np.float32)


def build_sample_weight_array(_target_raw, _target_scale, _threshold_array, _active_weight):
    _scaled_target = np.abs(safe_divide_array(_target_raw, _target_scale.reshape(-1, 1), 0.0))
    _signal_strength = _scaled_target.mean(axis=1)
    _reference = max(float(np.mean(_threshold_array)), 1.0e-6)
    _extra_weight = np.clip((_signal_strength / _reference) - 1.0, 0.0, 3.0)
    return (1.0 + (float(_active_weight) * _extra_weight)).astype(np.float32)


def build_pos_weight_array(_direction_y_train, _direction_active_train):
    _pos_weight_list = []
    for _horizon_index in range(_direction_y_train.shape[1]):
        _active_mask = _direction_active_train[:, _horizon_index] >= 0.5
        if not np.any(_active_mask):
            _pos_weight_list.append(1.0)
            continue

        _positive_count = float(_direction_y_train[_active_mask, _horizon_index].sum())
        _negative_count = float(np.sum(_active_mask) - _positive_count)
        if _positive_count <= 0.0 or _negative_count <= 0.0:
            _pos_weight_list.append(1.0)
            continue

        _pos_weight_list.append(float(np.clip(_negative_count / _positive_count, 0.25, 4.0)))

    return np.asarray(_pos_weight_list, dtype=np.float32)


class MultiHorizonDataset(Dataset):
    def __init__(
        self,
        _sequence_x_array,
        _static_x_array,
        _target_scaled_array,
        _target_raw_array,
        _direction_y_array,
        _direction_active_array,
        _target_scale_array,
        _sample_weight_array,
    ):
        self.sequence_x = torch.tensor(_sequence_x_array, dtype=torch.float32)
        self.static_x = torch.tensor(_static_x_array, dtype=torch.float32)
        self.target_scaled = torch.tensor(_target_scaled_array, dtype=torch.float32)
        self.target_raw = torch.tensor(_target_raw_array, dtype=torch.float32)
        self.direction_y = torch.tensor(_direction_y_array, dtype=torch.float32)
        self.direction_active = torch.tensor(_direction_active_array, dtype=torch.float32)
        self.target_scale = torch.tensor(_target_scale_array, dtype=torch.float32).view(-1, 1)
        self.sample_weight = torch.tensor(_sample_weight_array, dtype=torch.float32)

    def __len__(self):
        return len(self.sequence_x)

    def __getitem__(self, _index):
        return (
            self.sequence_x[_index],
            self.static_x[_index],
            self.target_scaled[_index],
            self.target_raw[_index],
            self.direction_y[_index],
            self.direction_active[_index],
            self.target_scale[_index],
            self.sample_weight[_index],
        )


def build_dataloader(
    _sequence_x,
    _static_x,
    _target_scaled,
    _target_raw,
    _direction_y,
    _direction_active,
    _target_scale,
    _sample_weight,
    _batch_size,
    _shuffle,
):
    _dataset = MultiHorizonDataset(
        _sequence_x,
        _static_x,
        _target_scaled,
        _target_raw,
        _direction_y,
        _direction_active,
        _target_scale,
        _sample_weight,
    )
    return DataLoader(_dataset, batch_size=int(_batch_size), shuffle=bool(_shuffle))


class MixerBlock(nn.Module):
    def __init__(self, _num_tokens, _hidden_size, _dropout):
        super().__init__()
        _token_hidden = max(_num_tokens * 2, 8)
        _channel_hidden = max(_hidden_size * 2, 32)
        self.token_norm = nn.LayerNorm(_hidden_size)
        self.token_mlp = nn.Sequential(
            nn.Linear(_num_tokens, _token_hidden),
            nn.GELU(),
            nn.Dropout(_dropout),
            nn.Linear(_token_hidden, _num_tokens),
            nn.Dropout(_dropout),
        )
        self.channel_norm = nn.LayerNorm(_hidden_size)
        self.channel_mlp = nn.Sequential(
            nn.Linear(_hidden_size, _channel_hidden),
            nn.GELU(),
            nn.Dropout(_dropout),
            nn.Linear(_channel_hidden, _hidden_size),
            nn.Dropout(_dropout),
        )

    def forward(self, _x):
        _token_residual = self.token_norm(_x).transpose(1, 2)
        _token_residual = self.token_mlp(_token_residual).transpose(1, 2)
        _x = _x + _token_residual
        _channel_residual = self.channel_mlp(self.channel_norm(_x))
        return _x + _channel_residual


class MultiHorizonPatchMixer(nn.Module):
    def __init__(
        self,
        _sequence_length,
        _input_size,
        _static_input_size,
        _horizon_count,
        _hidden_size,
        _mixer_layers,
        _patch_length,
        _dropout,
    ):
        super().__init__()

        if int(_sequence_length) % int(_patch_length) != 0:
            raise RuntimeError("sequence_length must be divisible by patch_length")

        self.sequence_length = int(_sequence_length)
        self.patch_length = int(_patch_length)
        self.num_patches = int(_sequence_length) // int(_patch_length)

        self.patch_projection = nn.Linear(int(_patch_length) * int(_input_size), int(_hidden_size))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, int(_hidden_size)))
        self.input_dropout = nn.Dropout(float(_dropout))
        self.blocks = nn.ModuleList(
            [MixerBlock(self.num_patches, int(_hidden_size), float(_dropout)) for _ in range(int(_mixer_layers))]
        )
        self.output_norm = nn.LayerNorm(int(_hidden_size))

        self.static_encoder = None
        self.static_gate = None
        if int(_static_input_size) > 0:
            self.static_encoder = nn.Sequential(
                nn.LayerNorm(int(_static_input_size)),
                nn.Linear(int(_static_input_size), int(_hidden_size)),
                nn.GELU(),
                nn.Dropout(float(_dropout)),
            )
            self.static_gate = nn.Sequential(
                nn.Linear(int(_hidden_size), int(_hidden_size)),
                nn.Sigmoid(),
            )

        _trunk_input_size = int(_hidden_size) * 2 + (int(_hidden_size) if int(_static_input_size) > 0 else 0)
        self.trunk = nn.Sequential(
            nn.Linear(_trunk_input_size, int(_hidden_size) * 2),
            nn.GELU(),
            nn.Dropout(float(_dropout)),
            nn.Linear(int(_hidden_size) * 2, int(_hidden_size)),
            nn.GELU(),
            nn.Dropout(float(_dropout)),
        )
        self.regression_head = nn.Linear(int(_hidden_size), int(_horizon_count))
        self.direction_head = nn.Linear(int(_hidden_size), int(_horizon_count))

    def forward(self, _sequence_x, _static_x):
        _batch_size, _, _feature_count = _sequence_x.shape
        _patches = _sequence_x.reshape(
            _batch_size,
            self.num_patches,
            self.patch_length * _feature_count,
        )
        _patches = self.patch_projection(_patches) + self.position_embedding
        _patches = self.input_dropout(_patches)

        for _block in self.blocks:
            _patches = _block(_patches)

        _patches = self.output_norm(_patches)
        _mean_pool = _patches.mean(dim=1)
        _last_token = _patches[:, -1, :]

        if self.static_encoder is not None and _static_x.shape[-1] > 0:
            _static_context = self.static_encoder(_static_x)
            _mean_pool = _mean_pool * self.static_gate(_static_context)
            _trunk_input = torch.cat([_mean_pool, _last_token, _static_context], dim=1)
        else:
            _trunk_input = torch.cat([_mean_pool, _last_token], dim=1)

        _shared_hidden = self.trunk(_trunk_input)
        _regression = self.regression_head(_shared_hidden)
        _direction_logits = self.direction_head(_shared_hidden)
        return _regression, _direction_logits


def compute_losses(
    _pred_scaled,
    _target_scaled,
    _direction_logits,
    _direction_y,
    _direction_active,
    _sample_weight,
    _huber_beta,
    _direction_loss_weight,
    _pos_weight_tensor,
):
    _regression_matrix = F.smooth_l1_loss(
        _pred_scaled,
        _target_scaled,
        beta=float(_huber_beta),
        reduction="none",
    )
    _regression_per_sample = _regression_matrix.mean(dim=1)
    _sample_weight_sum = _sample_weight.sum().clamp(min=1.0e-6)
    _regression_loss = (_regression_per_sample * _sample_weight).sum() / _sample_weight_sum

    _direction_loss = torch.tensor(0.0, device=_pred_scaled.device)
    _active_count = float(_direction_active.sum().item())
    if float(_direction_loss_weight) > 0.0:
        _direction_matrix = F.binary_cross_entropy_with_logits(
            _direction_logits,
            _direction_y,
            reduction="none",
            pos_weight=_pos_weight_tensor,
        )
        _direction_weight_matrix = _direction_active * _sample_weight.unsqueeze(1)
        _direction_weight_sum = _direction_weight_matrix.sum().clamp(min=1.0e-6)
        _direction_loss = (_direction_matrix * _direction_weight_matrix).sum() / _direction_weight_sum

    _total_loss = _regression_loss + (float(_direction_loss_weight) * _direction_loss)

    return {
        "total_loss": _total_loss,
        "regression_loss": _regression_loss,
        "direction_loss": _direction_loss,
        "active_count": _active_count,
    }


def train_one_epoch(
    _model,
    _loader,
    _optimizer,
    _device,
    _huber_beta,
    _direction_loss_weight,
    _pos_weight_tensor,
    _grad_clip,
):
    _model.train()
    _total_loss_sum = 0.0
    _regression_loss_sum = 0.0
    _direction_loss_sum = 0.0
    _sample_count = 0
    _active_count = 0.0

    for (
        _batch_sequence_x,
        _batch_static_x,
        _batch_target_scaled,
        _,
        _batch_direction_y,
        _batch_direction_active,
        _,
        _batch_sample_weight,
    ) in _loader:
        _batch_sequence_x = _batch_sequence_x.to(_device)
        _batch_static_x = _batch_static_x.to(_device)
        _batch_target_scaled = _batch_target_scaled.to(_device)
        _batch_direction_y = _batch_direction_y.to(_device)
        _batch_direction_active = _batch_direction_active.to(_device)
        _batch_sample_weight = _batch_sample_weight.to(_device)

        _optimizer.zero_grad()
        _pred_scaled, _direction_logits = _model(_batch_sequence_x, _batch_static_x)
        _loss_payload = compute_losses(
            _pred_scaled,
            _batch_target_scaled,
            _direction_logits,
            _batch_direction_y,
            _batch_direction_active,
            _batch_sample_weight,
            _huber_beta,
            _direction_loss_weight,
            _pos_weight_tensor,
        )
        _loss_payload["total_loss"].backward()
        if float(_grad_clip) > 0.0:
            nn.utils.clip_grad_norm_(_model.parameters(), float(_grad_clip))
        _optimizer.step()

        _batch_size = len(_batch_sequence_x)
        _total_loss_sum += float(_loss_payload["total_loss"].item()) * _batch_size
        _regression_loss_sum += float(_loss_payload["regression_loss"].item()) * _batch_size
        _direction_loss_sum += float(_loss_payload["direction_loss"].item()) * _batch_size
        _sample_count += _batch_size
        _active_count += float(_loss_payload["active_count"])

    return {
        "total_loss": _total_loss_sum / max(_sample_count, 1),
        "regression_loss": _regression_loss_sum / max(_sample_count, 1),
        "direction_loss": _direction_loss_sum / max(_sample_count, 1),
        "active_ratio": _active_count / float(max(_sample_count * _pos_weight_tensor.shape[0], 1)),
    }


def predict_loader(_model, _loader, _device):
    _model.eval()

    _true_scaled_list = []
    _pred_scaled_list = []
    _true_raw_list = []
    _direction_y_list = []
    _direction_active_list = []
    _direction_prob_list = []
    _target_scale_list = []
    _sample_weight_list = []

    with torch.no_grad():
        for (
            _batch_sequence_x,
            _batch_static_x,
            _batch_target_scaled,
            _batch_target_raw,
            _batch_direction_y,
            _batch_direction_active,
            _batch_target_scale,
            _batch_sample_weight,
        ) in _loader:
            _batch_sequence_x = _batch_sequence_x.to(_device)
            _batch_static_x = _batch_static_x.to(_device)

            _pred_scaled, _direction_logits = _model(_batch_sequence_x, _batch_static_x)
            _pred_scaled = _pred_scaled.cpu().numpy().astype(np.float32)
            _direction_prob = torch.sigmoid(_direction_logits).cpu().numpy().astype(np.float32)
            _target_scale = _batch_target_scale.cpu().numpy().astype(np.float32)

            _true_scaled_list.append(_batch_target_scaled.cpu().numpy().astype(np.float32))
            _pred_scaled_list.append(_pred_scaled)
            _true_raw_list.append(_batch_target_raw.cpu().numpy().astype(np.float32))
            _direction_y_list.append(_batch_direction_y.cpu().numpy().astype(np.float32))
            _direction_active_list.append(_batch_direction_active.cpu().numpy().astype(np.float32))
            _direction_prob_list.append(_direction_prob)
            _target_scale_list.append(_target_scale.reshape(-1))
            _sample_weight_list.append(_batch_sample_weight.cpu().numpy().astype(np.float32))

    _target_scale_array = np.concatenate(_target_scale_list, axis=0).astype(np.float32)
    return {
        "true_scaled": np.concatenate(_true_scaled_list, axis=0).astype(np.float32),
        "pred_scaled": np.concatenate(_pred_scaled_list, axis=0).astype(np.float32),
        "true_raw": np.concatenate(_true_raw_list, axis=0).astype(np.float32),
        "pred_raw": np.concatenate(_pred_scaled_list, axis=0).astype(np.float32) * _target_scale_array.reshape(-1, 1),
        "direction_y": np.concatenate(_direction_y_list, axis=0).astype(np.float32),
        "direction_active": np.concatenate(_direction_active_list, axis=0).astype(np.float32),
        "direction_prob_up": np.concatenate(_direction_prob_list, axis=0).astype(np.float32),
        "target_scale": _target_scale_array.astype(np.float32),
        "sample_weight": np.concatenate(_sample_weight_list, axis=0).astype(np.float32),
    }


def safe_correlation(_true_array, _pred_array):
    _true_array = np.asarray(_true_array, dtype=np.float32).reshape(-1)
    _pred_array = np.asarray(_pred_array, dtype=np.float32).reshape(-1)

    if len(_true_array) == 0 or len(_pred_array) == 0:
        return 0.0

    if float(np.std(_true_array)) <= 1.0e-8 or float(np.std(_pred_array)) <= 1.0e-8:
        return 0.0

    _corr = float(np.corrcoef(_true_array, _pred_array)[0, 1])
    if not np.isfinite(_corr):
        return 0.0

    return _corr


def compute_regression_metrics(_true_array, _pred_array, _baseline_constant_array=None, _baseline_drift_array=None):
    _true_array = np.asarray(_true_array, dtype=np.float32).reshape(-1)
    _pred_array = np.asarray(_pred_array, dtype=np.float32).reshape(-1)

    if len(_true_array) == 0:
        return {
            "count": 0,
            "mse": 0.0,
            "mae": 0.0,
            "direction_accuracy": 0.0,
            "pred_up_ratio": 0.0,
            "true_up_ratio": 0.0,
            "up_recall": 0.0,
            "down_recall": 0.0,
            "pred_std": 0.0,
            "true_std": 0.0,
            "correlation": 0.0,
            "baseline_constant_mse": None,
            "baseline_constant_mae": None,
            "baseline_constant_direction_accuracy": None,
            "baseline_constant_mae_gain": None,
            "baseline_drift_mse": None,
            "baseline_drift_mae": None,
            "baseline_drift_direction_accuracy": None,
            "baseline_drift_mae_gain": None,
        }

    _pred_error = _pred_array - _true_array
    _true_up_mask = _true_array >= 0.0
    _pred_up_mask = _pred_array >= 0.0
    _true_down_mask = ~_true_up_mask

    _metrics = {
        "count": int(len(_true_array)),
        "mse": float(np.mean(np.square(_pred_error))),
        "mae": float(np.mean(np.abs(_pred_error))),
        "direction_accuracy": float(np.mean(_pred_up_mask == _true_up_mask)),
        "pred_up_ratio": float(np.mean(_pred_up_mask)),
        "true_up_ratio": float(np.mean(_true_up_mask)),
        "up_recall": float(np.mean(_pred_up_mask[_true_up_mask])) if np.any(_true_up_mask) else 0.0,
        "down_recall": float(np.mean((~_pred_up_mask)[_true_down_mask])) if np.any(_true_down_mask) else 0.0,
        "pred_std": float(np.std(_pred_array)),
        "true_std": float(np.std(_true_array)),
        "correlation": safe_correlation(_true_array, _pred_array),
        "baseline_constant_mse": None,
        "baseline_constant_mae": None,
        "baseline_constant_direction_accuracy": None,
        "baseline_constant_mae_gain": None,
        "baseline_drift_mse": None,
        "baseline_drift_mae": None,
        "baseline_drift_direction_accuracy": None,
        "baseline_drift_mae_gain": None,
    }

    if _baseline_constant_array is not None:
        _baseline_constant_array = np.asarray(_baseline_constant_array, dtype=np.float32).reshape(-1)
        _baseline_error = _baseline_constant_array - _true_array
        _metrics["baseline_constant_mse"] = float(np.mean(np.square(_baseline_error)))
        _metrics["baseline_constant_mae"] = float(np.mean(np.abs(_baseline_error)))
        _metrics["baseline_constant_direction_accuracy"] = float(np.mean((_baseline_constant_array >= 0.0) == _true_up_mask))
        _metrics["baseline_constant_mae_gain"] = float(_metrics["baseline_constant_mae"] - _metrics["mae"])

    if _baseline_drift_array is not None:
        _baseline_drift_array = np.asarray(_baseline_drift_array, dtype=np.float32).reshape(-1)
        _baseline_drift_error = _baseline_drift_array - _true_array
        _metrics["baseline_drift_mse"] = float(np.mean(np.square(_baseline_drift_error)))
        _metrics["baseline_drift_mae"] = float(np.mean(np.abs(_baseline_drift_error)))
        _metrics["baseline_drift_direction_accuracy"] = float(np.mean((_baseline_drift_array >= 0.0) == _true_up_mask))
        _metrics["baseline_drift_mae_gain"] = float(_metrics["baseline_drift_mae"] - _metrics["mae"])

    return _metrics


def compute_direction_head_metrics(_direction_y_array, _direction_prob_array, _direction_active_array):
    _direction_y_array = np.asarray(_direction_y_array, dtype=np.float32).reshape(-1)
    _direction_prob_array = np.asarray(_direction_prob_array, dtype=np.float32).reshape(-1)
    _direction_active_array = np.asarray(_direction_active_array, dtype=np.float32).reshape(-1)

    _valid_mask = np.isfinite(_direction_prob_array) & (_direction_active_array >= 0.5)
    if not np.any(_valid_mask):
        return {
            "direction_head_accuracy": None,
            "direction_head_bce": None,
            "direction_head_pred_up_ratio": None,
            "direction_head_up_recall": None,
            "direction_head_down_recall": None,
            "direction_head_eval_count": 0,
            "direction_head_active_ratio": float(np.mean(_direction_active_array >= 0.5)) if len(_direction_active_array) > 0 else 0.0,
        }

    _prob = np.clip(_direction_prob_array[_valid_mask], 1.0e-6, 1.0 - 1.0e-6)
    _true = _direction_y_array[_valid_mask] >= 0.5
    _pred = _prob >= 0.5
    _true_down = ~_true
    _bce = -np.mean((_true.astype(np.float32) * np.log(_prob)) + ((~_true).astype(np.float32) * np.log(1.0 - _prob)))

    return {
        "direction_head_accuracy": float(np.mean(_pred == _true)),
        "direction_head_bce": float(_bce),
        "direction_head_pred_up_ratio": float(np.mean(_pred)),
        "direction_head_up_recall": float(np.mean(_pred[_true])) if np.any(_true) else 0.0,
        "direction_head_down_recall": float(np.mean((~_pred)[_true_down])) if np.any(_true_down) else 0.0,
        "direction_head_eval_count": int(np.sum(_valid_mask)),
        "direction_head_active_ratio": float(np.mean(_direction_active_array >= 0.5)),
    }


def summarize_predictions(
    _true_raw,
    _pred_raw,
    _baseline_constant,
    _baseline_drift,
    _direction_y,
    _direction_prob,
    _direction_active,
    _horizons,
):
    _per_horizon = {}
    for _horizon_index, _horizon_value in enumerate(_horizons):
        _metrics = compute_regression_metrics(
            _true_raw[:, _horizon_index],
            _pred_raw[:, _horizon_index],
            _baseline_constant[:, _horizon_index],
            _baseline_drift[:, _horizon_index],
        )
        _metrics.update(
            compute_direction_head_metrics(
                _direction_y[:, _horizon_index],
                _direction_prob[:, _horizon_index],
                _direction_active[:, _horizon_index],
            )
        )

        _active_mask = _direction_active[:, _horizon_index] >= 0.5
        _metrics["active_case_metrics"] = compute_regression_metrics(
            _true_raw[_active_mask, _horizon_index],
            _pred_raw[_active_mask, _horizon_index],
            _baseline_constant[_active_mask, _horizon_index],
            _baseline_drift[_active_mask, _horizon_index],
        )
        _per_horizon[str(_horizon_value)] = _metrics

    _flattened_metrics = compute_regression_metrics(
        _true_raw.reshape(-1),
        _pred_raw.reshape(-1),
        _baseline_constant.reshape(-1),
        _baseline_drift.reshape(-1),
    )
    _flattened_metrics.update(
        compute_direction_head_metrics(
            _direction_y.reshape(-1),
            _direction_prob.reshape(-1),
            _direction_active.reshape(-1),
        )
    )

    _active_flat_mask = _direction_active.reshape(-1) >= 0.5
    _active_flat_metrics = compute_regression_metrics(
        _true_raw.reshape(-1)[_active_flat_mask],
        _pred_raw.reshape(-1)[_active_flat_mask],
        _baseline_constant.reshape(-1)[_active_flat_mask],
        _baseline_drift.reshape(-1)[_active_flat_mask],
    )

    _aggregate = {
        "flattened_metrics": _flattened_metrics,
        "flattened_active_case_metrics": _active_flat_metrics,
        "mean_mae": float(np.mean([_per_horizon[str(_h)]["mae"] for _h in _horizons])),
        "mean_direction_accuracy": float(np.mean([_per_horizon[str(_h)]["direction_accuracy"] for _h in _horizons])),
        "mean_correlation": float(np.mean([_per_horizon[str(_h)]["correlation"] for _h in _horizons])),
        "mean_baseline_constant_mae_gain": float(np.mean([_per_horizon[str(_h)]["baseline_constant_mae_gain"] for _h in _horizons])),
        "mean_baseline_drift_mae_gain": float(np.mean([_per_horizon[str(_h)]["baseline_drift_mae_gain"] for _h in _horizons])),
    }

    return {
        "aggregate": _aggregate,
        "per_horizon": _per_horizon,
    }


def evaluate_model(
    _model,
    _loader,
    _device,
    _baseline_constant,
    _baseline_drift,
    _horizons,
):
    _prediction_payload = predict_loader(_model, _loader, _device)
    _prediction_payload["summary"] = summarize_predictions(
        _prediction_payload["true_raw"],
        _prediction_payload["pred_raw"],
        _baseline_constant,
        _baseline_drift,
        _prediction_payload["direction_y"],
        _prediction_payload["direction_prob_up"],
        _prediction_payload["direction_active"],
        _horizons,
    )
    return _prediction_payload


def format_datetime_series(_series):
    _timestamp_series = pd.to_datetime(_series, errors="coerce")
    return _timestamp_series.dt.strftime("%Y-%m-%d %H:%M:%S")


def build_prediction_dataframe(
    _context_df,
    _prediction_payload,
    _baseline_constant,
    _baseline_drift,
    _horizons,
):
    _prediction_df = _context_df.copy().reset_index(drop=True)
    _prediction_df["timestamp_jst"] = format_datetime_series(_prediction_df["timestamp_jst"])
    if "target_hours" in _prediction_df.columns:
        _prediction_df["target_hours"] = _prediction_df["target_hours"].astype(str)

    for _horizon_index, _horizon_value in enumerate(_horizons):
        _future_timestamp_column = f"future_timestamp_t_plus_{_horizon_value}_jst"
        if _future_timestamp_column in _prediction_df.columns:
            _prediction_df[_future_timestamp_column] = format_datetime_series(_prediction_df[_future_timestamp_column])

        _true_delta = _prediction_payload["true_raw"][:, _horizon_index]
        _pred_delta = _prediction_payload["pred_raw"][:, _horizon_index]
        _direction_prob = _prediction_payload["direction_prob_up"][:, _horizon_index]
        _direction_active = _prediction_payload["direction_active"][:, _horizon_index]
        _direction_target = _prediction_payload["direction_y"][:, _horizon_index]

        _prediction_df[f"true_delta_t_plus_{_horizon_value}"] = _true_delta
        _prediction_df[f"pred_delta_t_plus_{_horizon_value}"] = _pred_delta
        _prediction_df[f"baseline_constant_pred_delta_t_plus_{_horizon_value}"] = _baseline_constant[:, _horizon_index]
        _prediction_df[f"baseline_drift_pred_delta_t_plus_{_horizon_value}"] = _baseline_drift[:, _horizon_index]
        _prediction_df[f"direction_prob_up_t_plus_{_horizon_value}"] = _direction_prob
        _prediction_df[f"direction_active_t_plus_{_horizon_value}"] = _direction_active
        _prediction_df[f"direction_target_t_plus_{_horizon_value}"] = _direction_target
        _prediction_df[f"abs_error_t_plus_{_horizon_value}"] = np.abs(_pred_delta - _true_delta)
        _prediction_df[f"baseline_constant_abs_error_t_plus_{_horizon_value}"] = np.abs(_baseline_constant[:, _horizon_index] - _true_delta)
        _prediction_df[f"baseline_drift_abs_error_t_plus_{_horizon_value}"] = np.abs(_baseline_drift[:, _horizon_index] - _true_delta)
        _prediction_df[f"true_direction_t_plus_{_horizon_value}"] = np.where(_true_delta >= 0.0, "UP", "DOWN")
        _prediction_df[f"pred_direction_t_plus_{_horizon_value}"] = np.where(_pred_delta >= 0.0, "UP", "DOWN")
        _prediction_df[f"direction_head_t_plus_{_horizon_value}"] = np.where(_direction_prob >= 0.5, "UP", "DOWN")

    _abs_error_columns = [f"abs_error_t_plus_{_horizon_value}" for _horizon_value in _horizons]
    _prediction_df["mean_abs_error"] = _prediction_df[_abs_error_columns].mean(axis=1)
    _prediction_df["target_scale"] = _prediction_payload["target_scale"]
    _prediction_df["sample_weight"] = _prediction_payload["sample_weight"]
    return _prediction_df


def save_model(_model, _output_path):
    EnsureParentDirectory(_output_path)
    torch.save(_model.state_dict(), _output_path)


def save_json(_output_path, _payload):
    EnsureParentDirectory(_output_path)
    with open(_output_path, "w", encoding="utf-8") as _file:
        json.dump(_payload, _file, ensure_ascii=False, indent=2)
        _file.write("\n")


def save_prediction_csv(_output_path, _prediction_df):
    EnsureParentDirectory(_output_path)
    _prediction_df.to_csv(_output_path, index=False, encoding="utf-8")


def print_summary(_summary_payload):
    _aggregate = _summary_payload["test_summary"]["aggregate"]["flattened_metrics"]
    print("========== Multi-Horizon H1 Forecast Summary ==========")
    print(f"record_count={_summary_payload['test_record_count']}")
    print(f"flattened_mae={_aggregate['mae']:.8f}")
    print(f"flattened_baseline_constant_mae={float(_aggregate['baseline_constant_mae'] or 0.0):.8f}")
    print(f"flattened_baseline_constant_mae_gain={float(_aggregate['baseline_constant_mae_gain'] or 0.0):.8f}")
    print(f"flattened_baseline_drift_mae={float(_aggregate['baseline_drift_mae'] or 0.0):.8f}")
    print(f"flattened_baseline_drift_mae_gain={float(_aggregate['baseline_drift_mae_gain'] or 0.0):.8f}")
    print(f"flattened_direction_accuracy={_aggregate['direction_accuracy']:.4f}")
    print(f"flattened_correlation={_aggregate['correlation']:.4f}")

    for _horizon_value in _summary_payload["horizons"]:
        _metrics = _summary_payload["test_summary"]["per_horizon"][str(_horizon_value)]
        print(
            f"horizon={_horizon_value} "
            f"mae={_metrics['mae']:.8f} "
            f"baseline_constant_mae={float(_metrics['baseline_constant_mae'] or 0.0):.8f} "
            f"baseline_constant_gain={float(_metrics['baseline_constant_mae_gain'] or 0.0):.8f} "
            f"dir_acc={_metrics['direction_accuracy']:.4f} "
            f"corr={_metrics['correlation']:.4f}"
        )


def main():
    _args = parse_args()
    set_seed(int(_args.seed))

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _horizons = parse_horizon_list(_args.horizons)

    print("========== Multi-Horizon H1 Forecaster Train Start ==========")
    print(f"[INFO] cwd={Path.cwd()}")
    print(f"[INFO] device={_device}")
    print(f"[INFO] horizons={','.join(str(_horizon) for _horizon in _horizons)}")

    _df, _dataset_path = load_dataset(_args.dataset)
    _sequence_length = get_sequence_length(_df)
    _df = ensure_multihorizon_targets(_df, _horizons)

    _feature_payload = build_feature_payload(
        _df,
        _sequence_length,
        _horizons,
        int(_args.target_scale_lookback),
    )
    _sequence_x = _feature_payload["sequence_x"]
    _static_x = _feature_payload["static_x"]
    _target_raw = _feature_payload["target_raw"]
    _target_scale = _feature_payload["target_scale"]
    _drift_baseline = _feature_payload["drift_baseline"]

    _train_end, _valid_end = get_split_boundaries(len(_df))

    _sequence_x_train = _sequence_x[:_train_end]
    _sequence_x_valid = _sequence_x[_train_end:_valid_end]
    _sequence_x_test = _sequence_x[_valid_end:]
    _static_x_train = _static_x[:_train_end]
    _static_x_valid = _static_x[_train_end:_valid_end]
    _static_x_test = _static_x[_valid_end:]
    _target_raw_train = _target_raw[:_train_end]
    _target_raw_valid = _target_raw[_train_end:_valid_end]
    _target_raw_test = _target_raw[_valid_end:]
    _target_scale_train = _target_scale[:_train_end]
    _target_scale_valid = _target_scale[_train_end:_valid_end]
    _target_scale_test = _target_scale[_valid_end:]
    _drift_baseline_valid = _drift_baseline[_train_end:_valid_end]
    _drift_baseline_test = _drift_baseline[_valid_end:]

    _threshold_array = fit_active_thresholds(
        _target_raw_train,
        _target_scale_train,
        float(_args.active_quantile),
    )
    _direction_y_train, _direction_active_train, _target_scaled_train = build_direction_arrays(
        _target_raw_train,
        _target_scale_train,
        _threshold_array,
    )
    _direction_y_valid, _direction_active_valid, _target_scaled_valid = build_direction_arrays(
        _target_raw_valid,
        _target_scale_valid,
        _threshold_array,
    )
    _direction_y_test, _direction_active_test, _target_scaled_test = build_direction_arrays(
        _target_raw_test,
        _target_scale_test,
        _threshold_array,
    )

    _sample_weight_train = build_sample_weight_array(
        _target_raw_train,
        _target_scale_train,
        _threshold_array,
        float(_args.active_weight),
    )
    _sample_weight_valid = build_sample_weight_array(
        _target_raw_valid,
        _target_scale_valid,
        _threshold_array,
        float(_args.active_weight),
    )
    _sample_weight_test = build_sample_weight_array(
        _target_raw_test,
        _target_scale_test,
        _threshold_array,
        float(_args.active_weight),
    )

    _sequence_mean, _sequence_std = fit_feature_scaler(_sequence_x_train)
    _static_mean, _static_std = fit_feature_scaler(_static_x_train)
    _sequence_x_train = transform_feature_array(_sequence_x_train, _sequence_mean, _sequence_std)
    _sequence_x_valid = transform_feature_array(_sequence_x_valid, _sequence_mean, _sequence_std)
    _sequence_x_test = transform_feature_array(_sequence_x_test, _sequence_mean, _sequence_std)
    _static_x_train = transform_feature_array(_static_x_train, _static_mean, _static_std)
    _static_x_valid = transform_feature_array(_static_x_valid, _static_mean, _static_std)
    _static_x_test = transform_feature_array(_static_x_test, _static_mean, _static_std)

    _constant_baseline_train = np.mean(_target_raw_train, axis=0, keepdims=True).astype(np.float32)
    _constant_baseline_valid = np.repeat(_constant_baseline_train, len(_target_raw_valid), axis=0).astype(np.float32)
    _constant_baseline_test = np.repeat(_constant_baseline_train, len(_target_raw_test), axis=0).astype(np.float32)

    _train_loader = build_dataloader(
        _sequence_x_train,
        _static_x_train,
        _target_scaled_train,
        _target_raw_train,
        _direction_y_train,
        _direction_active_train,
        _target_scale_train,
        _sample_weight_train,
        int(_args.batch_size),
        True,
    )
    _valid_loader = build_dataloader(
        _sequence_x_valid,
        _static_x_valid,
        _target_scaled_valid,
        _target_raw_valid,
        _direction_y_valid,
        _direction_active_valid,
        _target_scale_valid,
        _sample_weight_valid,
        int(_args.batch_size),
        False,
    )
    _test_loader = build_dataloader(
        _sequence_x_test,
        _static_x_test,
        _target_scaled_test,
        _target_raw_test,
        _direction_y_test,
        _direction_active_test,
        _target_scale_test,
        _sample_weight_test,
        int(_args.batch_size),
        False,
    )

    _model = MultiHorizonPatchMixer(
        _sequence_length=_sequence_length,
        _input_size=_sequence_x_train.shape[-1],
        _static_input_size=_static_x_train.shape[-1],
        _horizon_count=len(_horizons),
        _hidden_size=int(_args.hidden_size),
        _mixer_layers=int(_args.mixer_layers),
        _patch_length=int(_args.patch_length),
        _dropout=float(_args.dropout),
    ).to(_device)
    _optimizer = torch.optim.AdamW(
        _model.parameters(),
        lr=float(_args.learning_rate),
        weight_decay=float(_args.weight_decay),
    )
    _pos_weight_tensor = torch.tensor(
        build_pos_weight_array(_direction_y_train, _direction_active_train),
        dtype=torch.float32,
        device=_device,
    )

    _best_valid_score = None
    _best_epoch = None
    _best_state_dict = None
    _best_valid_summary = None
    _no_improve_count = 0
    _history_rows = []

    print(f"[INFO] dataset={_dataset_path}")
    print(f"[INFO] dataset_rows={len(_df)}")
    print(f"[INFO] sequence_length={_sequence_length}")
    print(f"[INFO] sequence_feature_count={len(_feature_payload['sequence_feature_names'])}")
    print(f"[INFO] static_feature_count={len(_feature_payload['static_feature_names'])}")
    print(f"[INFO] train_size={len(_sequence_x_train)}")
    print(f"[INFO] valid_size={len(_sequence_x_valid)}")
    print(f"[INFO] test_size={len(_sequence_x_test)}")
    print(f"[INFO] target_scale_mean_train={float(np.mean(_target_scale_train)):.8f}")
    print(f"[INFO] active_thresholds={','.join(f'{float(_value):.6f}' for _value in _threshold_array.tolist())}")
    print(f"[INFO] direction_active_ratio_train={float(np.mean(_direction_active_train)):.4f}")
    print(f"[INFO] direction_active_ratio_valid={float(np.mean(_direction_active_valid)):.4f}")
    print(f"[INFO] direction_active_ratio_test={float(np.mean(_direction_active_test)):.4f}")
    print(f"[INFO] constant_baseline_train={','.join(f'{float(_value):.6f}' for _value in _constant_baseline_train.reshape(-1).tolist())}")
    print(f"[INFO] direction_pos_weight={','.join(f'{float(_value):.4f}' for _value in _pos_weight_tensor.cpu().tolist())}")

    for _epoch in range(1, int(_args.epochs) + 1):
        _train_metrics = train_one_epoch(
            _model,
            _train_loader,
            _optimizer,
            _device,
            float(_args.huber_beta),
            float(_args.direction_loss_weight),
            _pos_weight_tensor,
            float(_args.grad_clip),
        )
        _valid_payload = evaluate_model(
            _model,
            _valid_loader,
            _device,
            _constant_baseline_valid,
            _drift_baseline_valid,
            _horizons,
        )
        _valid_flat_metrics = _valid_payload["summary"]["aggregate"]["flattened_metrics"]
        _valid_score = float(_valid_flat_metrics["mae"])

        _history_rows.append(
            {
                "epoch": int(_epoch),
                "train_total_loss": float(_train_metrics["total_loss"]),
                "train_regression_loss": float(_train_metrics["regression_loss"]),
                "train_direction_loss": float(_train_metrics["direction_loss"]),
                "valid_flattened_mae": float(_valid_flat_metrics["mae"]),
                "valid_flattened_dir_acc": float(_valid_flat_metrics["direction_accuracy"]),
                "valid_flattened_corr": float(_valid_flat_metrics["correlation"]),
                "valid_mean_baseline_constant_mae_gain": float(_valid_payload["summary"]["aggregate"]["mean_baseline_constant_mae_gain"]),
            }
        )

        _is_improved = _best_valid_score is None or _valid_score < _best_valid_score
        if _is_improved:
            _best_valid_score = _valid_score
            _best_epoch = _epoch
            _best_state_dict = {key: value.cpu().clone() for key, value in _model.state_dict().items()}
            _best_valid_summary = _valid_payload["summary"]
            _no_improve_count = 0
        else:
            _no_improve_count += 1

        if bool(_args.verbose):
            print(
                f"[INFO] epoch={_epoch} "
                f"train_total_loss={_train_metrics['total_loss']:.8f} "
                f"train_regression_loss={_train_metrics['regression_loss']:.8f} "
                f"train_direction_loss={_train_metrics['direction_loss']:.8f} "
                f"valid_flattened_mae={_valid_flat_metrics['mae']:.8f} "
                f"valid_baseline_constant_gain={float(_valid_flat_metrics['baseline_constant_mae_gain'] or 0.0):.8f} "
                f"valid_dir_acc={_valid_flat_metrics['direction_accuracy']:.4f} "
                f"valid_corr={_valid_flat_metrics['correlation']:.4f}"
            )

        if _no_improve_count >= int(_args.patience):
            print(f"[INFO] early_stopped=1 best_epoch={_best_epoch} patience={int(_args.patience)}")
            break

    if _best_state_dict is None:
        raise RuntimeError("Training finished without a best model state")

    _model.load_state_dict(_best_state_dict)

    _test_payload = evaluate_model(
        _model,
        _test_loader,
        _device,
        _constant_baseline_test,
        _drift_baseline_test,
        _horizons,
    )
    _test_context_df = _df.iloc[_valid_end:].copy().reset_index(drop=True)
    _prediction_df = build_prediction_dataframe(
        _test_context_df,
        _test_payload,
        _constant_baseline_test,
        _drift_baseline_test,
        _horizons,
    )

    _summary_payload = {
        "dataset": str(_dataset_path),
        "horizons": [int(_horizon) for _horizon in _horizons],
        "sequence_length": int(_sequence_length),
        "train_record_count": int(len(_sequence_x_train)),
        "valid_record_count": int(len(_sequence_x_valid)),
        "test_record_count": int(len(_sequence_x_test)),
        "best_epoch": int(_best_epoch),
        "best_valid_score": float(_best_valid_score),
        "active_thresholds": [float(_value) for _value in _threshold_array.tolist()],
        "direction_pos_weight": [float(_value) for _value in _pos_weight_tensor.cpu().tolist()],
        "constant_baseline_train_mean": [float(_value) for _value in _constant_baseline_train.reshape(-1).tolist()],
        "validation_summary": _best_valid_summary,
        "test_summary": _test_payload["summary"],
        "history": _history_rows,
    }

    _metadata_payload = {
        "dataset": str(_dataset_path),
        "horizons": [int(_horizon) for _horizon in _horizons],
        "sequence_length": int(_sequence_length),
        "sequence_feature_names": list(_feature_payload["sequence_feature_names"]),
        "static_feature_names": list(_feature_payload["static_feature_names"]),
        "sequence_feature_mean": [float(_value) for _value in _sequence_mean.tolist()],
        "sequence_feature_std": [float(_value) for _value in _sequence_std.tolist()],
        "static_feature_mean": [float(_value) for _value in _static_mean.tolist()],
        "static_feature_std": [float(_value) for _value in _static_std.tolist()],
        "target_scale_lookback": int(_args.target_scale_lookback),
        "target_scale_floor": float(MIN_TARGET_SCALE),
        "active_thresholds": [float(_value) for _value in _threshold_array.tolist()],
        "direction_pos_weight": [float(_value) for _value in _pos_weight_tensor.cpu().tolist()],
        "hidden_size": int(_args.hidden_size),
        "mixer_layers": int(_args.mixer_layers),
        "patch_length": int(_args.patch_length),
        "dropout": float(_args.dropout),
        "learning_rate": float(_args.learning_rate),
        "weight_decay": float(_args.weight_decay),
        "batch_size": int(_args.batch_size),
        "epochs": int(_args.epochs),
        "huber_beta": float(_args.huber_beta),
        "active_quantile": float(_args.active_quantile),
        "active_weight": float(_args.active_weight),
        "direction_loss_weight": float(_args.direction_loss_weight),
        "grad_clip": float(_args.grad_clip),
        "seed": int(_args.seed),
        "best_epoch": int(_best_epoch),
    }

    save_model(_model, _args.model_output)
    save_json(_args.metadata_output, _metadata_payload)
    save_prediction_csv(_args.prediction_output, _prediction_df)
    save_json(_args.summary_output, _summary_payload)

    print(f"[INFO] model_saved={Path(_args.model_output).resolve()}")
    print(f"[INFO] metadata_saved={Path(_args.metadata_output).resolve()}")
    print(f"[INFO] prediction_saved={Path(_args.prediction_output).resolve()}")
    print(f"[INFO] summary_saved={Path(_args.summary_output).resolve()}")
    print_summary(_summary_payload)
    print("========== Multi-Horizon H1 Forecaster Train End ==========")


if __name__ == "__main__":
    main()
