# --------------------------------------------------
# LSTMModel.py
# 役割:
#   H1 multi-horizon patch mixer モデルの runtime 推論を担う
#
# 現段階:
#   学習済み metadata / model を読み込み、
#   runtime 側で特徴量を再構成して predicted_path を返す
# --------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Framework.Utility.Utility import Clamp01 as _clamp_01, JST, ParseJSTDateTime


BACKEND_NAME = "h1_multi_horizon_patch_mixer"
MIN_TARGET_SCALE = 1.0e-6
MAGNITUDE_CONFIDENCE_SCALE = 0.12
MAGNITUDE_CONFIDENCE_WEIGHT = 0.40
DOMINANCE_CONFIDENCE_WEIGHT = 0.60
SIGNAL_EPSILON = 1.0e-8
INDICATOR_WARMUP_BARS = 50

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_LEGACY_MODEL_PATH = _REPO_ROOT / "Asset/Models/h1_multi_horizon_patch_mixer.pt"
_LEGACY_METADATA_PATH = _REPO_ROOT / "Asset/Models/h1_multi_horizon_patch_mixer_metadata.json"
_ACTIVE_RUNTIME_POINTER_PATH = _REPO_ROOT / "Asset/Models/h1/active_runtime.json"

_RUNTIME_CACHE = None


class MixerBlock(nn.Module):
    def __init__(self, _num_tokens, _hidden_size, _dropout):
        super().__init__()
        _token_hidden = max(int(_num_tokens) * 2, 8)
        _channel_hidden = max(int(_hidden_size) * 2, 32)
        self.token_norm = nn.LayerNorm(int(_hidden_size))
        self.token_mlp = nn.Sequential(
            nn.Linear(int(_num_tokens), _token_hidden),
            nn.GELU(),
            nn.Dropout(float(_dropout)),
            nn.Linear(_token_hidden, int(_num_tokens)),
            nn.Dropout(float(_dropout)),
        )
        self.channel_norm = nn.LayerNorm(int(_hidden_size))
        self.channel_mlp = nn.Sequential(
            nn.Linear(int(_hidden_size), _channel_hidden),
            nn.GELU(),
            nn.Dropout(float(_dropout)),
            nn.Linear(_channel_hidden, int(_hidden_size)),
            nn.Dropout(float(_dropout)),
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


@dataclass
class ForecastRuntime:
    metadata: dict
    model: MultiHorizonPatchMixer
    device: torch.device
    artifact_info: dict

def _safe_divide_array(_numerator_array, _denominator_array, _default_value):
    _numerator_array = np.asarray(_numerator_array, dtype=np.float32)
    _denominator_array = np.asarray(_denominator_array, dtype=np.float32)
    _result = np.full_like(_numerator_array, float(_default_value), dtype=np.float32)
    _valid_mask = np.abs(_denominator_array) > 1.0e-8
    np.divide(_numerator_array, _denominator_array, out=_result, where=_valid_mask)
    return _result.astype(np.float32)


def _transform_feature_array(_x, _mean, _std):
    _x = np.asarray(_x, dtype=np.float32)
    _mean = np.asarray(_mean, dtype=np.float32)
    _std = np.asarray(_std, dtype=np.float32)
    _safe_std = np.where(np.abs(_std) < 1.0e-6, 1.0, _std).astype(np.float32)
    return ((_x - _mean) / _safe_std).astype(np.float32)


def _calc_rsi(_close_series, _period=14):
    _delta = _close_series.diff()
    _gain = _delta.clip(lower=0.0)
    _loss = -_delta.clip(upper=0.0)

    _avg_gain = _gain.rolling(window=int(_period), min_periods=int(_period)).mean()
    _avg_loss = _loss.rolling(window=int(_period), min_periods=int(_period)).mean()

    _rs = _avg_gain.div(_avg_loss.where(_avg_loss != 0.0))
    _rsi = 100.0 - (100.0 / (1.0 + _rs))

    _both_zero_mask = (_avg_gain == 0.0) & (_avg_loss == 0.0)
    _up_only_mask = (_avg_gain > 0.0) & (_avg_loss == 0.0)
    _down_only_mask = (_avg_gain == 0.0) & (_avg_loss > 0.0)

    _rsi = _rsi.mask(_both_zero_mask, 50.0)
    _rsi = _rsi.mask(_up_only_mask, 100.0)
    _rsi = _rsi.mask(_down_only_mask, 0.0)
    return _rsi.fillna(50.0)


def _calc_macd(_close_series):
    _ema_fast = _close_series.ewm(span=12, adjust=False).mean()
    _ema_slow = _close_series.ewm(span=26, adjust=False).mean()
    _macd = _ema_fast - _ema_slow
    _macd_signal = _macd.ewm(span=9, adjust=False).mean()
    _macd_hist = _macd - _macd_signal
    return _macd, _macd_signal, _macd_hist


def _build_trend_consistency(_diff_window):
    _up_count = (_diff_window > 0.0).sum(axis=1)
    _down_count = (_diff_window < 0.0).sum(axis=1)
    _dominant = np.maximum(_up_count, _down_count)
    _denominator = max(_diff_window.shape[1], 1)
    return (_dominant.astype(np.float32) / float(_denominator)).astype(np.float32)


def _build_close_position_feature(_close_array, _high_array, _low_array, _lookback):
    _lookback = int(min(_lookback, _close_array.shape[1]))
    _window_high = _high_array[:, -_lookback:].max(axis=1)
    _window_low = _low_array[:, -_lookback:].min(axis=1)
    return _safe_divide_array(_close_array[:, -1] - _window_low, _window_high - _window_low, 0.5)


def _extract_rate_value(_row, _field_name, _field_index):
    if isinstance(_row, dict):
        return float(_row.get(_field_name, 0.0))

    _dtype = getattr(_row, "dtype", None)
    _field_names = getattr(_dtype, "names", None)
    if _field_names and _field_name in _field_names:
        return float(_row[_field_name])

    return float(_row[_field_index])


def _normalize_rate_timestamp(_value):
    if _value is None:
        return None

    if isinstance(_value, pd.Timestamp):
        if _value.tzinfo is None:
            return _value.tz_localize(JST)
        return _value.tz_convert(JST)

    if isinstance(_value, (int, float, np.integer, np.floating)):
        return pd.to_datetime(float(_value), unit="s", utc=True).tz_convert(JST)

    _parsed = ParseJSTDateTime(_value)
    if _parsed is None:
        return None

    return pd.Timestamp(_parsed)


def _extract_rate_timestamp(_row):
    if isinstance(_row, dict):
        for _key in ["time", "timestamp", "datetime", "timestamp_jst"]:
            if _key in _row and _row.get(_key) is not None:
                return _normalize_rate_timestamp(_row.get(_key))

    _dtype = getattr(_row, "dtype", None)
    _field_names = getattr(_dtype, "names", None)
    if _field_names:
        for _key in ["time", "timestamp", "datetime", "timestamp_jst"]:
            if _key in _field_names:
                return _normalize_rate_timestamp(_row[_key])

    try:
        return _normalize_rate_timestamp(_row[0])
    except Exception:
        return None


def _build_history_dataframe(_ohlc, _anchor_timestamp_jst=""):
    _records = []

    for _row in _ohlc:
        _records.append(
            {
                "timestamp": _extract_rate_timestamp(_row),
                "open": _extract_rate_value(_row, "open", 1),
                "high": _extract_rate_value(_row, "high", 2),
                "low": _extract_rate_value(_row, "low", 3),
                "close": _extract_rate_value(_row, "close", 4),
            }
        )

    _history_df = pd.DataFrame(_records)
    if len(_history_df) == 0:
        return _history_df

    if _history_df["timestamp"].isna().all():
        _anchor_timestamp = ParseJSTDateTime(_anchor_timestamp_jst)
        if _anchor_timestamp is not None:
            _anchor_timestamp = pd.Timestamp(_anchor_timestamp).floor("h")
            _history_df["timestamp"] = pd.date_range(
                end=_anchor_timestamp,
                periods=len(_history_df),
                freq="h",
                tz=JST,
            )

    _history_df = _history_df.dropna(subset=["timestamp"]).copy()
    _history_df["timestamp"] = pd.to_datetime(_history_df["timestamp"], errors="coerce")
    _history_df = _history_df.dropna(subset=["timestamp"]).copy()
    _history_df = _history_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return _history_df


def _build_feature_arrays(_history_df, _metadata):
    _sequence_length = int(_metadata["sequence_length"])
    _horizons = [int(_value) for _value in _metadata["horizons"]]
    _target_scale_lookback = int(_metadata.get("target_scale_lookback", 12))
    _minimum_history_length = int(_sequence_length) + int(INDICATOR_WARMUP_BARS) - 1

    if len(_history_df) < _minimum_history_length:
        raise RuntimeError(f"At least {_minimum_history_length} H1 bars are required for inference")

    _feature_df = _history_df.copy().sort_values("timestamp").reset_index(drop=True)
    _feature_df["sma_20"] = _feature_df["close"].rolling(window=20, min_periods=20).mean()
    _feature_df["sma_50"] = _feature_df["close"].rolling(window=50, min_periods=50).mean()
    _feature_df["rsi_14"] = _calc_rsi(_feature_df["close"], 14)
    _macd, _, _macd_hist = _calc_macd(_feature_df["close"])
    _feature_df["macd"] = _macd
    _feature_df["macd_hist"] = _macd_hist
    _feature_df["ma_gap_sma20"] = _feature_df["close"] - _feature_df["sma_20"]
    _feature_df["ma_gap_sma50"] = _feature_df["close"] - _feature_df["sma_50"]

    _window_df = _feature_df.iloc[-_sequence_length:].copy().reset_index(drop=True)
    _required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "rsi_14",
        "macd",
        "macd_hist",
        "ma_gap_sma20",
        "ma_gap_sma50",
    ]
    if _window_df[_required_columns].isnull().any().any():
        raise RuntimeError("H1 feature window contains NaN values")

    _open_array = _window_df["open"].to_numpy(dtype=np.float32).reshape(1, -1)
    _high_array = _window_df["high"].to_numpy(dtype=np.float32).reshape(1, -1)
    _low_array = _window_df["low"].to_numpy(dtype=np.float32).reshape(1, -1)
    _close_array = _window_df["close"].to_numpy(dtype=np.float32).reshape(1, -1)
    _rsi_array = _window_df["rsi_14"].to_numpy(dtype=np.float32).reshape(1, -1)
    _macd_array = _window_df["macd"].to_numpy(dtype=np.float32).reshape(1, -1)
    _macd_hist_array = _window_df["macd_hist"].to_numpy(dtype=np.float32).reshape(1, -1)
    _ma_gap20_array = _window_df["ma_gap_sma20"].to_numpy(dtype=np.float32).reshape(1, -1)
    _ma_gap50_array = _window_df["ma_gap_sma50"].to_numpy(dtype=np.float32).reshape(1, -1)

    _range_array = (_high_array - _low_array).astype(np.float32)
    _body_array = (_close_array - _open_array).astype(np.float32)
    _upper_wick_array = (_high_array - np.maximum(_open_array, _close_array)).astype(np.float32)
    _lower_wick_array = (np.minimum(_open_array, _close_array) - _low_array).astype(np.float32)

    _close_prev_array = np.concatenate([_close_array[:, :1], _close_array[:, :-1]], axis=1)
    _close_return_array = _safe_divide_array(_close_array - _close_prev_array, _close_prev_array, 0.0)
    _close_return_array[:, 0] = 0.0

    _target_scale_lookback = max(1, min(_target_scale_lookback, _sequence_length))
    _recent_scale_array = _range_array[:, -_target_scale_lookback:].mean(axis=1).astype(np.float32)
    _recent_scale_array = np.maximum(_recent_scale_array, np.float32(MIN_TARGET_SCALE))
    _recent_scale_matrix = _recent_scale_array.reshape(-1, 1)

    _last_close_array = _close_array[:, -1:].astype(np.float32)
    _last_close_array = np.where(np.abs(_last_close_array) < 1.0e-8, 1.0, _last_close_array).astype(np.float32)

    _window_timestamp_series = pd.to_datetime(_window_df["timestamp"], errors="coerce")
    _hour_array = _window_timestamp_series.dt.hour.to_numpy(dtype=np.float32).reshape(1, -1)
    _weekday_array = _window_timestamp_series.dt.weekday.to_numpy(dtype=np.float32).reshape(1, -1)
    _hour_sin_array = np.sin((2.0 * np.pi * _hour_array) / 24.0).astype(np.float32)
    _hour_cos_array = np.cos((2.0 * np.pi * _hour_array) / 24.0).astype(np.float32)
    _weekday_sin_array = np.sin((2.0 * np.pi * _weekday_array) / 7.0).astype(np.float32)
    _weekday_cos_array = np.cos((2.0 * np.pi * _weekday_array) / 7.0).astype(np.float32)

    _sequence_x = np.stack(
        [
            _safe_divide_array(_open_array - _last_close_array, _last_close_array, 0.0),
            _safe_divide_array(_high_array - _last_close_array, _last_close_array, 0.0),
            _safe_divide_array(_low_array - _last_close_array, _last_close_array, 0.0),
            _safe_divide_array(_close_array - _last_close_array, _last_close_array, 0.0),
            _safe_divide_array(_body_array, _recent_scale_matrix, 0.0),
            _safe_divide_array(_range_array, _recent_scale_matrix, 0.0),
            _safe_divide_array(_upper_wick_array, _recent_scale_matrix, 0.0),
            _safe_divide_array(_lower_wick_array, _recent_scale_matrix, 0.0),
            _close_return_array,
            ((_rsi_array / 100.0) - 0.5).astype(np.float32),
            _safe_divide_array(_macd_array, _recent_scale_matrix, 0.0),
            _safe_divide_array(_macd_hist_array, _recent_scale_matrix, 0.0),
            _safe_divide_array(_ma_gap20_array, _recent_scale_matrix, 0.0),
            _safe_divide_array(_ma_gap50_array, _recent_scale_matrix, 0.0),
            _hour_sin_array,
            _hour_cos_array,
            _weekday_sin_array,
            _weekday_cos_array,
        ],
        axis=-1,
    ).astype(np.float32)

    _diff_array = np.diff(_close_array, axis=1).astype(np.float32)
    _return_mean_6 = _close_return_array[:, -6:].mean(axis=1)
    _return_mean_12 = _close_return_array[:, -12:].mean(axis=1)
    _return_mean_24 = _close_return_array[:, -24:].mean(axis=1)
    _return_std_6 = _close_return_array[:, -6:].std(axis=1)
    _return_std_12 = _close_return_array[:, -12:].std(axis=1)
    _return_std_24 = _close_return_array[:, -24:].std(axis=1)
    _range_mean_6 = _safe_divide_array(_range_array[:, -6:].mean(axis=1), _recent_scale_array, 0.0)
    _range_mean_12 = _safe_divide_array(_range_array[:, -12:].mean(axis=1), _recent_scale_array, 0.0)
    _range_mean_24 = _safe_divide_array(_range_array[:, -24:].mean(axis=1), _recent_scale_array, 0.0)
    _range_ratio_last_6 = _safe_divide_array(_range_array[:, -1], _range_array[:, -6:].mean(axis=1), 1.0)
    _range_ratio_last_12 = _safe_divide_array(_range_array[:, -1], _range_array[:, -12:].mean(axis=1), 1.0)
    _range_ratio_last_24 = _safe_divide_array(_range_array[:, -1], _range_array[:, -24:].mean(axis=1), 1.0)
    _slope_6 = _safe_divide_array(_close_array[:, -1] - _close_array[:, -6], _recent_scale_array, 0.0)
    _slope_12 = _safe_divide_array(_close_array[:, -1] - _close_array[:, -12], _recent_scale_array, 0.0)
    _slope_24 = _safe_divide_array(_close_array[:, -1] - _close_array[:, -24], _recent_scale_array, 0.0)
    _close_position_12 = _build_close_position_feature(_close_array, _high_array, _low_array, 12)
    _close_position_24 = _build_close_position_feature(_close_array, _high_array, _low_array, 24)
    _close_position_full = _build_close_position_feature(_close_array, _high_array, _low_array, _sequence_length)
    _up_ratio_6 = (_diff_array[:, -5:] > 0.0).mean(axis=1).astype(np.float32)
    _up_ratio_12 = (_diff_array[:, -11:] > 0.0).mean(axis=1).astype(np.float32)
    _up_ratio_24 = (_diff_array[:, -23:] > 0.0).mean(axis=1).astype(np.float32)
    _trend_consistency_12 = _build_trend_consistency(_diff_array[:, -11:])
    _trend_consistency_24 = _build_trend_consistency(_diff_array[:, -23:])
    _recent_diff_mean = _diff_array[:, -6:].mean(axis=1).astype(np.float32)
    _current_timestamp = _window_timestamp_series.iloc[-1]
    _current_hour = np.asarray([float(_current_timestamp.hour)], dtype=np.float32)
    _current_weekday = np.asarray([float(_current_timestamp.weekday())], dtype=np.float32)
    _current_hour_sin = np.sin((2.0 * np.pi * _current_hour) / 24.0).astype(np.float32)
    _current_hour_cos = np.cos((2.0 * np.pi * _current_hour) / 24.0).astype(np.float32)
    _current_weekday_sin = np.sin((2.0 * np.pi * _current_weekday) / 7.0).astype(np.float32)
    _current_weekday_cos = np.cos((2.0 * np.pi * _current_weekday) / 7.0).astype(np.float32)

    _horizon_array = np.asarray(_horizons, dtype=np.float32).reshape(1, -1)
    _drift_baseline = (_recent_diff_mean.reshape(-1, 1) * _horizon_array).astype(np.float32)

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
    for _horizon_index in range(len(_horizons)):
        _static_feature_list.append(
            _safe_divide_array(_drift_baseline[:, _horizon_index], _recent_scale_array, 0.0).astype(np.float32)
        )

    _static_x = np.stack(_static_feature_list, axis=1).astype(np.float32)

    if _sequence_x.shape[-1] != len(_metadata["sequence_feature_names"]):
        raise RuntimeError("Sequence feature count does not match metadata")
    if _static_x.shape[-1] != len(_metadata["static_feature_names"]):
        raise RuntimeError("Static feature count does not match metadata")

    _sequence_x = _transform_feature_array(
        _sequence_x,
        np.asarray(_metadata["sequence_feature_mean"], dtype=np.float32).reshape(1, 1, -1),
        np.asarray(_metadata["sequence_feature_std"], dtype=np.float32).reshape(1, 1, -1),
    )
    _static_x = _transform_feature_array(
        _static_x,
        np.asarray(_metadata["static_feature_mean"], dtype=np.float32).reshape(1, -1),
        np.asarray(_metadata["static_feature_std"], dtype=np.float32).reshape(1, -1),
    )

    return {
        "sequence_x": _sequence_x,
        "static_x": _static_x,
        "target_scale": float(_recent_scale_array[0]),
        "last_close": float(_close_array[0, -1]),
        "horizons": _horizons,
        "history_end_timestamp_jst": _window_timestamp_series.iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "close_list": [float(_value) for _value in _close_array.reshape(-1).tolist()],
        "drift_baseline": [float(_value) for _value in _drift_baseline.reshape(-1).tolist()],
    }


def _build_predicted_path(_last_close, _predicted_close_by_horizon, _max_horizon):
    _known_points = {0: float(_last_close)}
    for _horizon_value, _predicted_close in _predicted_close_by_horizon.items():
        _known_points[int(_horizon_value)] = float(_predicted_close)

    _sorted_points = sorted(_known_points.items())
    _predicted_path = []

    for _step in range(1, int(_max_horizon) + 1):
        _previous_step = 0
        _previous_value = float(_last_close)
        _next_step = int(_max_horizon)
        _next_value = float(_predicted_close_by_horizon.get(int(_max_horizon), _last_close))

        for _point_step, _point_value in _sorted_points:
            if _point_step <= _step:
                _previous_step = int(_point_step)
                _previous_value = float(_point_value)
            if _point_step >= _step:
                _next_step = int(_point_step)
                _next_value = float(_point_value)
                break

        if _next_step == _previous_step:
            _predicted_path.append(float(_next_value))
            continue

        _ratio = float((_step - _previous_step) / float(_next_step - _previous_step))
        _interpolated_value = _previous_value + ((_next_value - _previous_value) * _ratio)
        _predicted_path.append(float(_interpolated_value))

    return _predicted_path


def _resolve_repo_path(_path_text):
    _path_text = str(_path_text or "").strip()
    if _path_text == "":
        return None

    _path = Path(_path_text)
    if not _path.is_absolute():
        _path = _REPO_ROOT / _path

    return _path.resolve()


def _build_legacy_runtime_descriptor():
    return {
        "role": "H1_TACTICAL_BIAS",
        "active_model_id": "legacy_h1_multi_horizon_patch_mixer",
        "model_path": _LEGACY_MODEL_PATH.resolve(),
        "metadata_path": _LEGACY_METADATA_PATH.resolve(),
        "summary_path": None,
        "predictions_path": None,
        "dataset_id": "",
        "dataset_path": None,
        "promotion_stage": "legacy_fallback",
        "selection_source": "legacy_default_path",
        "pointer_path": None,
    }


def _resolve_runtime_descriptor():
    if not _ACTIVE_RUNTIME_POINTER_PATH.exists():
        return _build_legacy_runtime_descriptor()

    with open(_ACTIVE_RUNTIME_POINTER_PATH, "r", encoding="utf-8") as _file:
        _pointer_payload = json.load(_file)

    _model_path = _resolve_repo_path(_pointer_payload.get("model_path"))
    _metadata_path = _resolve_repo_path(_pointer_payload.get("metadata_path"))
    _summary_path = _resolve_repo_path(_pointer_payload.get("summary_path"))
    _predictions_path = _resolve_repo_path(_pointer_payload.get("predictions_path"))
    _dataset_path = _resolve_repo_path(_pointer_payload.get("dataset_path"))

    if _metadata_path is None:
        raise RuntimeError(f"H1 active runtime pointer is missing metadata_path: {_ACTIVE_RUNTIME_POINTER_PATH}")
    if _model_path is None:
        raise RuntimeError(f"H1 active runtime pointer is missing model_path: {_ACTIVE_RUNTIME_POINTER_PATH}")
    if not _metadata_path.exists():
        raise RuntimeError(f"H1 active metadata file was not found: {_metadata_path}")
    if not _model_path.exists():
        raise RuntimeError(f"H1 active model file was not found: {_model_path}")

    return {
        "role": str(_pointer_payload.get("role") or "H1_TACTICAL_BIAS"),
        "active_model_id": str(_pointer_payload.get("active_model_id") or "unknown_h1_model"),
        "model_path": _model_path,
        "metadata_path": _metadata_path,
        "summary_path": _summary_path,
        "predictions_path": _predictions_path,
        "dataset_id": str(_pointer_payload.get("dataset_id") or ""),
        "dataset_path": _dataset_path,
        "promotion_stage": str(_pointer_payload.get("promotion_stage") or "active"),
        "selection_source": "active_runtime_pointer",
        "pointer_path": _ACTIVE_RUNTIME_POINTER_PATH.resolve(),
    }


def _load_runtime():
    global _RUNTIME_CACHE

    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    _descriptor = _resolve_runtime_descriptor()
    _metadata_path = _descriptor["metadata_path"]
    _model_path = _descriptor["model_path"]

    with open(_metadata_path, "r", encoding="utf-8") as _file:
        _metadata = json.load(_file)

    _horizons = [int(_value) for _value in _metadata["horizons"]]
    _model = MultiHorizonPatchMixer(
        _sequence_length=int(_metadata["sequence_length"]),
        _input_size=len(_metadata["sequence_feature_names"]),
        _static_input_size=len(_metadata["static_feature_names"]),
        _horizon_count=len(_horizons),
        _hidden_size=int(_metadata["hidden_size"]),
        _mixer_layers=int(_metadata["mixer_layers"]),
        _patch_length=int(_metadata["patch_length"]),
        _dropout=float(_metadata["dropout"]),
    )
    _state_dict = torch.load(_model_path, map_location="cpu")
    _model.load_state_dict(_state_dict)
    _model.eval()

    _RUNTIME_CACHE = ForecastRuntime(
        metadata=_metadata,
        model=_model,
        device=torch.device("cpu"),
        artifact_info={
            "role": _descriptor["role"],
            "active_model_id": _descriptor["active_model_id"],
            "model_path": str(_model_path),
            "metadata_path": str(_metadata_path),
            "summary_path": str(_descriptor["summary_path"]) if _descriptor["summary_path"] is not None else "",
            "predictions_path": str(_descriptor["predictions_path"]) if _descriptor["predictions_path"] is not None else "",
            "dataset_id": _descriptor["dataset_id"],
            "dataset_path": str(_descriptor["dataset_path"]) if _descriptor["dataset_path"] is not None else "",
            "promotion_stage": _descriptor["promotion_stage"],
            "selection_source": _descriptor["selection_source"],
            "pointer_path": str(_descriptor["pointer_path"]) if _descriptor["pointer_path"] is not None else "",
        },
    )
    return _RUNTIME_CACHE


def PredictMultiHorizonForecast(_ohlc, _timestamp_jst=""):
    _runtime = _load_runtime()
    _history_df = _build_history_dataframe(_ohlc, _timestamp_jst)
    _payload = _build_feature_arrays(_history_df, _runtime.metadata)

    _sequence_tensor = torch.tensor(_payload["sequence_x"], dtype=torch.float32, device=_runtime.device)
    _static_tensor = torch.tensor(_payload["static_x"], dtype=torch.float32, device=_runtime.device)

    with torch.no_grad():
        _predicted_scaled_tensor, _ = _runtime.model(_sequence_tensor, _static_tensor)

    _predicted_scaled = _predicted_scaled_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
    _target_scale = float(_payload["target_scale"])
    _predicted_delta = (_predicted_scaled * _target_scale).astype(np.float32)
    _horizons = [int(_value) for _value in _payload["horizons"]]
    _last_close = float(_payload["last_close"])
    _predicted_close = (_last_close + _predicted_delta).astype(np.float32)

    _horizon_weight_array = np.asarray(_horizons, dtype=np.float32)
    _horizon_weight_array = (_horizon_weight_array / float(np.mean(_horizon_weight_array))).astype(np.float32)
    _weighted_scaled_delta = (_predicted_scaled * _horizon_weight_array).astype(np.float32)
    _positive_strength = float(np.clip(_weighted_scaled_delta, 0.0, None).sum())
    _negative_strength = float(np.clip(-_weighted_scaled_delta, 0.0, None).sum())
    _absolute_strength = float(np.abs(_weighted_scaled_delta).sum())
    _direction_score_long = 0.0
    _direction_score_short = 0.0
    _direction_dominance = 0.0
    if _absolute_strength > SIGNAL_EPSILON:
        _direction_score_long = float(_positive_strength / _absolute_strength)
        _direction_score_short = float(_negative_strength / _absolute_strength)
        _direction_dominance = float(max(_direction_score_long, _direction_score_short))

    _signal_strength = float(_weighted_scaled_delta.sum() / float(_horizon_weight_array.sum()))
    _magnitude_score = _clamp_01(abs(_signal_strength) / MAGNITUDE_CONFIDENCE_SCALE)
    _confidence = _clamp_01(
        (DOMINANCE_CONFIDENCE_WEIGHT * _direction_dominance) +
        (MAGNITUDE_CONFIDENCE_WEIGHT * _magnitude_score)
    )

    if _signal_strength > SIGNAL_EPSILON:
        _net_direction = "LONG_BIAS"
    elif _signal_strength < -SIGNAL_EPSILON:
        _net_direction = "SHORT_BIAS"
    else:
        _net_direction = "NEUTRAL"

    _predicted_delta_by_horizon = {
        int(_horizon_value): float(_predicted_delta[_index])
        for _index, _horizon_value in enumerate(_horizons)
    }
    _predicted_close_by_horizon = {
        int(_horizon_value): float(_predicted_close[_index])
        for _index, _horizon_value in enumerate(_horizons)
    }
    _predicted_path = _build_predicted_path(
        _last_close=_last_close,
        _predicted_close_by_horizon=_predicted_close_by_horizon,
        _max_horizon=max(_horizons),
    )

    return {
        "backend_name": BACKEND_NAME,
        "artifact_role": str(_runtime.artifact_info.get("role") or ""),
        "active_model_id": str(_runtime.artifact_info.get("active_model_id") or ""),
        "artifact_selection_source": str(_runtime.artifact_info.get("selection_source") or ""),
        "dataset_id": str(_runtime.artifact_info.get("dataset_id") or ""),
        "horizons": list(_horizons),
        "sequence_length": int(_runtime.metadata["sequence_length"]),
        "history_end_timestamp_jst": _payload["history_end_timestamp_jst"],
        "last_close": _last_close,
        "target_scale": _target_scale,
        "predicted_delta_by_horizon": _predicted_delta_by_horizon,
        "predicted_close_by_horizon": _predicted_close_by_horizon,
        "predicted_path": _predicted_path,
        "net_direction": _net_direction,
        "direction_score_long": float(_direction_score_long),
        "direction_score_short": float(_direction_score_short),
        "direction_dominance": float(_direction_dominance),
        "confidence": float(_confidence),
        "signal_strength": float(_signal_strength),
        "magnitude_score": float(_magnitude_score),
        "close_list": list(_payload["close_list"]),
        "drift_baseline": list(_payload["drift_baseline"]),
    }
def GetForecastBackendName():
    return BACKEND_NAME


def GetForecastSequenceLength():
    _sequence_length = int(_load_runtime().metadata["sequence_length"])
    return int(_sequence_length) + int(INDICATOR_WARMUP_BARS) - 1
