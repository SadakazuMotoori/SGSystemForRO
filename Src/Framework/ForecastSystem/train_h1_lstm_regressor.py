import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.Utility.Utility import EnsureParentDirectory


MODEL_OUTPUT_PATH = "Asset/Models/h1_lstm_regressor.pt"
METADATA_OUTPUT_PATH = "Asset/Models/h1_lstm_regressor_metadata.json"
PREDICTION_OUTPUT_PATH = "Asset/Models/h1_lstm_regressor_test_predictions.csv"
DEFAULT_DATASET_PATH = "Src/Backtest/Output/datasets/h1_training_dataset.csv"
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.0
DEFAULT_TARGET_SCALE_LOOKBACK = 10
DEFAULT_DIRECTION_LOSS_WEIGHT = 0.5
DEFAULT_DIRECTION_NEUTRAL_QUANTILE = 0.0
MIN_TARGET_SCALE = 1.0e-6
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
PREDICTION_CONTEXT_COLUMNS = [
    "timestamp_jst",
    "future_timestamp_jst",
    "symbol",
    "entry_price",
    "future_hours",
    "h1_last_close",
    "h1_recent_momentum",
    "h1_trend_consistency",
    "h1_window_range_mean",
    "h1_window_range_std",
    "h1_window_return_std",
    "h1_window_up_ratio",
    "h1_window_range_ratio_5bar",
    "h1_window_range_ratio_10bar",
    "h1_window_close_slope_5bar",
    "h1_window_close_slope_10bar",
    "h1_window_close_position_10bar",
    "h1_window_close_position_full",
    "h1_current_hour_sin",
    "h1_current_hour_cos",
    "h1_current_weekday_sin",
    "h1_current_weekday_cos",
    "h1_range_31",
    "h1_range_mean_5_31",
    "h1_range_mean_10_31",
    "h1_range_ratio_5_31",
    "h1_range_ratio_10_31",
    "h1_rsi_14_31",
    "h1_macd_31",
    "h1_macd_hist_31",
    "h1_ma_gap_sma20_31",
    "h1_ma_gap_sma50_31",
    "h1_close_return_1_31",
    "h1_return_std_5_31",
    "h1_return_std_10_31",
    "h1_close_position_5_31",
    "h1_close_position_10_31",
    "h1_hour_sin_31",
    "h1_hour_cos_31",
    "h1_weekday_sin_31",
    "h1_weekday_cos_31",
    "target_scale",
    "true_scaled_delta",
    "pred_scaled_delta",
]


# --------------------------------------------------
# CLI引数を読み込む
# 役割:
#   H1回帰モデル学習に必要な入力CSVや学習設定を受け取る
# --------------------------------------------------
def parse_args():
    # --------------------------------------------------
    # VSCodeでF5実行した場合を想定し、
    # 引数未指定時はデバッグ用既定値でそのまま走らせる
    # --------------------------------------------------
    if len(sys.argv) == 1:
        return SimpleNamespace(
            dataset=DEFAULT_DATASET_PATH,
            model_output=MODEL_OUTPUT_PATH,
            metadata_output=METADATA_OUTPUT_PATH,
            prediction_output=PREDICTION_OUTPUT_PATH,
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
            hidden_size=DEFAULT_HIDDEN_SIZE,
            num_layers=DEFAULT_NUM_LAYERS,
            dropout=DEFAULT_DROPOUT,
            seed=DEFAULT_SEED,
            patience=DEFAULT_PATIENCE,
            min_delta=DEFAULT_MIN_DELTA,
            target_scale_lookback=DEFAULT_TARGET_SCALE_LOOKBACK,
            use_multitask_direction_head=False,
            direction_loss_weight=DEFAULT_DIRECTION_LOSS_WEIGHT,
            direction_neutral_quantile=DEFAULT_DIRECTION_NEUTRAL_QUANTILE,
            use_target_range_normalization=False,
            use_derived_static_features=False,
            verbose=True,
        )

    _parser = argparse.ArgumentParser(description="Train H1 LSTM regressor for SGSystemForRO")
    _parser.add_argument("--dataset", required=True, help="Path to h1_training_dataset.csv")
    _parser.add_argument("--model-output", default=MODEL_OUTPUT_PATH, help="Path to save trained model")
    _parser.add_argument("--metadata-output", default=METADATA_OUTPUT_PATH, help="Path to save metadata json")
    _parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size")
    _parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    _parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    _parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help="LSTM hidden size")
    _parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of LSTM layers")
    _parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout ratio")
    _parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    _parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    _parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA, help="Minimum valid MSE improvement")
    _parser.add_argument(
        "--target-scale-lookback",
        type=int,
        default=DEFAULT_TARGET_SCALE_LOOKBACK,
        help="Lookback window for recent H1 range mean used in target normalization",
    )
    _parser.add_argument(
        "--use-multitask-direction-head",
        action="store_true",
        help="Train an additional direction classification head together with delta regression",
    )
    _parser.add_argument(
        "--direction-loss-weight",
        type=float,
        default=DEFAULT_DIRECTION_LOSS_WEIGHT,
        help="Loss weight for the optional direction classification head",
    )
    _parser.add_argument(
        "--direction-neutral-quantile",
        type=float,
        default=DEFAULT_DIRECTION_NEUTRAL_QUANTILE,
        help="Train-set quantile of abs(delta / recent_range_mean) used as a neutral-zone threshold for the direction head",
    )
    _parser.add_argument(
        "--use-target-range-normalization",
        action="store_true",
        help="Train on delta normalized by recent H1 range mean",
    )
    _parser.add_argument("--prediction-output", default=PREDICTION_OUTPUT_PATH, help="Path to save test predictions csv")
    _parser.add_argument(
        "--use-derived-static-features",
        action="store_true",
        help="Add experimental static features derived from the existing 32-bar window",
    )
    _parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

    return _parser.parse_args()


# --------------------------------------------------
# 乱数シードを固定する
# 役割:
#   学習結果の再現性を高める
# --------------------------------------------------
def set_seed(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)


# --------------------------------------------------
# 学習用データセットを表すクラス
# 役割:
#   DataLoaderへ渡すため、特徴量と教師値を管理する
# --------------------------------------------------
class SequenceRegressionDataset(Dataset):
    def __init__(
        self,
        _sequence_x_array,
        _static_x_array,
        _y_array,
        _raw_y_array,
        _direction_y_array,
        _direction_active_array,
        _direction_scaled_delta_array,
        _target_scale_array,
    ):
        self.sequence_x = torch.tensor(_sequence_x_array, dtype=torch.float32)
        self.static_x = torch.tensor(_static_x_array, dtype=torch.float32)
        self.y = torch.tensor(_y_array, dtype=torch.float32).view(-1, 1)
        self.raw_y = torch.tensor(_raw_y_array, dtype=torch.float32).view(-1, 1)
        self.direction_y = torch.tensor(_direction_y_array, dtype=torch.float32).view(-1, 1)
        self.direction_active = torch.tensor(_direction_active_array, dtype=torch.float32).view(-1, 1)
        self.direction_scaled_delta = torch.tensor(_direction_scaled_delta_array, dtype=torch.float32).view(-1, 1)
        self.target_scale = torch.tensor(_target_scale_array, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.sequence_x)

    def __getitem__(self, _index):
        return (
            self.sequence_x[_index],
            self.static_x[_index],
            self.y[_index],
            self.raw_y[_index],
            self.direction_y[_index],
            self.direction_active[_index],
            self.direction_scaled_delta[_index],
            self.target_scale[_index],
        )


# --------------------------------------------------
# LSTM回帰モデル
# 役割:
#   H1系列特徴量から2本先のdeltaを予測する
# --------------------------------------------------
class H1LSTMRegressor(nn.Module):
    def __init__(
        self,
        _input_size,
        _hidden_size,
        _num_layers,
        _dropout,
        _static_input_size=0,
        _use_multitask_direction_head=False,
    ):
        super().__init__()

        _effective_dropout = _dropout if _num_layers > 1 else 0.0
        _head_input_size = _hidden_size + int(_static_input_size)
        self.use_multitask_direction_head = bool(_use_multitask_direction_head)

        self.lstm = nn.LSTM(
            input_size=_input_size,
            hidden_size=_hidden_size,
            num_layers=_num_layers,
            batch_first=True,
            dropout=_effective_dropout,
        )
        self.head_dropout = nn.Dropout(_dropout)
        self.fc = nn.Linear(_head_input_size, 1)
        self.direction_fc = nn.Linear(_head_input_size, 1) if self.use_multitask_direction_head else None

    def forward(self, _sequence_x, _static_x):
        _output, _ = self.lstm(_sequence_x)
        _last_hidden = _output[:, -1, :]

        if _static_x.shape[-1] > 0:
            _last_hidden = torch.cat([_last_hidden, _static_x], dim=1)

        _head_input = self.head_dropout(_last_hidden)
        _pred = self.fc(_head_input)
        _direction_logit = self.direction_fc(_head_input) if self.direction_fc is not None else None
        return _pred, _direction_logit


# --------------------------------------------------
# CSVを読み込む
# 役割:
#   学習対象のDataFrameを読み込み、空や欠損を確認する
# --------------------------------------------------
def load_dataset(_dataset_path):
    _dataset_path = Path(_dataset_path)

    if not _dataset_path.exists():
        raise RuntimeError(f"Dataset file not found: {_dataset_path}")

    _df = pd.read_csv(_dataset_path)

    if len(_df) == 0:
        raise RuntimeError(f"Dataset is empty: {_dataset_path}")

    return _df


# --------------------------------------------------
# 安全に割り算する
# 役割:
#   0除算やNaNを避けて派生特徴を作る
# --------------------------------------------------
def safe_divide_array(_numerator_array, _denominator_array, _default_value):
    _result = np.full_like(_numerator_array, float(_default_value), dtype=np.float32)
    _valid_mask = np.abs(_denominator_array) > 1.0e-8
    np.divide(
        _numerator_array,
        _denominator_array,
        out=_result,
        where=_valid_mask,
    )
    return _result.astype(np.float32)


# --------------------------------------------------
# 既存CSVから静的派生特徴を補完する
# 役割:
#   データ再生成なしでも分析しやすい局面特徴を増やす
# --------------------------------------------------
def ensure_derived_static_features(_df, _sequence_length):
    _work_df = _df.copy()
    _last_step = int(_sequence_length) - 1

    _range_columns = [f"h1_range_{_index:02d}" for _index in range(_sequence_length)]
    _return_columns = [f"h1_close_return_1_{_index:02d}" for _index in range(_sequence_length)]
    _close_columns = [f"h1_close_{_index:02d}" for _index in range(_sequence_length)]
    _high_columns = [f"h1_high_{_index:02d}" for _index in range(_sequence_length)]
    _low_columns = [f"h1_low_{_index:02d}" for _index in range(_sequence_length)]

    _required_columns = [
        *_range_columns,
        *_return_columns,
        *_close_columns,
        *_high_columns,
        *_low_columns,
    ]

    if not all(_column_name in _work_df.columns for _column_name in _required_columns):
        return _work_df

    _range_array = _work_df[_range_columns].to_numpy(dtype=np.float32)
    _return_array = _work_df[_return_columns].to_numpy(dtype=np.float32)
    _close_array = _work_df[_close_columns].to_numpy(dtype=np.float32)
    _high_array = _work_df[_high_columns].to_numpy(dtype=np.float32)
    _low_array = _work_df[_low_columns].to_numpy(dtype=np.float32)

    if "h1_window_range_mean" not in _work_df.columns:
        _work_df["h1_window_range_mean"] = _range_array.mean(axis=1)

    if "h1_window_range_std" not in _work_df.columns:
        _work_df["h1_window_range_std"] = _range_array.std(axis=1)

    if "h1_window_return_std" not in _work_df.columns:
        _work_df["h1_window_return_std"] = _return_array.std(axis=1)

    if "h1_window_up_ratio" not in _work_df.columns:
        _up_ratio = (_return_array[:, 1:] > 0.0).mean(axis=1)
        _work_df["h1_window_up_ratio"] = _up_ratio.astype(np.float32)

    _range_last_5 = _range_array[:, max(0, _sequence_length - 5):]
    _range_last_10 = _range_array[:, max(0, _sequence_length - 10):]

    if "h1_window_range_ratio_5bar" not in _work_df.columns:
        _work_df["h1_window_range_ratio_5bar"] = safe_divide_array(
            _range_array[:, _last_step],
            _range_last_5.mean(axis=1),
            1.0,
        )

    if "h1_window_range_ratio_10bar" not in _work_df.columns:
        _work_df["h1_window_range_ratio_10bar"] = safe_divide_array(
            _range_array[:, _last_step],
            _range_last_10.mean(axis=1),
            1.0,
        )

    if "h1_window_close_slope_5bar" not in _work_df.columns:
        _work_df["h1_window_close_slope_5bar"] = (
            _close_array[:, _last_step] - _close_array[:, max(0, _last_step - 5)]
        ).astype(np.float32)

    if "h1_window_close_slope_10bar" not in _work_df.columns:
        _work_df["h1_window_close_slope_10bar"] = (
            _close_array[:, _last_step] - _close_array[:, max(0, _last_step - 10)]
        ).astype(np.float32)

    if "h1_window_close_position_10bar" not in _work_df.columns:
        _high_last_10 = _high_array[:, max(0, _sequence_length - 10):].max(axis=1)
        _low_last_10 = _low_array[:, max(0, _sequence_length - 10):].min(axis=1)
        _work_df["h1_window_close_position_10bar"] = safe_divide_array(
            _close_array[:, _last_step] - _low_last_10,
            _high_last_10 - _low_last_10,
            0.5,
        )

    if "h1_window_close_position_full" not in _work_df.columns:
        _high_window = _high_array.max(axis=1)
        _low_window = _low_array.min(axis=1)
        _work_df["h1_window_close_position_full"] = safe_divide_array(
            _close_array[:, _last_step] - _low_window,
            _high_window - _low_window,
            0.5,
        )

    if "timestamp_jst" in _work_df.columns:
        _timestamp_series = pd.to_datetime(_work_df["timestamp_jst"], errors="coerce")
        _hour_series = _timestamp_series.dt.hour.astype(float).fillna(0.0)
        _weekday_series = _timestamp_series.dt.weekday.astype(float).fillna(0.0)

        if "h1_current_hour_sin" not in _work_df.columns:
            _work_df["h1_current_hour_sin"] = np.sin((2.0 * np.pi * _hour_series) / 24.0).astype(np.float32)

        if "h1_current_hour_cos" not in _work_df.columns:
            _work_df["h1_current_hour_cos"] = np.cos((2.0 * np.pi * _hour_series) / 24.0).astype(np.float32)

        if "h1_current_weekday_sin" not in _work_df.columns:
            _work_df["h1_current_weekday_sin"] = np.sin((2.0 * np.pi * _weekday_series) / 7.0).astype(np.float32)

        if "h1_current_weekday_cos" not in _work_df.columns:
            _work_df["h1_current_weekday_cos"] = np.cos((2.0 * np.pi * _weekday_series) / 7.0).astype(np.float32)

    return _work_df


# --------------------------------------------------
# 特徴量列を抽出する
# 役割:
#   h1_ で始まる時系列特徴量列だけを学習入力として使う
# --------------------------------------------------
def get_feature_columns(_df):
    _all_feature_columns = [c for c in _df.columns if c.startswith("h1_")]

    if len(_all_feature_columns) == 0:
        raise RuntimeError("No feature columns starting with 'h1_' were found")

    _sequence_feature_columns = sorted(
        [
            _column_name
            for _column_name in _all_feature_columns
            if _column_name.rsplit("_", 1)[-1].isdigit()
        ]
    )
    _static_feature_columns = sorted(
        [
            _column_name
            for _column_name in _all_feature_columns
            if not _column_name.rsplit("_", 1)[-1].isdigit()
        ]
    )

    if len(_sequence_feature_columns) == 0:
        raise RuntimeError("No time-step feature columns ending with _00 style suffixes were found")

    return _sequence_feature_columns, _static_feature_columns


# --------------------------------------------------
# ターゲット列名を解決する
# 役割:
#   現在のデータセットに存在する回帰ターゲット列を見つける
# --------------------------------------------------
def get_target_column(_df):
    _candidates = [
        "target_delta_t_plus_2",
        "target_delta",
    ]

    for _column_name in _candidates:
        if _column_name in _df.columns:
            return _column_name

    raise RuntimeError("No target delta column was found")


# --------------------------------------------------
# sequence_length を取得する
# 役割:
#   reshape時に必要な timesteps を確定する
# --------------------------------------------------
def get_sequence_length(_df):
    if "sequence_length" not in _df.columns:
        raise RuntimeError("sequence_length column was not found")

    _unique_values = sorted(_df["sequence_length"].dropna().unique().tolist())

    if len(_unique_values) != 1:
        raise RuntimeError(f"sequence_length must be unique, found: {_unique_values}")

    return int(_unique_values[0])


# --------------------------------------------------
# 特徴量テンソルを構築する
# 役割:
#   横持ちCSVを (samples, timesteps, features) へ変換する
# --------------------------------------------------
def build_feature_tensor(_df, _sequence_feature_columns, _sequence_length):
    _feature_df = _df[_sequence_feature_columns].copy()

    if _feature_df.isnull().any().any():
        raise RuntimeError("Feature columns include NaN values")

    _base_feature_names = sorted(
        {
            _column_name.rsplit("_", 1)[0]
            for _column_name in _sequence_feature_columns
            if _column_name.rsplit("_", 1)[-1].isdigit()
        }
    )

    if len(_base_feature_names) == 0:
        raise RuntimeError("Failed to infer base feature names from h1_* columns")

    _ordered_feature_columns = []
    for _step_index in range(_sequence_length):
        for _base_name in _base_feature_names:
            _column_name = f"{_base_name}_{_step_index:02d}"
            if _column_name not in _feature_df.columns:
                raise RuntimeError(f"Missing feature column: {_column_name}")
            _ordered_feature_columns.append(_column_name)

    _x_flat = _feature_df[_ordered_feature_columns].to_numpy(dtype=np.float32)
    _feature_count = len(_base_feature_names)
    _x_tensor = _x_flat.reshape(len(_df), _sequence_length, _feature_count)

    return _x_tensor, _base_feature_names


# --------------------------------------------------
# 要約特徴量配列を構築する
# 役割:
#   時系列窓の外にある補助特徴を2次元配列へ変換する
# --------------------------------------------------
def build_static_feature_array(_df, _static_feature_columns):
    if len(_static_feature_columns) == 0:
        return np.zeros((len(_df), 0), dtype=np.float32), []

    _static_feature_df = _df[_static_feature_columns].copy()

    if _static_feature_df.isnull().any().any():
        raise RuntimeError("Static feature columns include NaN values")

    return _static_feature_df.to_numpy(dtype=np.float32), list(_static_feature_columns)


# --------------------------------------------------
# ターゲット配列を構築する
# 役割:
#   回帰教師値を numpy 配列へ変換する
# --------------------------------------------------
def build_target_array(_df, _target_column):
    _target_series = _df[_target_column]

    if _target_series.isnull().any():
        raise RuntimeError(f"Target column includes NaN values: {_target_column}")

    return _target_series.to_numpy(dtype=np.float32)


# --------------------------------------------------
# ターゲット正規化スケールを構築する
# 役割:
#   直近の H1 range 平均を使って delta のスケールを揃える
# --------------------------------------------------
def build_target_scale_array(_df, _sequence_length, _lookback):
    _lookback = int(_lookback)
    if _lookback <= 0:
        raise RuntimeError("target_scale_lookback must be 1 or greater")

    _lookback = min(_lookback, int(_sequence_length))
    _start_index = int(_sequence_length) - _lookback
    _range_columns = [f"h1_range_{_index:02d}" for _index in range(_start_index, int(_sequence_length))]

    for _column_name in _range_columns:
        if _column_name not in _df.columns:
            raise RuntimeError(f"Target scale feature column was not found: {_column_name}")

    _scale_array = _df[_range_columns].mean(axis=1).to_numpy(dtype=np.float32)
    _scale_array = np.maximum(_scale_array, np.float32(MIN_TARGET_SCALE))
    return _scale_array


# --------------------------------------------------
# ターゲットをモデル学習用スケールへ変換する
# 役割:
#   raw delta を recent range で割って学習しやすい値域へ揃える
# --------------------------------------------------
def transform_target_array(_raw_y_array, _target_scale_array, _use_range_normalization):
    if not _use_range_normalization:
        return _raw_y_array.astype(np.float32)

    return (_raw_y_array / _target_scale_array).astype(np.float32)


# --------------------------------------------------
# モデル出力を raw delta へ戻す
# 役割:
#   評価や保存時に人間が解釈しやすい価格差へ復元する
# --------------------------------------------------
def inverse_transform_target_array(_model_y_array, _target_scale_array, _use_range_normalization):
    if not _use_range_normalization:
        return _model_y_array.astype(np.float32)

    return (_model_y_array * _target_scale_array).astype(np.float32)


# --------------------------------------------------
# 方向ヘッド用の中立帯閾値を学習する
# 方針:
#   train split の abs(delta / recent_range_mean) 分布から quantile 閾値を決める
# --------------------------------------------------
def fit_direction_neutral_threshold(_raw_y_train, _target_scale_train, _neutral_quantile):
    _neutral_quantile = float(_neutral_quantile)

    if _neutral_quantile <= 0.0:
        return 0.0

    if _neutral_quantile >= 1.0:
        raise RuntimeError("direction_neutral_quantile must be smaller than 1.0")

    _scaled_delta = safe_divide_array(
        np.asarray(_raw_y_train, dtype=np.float32),
        np.asarray(_target_scale_train, dtype=np.float32),
        0.0,
    )
    _abs_scaled_delta = np.abs(_scaled_delta)
    return float(np.quantile(_abs_scaled_delta, _neutral_quantile))


# --------------------------------------------------
# 方向ヘッドの教師ラベルと有効マスクを作る
# 方針:
#   abs(delta / recent_range_mean) が閾値未満の局面は中立として分類損失から外す
# --------------------------------------------------
def build_direction_target_arrays(_raw_y_array, _target_scale_array, _neutral_threshold):
    _raw_y_array = np.asarray(_raw_y_array, dtype=np.float32)
    _target_scale_array = np.asarray(_target_scale_array, dtype=np.float32)
    _scaled_delta_array = safe_divide_array(_raw_y_array, _target_scale_array, 0.0)
    _direction_y_array = (_scaled_delta_array >= 0.0).astype(np.float32)
    _direction_active_array = (np.abs(_scaled_delta_array) >= float(_neutral_threshold)).astype(np.float32)
    return (
        _direction_y_array.astype(np.float32),
        _direction_active_array.astype(np.float32),
        _scaled_delta_array.astype(np.float32),
    )


# --------------------------------------------------
# 時系列で train / valid / test に分割する
# 役割:
#   未来情報混入を防ぐため、順序を保ったまま分割する
# --------------------------------------------------
def get_split_boundaries(_total_count):
    if _total_count < 10:
        raise RuntimeError("Dataset is too small for time-based split")

    _train_end = int(_total_count * TRAIN_RATIO)
    _valid_end = int(_total_count * (TRAIN_RATIO + VALID_RATIO))

    if _train_end <= 0 or _valid_end <= _train_end or _valid_end >= _total_count:
        raise RuntimeError("Invalid split boundaries were produced")

    return _train_end, _valid_end


# --------------------------------------------------
# 標準化を行う
# 役割:
#   train統計量で入力特徴量のみを正規化する
# --------------------------------------------------
def fit_feature_scaler(_x_train):
    if _x_train.ndim == 3:
        _flat = _x_train.reshape(-1, _x_train.shape[-1])
    elif _x_train.ndim == 2:
        _flat = _x_train
    else:
        raise RuntimeError(f"Unsupported feature ndim: {_x_train.ndim}")

    _mean = _flat.mean(axis=0)
    _std = _flat.std(axis=0)
    _std = np.where(_std == 0.0, 1.0, _std)

    return _mean.astype(np.float32), _std.astype(np.float32)


# --------------------------------------------------
# 標準化を適用する
# 役割:
#   学習・検証・テストへ同じスケーラーを適用する
# --------------------------------------------------
def transform_feature_tensor(_x, _mean, _std):
    if _x.shape[-1] == 0:
        return _x.astype(np.float32)

    if _x.ndim == 3:
        return ((_x - _mean.reshape(1, 1, -1)) / _std.reshape(1, 1, -1)).astype(np.float32)

    if _x.ndim == 2:
        return ((_x - _mean.reshape(1, -1)) / _std.reshape(1, -1)).astype(np.float32)

    raise RuntimeError(f"Unsupported feature ndim: {_x.ndim}")


# --------------------------------------------------
# DataLoaderを構築する
# 役割:
#   学習・検証・テスト用のローダを作る
# --------------------------------------------------
def build_dataloader(
    _sequence_x,
    _static_x,
    _y,
    _raw_y,
    _direction_y,
    _direction_active,
    _direction_scaled_delta,
    _target_scale,
    _batch_size,
    _shuffle,
):
    _dataset = SequenceRegressionDataset(
        _sequence_x,
        _static_x,
        _y,
        _raw_y,
        _direction_y,
        _direction_active,
        _direction_scaled_delta,
        _target_scale,
    )
    return DataLoader(_dataset, batch_size=_batch_size, shuffle=_shuffle)


# --------------------------------------------------
# 1epoch 学習する
# 役割:
#   train loader を1周して損失平均を返す
# --------------------------------------------------
def train_one_epoch(
    _model,
    _loader,
    _regression_criterion,
    _direction_criterion,
    _optimizer,
    _device,
    _use_multitask_direction_head,
    _direction_loss_weight,
):
    _model.train()
    _total_loss_sum = 0.0
    _regression_loss_sum = 0.0
    _direction_loss_sum = 0.0
    _sample_count = 0
    _direction_active_count = 0

    for (
        _batch_sequence_x,
        _batch_static_x,
        _batch_y,
        _,
        _batch_direction_y,
        _batch_direction_active,
        _,
        _,
    ) in _loader:
        _batch_sequence_x = _batch_sequence_x.to(_device)
        _batch_static_x = _batch_static_x.to(_device)
        _batch_y = _batch_y.to(_device)
        _batch_direction_y = _batch_direction_y.to(_device)
        _batch_direction_active = _batch_direction_active.to(_device)

        _optimizer.zero_grad()
        _pred, _direction_logit = _model(_batch_sequence_x, _batch_static_x)
        _regression_loss = _regression_criterion(_pred, _batch_y)
        _loss = _regression_loss
        _direction_loss_value = torch.tensor(0.0, device=_device)

        if _use_multitask_direction_head:
            _direction_active_mask = _batch_direction_active.view(-1) >= 0.5
            if bool(torch.any(_direction_active_mask)):
                _direction_loss_value = _direction_criterion(
                    _direction_logit.view(-1)[_direction_active_mask],
                    _batch_direction_y.view(-1)[_direction_active_mask],
                )
                _loss = _loss + (float(_direction_loss_weight) * _direction_loss_value)
                _direction_active_count += int(_direction_active_mask.sum().item())

        _loss.backward()
        _optimizer.step()

        _batch_size = len(_batch_sequence_x)
        _total_loss_sum += float(_loss.item()) * _batch_size
        _regression_loss_sum += float(_regression_loss.item()) * _batch_size
        if _use_multitask_direction_head and float(_direction_loss_value.item()) > 0.0:
            _direction_loss_sum += float(_direction_loss_value.item()) * int(_direction_active_mask.sum().item())
        _sample_count += _batch_size

    return {
        "total_loss": _total_loss_sum / _sample_count,
        "target_loss": _regression_loss_sum / _sample_count,
        "direction_loss": (_direction_loss_sum / _direction_active_count) if _direction_active_count > 0 else 0.0,
        "direction_active_ratio": float(_direction_active_count) / float(_sample_count) if _sample_count > 0 else 0.0,
    }


# --------------------------------------------------
# 評価する
# 役割:
#   loader を走査して MSE / MAE / 方向一致率を返す
# --------------------------------------------------
def predict_loader(_model, _loader, _device, _use_range_normalization):
    _model.eval()

    _true_raw_list = []
    _pred_raw_list = []
    _true_model_list = []
    _pred_model_list = []
    _true_direction_list = []
    _direction_active_list = []
    _direction_scaled_delta_list = []
    _direction_logit_list = []
    _direction_prob_list = []
    _target_scale_list = []

    with torch.no_grad():
        for (
            _batch_sequence_x,
            _batch_static_x,
            _batch_y,
            _batch_raw_y,
            _batch_direction_y,
            _batch_direction_active,
            _batch_direction_scaled_delta,
            _batch_target_scale,
        ) in _loader:
            _batch_sequence_x = _batch_sequence_x.to(_device)
            _batch_static_x = _batch_static_x.to(_device)

            _pred_model, _direction_logit = _model(_batch_sequence_x, _batch_static_x)
            _pred_model = _pred_model.cpu().numpy().reshape(-1)
            _true_model = _batch_y.cpu().numpy().reshape(-1)
            _true_raw = _batch_raw_y.cpu().numpy().reshape(-1)
            _true_direction = _batch_direction_y.cpu().numpy().reshape(-1)
            _direction_active = _batch_direction_active.cpu().numpy().reshape(-1)
            _direction_scaled_delta = _batch_direction_scaled_delta.cpu().numpy().reshape(-1)
            _target_scale = _batch_target_scale.cpu().numpy().reshape(-1)
            _pred_raw = inverse_transform_target_array(_pred_model, _target_scale, _use_range_normalization)
            _direction_logit_array = (
                _direction_logit.cpu().numpy().reshape(-1)
                if _direction_logit is not None
                else np.full(len(_pred_model), np.nan, dtype=np.float32)
            )
            _direction_prob_array = (
                1.0 / (1.0 + np.exp(-_direction_logit_array))
                if _direction_logit is not None
                else np.full(len(_pred_model), np.nan, dtype=np.float32)
            )

            _pred_model_list.extend(_pred_model.tolist())
            _true_model_list.extend(_true_model.tolist())
            _true_raw_list.extend(_true_raw.tolist())
            _pred_raw_list.extend(_pred_raw.tolist())
            _true_direction_list.extend(_true_direction.tolist())
            _direction_active_list.extend(_direction_active.tolist())
            _direction_scaled_delta_list.extend(_direction_scaled_delta.tolist())
            _direction_logit_list.extend(_direction_logit_array.tolist())
            _direction_prob_list.extend(_direction_prob_array.tolist())
            _target_scale_list.extend(_target_scale.tolist())

    return {
        "true_raw": np.asarray(_true_raw_list, dtype=np.float32),
        "pred_raw": np.asarray(_pred_raw_list, dtype=np.float32),
        "true_model": np.asarray(_true_model_list, dtype=np.float32),
        "pred_model": np.asarray(_pred_model_list, dtype=np.float32),
        "true_direction": np.asarray(_true_direction_list, dtype=np.float32),
        "direction_active": np.asarray(_direction_active_list, dtype=np.float32),
        "direction_scaled_delta": np.asarray(_direction_scaled_delta_list, dtype=np.float32),
        "direction_logit": np.asarray(_direction_logit_list, dtype=np.float32),
        "direction_prob_up": np.asarray(_direction_prob_list, dtype=np.float32),
        "target_scale": np.asarray(_target_scale_list, dtype=np.float32),
    }


# --------------------------------------------------
# 回帰評価指標を算出する
# 役割:
#   delta予測の精度と方向性バイアスをまとめて返す
# --------------------------------------------------
def compute_regression_metrics(_true_array, _pred_array):
    _true_array = np.asarray(_true_array, dtype=np.float32).reshape(-1)
    _pred_array = np.asarray(_pred_array, dtype=np.float32).reshape(-1)

    if len(_true_array) == 0:
        raise RuntimeError("Metrics cannot be computed on an empty array")

    _error = _pred_array - _true_array
    _true_up_mask = _true_array >= 0.0
    _true_down_mask = ~_true_up_mask
    _pred_up_mask = _pred_array >= 0.0
    _pred_down_mask = ~_pred_up_mask

    _true_std = float(np.std(_true_array))
    _pred_std = float(np.std(_pred_array))

    _correlation = 0.0
    if _true_std > 1.0e-8 and _pred_std > 1.0e-8:
        _correlation = float(np.corrcoef(_true_array, _pred_array)[0, 1])
        if not np.isfinite(_correlation):
            _correlation = 0.0

    return {
        "mse": float(np.mean(np.square(_error))),
        "mae": float(np.mean(np.abs(_error))),
        "directional_accuracy": float(np.mean(_pred_up_mask == _true_up_mask)),
        "pred_up_ratio": float(np.mean(_pred_up_mask)),
        "true_up_ratio": float(np.mean(_true_up_mask)),
        "up_recall": float(np.mean(_pred_up_mask[_true_up_mask])) if _true_up_mask.any() else 0.0,
        "down_recall": float(np.mean(_pred_down_mask[_true_down_mask])) if _true_down_mask.any() else 0.0,
        "pred_std": _pred_std,
        "true_std": _true_std,
        "correlation": _correlation,
    }


# --------------------------------------------------
# 方向分類ヘッドの評価指標を算出する
# 役割:
#   符号専用ヘッドがどれだけ方向を捉えたかを返す
# --------------------------------------------------
def compute_direction_head_metrics(_true_direction_array, _direction_prob_array, _direction_active_array=None):
    _true_direction_array = np.asarray(_true_direction_array, dtype=np.float32).reshape(-1)
    _direction_prob_array = np.asarray(_direction_prob_array, dtype=np.float32).reshape(-1)
    _direction_active_array = (
        np.asarray(_direction_active_array, dtype=np.float32).reshape(-1)
        if _direction_active_array is not None
        else np.ones_like(_true_direction_array, dtype=np.float32)
    )

    _valid_mask = np.isfinite(_direction_prob_array) & (_direction_active_array >= 0.5)
    if not _valid_mask.any():
        return {
            "direction_head_bce": None,
            "direction_head_accuracy": None,
            "direction_head_pred_up_ratio": None,
            "direction_head_up_recall": None,
            "direction_head_down_recall": None,
            "direction_head_prob_mean": None,
            "direction_head_prob_std": None,
            "direction_head_eval_count": 0,
            "direction_head_active_ratio": 0.0,
        }

    _true_direction_array = _true_direction_array[_valid_mask]
    _direction_prob_array = np.clip(_direction_prob_array[_valid_mask], 1.0e-6, 1.0 - 1.0e-6)
    _pred_up_mask = _direction_prob_array >= 0.5
    _true_up_mask = _true_direction_array >= 0.5
    _true_down_mask = ~_true_up_mask

    _bce = -np.mean(
        (_true_direction_array * np.log(_direction_prob_array)) +
        ((1.0 - _true_direction_array) * np.log(1.0 - _direction_prob_array))
    )

    return {
        "direction_head_bce": float(_bce),
        "direction_head_accuracy": float(np.mean(_pred_up_mask == _true_up_mask)),
        "direction_head_pred_up_ratio": float(np.mean(_pred_up_mask)),
        "direction_head_up_recall": float(np.mean(_pred_up_mask[_true_up_mask])) if _true_up_mask.any() else 0.0,
        "direction_head_down_recall": float(np.mean((~_pred_up_mask)[_true_down_mask])) if _true_down_mask.any() else 0.0,
        "direction_head_prob_mean": float(np.mean(_direction_prob_array)),
        "direction_head_prob_std": float(np.std(_direction_prob_array)),
        "direction_head_eval_count": int(_valid_mask.sum()),
        "direction_head_active_ratio": float(np.mean(_direction_active_array >= 0.5)),
    }


# --------------------------------------------------
# モデルを評価する
# 役割:
#   loaderを走査して回帰指標をまとめて返す
# --------------------------------------------------
def evaluate_model(_model, _loader, _device, _use_range_normalization):
    _prediction_payload = predict_loader(_model, _loader, _device, _use_range_normalization)
    _metrics = compute_regression_metrics(
        _prediction_payload["true_raw"],
        _prediction_payload["pred_raw"],
    )
    _model_error = _prediction_payload["pred_model"] - _prediction_payload["true_model"]
    _metrics["target_mse"] = float(np.mean(np.square(_model_error)))
    _metrics["target_mae"] = float(np.mean(np.abs(_model_error)))
    _metrics.update(
        compute_direction_head_metrics(
            _prediction_payload["true_direction"],
            _prediction_payload["direction_prob_up"],
            _prediction_payload["direction_active"],
        )
    )
    return _metrics


# --------------------------------------------------
# 定数ベースラインを評価する
# 役割:
#   train平均予測との比較基準を用意する
# --------------------------------------------------
def evaluate_constant_baseline(_true_array, _constant_prediction):
    _pred_array = np.full(len(_true_array), float(_constant_prediction), dtype=np.float32)
    return compute_regression_metrics(_true_array, _pred_array)


# --------------------------------------------------
# モデルを保存する
# 役割:
#   学習済み重みをファイルへ保存する
# --------------------------------------------------
def save_model(_model, _model_output_path):
    EnsureParentDirectory(_model_output_path)
    torch.save(_model.state_dict(), _model_output_path)


# --------------------------------------------------
# メタ情報を保存する
# 役割:
#   推論時に必要な特徴量情報や標準化統計量を保持する
# --------------------------------------------------
def save_metadata(
    _metadata_output_path,
    _sequence_feature_names,
    _static_feature_names,
    _sequence_length,
    _target_column,
    _target_scale_lookback,
    _use_target_range_normalization,
    _use_multitask_direction_head,
    _direction_loss_weight,
    _direction_neutral_quantile,
    _direction_neutral_threshold,
    _direction_active_ratios,
    _sequence_mean,
    _sequence_std,
    _static_mean,
    _static_std,
    _args,
    _split_sizes,
):
    EnsureParentDirectory(_metadata_output_path)

    _metadata = {
        "feature_names": list(_sequence_feature_names),
        "sequence_feature_names": list(_sequence_feature_names),
        "static_feature_names": list(_static_feature_names),
        "sequence_length": int(_sequence_length),
        "target_column": _target_column,
        "target_transform": (
            "divide_by_recent_mean_h1_range"
            if _use_target_range_normalization
            else "none"
        ),
        "target_scale_lookback": int(_target_scale_lookback),
        "target_scale_floor": float(MIN_TARGET_SCALE),
        "use_multitask_direction_head": bool(_use_multitask_direction_head),
        "direction_loss_weight": float(_direction_loss_weight),
        "direction_neutral_quantile": float(_direction_neutral_quantile),
        "direction_neutral_threshold": float(_direction_neutral_threshold),
        "direction_active_ratio_train": float(_direction_active_ratios["train"]),
        "direction_active_ratio_valid": float(_direction_active_ratios["valid"]),
        "direction_active_ratio_test": float(_direction_active_ratios["test"]),
        "feature_mean": [float(v) for v in _sequence_mean.tolist()],
        "feature_std": [float(v) for v in _sequence_std.tolist()],
        "static_feature_mean": [float(v) for v in _static_mean.tolist()],
        "static_feature_std": [float(v) for v in _static_std.tolist()],
        "batch_size": int(_args.batch_size),
        "epochs": int(_args.epochs),
        "learning_rate": float(_args.learning_rate),
        "hidden_size": int(_args.hidden_size),
        "num_layers": int(_args.num_layers),
        "dropout": float(_args.dropout),
        "static_feature_count": int(len(_static_feature_names)),
        "train_size": int(_split_sizes["train"]),
        "valid_size": int(_split_sizes["valid"]),
        "test_size": int(_split_sizes["test"]),
    }

    with open(_metadata_output_path, "w", encoding="utf-8") as _file:
        json.dump(_metadata, _file, ensure_ascii=False, indent=2)

# --------------------------------------------------
# 予測結果を作る
# 役割:
#   test loader を走査して予測値と正解値を一覧化する
# --------------------------------------------------
def predict_regression(
    _model,
    _loader,
    _device,
    _context_df,
    _baseline_prediction,
    _use_range_normalization,
):
    _prediction_payload = predict_loader(_model, _loader, _device, _use_range_normalization)
    _true_array = _prediction_payload["true_raw"]
    _pred_array = _prediction_payload["pred_raw"]

    _context_df = _context_df.copy().reset_index().rename(columns={"index": "dataset_row_index"})
    if len(_context_df) != len(_true_array):
        raise RuntimeError("Prediction context row count does not match prediction count")

    _prediction_df = pd.DataFrame(
        {
            "test_row_index": list(range(len(_true_array))),
            "dataset_row_index": _context_df["dataset_row_index"].tolist(),
            "true_delta": _true_array.tolist(),
            "pred_delta": _pred_array.tolist(),
            "target_scale": _prediction_payload["target_scale"].tolist(),
            "true_scaled_delta": _prediction_payload["true_model"].tolist(),
            "pred_scaled_delta": _prediction_payload["pred_model"].tolist(),
            "direction_target": _prediction_payload["true_direction"].tolist(),
            "direction_target_active": _prediction_payload["direction_active"].tolist(),
            "direction_target_scaled_delta": _prediction_payload["direction_scaled_delta"].tolist(),
            "direction_logit": _prediction_payload["direction_logit"].tolist(),
            "direction_prob_up": _prediction_payload["direction_prob_up"].tolist(),
            "baseline_constant_pred_delta": [float(_baseline_prediction)] * len(_true_array),
        }
    )

    for _column_name in PREDICTION_CONTEXT_COLUMNS:
        if _column_name in _context_df.columns:
            _prediction_df[_column_name] = _context_df[_column_name].tolist()

    _prediction_df["true_direction"] = np.where(_prediction_df["true_delta"] >= 0.0, "UP", "DOWN")
    _prediction_df["pred_direction"] = np.where(_prediction_df["pred_delta"] >= 0.0, "UP", "DOWN")
    _direction_head_available = _prediction_df["direction_prob_up"].notna()
    _direction_target_active = _prediction_df["direction_target_active"] >= 0.5
    _prediction_df["direction_target_label"] = np.where(
        _direction_target_active,
        np.where(_prediction_df["direction_target"] >= 0.5, "UP", "DOWN"),
        "NEUTRAL",
    )
    _prediction_df["pred_direction_head"] = np.where(
        _direction_head_available,
        np.where(_prediction_df["direction_prob_up"] >= 0.5, "UP", "DOWN"),
        "",
    )
    _prediction_df["baseline_direction"] = np.where(
        _prediction_df["baseline_constant_pred_delta"] >= 0.0,
        "UP",
        "DOWN",
    )
    _prediction_df["direction_match"] = (
        _prediction_df["true_direction"] == _prediction_df["pred_direction"]
    )
    _prediction_df["baseline_direction_match"] = (
        _prediction_df["true_direction"] == _prediction_df["baseline_direction"]
    )
    _prediction_df["direction_head_match"] = np.where(
        _direction_head_available & _direction_target_active,
        _prediction_df["direction_target_label"] == _prediction_df["pred_direction_head"],
        np.nan,
    )
    _prediction_df["direction_head_agrees_with_regression"] = np.where(
        _direction_head_available,
        _prediction_df["pred_direction_head"] == _prediction_df["pred_direction"],
        np.nan,
    )
    _prediction_df["error"] = _prediction_df["pred_delta"] - _prediction_df["true_delta"]
    _prediction_df["abs_error"] = (_prediction_df["pred_delta"] - _prediction_df["true_delta"]).abs()
    _prediction_df["baseline_abs_error"] = (
        _prediction_df["baseline_constant_pred_delta"] - _prediction_df["true_delta"]
    ).abs()
    _prediction_df["pred_abs_delta"] = _prediction_df["pred_delta"].abs()
    _prediction_df["true_abs_delta"] = _prediction_df["true_delta"].abs()

    return _prediction_df


# --------------------------------------------------
# test予測結果をCSV保存する
# 役割:
#   回帰予測の個別結果を後から確認できるようにする
# --------------------------------------------------
def save_test_predictions(_output_path, _prediction_df):
    EnsureParentDirectory(_output_path)
    _prediction_df.to_csv(_output_path, index=False, encoding="utf-8")

# --------------------------------------------------
# メイン処理
# 役割:
#   CSV読込から学習、評価、保存までを一通り実行する
# --------------------------------------------------
def main():
    _args = parse_args()
    set_seed(int(_args.seed))

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== H1 LSTM Regressor Train Start ==========")
    print(f"[INFO] cwd={Path.cwd()}")
    print(f"[INFO] device={_device}")

    _df = load_dataset(_args.dataset)
    _target_column = get_target_column(_df)
    _sequence_length = get_sequence_length(_df)
    if bool(_args.use_derived_static_features):
        _df = ensure_derived_static_features(_df, _sequence_length)
    _sequence_feature_columns, _static_feature_columns = get_feature_columns(_df)
    _use_target_range_normalization = bool(_args.use_target_range_normalization)
    _use_multitask_direction_head = bool(_args.use_multitask_direction_head)

    _sequence_x, _sequence_feature_names = build_feature_tensor(
        _df,
        _sequence_feature_columns,
        _sequence_length,
    )
    _static_x, _static_feature_names = build_static_feature_array(_df, _static_feature_columns)
    _target_scale = build_target_scale_array(_df, _sequence_length, int(_args.target_scale_lookback))
    _raw_y = build_target_array(_df, _target_column)
    _y = transform_target_array(_raw_y, _target_scale, _use_target_range_normalization)
    _train_end, _valid_end = get_split_boundaries(len(_sequence_x))

    _sequence_x_train = _sequence_x[:_train_end]
    _sequence_x_valid = _sequence_x[_train_end:_valid_end]
    _sequence_x_test = _sequence_x[_valid_end:]
    _static_x_train = _static_x[:_train_end]
    _static_x_valid = _static_x[_train_end:_valid_end]
    _static_x_test = _static_x[_valid_end:]
    _target_scale_train = _target_scale[:_train_end]
    _target_scale_valid = _target_scale[_train_end:_valid_end]
    _target_scale_test = _target_scale[_valid_end:]
    _y_train = _y[:_train_end]
    _y_valid = _y[_train_end:_valid_end]
    _y_test = _y[_valid_end:]
    _raw_y_train = _raw_y[:_train_end]
    _raw_y_valid = _raw_y[_train_end:_valid_end]
    _raw_y_test = _raw_y[_valid_end:]
    _direction_neutral_threshold = fit_direction_neutral_threshold(
        _raw_y_train,
        _target_scale_train,
        float(_args.direction_neutral_quantile),
    )
    _direction_y_train, _direction_active_train, _direction_scaled_delta_train = build_direction_target_arrays(
        _raw_y_train,
        _target_scale_train,
        _direction_neutral_threshold,
    )
    _direction_y_valid, _direction_active_valid, _direction_scaled_delta_valid = build_direction_target_arrays(
        _raw_y_valid,
        _target_scale_valid,
        _direction_neutral_threshold,
    )
    _direction_y_test, _direction_active_test, _direction_scaled_delta_test = build_direction_target_arrays(
        _raw_y_test,
        _target_scale_test,
        _direction_neutral_threshold,
    )
    _test_context_df = _df.iloc[_valid_end:].copy()

    _sequence_mean, _sequence_std = fit_feature_scaler(_sequence_x_train)
    _static_mean, _static_std = fit_feature_scaler(_static_x_train)

    _sequence_x_train = transform_feature_tensor(_sequence_x_train, _sequence_mean, _sequence_std)
    _sequence_x_valid = transform_feature_tensor(_sequence_x_valid, _sequence_mean, _sequence_std)
    _sequence_x_test = transform_feature_tensor(_sequence_x_test, _sequence_mean, _sequence_std)
    _static_x_train = transform_feature_tensor(_static_x_train, _static_mean, _static_std)
    _static_x_valid = transform_feature_tensor(_static_x_valid, _static_mean, _static_std)
    _static_x_test = transform_feature_tensor(_static_x_test, _static_mean, _static_std)

    _train_loader = build_dataloader(
        _sequence_x_train,
        _static_x_train,
        _y_train,
        _raw_y_train,
        _direction_y_train,
        _direction_active_train,
        _direction_scaled_delta_train,
        _target_scale_train,
        int(_args.batch_size),
        True,
    )
    _valid_loader = build_dataloader(
        _sequence_x_valid,
        _static_x_valid,
        _y_valid,
        _raw_y_valid,
        _direction_y_valid,
        _direction_active_valid,
        _direction_scaled_delta_valid,
        _target_scale_valid,
        int(_args.batch_size),
        False,
    )
    _test_loader = build_dataloader(
        _sequence_x_test,
        _static_x_test,
        _y_test,
        _raw_y_test,
        _direction_y_test,
        _direction_active_test,
        _direction_scaled_delta_test,
        _target_scale_test,
        int(_args.batch_size),
        False,
    )

    _model = H1LSTMRegressor(
        _input_size=len(_sequence_feature_names),
        _hidden_size=int(_args.hidden_size),
        _num_layers=int(_args.num_layers),
        _dropout=float(_args.dropout),
        _static_input_size=len(_static_feature_names),
        _use_multitask_direction_head=_use_multitask_direction_head,
    ).to(_device)

    _criterion = nn.MSELoss()
    _direction_criterion = nn.BCEWithLogitsLoss()
    _optimizer = torch.optim.Adam(_model.parameters(), lr=float(_args.learning_rate))
    _baseline_prediction = float(np.mean(_raw_y_train))
    _valid_baseline_metrics = evaluate_constant_baseline(_raw_y_valid, _baseline_prediction)
    _test_baseline_metrics = evaluate_constant_baseline(_raw_y_test, _baseline_prediction)

    print(f"[INFO] dataset_rows={len(_df)}")
    print(f"[INFO] sequence_length={_sequence_length}")
    print(f"[INFO] sequence_feature_count={len(_sequence_feature_names)}")
    print(f"[INFO] static_feature_count={len(_static_feature_names)}")
    print(f"[INFO] use_derived_static_features={int(bool(_args.use_derived_static_features))}")
    print(f"[INFO] use_target_range_normalization={int(_use_target_range_normalization)}")
    print(f"[INFO] use_multitask_direction_head={int(_use_multitask_direction_head)}")
    print(f"[INFO] direction_loss_weight={float(_args.direction_loss_weight):.4f}")
    print(f"[INFO] direction_neutral_quantile={float(_args.direction_neutral_quantile):.4f}")
    print(f"[INFO] direction_neutral_threshold={float(_direction_neutral_threshold):.8f}")
    print(f"[INFO] target_scale_lookback={int(_args.target_scale_lookback)}")
    print(f"[INFO] train_target_scale_mean={float(np.mean(_target_scale_train)):.8f}")
    print(f"[INFO] train_target_scale_std={float(np.std(_target_scale_train)):.8f}")
    print(f"[INFO] direction_active_ratio_train={float(np.mean(_direction_active_train)):.4f}")
    print(f"[INFO] direction_active_ratio_valid={float(np.mean(_direction_active_valid)):.4f}")
    print(f"[INFO] direction_active_ratio_test={float(np.mean(_direction_active_test)):.4f}")
    print(f"[INFO] target_column={_target_column}")
    print(f"[INFO] train_size={len(_sequence_x_train)}")
    print(f"[INFO] valid_size={len(_sequence_x_valid)}")
    print(f"[INFO] test_size={len(_sequence_x_test)}")
    print(f"[INFO] baseline_constant_pred={_baseline_prediction:.8f}")
    print(
        f"[INFO] valid_baseline_mse={_valid_baseline_metrics['mse']:.8f} "
        f"valid_baseline_mae={_valid_baseline_metrics['mae']:.8f} "
        f"valid_baseline_dir_acc={_valid_baseline_metrics['directional_accuracy']:.4f}"
    )
    print(
        f"[INFO] test_baseline_mse={_test_baseline_metrics['mse']:.8f} "
        f"test_baseline_mae={_test_baseline_metrics['mae']:.8f} "
        f"test_baseline_dir_acc={_test_baseline_metrics['directional_accuracy']:.4f}"
    )

    _best_valid_score = None
    _best_valid_mse = None
    _best_state_dict = None
    _best_epoch = None
    _no_improve_count = 0

    for _epoch in range(1, int(_args.epochs) + 1):
        _train_metrics = train_one_epoch(
            _model,
            _train_loader,
            _criterion,
            _direction_criterion,
            _optimizer,
            _device,
            _use_multitask_direction_head,
            float(_args.direction_loss_weight),
        )
        _valid_metrics = evaluate_model(_model, _valid_loader, _device, _use_target_range_normalization)
        _valid_selection_score = _valid_metrics["target_mse"]
        if _use_multitask_direction_head:
            _direction_head_bce = _valid_metrics.get("direction_head_bce")
            if _direction_head_bce is not None:
                _valid_selection_score += float(_args.direction_loss_weight) * float(_direction_head_bce)

        _is_improved = (
            _best_valid_score is None or
            (_best_valid_score - _valid_selection_score) > float(_args.min_delta)
        )

        if _is_improved:
            _best_valid_score = _valid_selection_score
            _best_valid_mse = _valid_metrics["mse"]
            _best_state_dict = {k: v.cpu().clone() for k, v in _model.state_dict().items()}
            _best_epoch = _epoch
            _no_improve_count = 0
        else:
            _no_improve_count += 1

        if _args.verbose:
            _message = (
                f"[INFO] epoch={_epoch} "
                f"train_total_loss={_train_metrics['total_loss']:.8f} "
                f"train_target_mse={_train_metrics['target_loss']:.8f} "
                f"valid_target_mse={_valid_metrics['target_mse']:.8f} "
                f"valid_mse={_valid_metrics['mse']:.8f} "
                f"valid_mae={_valid_metrics['mae']:.8f} "
                f"valid_dir_acc={_valid_metrics['directional_accuracy']:.4f} "
                f"valid_pred_up_ratio={_valid_metrics['pred_up_ratio']:.4f} "
                f"valid_corr={_valid_metrics['correlation']:.4f}"
            )
            if _use_multitask_direction_head:
                _valid_direction_head_bce = _valid_metrics.get("direction_head_bce")
                _valid_direction_head_acc = _valid_metrics.get("direction_head_accuracy")
                _message += (
                    f" train_direction_loss={_train_metrics['direction_loss']:.8f} "
                    f"train_direction_active_ratio={_train_metrics['direction_active_ratio']:.4f} "
                    f"valid_direction_head_bce={float(_valid_direction_head_bce or 0.0):.8f} "
                    f"valid_direction_head_acc={float(_valid_direction_head_acc or 0.0):.4f} "
                    f"valid_direction_head_eval_count={int(_valid_metrics.get('direction_head_eval_count', 0))}"
                )
            print(_message)

        if _no_improve_count >= int(_args.patience):
            print(
                f"[INFO] early_stopped=1 "
                f"best_epoch={_best_epoch} "
                f"patience={int(_args.patience)}"
            )
            break

    if _best_state_dict is None:
        raise RuntimeError("Training finished without a best model state")

    _model.load_state_dict(_best_state_dict)

    _test_metrics = evaluate_model(_model, _test_loader, _device, _use_target_range_normalization)

    _prediction_df = predict_regression(
        _model,
        _test_loader,
        _device,
        _test_context_df,
        _baseline_prediction,
        _use_target_range_normalization,
    )

    print(f"[INFO] best_epoch={_best_epoch}")
    if _best_valid_score is not None:
        print(f"[INFO] best_valid_selection_score={_best_valid_score:.8f}")
    print(f"[INFO] best_valid_mse={_best_valid_mse:.8f}")
    print(f"[INFO] test_target_mse={_test_metrics['target_mse']:.8f}")
    print(f"[INFO] test_target_mae={_test_metrics['target_mae']:.8f}")
    print(f"[INFO] test_mse={_test_metrics['mse']:.8f}")
    print(f"[INFO] test_mae={_test_metrics['mae']:.8f}")
    print(f"[INFO] test_dir_acc={_test_metrics['directional_accuracy']:.4f}")
    print(f"[INFO] test_pred_up_ratio={_test_metrics['pred_up_ratio']:.4f}")
    print(f"[INFO] test_true_up_ratio={_test_metrics['true_up_ratio']:.4f}")
    print(f"[INFO] test_up_recall={_test_metrics['up_recall']:.4f}")
    print(f"[INFO] test_down_recall={_test_metrics['down_recall']:.4f}")
    print(f"[INFO] test_pred_std={_test_metrics['pred_std']:.8f}")
    print(f"[INFO] test_true_std={_test_metrics['true_std']:.8f}")
    print(f"[INFO] test_corr={_test_metrics['correlation']:.4f}")
    if _use_multitask_direction_head:
        print(f"[INFO] test_direction_head_bce={float(_test_metrics.get('direction_head_bce') or 0.0):.8f}")
        print(f"[INFO] test_direction_head_acc={float(_test_metrics.get('direction_head_accuracy') or 0.0):.4f}")
        print(f"[INFO] test_direction_head_pred_up_ratio={float(_test_metrics.get('direction_head_pred_up_ratio') or 0.0):.4f}")
        print(f"[INFO] test_direction_head_up_recall={float(_test_metrics.get('direction_head_up_recall') or 0.0):.4f}")
        print(f"[INFO] test_direction_head_down_recall={float(_test_metrics.get('direction_head_down_recall') or 0.0):.4f}")
        print(f"[INFO] test_direction_head_eval_count={int(_test_metrics.get('direction_head_eval_count', 0))}")
        print(f"[INFO] test_direction_head_active_ratio={float(_test_metrics.get('direction_head_active_ratio', 0.0)):.4f}")

    save_model(_model, _args.model_output)
    save_metadata(
        _args.metadata_output,
        _sequence_feature_names,
        _static_feature_names,
        _sequence_length,
        _target_column,
        int(_args.target_scale_lookback),
        _use_target_range_normalization,
        _use_multitask_direction_head,
        float(_args.direction_loss_weight),
        float(_args.direction_neutral_quantile),
        float(_direction_neutral_threshold),
        {
            "train": float(np.mean(_direction_active_train)),
            "valid": float(np.mean(_direction_active_valid)),
            "test": float(np.mean(_direction_active_test)),
        },
        _sequence_mean,
        _sequence_std,
        _static_mean,
        _static_std,
        _args,
        {
            "train": len(_sequence_x_train),
            "valid": len(_sequence_x_valid),
            "test": len(_sequence_x_test),
        },
    )
    save_test_predictions(_args.prediction_output, _prediction_df)

    print(f"[INFO] model_saved={Path(_args.model_output).resolve()}")
    print(f"[INFO] metadata_saved={Path(_args.metadata_output).resolve()}")
    print(f"[INFO] prediction_saved={Path(_args.prediction_output).resolve()}")
    print("========== H1 LSTM Regressor Train End ==========")


if __name__ == "__main__":
    main()
