import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.Utility.Utility import EnsureParentDirectory


DEFAULT_INPUT_CSV = "Asset/Models/h1_lstm_regressor_test_predictions.csv"
DEFAULT_OUTPUT_DIR = "Asset/Models/h1_lstm_regressor_analysis"
DEFAULT_TOP_N_ERRORS = 20
ABS_DELTA_BINS = [0.0, 0.05, 0.10, 0.20, 0.30, np.inf]
ABS_DELTA_LABELS = ["0.00_0.05", "0.05_0.10", "0.10_0.20", "0.20_0.30", "0.30_plus"]
FEATURE_BAND_COLUMNS = [
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
    "h1_recent_momentum",
    "h1_trend_consistency",
]
TOP_ERROR_COLUMNS = [
    "test_row_index",
    "dataset_row_index",
    "timestamp_jst",
    "future_timestamp_jst",
    "symbol",
    "entry_price",
    "true_delta",
    "pred_delta",
    "baseline_constant_pred_delta",
    "direction_target",
    "direction_target_label",
    "direction_target_active",
    "direction_target_scaled_delta",
    "direction_prob_up",
    "direction_logit",
    "error",
    "abs_error",
    "baseline_abs_error",
    "abs_error_improvement_vs_baseline",
    "true_direction",
    "pred_direction",
    "pred_direction_head",
    "direction_match",
    "direction_head_match",
    "direction_head_agrees_with_regression",
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
    "h1_recent_momentum",
    "h1_trend_consistency",
]
WEEKDAY_NAME_MAP = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


# --------------------------------------------------
# CLI引数を読み込む
# 役割:
#   予測CSVと分析出力先を受け取る
# --------------------------------------------------
def parse_args():
    if len(sys.argv) == 1:
        return SimpleNamespace(
            input_csv=DEFAULT_INPUT_CSV,
            output_dir=DEFAULT_OUTPUT_DIR,
            top_n_errors=DEFAULT_TOP_N_ERRORS,
            verbose=True,
        )

    _parser = argparse.ArgumentParser(description="Analyze H1 LSTM regressor prediction CSV")
    _parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Prediction CSV path")
    _parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save analysis outputs")
    _parser.add_argument("--top-n-errors", type=int, default=DEFAULT_TOP_N_ERRORS, help="Number of worst cases to save")
    _parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    return _parser.parse_args()


# --------------------------------------------------
# 予測CSVを読み込む
# 役割:
#   学習後に出力された test prediction CSV をDataFrameへ変換する
# --------------------------------------------------
def load_prediction_csv(_input_csv):
    _input_path = Path(_input_csv).resolve()

    if not _input_path.exists():
        raise RuntimeError(f"Prediction CSV was not found: {_input_path}")

    _df = pd.read_csv(_input_path)

    if len(_df) == 0:
        raise RuntimeError(f"Prediction CSV is empty: {_input_path}")

    return _df, _input_path


# --------------------------------------------------
# 分析用に列を整形する
# 役割:
#   数値列や時刻列を集計しやすい形へ統一する
# --------------------------------------------------
def prepare_prediction_df(_df):
    _required_columns = [
        "true_delta",
        "pred_delta",
        "baseline_constant_pred_delta",
        "abs_error",
        "baseline_abs_error",
        "true_direction",
        "pred_direction",
    ]

    for _column_name in _required_columns:
        if _column_name not in _df.columns:
            raise RuntimeError(f"Required prediction column was not found: {_column_name}")

    _work_df = _df.copy()

    _numeric_columns = [
        "true_delta",
        "pred_delta",
        "baseline_constant_pred_delta",
        "direction_target",
        "direction_target_active",
        "direction_target_scaled_delta",
        "direction_prob_up",
        "direction_logit",
        "error",
        "abs_error",
        "baseline_abs_error",
        "pred_abs_delta",
        "true_abs_delta",
        "entry_price",
        *FEATURE_BAND_COLUMNS,
    ]

    for _column_name in _numeric_columns:
        if _column_name in _work_df.columns:
            _work_df[_column_name] = pd.to_numeric(_work_df[_column_name], errors="coerce")

    for _column_name in ["timestamp_jst", "future_timestamp_jst"]:
        if _column_name in _work_df.columns:
            _work_df[_column_name] = pd.to_datetime(_work_df[_column_name], errors="coerce")

    if "direction_match" in _work_df.columns:
        _work_df["direction_match"] = _work_df["direction_match"].astype(str).str.lower().eq("true")

    if "baseline_direction_match" in _work_df.columns:
        _work_df["baseline_direction_match"] = _work_df["baseline_direction_match"].astype(str).str.lower().eq("true")

    if "direction_head_match" in _work_df.columns:
        _direction_head_match_text = _work_df["direction_head_match"].astype(str).str.lower()
        _work_df["direction_head_match"] = np.where(
            _direction_head_match_text.isin(["true", "false"]),
            _direction_head_match_text.eq("true"),
            np.nan,
        )

    if "direction_head_agrees_with_regression" in _work_df.columns:
        _direction_head_agree_text = _work_df["direction_head_agrees_with_regression"].astype(str).str.lower()
        _work_df["direction_head_agrees_with_regression"] = np.where(
            _direction_head_agree_text.isin(["true", "false"]),
            _direction_head_agree_text.eq("true"),
            np.nan,
        )

    if "direction_target_active" in _work_df.columns:
        _work_df["direction_target_active"] = pd.to_numeric(_work_df["direction_target_active"], errors="coerce")

    if "timestamp_jst" in _work_df.columns:
        _work_df["hour"] = _work_df["timestamp_jst"].dt.hour
        _work_df["weekday"] = _work_df["timestamp_jst"].dt.weekday
        _work_df["weekday_name"] = _work_df["weekday"].map(WEEKDAY_NAME_MAP)

    if "baseline_abs_error" in _work_df.columns and "abs_error" in _work_df.columns:
        _work_df["abs_error_improvement_vs_baseline"] = _work_df["baseline_abs_error"] - _work_df["abs_error"]

    if "true_abs_delta" not in _work_df.columns:
        _work_df["true_abs_delta"] = _work_df["true_delta"].abs()

    if "pred_abs_delta" not in _work_df.columns:
        _work_df["pred_abs_delta"] = _work_df["pred_delta"].abs()

    return _work_df


# --------------------------------------------------
# 配列相関を安全に計算する
# 役割:
#   分散がない場合のNaNを避けて相関を返す
# --------------------------------------------------
def safe_correlation(_x_array, _y_array):
    _x_array = np.asarray(_x_array, dtype=np.float32)
    _y_array = np.asarray(_y_array, dtype=np.float32)

    if len(_x_array) == 0 or len(_y_array) == 0:
        return 0.0

    if float(np.std(_x_array)) <= 1.0e-8 or float(np.std(_y_array)) <= 1.0e-8:
        return 0.0

    _corr = float(np.corrcoef(_x_array, _y_array)[0, 1])
    if not np.isfinite(_corr):
        return 0.0

    return _corr


# --------------------------------------------------
# サブセット単位の評価指標を作る
# 役割:
#   モデルとベースラインを同じ物差しで比較する
# --------------------------------------------------
def build_metrics_for_subset(_df):
    if len(_df) == 0:
        return {
            "count": 0,
            "mse": 0.0,
            "mae": 0.0,
            "baseline_mse": 0.0,
            "baseline_mae": 0.0,
            "mae_delta_vs_baseline": 0.0,
            "direction_accuracy": 0.0,
            "baseline_direction_accuracy": 0.0,
            "direction_accuracy_delta_vs_baseline": 0.0,
            "pred_up_ratio": 0.0,
            "true_up_ratio": 0.0,
            "up_recall": 0.0,
            "down_recall": 0.0,
            "pred_mean": 0.0,
            "true_mean": 0.0,
            "pred_std": 0.0,
            "true_std": 0.0,
            "correlation": 0.0,
            "baseline_correlation": 0.0,
            "mean_abs_error_improvement_vs_baseline": 0.0,
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

    _true = _df["true_delta"].to_numpy(dtype=np.float32)
    _pred = _df["pred_delta"].to_numpy(dtype=np.float32)
    _baseline = _df["baseline_constant_pred_delta"].to_numpy(dtype=np.float32)

    _true_up = _true >= 0.0
    _pred_up = _pred >= 0.0
    _baseline_up = _baseline >= 0.0
    _true_down = ~_true_up

    _pred_error = _pred - _true
    _baseline_error = _baseline - _true

    _metrics = {
        "count": int(len(_df)),
        "mse": float(np.mean(np.square(_pred_error))),
        "mae": float(np.mean(np.abs(_pred_error))),
        "baseline_mse": float(np.mean(np.square(_baseline_error))),
        "baseline_mae": float(np.mean(np.abs(_baseline_error))),
        "mae_delta_vs_baseline": float(np.mean(np.abs(_baseline_error)) - np.mean(np.abs(_pred_error))),
        "direction_accuracy": float(np.mean(_pred_up == _true_up)),
        "baseline_direction_accuracy": float(np.mean(_baseline_up == _true_up)),
        "direction_accuracy_delta_vs_baseline": float(np.mean(_pred_up == _true_up) - np.mean(_baseline_up == _true_up)),
        "pred_up_ratio": float(np.mean(_pred_up)),
        "true_up_ratio": float(np.mean(_true_up)),
        "up_recall": float(np.mean(_pred_up[_true_up])) if _true_up.any() else 0.0,
        "down_recall": float(np.mean((~_pred_up)[_true_down])) if _true_down.any() else 0.0,
        "pred_mean": float(np.mean(_pred)),
        "true_mean": float(np.mean(_true)),
        "pred_std": float(np.std(_pred)),
        "true_std": float(np.std(_true)),
        "correlation": safe_correlation(_true, _pred),
        "baseline_correlation": safe_correlation(_true, _baseline),
        "mean_abs_error_improvement_vs_baseline": float(_df["abs_error_improvement_vs_baseline"].mean()),
    }

    if "direction_prob_up" not in _df.columns:
        _metrics.update(
            {
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
        )
        return _metrics

    _direction_prob = _df["direction_prob_up"].to_numpy(dtype=np.float32)
    if "direction_target_active" in _df.columns:
        _direction_active = _df["direction_target_active"].to_numpy(dtype=np.float32) >= 0.5
    else:
        _direction_active = np.ones(len(_df), dtype=bool)

    if "direction_target" in _df.columns:
        _direction_target = _df["direction_target"].to_numpy(dtype=np.float32) >= 0.5
    else:
        _direction_target = _true_up

    _valid_mask = np.isfinite(_direction_prob) & _direction_active

    if not _valid_mask.any():
        _metrics.update(
            {
                "direction_head_bce": None,
                "direction_head_accuracy": None,
                "direction_head_pred_up_ratio": None,
                "direction_head_up_recall": None,
                "direction_head_down_recall": None,
                "direction_head_prob_mean": None,
                "direction_head_prob_std": None,
                "direction_head_eval_count": 0,
                "direction_head_active_ratio": float(np.mean(_direction_active)) if len(_direction_active) > 0 else 0.0,
            }
        )
        return _metrics

    _direction_prob = np.clip(_direction_prob[_valid_mask], 1.0e-6, 1.0 - 1.0e-6)
    _true_up_head = _direction_target[_valid_mask]
    _true_down_head = ~_true_up_head
    _pred_up_head = _direction_prob >= 0.5
    _direction_bce = -np.mean(
        (_true_up_head.astype(np.float32) * np.log(_direction_prob)) +
        ((~_true_up_head).astype(np.float32) * np.log(1.0 - _direction_prob))
    )

    _metrics.update(
        {
            "direction_head_bce": float(_direction_bce),
            "direction_head_accuracy": float(np.mean(_pred_up_head == _true_up_head)),
            "direction_head_pred_up_ratio": float(np.mean(_pred_up_head)),
            "direction_head_up_recall": float(np.mean(_pred_up_head[_true_up_head])) if _true_up_head.any() else 0.0,
            "direction_head_down_recall": float(np.mean((~_pred_up_head)[_true_down_head])) if _true_down_head.any() else 0.0,
            "direction_head_prob_mean": float(np.mean(_direction_prob)),
            "direction_head_prob_std": float(np.std(_direction_prob)),
            "direction_head_eval_count": int(_valid_mask.sum()),
            "direction_head_active_ratio": float(np.mean(_direction_active)),
        }
    )

    return _metrics


# --------------------------------------------------
# グループ集計DataFrameを作る
# 役割:
#   任意列ごとにモデル性能を比較できる表を作る
# --------------------------------------------------
def build_group_summary(_df, _group_column):
    _rows = []

    for _group_value, _subset_df in _df.groupby(_group_column, dropna=False, sort=True, observed=False):
        _metrics = build_metrics_for_subset(_subset_df)
        _metrics[_group_column] = _group_value
        _rows.append(_metrics)

    if len(_rows) == 0:
        return pd.DataFrame()

    _summary_df = pd.DataFrame(_rows)
    _ordered_columns = [_group_column, *[c for c in _summary_df.columns if c != _group_column]]
    return _summary_df[_ordered_columns].copy()


# --------------------------------------------------
# 曜日順へ整列する
# 役割:
#   集計表の並びを人間が読みやすい順に揃える
# --------------------------------------------------
def sort_weekday_summary(_weekday_summary_df):
    if len(_weekday_summary_df) == 0:
        return _weekday_summary_df

    _weekday_summary_df = _weekday_summary_df.copy()
    _weekday_summary_df["weekday_sort"] = _weekday_summary_df["weekday"].astype(int)
    _weekday_summary_df = _weekday_summary_df.sort_values("weekday_sort").drop(columns=["weekday_sort"])
    return _weekday_summary_df.reset_index(drop=True)


# --------------------------------------------------
# 方向別集計を作る
# 役割:
#   UP局面とDOWN局面での外し方の差を見る
# --------------------------------------------------
def build_true_direction_summary(_df):
    return build_group_summary(_df, "true_direction")


# --------------------------------------------------
# 時間帯別集計を作る
# 役割:
#   何時台が苦手かを確認する
# --------------------------------------------------
def build_hour_summary(_df):
    if "hour" not in _df.columns:
        return pd.DataFrame()

    _summary_df = build_group_summary(_df, "hour")
    return _summary_df.sort_values("hour").reset_index(drop=True)


# --------------------------------------------------
# 曜日別集計を作る
# 役割:
#   曜日ごとの相場癖との相性を見る
# --------------------------------------------------
def build_weekday_summary(_df):
    if "weekday" not in _df.columns or "weekday_name" not in _df.columns:
        return pd.DataFrame()

    _summary_df = build_group_summary(_df, "weekday")
    _summary_df["weekday_name"] = _summary_df["weekday"].map(WEEKDAY_NAME_MAP)
    return sort_weekday_summary(_summary_df)


# --------------------------------------------------
# 変動幅帯の集計を作る
# 役割:
#   大きく動く局面ほど崩れていないかを確認する
# --------------------------------------------------
def build_abs_delta_band_summary(_df):
    _work_df = _df.copy()
    _work_df["true_abs_delta_band"] = pd.cut(
        _work_df["true_abs_delta"],
        bins=ABS_DELTA_BINS,
        labels=ABS_DELTA_LABELS,
        include_lowest=True,
        right=False,
    )

    _summary_df = build_group_summary(_work_df, "true_abs_delta_band")

    if len(_summary_df) == 0:
        return _summary_df

    _summary_df["true_abs_delta_band"] = pd.Categorical(
        _summary_df["true_abs_delta_band"],
        categories=ABS_DELTA_LABELS,
        ordered=True,
    )
    return _summary_df.sort_values("true_abs_delta_band").reset_index(drop=True)


# --------------------------------------------------
# 特徴量帯の集計を作る
# 役割:
#   どの特徴量レンジで外しているかを横断的に見る
# --------------------------------------------------
def build_feature_band_summary(_df, _feature_columns):
    _rows = []

    for _column_name in _feature_columns:
        if _column_name not in _df.columns:
            continue

        _subset_df = _df[_df[_column_name].notna()].copy()
        if len(_subset_df) < 8 or _subset_df[_column_name].nunique() < 4:
            continue

        try:
            _subset_df["feature_band"] = pd.qcut(_subset_df[_column_name], q=4, duplicates="drop")
        except Exception:
            continue

        for _band_value, _band_df in _subset_df.groupby("feature_band", dropna=False, sort=True, observed=False):
            _metrics = build_metrics_for_subset(_band_df)
            _metrics["feature_name"] = _column_name
            _metrics["feature_band"] = str(_band_value)
            _metrics["feature_min"] = float(_band_df[_column_name].min())
            _metrics["feature_max"] = float(_band_df[_column_name].max())
            _rows.append(_metrics)

    if len(_rows) == 0:
        return pd.DataFrame()

    _summary_df = pd.DataFrame(_rows)
    _ordered_columns = [
        "feature_name",
        "feature_band",
        "feature_min",
        "feature_max",
        *[c for c in _summary_df.columns if c not in {"feature_name", "feature_band", "feature_min", "feature_max"}],
    ]
    return _summary_df[_ordered_columns].sort_values(["feature_name", "feature_min"]).reset_index(drop=True)


# --------------------------------------------------
# 大外しケースを抽出する
# 役割:
#   個別局面をあとで読み返せるようにする
# --------------------------------------------------
def build_top_error_cases(_df, _top_n_errors):
    _work_df = _df.copy().sort_values("abs_error", ascending=False).head(int(_top_n_errors))
    _columns = [c for c in TOP_ERROR_COLUMNS if c in _work_df.columns]
    return _work_df[_columns].reset_index(drop=True)


# --------------------------------------------------
# 方向外しケースを抽出する
# 役割:
#   符号を逆に読んだ場面だけを重点確認できるようにする
# --------------------------------------------------
def build_direction_miss_cases(_df, _top_n_errors):
    if "direction_match" not in _df.columns:
        return pd.DataFrame()

    _work_df = _df[~_df["direction_match"]].copy().sort_values("abs_error", ascending=False).head(int(_top_n_errors))
    _columns = [c for c in TOP_ERROR_COLUMNS if c in _work_df.columns]
    return _work_df[_columns].reset_index(drop=True)


# --------------------------------------------------
# JSONサマリを作る
# 役割:
#   後からざっと比較できる概要情報を残す
# --------------------------------------------------
def build_summary_payload(
    _input_path,
    _output_dir,
    _prediction_df,
    _direction_summary_df,
    _hour_summary_df,
    _weekday_summary_df,
    _abs_delta_band_summary_df,
    _feature_band_summary_df,
):
    _overall_metrics = build_metrics_for_subset(_prediction_df)

    _summary = {
        "input_csv": str(_input_path),
        "output_dir": str(Path(_output_dir).resolve()),
        "record_count": int(len(_prediction_df)),
        "overall_metrics": _overall_metrics,
        "direction_summary": _direction_summary_df.to_dict(orient="records"),
        "hour_worst_mae_top5": _hour_summary_df.sort_values("mae", ascending=False).head(5).to_dict(orient="records"),
        "hour_best_mae_gain_top5": _hour_summary_df.sort_values("mae_delta_vs_baseline", ascending=False).head(5).to_dict(orient="records"),
        "weekday_summary": _weekday_summary_df.to_dict(orient="records"),
        "abs_delta_band_summary": _abs_delta_band_summary_df.to_dict(orient="records"),
        "feature_band_worst_mae_delta_top10": _feature_band_summary_df.sort_values("mae_delta_vs_baseline", ascending=True).head(10).to_dict(orient="records"),
        "feature_band_best_mae_delta_top10": _feature_band_summary_df.sort_values("mae_delta_vs_baseline", ascending=False).head(10).to_dict(orient="records"),
    }

    return _summary


# --------------------------------------------------
# DataFrameをCSV保存する
# 役割:
#   集計結果を比較しやすい表として残す
# --------------------------------------------------
def save_dataframe(_output_path, _df):
    EnsureParentDirectory(_output_path)
    _df.to_csv(_output_path, index=False, encoding="utf-8")


# --------------------------------------------------
# JSONを保存する
# 役割:
#   概要を機械可読な形で残す
# --------------------------------------------------
def save_json(_output_path, _payload):
    EnsureParentDirectory(_output_path)

    with open(_output_path, "w", encoding="utf-8") as _file:
        json.dump(_payload, _file, ensure_ascii=False, indent=2)
        _file.write("\n")


# --------------------------------------------------
# 主要サマリを表示する
# 役割:
#   実行直後に重要な気づきを確認できるようにする
# --------------------------------------------------
def print_summary(_summary_payload):
    _metrics = _summary_payload["overall_metrics"]

    print("========== H1 Prediction Analysis Summary ==========")
    print(f"record_count={_summary_payload['record_count']}")
    print(f"mae={_metrics['mae']:.8f}")
    print(f"baseline_mae={_metrics['baseline_mae']:.8f}")
    print(f"mae_delta_vs_baseline={_metrics['mae_delta_vs_baseline']:.8f}")
    print(f"direction_accuracy={_metrics['direction_accuracy']:.4f}")
    print(f"baseline_direction_accuracy={_metrics['baseline_direction_accuracy']:.4f}")
    print(f"direction_accuracy_delta_vs_baseline={_metrics['direction_accuracy_delta_vs_baseline']:.4f}")
    print(f"pred_up_ratio={_metrics['pred_up_ratio']:.4f}")
    print(f"true_up_ratio={_metrics['true_up_ratio']:.4f}")
    print(f"up_recall={_metrics['up_recall']:.4f}")
    print(f"down_recall={_metrics['down_recall']:.4f}")
    print(f"correlation={_metrics['correlation']:.4f}")
    if _metrics.get("direction_head_accuracy") is not None:
        print(f"direction_head_bce={_metrics['direction_head_bce']:.8f}")
        print(f"direction_head_accuracy={_metrics['direction_head_accuracy']:.4f}")
        print(f"direction_head_pred_up_ratio={_metrics['direction_head_pred_up_ratio']:.4f}")
        print(f"direction_head_up_recall={_metrics['direction_head_up_recall']:.4f}")
        print(f"direction_head_down_recall={_metrics['direction_head_down_recall']:.4f}")
        print(f"direction_head_eval_count={int(_metrics['direction_head_eval_count'])}")
        print(f"direction_head_active_ratio={_metrics['direction_head_active_ratio']:.4f}")
    print("----- worst_hour_by_mae -----")
    for _row in _summary_payload["hour_worst_mae_top5"]:
        print(
            f"hour={int(_row['hour'])} "
            f"count={_row['count']} "
            f"mae={_row['mae']:.6f} "
            f"mae_delta_vs_baseline={_row['mae_delta_vs_baseline']:.6f} "
            f"dir_acc={_row['direction_accuracy']:.4f}"
        )

    print("----- worst_feature_band_by_mae_delta -----")
    for _row in _summary_payload["feature_band_worst_mae_delta_top10"][:5]:
        print(
            f"feature={_row['feature_name']} "
            f"band={_row['feature_band']} "
            f"count={_row['count']} "
            f"mae_delta_vs_baseline={_row['mae_delta_vs_baseline']:.6f} "
            f"dir_acc={_row['direction_accuracy']:.4f}"
        )


# --------------------------------------------------
# メイン処理
# 役割:
#   予測CSVを読み込み、各種集計を出力する
# --------------------------------------------------
def main():
    _args = parse_args()

    _prediction_df, _input_path = load_prediction_csv(_args.input_csv)
    _prediction_df = prepare_prediction_df(_prediction_df)

    _output_dir = Path(_args.output_dir).resolve()
    _direction_summary_df = build_true_direction_summary(_prediction_df)
    _hour_summary_df = build_hour_summary(_prediction_df)
    _weekday_summary_df = build_weekday_summary(_prediction_df)
    _abs_delta_band_summary_df = build_abs_delta_band_summary(_prediction_df)
    _feature_band_summary_df = build_feature_band_summary(_prediction_df, FEATURE_BAND_COLUMNS)
    _top_error_cases_df = build_top_error_cases(_prediction_df, int(_args.top_n_errors))
    _direction_miss_cases_df = build_direction_miss_cases(_prediction_df, int(_args.top_n_errors))

    _summary_payload = build_summary_payload(
        _input_path,
        _output_dir,
        _prediction_df,
        _direction_summary_df,
        _hour_summary_df,
        _weekday_summary_df,
        _abs_delta_band_summary_df,
        _feature_band_summary_df,
    )

    save_json(str(_output_dir / "summary.json"), _summary_payload)
    save_dataframe(str(_output_dir / "direction_summary.csv"), _direction_summary_df)
    save_dataframe(str(_output_dir / "hour_summary.csv"), _hour_summary_df)
    save_dataframe(str(_output_dir / "weekday_summary.csv"), _weekday_summary_df)
    save_dataframe(str(_output_dir / "abs_delta_band_summary.csv"), _abs_delta_band_summary_df)
    save_dataframe(str(_output_dir / "feature_band_summary.csv"), _feature_band_summary_df)
    save_dataframe(str(_output_dir / "top_abs_error_cases.csv"), _top_error_cases_df)
    save_dataframe(str(_output_dir / "top_direction_miss_cases.csv"), _direction_miss_cases_df)

    if _args.verbose:
        print(f"[INFO] input_csv={_input_path}")
        print(f"[INFO] output_dir={_output_dir}")
        print(f"[INFO] feature_band_rows={len(_feature_band_summary_df)}")
        print(f"[INFO] top_error_cases={len(_top_error_cases_df)}")
        print(f"[INFO] direction_miss_cases={len(_direction_miss_cases_df)}")

    print_summary(_summary_payload)


if __name__ == "__main__":
    main()
