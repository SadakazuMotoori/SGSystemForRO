import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


# --------------------------------------------------
# デバッグ実行用の既定値を返す
# 役割:
#   VSCodeでF5実行した時に、そのまま集計を流せるようにする
# --------------------------------------------------
def get_debug_args():
    return SimpleNamespace(
        input_csv="Src/Backtest/Output/raw_signals/raw_signals_mt5.csv",
        top_n_reason_codes=10,
    )


# --------------------------------------------------
# 実行引数を取得する
# 役割:
#   引数なし実行時はデバッグ用既定値、
#   引数あり実行時はCLI指定値を使う
# --------------------------------------------------
def parse_args():
    if len(sys.argv) == 1:
        return get_debug_args()

    if len(sys.argv) < 2:
        raise RuntimeError("input_csv を指定してください。")

    _input_csv = sys.argv[1]
    _top_n_reason_codes = 10

    if len(sys.argv) >= 3:
        try:
            _top_n_reason_codes = int(sys.argv[2])
        except Exception:
            _top_n_reason_codes = 10

    return SimpleNamespace(
        input_csv=_input_csv,
        top_n_reason_codes=_top_n_reason_codes,
    )


# --------------------------------------------------
# 入力CSVを読み込む
# 役割:
#   run_backtest.py が出力した raw_signals をDataFrameへ変換する
# --------------------------------------------------
def load_raw_signals(_input_csv):
    _input_path = Path(_input_csv).resolve()

    if not _input_path.exists():
        raise RuntimeError(f"入力CSVが見つかりません: {_input_path}")

    _df = pd.read_csv(_input_path)

    if len(_df) == 0:
        raise RuntimeError(f"入力CSVが空です: {_input_path}")

    return _df, _input_path


# --------------------------------------------------
# 真偽値列を安全に bool へ揃える
# 役割:
#   CSV読み込み後に文字列化された True/False を扱いやすくする
# --------------------------------------------------
def normalize_bool_series(_series):
    return _series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def convert_final_action_to_predicted_direction(_final_action):
    if _final_action == "ENTER_LONG":
        return "UP"

    if _final_action == "ENTER_SHORT":
        return "DOWN"

    return "NO_SIGNAL"


def calculate_direction_hit_rate(_df, _direction_column):
    if _direction_column not in _df.columns:
        return 0, 0, 0.0

    _work_df = _df.copy()
    _signal_mask = _work_df[_direction_column].isin(["UP", "DOWN"])
    _non_draw_mask = _work_df["actual_direction"].fillna("DRAW") != "DRAW"
    _evaluated_df = _work_df[_signal_mask & _non_draw_mask].copy()

    if len(_evaluated_df) == 0:
        return 0, 0, 0.0

    _correct_mask = _evaluated_df[_direction_column] == _evaluated_df["actual_direction"]
    return int(len(_evaluated_df)), int(_correct_mask.sum()), float(_correct_mask.mean())


# --------------------------------------------------
# サマリーを辞書形式で作る
# 役割:
#   後続の表示処理で使う集計結果をまとめる
# --------------------------------------------------
def build_summary(_df):
    _summary = {}

    _summary["total_records"] = int(len(_df))

    _summary["final_action_counts"] = (
        _df["final_action"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .to_dict()
    )

    if "base_final_action" in _df.columns:
        _summary["base_final_action_counts"] = (
            _df["base_final_action"]
            .fillna("UNKNOWN")
            .value_counts(dropna=False)
            .to_dict()
        )
    else:
        _summary["base_final_action_counts"] = {}

    _summary["external_filter_status_counts"] = (
        _df["external_filter_status"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .to_dict()
    )

    _summary["env_direction_counts"] = (
        _df["env_direction"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .to_dict()
    )

    _summary["predicted_direction_counts"] = (
        _df["predicted_direction"]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .to_dict()
    )

    _signal_mask = _df["predicted_direction"].isin(["UP", "DOWN"])
    _non_draw_mask = ~normalize_bool_series(_df["is_draw"])

    _signals_df = _df[_signal_mask].copy()
    _evaluated_df = _df[_signal_mask & _non_draw_mask].copy()

    _summary["signal_count"] = int(len(_signals_df))
    _summary["evaluated_signal_count"] = int(len(_evaluated_df))
    _summary["no_signal_count"] = int(len(_df) - len(_signals_df))

    if len(_df) > 0:
        _summary["no_signal_ratio"] = float((len(_df) - len(_signals_df)) / len(_df))
    else:
        _summary["no_signal_ratio"] = 0.0

    if len(_evaluated_df) > 0:
        _correct_mask = normalize_bool_series(_evaluated_df["is_correct"])
        _summary["enter_hit_rate"] = float(_correct_mask.mean())
        _summary["correct_count"] = int(_correct_mask.sum())
    else:
        _summary["enter_hit_rate"] = 0.0
        _summary["correct_count"] = 0

    return _summary

# --------------------------------------------------
# 有効シグナル行だけを抽出する
# 役割:
#   predicted_direction が UP / DOWN で、
#   かつ DRAW ではない行だけを勝率集計対象にする
# --------------------------------------------------
def build_evaluated_signals_df(_df):
    _signal_mask = _df["predicted_direction"].isin(["UP", "DOWN"])
    _non_draw_mask = ~normalize_bool_series(_df["is_draw"])
    return _df[_signal_mask & _non_draw_mask].copy()


# --------------------------------------------------
# LONG / SHORT 別の件数と勝率を集計する
# 役割:
#   売り買いどちらに偏りや弱さがあるかを把握する
# --------------------------------------------------
def build_direction_side_summary(_evaluated_df):
    _result = {}

    for _action in ["ENTER_LONG", "ENTER_SHORT"]:
        _subset = _evaluated_df[_evaluated_df["final_action"] == _action].copy()

        if len(_subset) == 0:
            _result[_action] = {
                "count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
            }
            continue

        _correct_mask = normalize_bool_series(_subset["is_correct"])

        _result[_action] = {
            "count": int(len(_subset)),
            "correct_count": int(_correct_mask.sum()),
            "hit_rate": float(_correct_mask.mean()),
        }

    return _result


# --------------------------------------------------
# スコアを帯へ変換する
# 役割:
#   decision_score / entry_score を帯別集計しやすくする
# --------------------------------------------------
def classify_score_band(_value):
    try:
        _score = int(float(_value))
    except Exception:
        return "UNKNOWN"

    if _score < 0:
        return "UNKNOWN"
    if _score <= 39:
        return "00_39"
    if _score <= 49:
        return "40_49"
    if _score <= 59:
        return "50_59"
    if _score <= 69:
        return "60_69"
    if _score <= 79:
        return "70_79"
    if _score <= 89:
        return "80_89"
    return "90_100"


# --------------------------------------------------
# 指定スコア列の帯別件数と勝率を集計する
# 役割:
#   どのスコア帯が有効かを把握する
# --------------------------------------------------
def build_score_band_summary(_evaluated_df, _column_name):
    if _column_name not in _evaluated_df.columns:
        return {}

    _work_df = _evaluated_df.copy()
    _work_df["score_band"] = _work_df[_column_name].apply(classify_score_band)

    _result = {}
    for _band in ["00_39", "40_49", "50_59", "60_69", "70_79", "80_89", "90_100", "UNKNOWN"]:
        _subset = _work_df[_work_df["score_band"] == _band].copy()

        if len(_subset) == 0:
            continue

        _correct_mask = normalize_bool_series(_subset["is_correct"])

        _result[_band] = {
            "count": int(len(_subset)),
            "correct_count": int(_correct_mask.sum()),
            "hit_rate": float(_correct_mask.mean()),
        }

    return _result


# --------------------------------------------------
# 件数と勝率の辞書集計を表示する
# 役割:
#   LONG/SHORT別・score帯別の結果を見やすくする
# --------------------------------------------------
def print_count_hit_rate_block(_title, _data):
    print(f"----- {_title} -----")

    if not _data:
        print("(empty)")
        return

    for _key, _value in _data.items():
        print(
            f"{_key}: "
            f"count={_value['count']}, "
            f"correct_count={_value['correct_count']}, "
            f"hit_rate={_value['hit_rate']:.4f}"
        )

# --------------------------------------------------
# 採用閾値別の件数と勝率を集計する
# 役割:
#   decision_score の最低採用ラインごとに、
#   使えるシグナルの質を比較できるようにする
# --------------------------------------------------
def build_decision_score_threshold_summary(_evaluated_df, _thresholds):
    _result = {}

    if "decision_score" not in _evaluated_df.columns:
        return _result

    for _threshold in _thresholds:
        _subset = _evaluated_df[
            pd.to_numeric(_evaluated_df["decision_score"], errors="coerce").fillna(-1) >= _threshold
        ].copy()

        if len(_subset) == 0:
            _result[f"decision_score>={_threshold}"] = {
                "count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
                "long_count": 0,
                "long_hit_rate": 0.0,
                "short_count": 0,
                "short_hit_rate": 0.0,
            }
            continue

        _correct_mask = normalize_bool_series(_subset["is_correct"])

        _long_subset = _subset[_subset["final_action"] == "ENTER_LONG"].copy()
        _short_subset = _subset[_subset["final_action"] == "ENTER_SHORT"].copy()

        if len(_long_subset) > 0:
            _long_hit_rate = float(normalize_bool_series(_long_subset["is_correct"]).mean())
        else:
            _long_hit_rate = 0.0

        if len(_short_subset) > 0:
            _short_hit_rate = float(normalize_bool_series(_short_subset["is_correct"]).mean())
        else:
            _short_hit_rate = 0.0

        _result[f"decision_score>={_threshold}"] = {
            "count": int(len(_subset)),
            "correct_count": int(_correct_mask.sum()),
            "hit_rate": float(_correct_mask.mean()),
            "long_count": int(len(_long_subset)),
            "long_hit_rate": _long_hit_rate,
            "short_count": int(len(_short_subset)),
            "short_hit_rate": _short_hit_rate,
        }

    return _result


# --------------------------------------------------
# 採用閾値別サマリーを表示する
# 役割:
#   decision_score の採用ラインごとの差を見やすくする
# --------------------------------------------------
def print_threshold_summary_block(_title, _data):
    print(f"----- {_title} -----")

    if not _data:
        print("(empty)")
        return

    for _key, _value in _data.items():
        print(
            f"{_key}: "
            f"count={_value['count']}, "
            f"correct_count={_value['correct_count']}, "
            f"hit_rate={_value['hit_rate']:.4f}, "
            f"long_count={_value['long_count']}, "
            f"long_hit_rate={_value['long_hit_rate']:.4f}, "
            f"short_count={_value['short_count']}, "
            f"short_hit_rate={_value['short_hit_rate']:.4f}"
        )

# --------------------------------------------------
# 仮採用条件サマリーを構築する
# 役割:
#   現時点で実戦候補になりうる採用条件を、
#   そのまま比較できる形で集計する
# --------------------------------------------------
def build_candidate_strategy_summary(_evaluated_df):
    _result = {}

    if "decision_score" not in _evaluated_df.columns:
        return _result

    _work_df = _evaluated_df.copy()
    _work_df["decision_score_num"] = pd.to_numeric(
        _work_df["decision_score"],
        errors="coerce",
    ).fillna(-1)

    _candidates = {
        "decision_score>=70": _work_df["decision_score_num"] >= 70,
        "decision_score>=70_AND_ENTER_LONG": (
            (_work_df["decision_score_num"] >= 70) &
            (_work_df["final_action"] == "ENTER_LONG")
        ),
        "decision_score>=70_AND_ENTER_SHORT": (
            (_work_df["decision_score_num"] >= 70) &
            (_work_df["final_action"] == "ENTER_SHORT")
        ),
    }

    for _name, _mask in _candidates.items():
        _subset = _work_df[_mask].copy()

        if len(_subset) == 0:
            _result[_name] = {
                "count": 0,
                "correct_count": 0,
                "hit_rate": 0.0,
            }
            continue

        _correct_mask = normalize_bool_series(_subset["is_correct"])

        _result[_name] = {
            "count": int(len(_subset)),
            "correct_count": int(_correct_mask.sum()),
            "hit_rate": float(_correct_mask.mean()),
        }

    return _result


def build_main_flow_gate_summary(_df):
    if "base_final_action" not in _df.columns:
        return {}

    _work_df = _df.copy()
    _work_df["base_predicted_direction"] = _work_df["base_final_action"].fillna("").apply(
        convert_final_action_to_predicted_direction
    )
    _work_df["final_predicted_direction"] = _work_df["final_action"].fillna("").apply(
        convert_final_action_to_predicted_direction
    )

    _base_signal_mask = _work_df["base_predicted_direction"].isin(["UP", "DOWN"])
    _final_signal_mask = _work_df["final_predicted_direction"].isin(["UP", "DOWN"])
    _gate_changed_mask = _work_df["base_final_action"].fillna("") != _work_df["final_action"].fillna("")
    _blocked_signal_mask = _base_signal_mask & (~_final_signal_mask)
    _retained_signal_mask = _base_signal_mask & _final_signal_mask & (~_gate_changed_mask)

    _base_eval_count, _base_correct_count, _base_hit_rate = calculate_direction_hit_rate(
        _work_df,
        "base_predicted_direction",
    )
    _final_eval_count, _final_correct_count, _final_hit_rate = calculate_direction_hit_rate(
        _work_df,
        "final_predicted_direction",
    )

    _summary = {
        "gate_changed_count": int(_gate_changed_mask.sum()),
        "gate_changed_ratio": float(_gate_changed_mask.mean()) if len(_work_df) > 0 else 0.0,
        "base_signal_count": int(_base_signal_mask.sum()),
        "final_signal_count": int(_final_signal_mask.sum()),
        "blocked_signal_count": int(_blocked_signal_mask.sum()),
        "retained_signal_count": int(_retained_signal_mask.sum()),
        "base_evaluated_signal_count": _base_eval_count,
        "base_correct_count": _base_correct_count,
        "base_enter_hit_rate": _base_hit_rate,
        "final_evaluated_signal_count": _final_eval_count,
        "final_correct_count": _final_correct_count,
        "final_enter_hit_rate": _final_hit_rate,
        "hit_rate_delta": float(_final_hit_rate - _base_hit_rate),
    }

    if "m15_path_signal_ready" in _work_df.columns:
        _path_ready_mask = normalize_bool_series(_work_df["m15_path_signal_ready"])
        _summary["path_signal_ready_count"] = int(_path_ready_mask.sum())
        _summary["path_signal_ready_ratio"] = float(_path_ready_mask.mean()) if len(_work_df) > 0 else 0.0

    if "m15_path_gap_threshold_passed" in _work_df.columns:
        _gap_pass_mask = normalize_bool_series(_work_df["m15_path_gap_threshold_passed"])
        _summary["path_gap_threshold_passed_count"] = int(_gap_pass_mask.sum())
        _summary["path_gap_threshold_passed_ratio"] = float(_gap_pass_mask.mean()) if len(_work_df) > 0 else 0.0

    if "m15_path_signal_side" in _work_df.columns:
        _side_counts = (
            _work_df["m15_path_signal_side"]
            .fillna("UNKNOWN")
            .value_counts(dropna=False)
            .to_dict()
        )
        _summary["path_signal_side_counts"] = _side_counts

    return _summary


# --------------------------------------------------
# 仮採用条件サマリーを表示する
# 役割:
#   実戦候補の採用条件を比較しやすくする
# --------------------------------------------------
def print_candidate_strategy_summary(_title, _data):
    print(f"----- {_title} -----")

    if not _data:
        print("(empty)")
        return

    for _key, _value in _data.items():
        print(
            f"{_key}: "
            f"count={_value['count']}, "
            f"correct_count={_value['correct_count']}, "
            f"hit_rate={_value['hit_rate']:.4f}"
        )

# --------------------------------------------------
# reason code を分解して頻度集計する
# 役割:
#   どの理由で止まりやすいかを把握できるようにする
# --------------------------------------------------
def build_reason_code_counts(_df, _column_name):
    if _column_name not in _df.columns:
        return {}

    _series = _df[_column_name].fillna("").astype(str)

    _counter = {}
    for _value in _series:
        if _value.strip() == "":
            continue

        for _code in _value.split(";"):
            _code = _code.strip()
            if _code == "":
                continue

            _counter[_code] = _counter.get(_code, 0) + 1

    _sorted = dict(sorted(_counter.items(), key=lambda _item: (-_item[1], _item[0])))
    return _sorted


# --------------------------------------------------
# 辞書集計を見やすく表示する
# 役割:
#   カウント系の結果をコンソールで確認しやすくする
# --------------------------------------------------
def print_dict_block(_title, _data):
    print(f"----- {_title} -----")

    if not _data:
        print("(empty)")
        return

    for _key, _value in _data.items():
        print(f"{_key}: {_value}")


# --------------------------------------------------
# reason code 上位件数を表示する
# 役割:
#   停止要因や通過要因の頻出傾向を把握しやすくする
# --------------------------------------------------
def print_top_reason_codes(_title, _data, _top_n):
    print(f"----- {_title} -----")

    if not _data:
        print("(empty)")
        return

    _items = list(_data.items())[:_top_n]
    for _key, _value in _items:
        print(f"{_key}: {_value}")


# --------------------------------------------------
# サマリーを表示する
# 役割:
#   バックテスト結果の全体像を即座に確認できるようにする
# --------------------------------------------------
def print_summary(_summary):
    print("========== Evaluate Backtest Signals ==========")
    print(f"total_records: {_summary['total_records']}")
    print(f"signal_count: {_summary['signal_count']}")
    print(f"evaluated_signal_count: {_summary['evaluated_signal_count']}")
    print(f"correct_count: {_summary['correct_count']}")
    print(f"no_signal_count: {_summary['no_signal_count']}")
    print(f"no_signal_ratio: {_summary['no_signal_ratio']:.4f}")
    print(f"enter_hit_rate: {_summary['enter_hit_rate']:.4f}")


# --------------------------------------------------
# メイン処理
# 役割:
#   raw_signals を読み込み、バックテスト集計結果を表示する
# --------------------------------------------------
def main():
    _args = parse_args()
    _df, _input_path = load_raw_signals(_args.input_csv)

    print(f"[INFO] input_csv={_input_path}")
    print(f"[INFO] rows={len(_df)}")

    _summary = build_summary(_df)
    _evaluated_df = build_evaluated_signals_df(_df)
    _direction_side_summary = build_direction_side_summary(_evaluated_df)
    _decision_score_band_summary = build_score_band_summary(_evaluated_df, "decision_score")
    _entry_score_band_summary = build_score_band_summary(_evaluated_df, "entry_score")
    _decision_score_threshold_summary = build_decision_score_threshold_summary(_evaluated_df,[60, 70, 80],)
    _candidate_strategy_summary = build_candidate_strategy_summary(_evaluated_df)
    _main_flow_gate_summary = build_main_flow_gate_summary(_df)

    _final_reason_counts = build_reason_code_counts(_df, "final_reason_codes")
    _external_reason_counts = build_reason_code_counts(_df, "external_reason_codes")
    _h2_reason_counts = build_reason_code_counts(_df, "h2_reason_codes")
    _h1_reason_counts = build_reason_code_counts(_df, "h1_reason_codes")
    _m15_path_reason_counts = build_reason_code_counts(_df, "m15_path_reason_codes")
    _m15_reason_counts = build_reason_code_counts(_df, "m15_reason_codes")

    print_summary(_summary)
    print_dict_block("final_action_counts", _summary["final_action_counts"])
    print_dict_block("base_final_action_counts", _summary["base_final_action_counts"])
    print_dict_block("external_filter_status_counts", _summary["external_filter_status_counts"])
    print_dict_block("env_direction_counts", _summary["env_direction_counts"])
    print_dict_block("predicted_direction_counts", _summary["predicted_direction_counts"])
    print_dict_block("main_flow_gate_summary", _main_flow_gate_summary)

    print_top_reason_codes("top_final_reason_codes", _final_reason_counts, _args.top_n_reason_codes)
    print_top_reason_codes("top_external_reason_codes", _external_reason_counts, _args.top_n_reason_codes)
    print_top_reason_codes("top_h2_reason_codes", _h2_reason_counts, _args.top_n_reason_codes)
    print_top_reason_codes("top_h1_reason_codes", _h1_reason_counts, _args.top_n_reason_codes)
    print_top_reason_codes("top_m15_path_reason_codes", _m15_path_reason_counts, _args.top_n_reason_codes)
    print_top_reason_codes("top_m15_reason_codes", _m15_reason_counts, _args.top_n_reason_codes)

    print_count_hit_rate_block("direction_side_summary", _direction_side_summary)
    print_count_hit_rate_block("decision_score_band_summary", _decision_score_band_summary)
    print_count_hit_rate_block("entry_score_band_summary", _entry_score_band_summary)
    print_threshold_summary_block("decision_score_threshold_summary",_decision_score_threshold_summary,)
    print_candidate_strategy_summary("candidate_strategy_summary",_candidate_strategy_summary,)

if __name__ == "__main__":
    main()
