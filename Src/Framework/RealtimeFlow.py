import json
import time

from Framework.ContextSystem.ExternalContextBuilder import BuildExternalContext
from Framework.MTSystem.MTManager import (
    MTManager_BuildConfirmedMarketData,
    MTManager_Initialize,
)
from Framework.ROModule.external_filter import evaluate_external_filter
from Framework.ROModule.final_decision import evaluate_final_decision
from Framework.ROModule.h1_forecast import build_h1_runtime_view, evaluate_h1_forecast
from Framework.ROModule.h2_environment import evaluate_h2_environment
from Framework.ROModule.h2_environment_contract import build_h2_runtime_view
from Framework.ROModule.main_flow_gate import (
    apply_main_flow_gate as apply_shared_main_flow_gate,
    evaluate_main_m15_path_signal as evaluate_shared_main_m15_path_signal,
)
from Framework.ROModule.m15_entry import evaluate_m15_entry
from Framework.Utility.Utility import GetJSTNowStr, LoadJson, ToFloat as _to_float


DEFAULT_SYMBOL = "USDJPY"
DEFAULT_M15_FETCH_COUNT = 200
DEFAULT_H1_FETCH_COUNT = 200
DEFAULT_H2_FETCH_COUNT = 200
DEFAULT_LOOP_SLEEP_SECONDS = 1.0
DEFAULT_ACTUAL_TRADE_ENABLED = False
DEFAULT_M15_PREDICTED_PATH_GAP_THRESHOLD_PIPS = 30.0


# --------------------------------------------------
# 設定読込:
#   runtime loop 全体で使う threshold JSON を共通の入口で読む
# 目的:
#   main.py と realtime loop の双方から同じ設定取得関数を使えるようにする
# --------------------------------------------------
def LoadThresholds(_path="Asset/Config/thresholds.json"):
    return LoadJson(_path)


# --------------------------------------------------
# 市場データ構築:
#   H2 / H1 / M15 の confirmed market data を MTManager からまとめて受け取る
# 目的:
#   realtime loop が MT5 API を直接触らずに済むようにする
# --------------------------------------------------
def BuildMarketData(
    _symbol=DEFAULT_SYMBOL,
    _timestamp_jst="",
    _m15_count=DEFAULT_M15_FETCH_COUNT,
    _h1_count=DEFAULT_H1_FETCH_COUNT,
    _h2_count=DEFAULT_H2_FETCH_COUNT,
):
    return MTManager_BuildConfirmedMarketData(
        _symbol=_symbol,
        _timestamp_jst=_timestamp_jst,
        _m15_count=_m15_count,
        _h1_count=_h1_count,
        _h2_count=_h2_count,
    )


# --------------------------------------------------
# システム文脈構築:
#   直前の判断結果を引き継ぎつつ、今回ループの system_context を組み立てる
# 目的:
#   realtime loop の各更新フェーズで必要な共通コンテキストを揃える
# --------------------------------------------------
def BuildSystemContext(_last_decision=None, _last_entry_result=None, _position_state="FLAT"):
    _now_jst = GetJSTNowStr()

    return {
        "round_id": _now_jst.replace("-", "").replace(" ", "").replace(":", ""),
        "latest_update_jst": _now_jst,
        "last_decision": _last_decision,
        "last_entry_result": _last_entry_result,
        "position_state": _position_state,
    }

# --------------------------------------------------
# runtime 設定構築:
#   loop 全体で使う運用パラメータを 1 箇所へ集約する
# 目的:
#   閾値 JSON とコード内既定値の責務を整理し、調整点を見通しやすくする
# --------------------------------------------------
def _build_runtime_config(_thresholds, _symbol=DEFAULT_SYMBOL):
    return {
        "actual_trade_enabled": bool(DEFAULT_ACTUAL_TRADE_ENABLED),
        "symbol": _symbol,
        "m15_fetch_count": int(DEFAULT_M15_FETCH_COUNT),
        "h1_fetch_count": int(DEFAULT_H1_FETCH_COUNT),
        "h2_fetch_count": int(DEFAULT_H2_FETCH_COUNT),
        "loop_sleep_seconds": float(DEFAULT_LOOP_SLEEP_SECONDS),
        "m15_predicted_path_gap_threshold_pips": _to_float(
            _thresholds.get("m15_predicted_path_gap_threshold_pips"),
            DEFAULT_M15_PREDICTED_PATH_GAP_THRESHOLD_PIPS,
        ),
    }


# --------------------------------------------------
# confirmed bar 時計:
#   各 timeframe の最新 confirmed bar 時刻だけを比較用に抜き出す
# 目的:
#   1 秒ループ中でも、バーが更新された timeframe だけを再計算する
# --------------------------------------------------
def _build_bar_clock(_market_data):
    return {
        _timeframe_name: _timeframe_data.get("confirmed_bar_jst", "")
        for _timeframe_name, _timeframe_data in _market_data.items()
    }


# --------------------------------------------------
# 追跡状態構築:
#   各 timeframe を最後に評価した confirmed bar 時刻を管理する
# 目的:
#   モジュール返却値に loop 用メタ情報を混ぜず、再計算条件を state 側で持つ
# --------------------------------------------------
def _build_tracking_state(_bar_clock):
    return {
        "evaluated_bar_jst": {
            "H2": _bar_clock.get("H2", ""),
            "H1": _bar_clock.get("H1", ""),
            "M15": "",
        },
        "latest_decision_m15_bar_jst": "",
        "latest_decision_update_jst": "",
    }


def _build_h1_runtime_state(_h1_forecast_result, _bar_clock, _system_context):
    return {
        "result": _h1_forecast_result,
        "runtime_view": build_h1_runtime_view(_h1_forecast_result),
        "latest_forecast_bar_jst": _bar_clock.get("H1", ""),
        "latest_forecast_update_jst": _system_context.get("latest_update_jst", ""),
    }


def _sync_h1_runtime_state(_state, _h1_forecast_result):
    _h1_state = _build_h1_runtime_state(
        _h1_forecast_result=_h1_forecast_result,
        _bar_clock=_state["bar_clock"],
        _system_context=_state["system_context"],
    )
    _state["h1_state"] = _h1_state
    _state["h1_forecast_result"] = _h1_state["result"]
    _state["h1_runtime_view"] = _h1_state["runtime_view"]


def _get_h1_forecast_result(_state):
    return _state.get("h1_state", {}).get("result", {})


def _get_h1_runtime_view(_state):
    return _state.get("h1_state", {}).get("runtime_view", build_h1_runtime_view(None))


def _has_new_confirmed_bar(_state, _timeframe_name):
    _current_bar_jst = _state["bar_clock"].get(_timeframe_name, "")
    _last_evaluated_bar_jst = _state["tracking"]["evaluated_bar_jst"].get(_timeframe_name, "")
    return _current_bar_jst != _last_evaluated_bar_jst


def _mark_confirmed_bar_evaluated(_state, _timeframe_name):
    _state["tracking"]["evaluated_bar_jst"][_timeframe_name] = _state["bar_clock"].get(_timeframe_name, "")


# --------------------------------------------------
# 外部文脈更新:
#   external_context と external_filter を現在の market/system 状態から作り直す
# 目的:
#   loop の各周回で、外部イベント判定を最新状態へ合わせる
# --------------------------------------------------
def _refresh_external_state(_market_data, _system_context, _thresholds):
    _external_context = BuildExternalContext(
        _marketData=_market_data,
        _systemContext=_system_context,
        _thresholds=_thresholds,
    )
    _external_filter_result = evaluate_external_filter(
        market_data=_market_data["M15"],
        external_context=_external_context,
        system_context=_system_context,
        thresholds=_thresholds,
    )

    return _external_context, _external_filter_result


def _build_market_data_from_runtime_config(_runtime_config, _timestamp_jst=""):
    return BuildMarketData(
        _symbol=_runtime_config["symbol"],
        _timestamp_jst=_timestamp_jst,
        _m15_count=_runtime_config["m15_fetch_count"],
        _h1_count=_runtime_config["h1_fetch_count"],
        _h2_count=_runtime_config["h2_fetch_count"],
    )


def _refresh_loop_inputs(_state):
    _system_context = BuildSystemContext(
        _last_decision=_state.get("final_decision_result"),
        _last_entry_result=_state.get("m15_entry_result"),
        _position_state=_state.get("position_state", "FLAT"),
    )
    _market_data = _build_market_data_from_runtime_config(
        _runtime_config=_state["runtime_config"],
        _timestamp_jst=_system_context["latest_update_jst"],
    )
    _bar_clock = _build_bar_clock(_market_data)

    _external_context, _external_filter_result = _refresh_external_state(
        _market_data=_market_data,
        _system_context=_system_context,
        _thresholds=_state["thresholds"],
    )

    _state["system_context"] = _system_context
    _state["market_data"] = _market_data
    _state["bar_clock"] = _bar_clock
    _state["external_context"] = _external_context
    _state["external_filter_result"] = _external_filter_result


# --------------------------------------------------
# pip サイズ解決:
#   通貨ペアの価格差を pips へ揃えるための基本単位を返す
# 目的:
#   predicted_path と現在値の価格差を timeframe 共通の尺度で判定する
# --------------------------------------------------
# --------------------------------------------------
# M15 path-gap 判定:
#   H2 方向に沿う predicted_path の最大乖離が十分かを M15 更新時点で評価する
# 目的:
#   final_decision に加える realtime flow 固有の追加ゲートを管理する
# --------------------------------------------------
# --------------------------------------------------
# final_decision 追加ゲート:
#   M15 path-gap 判定が未成立なら entry を WAIT へ落とす
# 目的:
#   既存の final_decision を活かしたまま、realtime flow 固有の最終ゲートだけを追加する
# --------------------------------------------------
# --------------------------------------------------
# 起動時サマリ表示:
#   ループ開始時の H2 / H1 初期状態を見やすく表示する
# 目的:
#   起動直後の前提状態をログからすぐ確認できるようにする
# --------------------------------------------------
def _print_startup_summary(_state):
    _h2_runtime_view = build_h2_runtime_view(_state["h2_environment_result"])
    _h1_runtime_view = _get_h1_runtime_view(_state)
    _h1_state = _state.get("h1_state", {})

    _startup_payload = {
        "latest_update_jst": _state["system_context"]["latest_update_jst"],
        "confirmed_bar_clock": _state["bar_clock"],
        "decision_update_policy": "M15 confirmed bar only",
        "h2_environment": _h2_runtime_view,
        "h1_forecast": {
            "forecast_role": _h1_runtime_view["forecast_role"],
            "forecast_status": _h1_runtime_view["forecast_status"],
            "net_direction": _h1_runtime_view["net_direction"],
            "bias_direction": _h1_runtime_view["bias_direction"],
            "bias_ready": _h1_runtime_view["bias_ready"],
            "bias_alignment_hint": _h1_runtime_view["bias_alignment_hint"],
            "confidence": _h1_runtime_view["confidence"],
            "predicted_path_type": _h1_runtime_view["predicted_path_type"],
            "predicted_path_source_horizons": _h1_runtime_view["predicted_path_source_horizons"],
            "summary": _h1_runtime_view["summary"],
            "active_model_id": _h1_runtime_view["active_model_id"],
            "dataset_id": _h1_runtime_view["dataset_id"],
            "confirmed_bar_jst": _h1_state.get("latest_forecast_bar_jst"),
            "latest_update_jst": _h1_state.get("latest_forecast_update_jst"),
        },
    }

    print("----- Startup Summary -----")
    print(json.dumps(_startup_payload, ensure_ascii=False, indent=2))


# --------------------------------------------------
# 更新ログ表示:
#   H2 / H1 のどちらが更新されたかをコンパクトに出力する
# 目的:
#   1 秒ループ中の再計算タイミングをログから追いやすくする
# --------------------------------------------------
def _print_module_update(_title, _payload):
    print(f"----- {_title} -----")
    print(json.dumps(_payload, ensure_ascii=False, indent=2))


# --------------------------------------------------
# 最終判断表示:
#   M15 path-gap 判定と final_decision の最新結果を表示する
# 目的:
#   realtime loop の最終出力を毎回同じ形式で確認できるようにする
# --------------------------------------------------
def PrintFinalDecision(_realtime_state):
    print("----- M15 Path Signal -----")
    print(json.dumps(_realtime_state["m15_path_signal_result"], ensure_ascii=False, indent=2))
    print("----- Final Decision -----")
    print(json.dumps(_realtime_state["final_decision_result"], ensure_ascii=False, indent=2))


def _update_decision_snapshot(_state):
    _state["tracking"]["latest_decision_m15_bar_jst"] = _state["bar_clock"].get("M15", "")
    _state["tracking"]["latest_decision_update_jst"] = _state["system_context"].get("latest_update_jst", "")


def _print_runtime_error(_error, _state=None):
    _error_payload = {
        "error_type": type(_error).__name__,
        "error_message": str(_error),
    }

    if isinstance(_state, dict):
        _error_payload["latest_update_jst"] = _state.get("system_context", {}).get("latest_update_jst")
        _error_payload["confirmed_bar_clock"] = _state.get("bar_clock")
        _error_payload["decision_snapshot"] = {
            "latest_decision_m15_bar_jst": _state.get("tracking", {}).get("latest_decision_m15_bar_jst"),
            "latest_decision_update_jst": _state.get("tracking", {}).get("latest_decision_update_jst"),
        }
        _error_payload["h1_snapshot"] = {
            "latest_forecast_bar_jst": _state.get("h1_state", {}).get("latest_forecast_bar_jst"),
            "latest_forecast_update_jst": _state.get("h1_state", {}).get("latest_forecast_update_jst"),
            "runtime_view": _state.get("h1_state", {}).get("runtime_view"),
        }

    print("[ERROR] realtime flow failed.")
    print(json.dumps(_error_payload, ensure_ascii=False, indent=2))


def _update_h2_phase(_state):
    if not _has_new_confirmed_bar(_state, "H2"):
        return False

    _state["h2_environment_result"] = evaluate_h2_environment(
        market_data_h2=_state["market_data"]["H2"],
        external_filter_result=_state["external_filter_result"],
        thresholds=_state["thresholds"],
    )
    _state["h2_regime_result"] = _state["h2_environment_result"]
    _mark_confirmed_bar_evaluated(_state, "H2")
    _h2_runtime_view = build_h2_runtime_view(_state["h2_environment_result"])

    _print_module_update(
        "H2 Update",
        {
            "confirmed_bar_jst": _state["bar_clock"]["H2"],
            "env_direction": _h2_runtime_view["env_direction"],
            "trend_strength": _h2_runtime_view["trend_strength"],
            "regime_direction": _h2_runtime_view["regime_direction"],
            "regime_score": _h2_runtime_view["regime_score"],
            "regime_quality": _h2_runtime_view["regime_quality"],
            "summary": _h2_runtime_view["summary"],
        },
    )
    return True


def _update_h1_phase(_state):
    if not _has_new_confirmed_bar(_state, "H1"):
        return False

    _h1_forecast_result = evaluate_h1_forecast(
        _h1_data=_state["market_data"]["H1"],
        _thresholds=_state["thresholds"],
    )
    _sync_h1_runtime_state(_state, _h1_forecast_result)
    _mark_confirmed_bar_evaluated(_state, "H1")
    _h1_runtime_view = _get_h1_runtime_view(_state)

    _print_module_update(
        "H1 Update",
        {
            "confirmed_bar_jst": _state["bar_clock"]["H1"],
            "forecast_role": _h1_runtime_view["forecast_role"],
            "forecast_status": _h1_runtime_view["forecast_status"],
            "net_direction": _h1_runtime_view["net_direction"],
            "bias_direction": _h1_runtime_view["bias_direction"],
            "bias_ready": _h1_runtime_view["bias_ready"],
            "bias_alignment_hint": _h1_runtime_view["bias_alignment_hint"],
            "confidence": _h1_runtime_view["confidence"],
            "predicted_path": _h1_runtime_view["predicted_path"],
            "predicted_path_type": _h1_runtime_view["predicted_path_type"],
            "predicted_path_source_horizons": _h1_runtime_view["predicted_path_source_horizons"],
            "active_model_id": _h1_runtime_view["active_model_id"],
            "dataset_id": _h1_runtime_view["dataset_id"],
        },
    )
    return True


def _update_m15_phase(_state):
    if not _has_new_confirmed_bar(_state, "M15"):
        return False

    _state["m15_path_signal_result"] = evaluate_shared_main_m15_path_signal(
        _market_data=_state["market_data"],
        _h2_environment_result=_state["h2_environment_result"],
        _h1_forecast_result=_get_h1_forecast_result(_state),
        _gap_threshold_pips=_state["runtime_config"]["m15_predicted_path_gap_threshold_pips"],
    )
    _state["m15_entry_result"] = evaluate_m15_entry(
        market_data_m15=_state["market_data"]["M15"],
        h2_environment_result=_state["h2_environment_result"],
        h1_forecast_result=_get_h1_forecast_result(_state),
        external_filter_result=_state["external_filter_result"],
        thresholds=_state["thresholds"],
    )

    _base_final_decision_result = evaluate_final_decision(
        external_filter_result=_state["external_filter_result"],
        h2_environment_result=_state["h2_environment_result"],
        h1_forecast_result=_get_h1_forecast_result(_state),
        m15_entry_result=_state["m15_entry_result"],
        thresholds=_state["thresholds"],
    )
    _state["final_decision_result"] = apply_shared_main_flow_gate(
        _base_final_decision_result=_base_final_decision_result,
        _m15_path_signal_result=_state["m15_path_signal_result"],
    )

    _mark_confirmed_bar_evaluated(_state, "M15")
    _update_decision_snapshot(_state)
    PrintFinalDecision(_state)
    return True


# --------------------------------------------------
# 初期状態構築:
#   起動直後に H2 / H1 / external 系の初期評価結果をまとめて作る
# 目的:
#   ループ開始前に必要な基準状態を揃え、各更新フェーズへ引き渡す
# --------------------------------------------------
def _initialize_realtime_state(_thresholds, _symbol=DEFAULT_SYMBOL):
    _runtime_config = _build_runtime_config(_thresholds=_thresholds, _symbol=_symbol)
    _system_context = BuildSystemContext()
    _market_data = _build_market_data_from_runtime_config(
        _runtime_config=_runtime_config,
        _timestamp_jst=_system_context["latest_update_jst"],
    )
    _bar_clock = _build_bar_clock(_market_data)

    _external_context, _external_filter_result = _refresh_external_state(
        _market_data=_market_data,
        _system_context=_system_context,
        _thresholds=_thresholds,
    )

    _h2_environment_result = evaluate_h2_environment(
        market_data_h2=_market_data["H2"],
        external_filter_result=_external_filter_result,
        thresholds=_thresholds,
    )
    _h1_forecast_result = evaluate_h1_forecast(
        _h1_data=_market_data["H1"],
        _thresholds=_thresholds,
    )
    _h1_state = _build_h1_runtime_state(
        _h1_forecast_result=_h1_forecast_result,
        _bar_clock=_bar_clock,
        _system_context=_system_context,
    )

    return {
        "runtime_config": _runtime_config,
        "symbol": _runtime_config["symbol"],
        "thresholds": _thresholds,
        "market_data": _market_data,
        "bar_clock": _bar_clock,
        "tracking": _build_tracking_state(_bar_clock),
        "system_context": _system_context,
        "external_context": _external_context,
        "external_filter_result": _external_filter_result,
        "h2_environment_result": _h2_environment_result,
        "h2_regime_result": _h2_environment_result,
        "h1_state": _h1_state,
        "h1_forecast_result": _h1_state["result"],
        "h1_runtime_view": _h1_state["runtime_view"],
        "m15_entry_result": None,
        "m15_path_signal_result": None,
        "final_decision_result": None,
        "position_state": "FLAT",
    }


# --------------------------------------------------
# realtime main loop:
#   起動時初期化の後、1 秒ごとに H2 / H1 / M15 の更新判定を回す
# 目的:
#   SGSystem の本番前提フローを 1 つの制御関数へまとめ、main.py は薄い入口だけにする
# --------------------------------------------------
def main():
    _state = None

    print("==========SGSystem Start==========")

    try:
        if not MTManager_Initialize():
            print("[ERROR] MT5 initialize failed. Trading loop will not start.")
            return False

        _thresholds = LoadThresholds()
        _state = _initialize_realtime_state(_thresholds=_thresholds, _symbol=DEFAULT_SYMBOL)
        print(f"[INFO] actual_trade_enabled={_state['runtime_config']['actual_trade_enabled']}")
        _print_startup_summary(_state)

        while True:
            _refresh_loop_inputs(_state)
            _update_h2_phase(_state)
            _update_h1_phase(_state)
            _update_m15_phase(_state)
            time.sleep(_state["runtime_config"]["loop_sleep_seconds"])

    except KeyboardInterrupt:
        print("[INFO] SGSystem loop stopped by user.")
        return True

    except Exception as e:
        _print_runtime_error(_error=e, _state=_state)
        return False

    finally:
        print("==========SGSystem End==========")
