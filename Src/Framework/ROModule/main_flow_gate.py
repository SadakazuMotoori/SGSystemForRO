import copy

from Framework.ROModule.h1_forecast import build_h1_runtime_view
from Framework.ROModule.h2_environment_contract import build_h2_runtime_view
from Framework.Utility.Utility import ToFloat as _to_float


DEFAULT_MAIN_FLOW_M15_PREDICTED_PATH_GAP_THRESHOLD_PIPS = 30.0


def resolve_m15_predicted_path_gap_threshold_pips(
    _thresholds,
    _default_value=DEFAULT_MAIN_FLOW_M15_PREDICTED_PATH_GAP_THRESHOLD_PIPS,
):
    if not isinstance(_thresholds, dict):
        return float(_default_value)

    return _to_float(_thresholds.get("m15_predicted_path_gap_threshold_pips"), _default_value)


def _resolve_pip_size(_symbol):
    if str(_symbol).upper().endswith("JPY"):
        return 0.01

    return 0.0001


def _extract_close_price(_price_row):
    if isinstance(_price_row, dict):
        return float(_price_row.get("close", 0.0))

    return float(_price_row[4])


def evaluate_main_m15_path_signal(
    _market_data,
    _h2_environment_result,
    _h1_forecast_result,
    _gap_threshold_pips,
):
    _m15_ohlc = _market_data.get("M15", {}).get("ohlc", [])
    _symbol = _market_data.get("M15", {}).get("symbol", "")
    _pip_size = _resolve_pip_size(_symbol)
    _reason_codes = []

    if len(_m15_ohlc) == 0:
        return {
            "module_name": "main_m15_path_signal",
            "status": "ERROR",
            "signal_ready": False,
            "signal_side": "NONE",
            "reason_codes": ["MAIN_M15_PATH_SIGNAL_M15_DATA_UNAVAILABLE"],
            "summary": "M15 confirmed data is unavailable.",
        }

    _current_m15_close = _extract_close_price(_m15_ohlc[-1])
    _h1_runtime_view = build_h1_runtime_view(_h1_forecast_result)
    _predicted_path = [float(_predicted_price) for _predicted_price in _h1_runtime_view["predicted_path"]]
    _h2_runtime_view = build_h2_runtime_view(_h2_environment_result)
    _env_direction = _h2_runtime_view["env_direction"]
    _h1_forecast_role = _h1_runtime_view["forecast_role"]
    _h1_net_direction = _h1_runtime_view["net_direction"]
    _h1_bias_direction = _h1_runtime_view["bias_direction"]
    _h1_bias_ready = bool(_h1_runtime_view["bias_ready"])
    _h1_bias_alignment_hint = _h1_runtime_view["bias_alignment_hint"]
    _h1_forecast_status = _h1_runtime_view["forecast_status"]
    _predicted_path_type = _h1_runtime_view["predicted_path_type"]
    _predicted_path_source_horizons = list(_h1_runtime_view["predicted_path_source_horizons"])
    _m15_bar_timestamp_jst = _market_data.get("M15", {}).get("confirmed_bar_jst", "")

    if len(_predicted_path) == 0:
        return {
            "module_name": "main_m15_path_signal",
            "status": "OK",
            "signal_ready": False,
            "signal_side": "NONE",
            "h1_forecast_role": _h1_forecast_role,
            "h1_bias_direction": _h1_bias_direction,
            "h1_bias_ready": _h1_bias_ready,
            "h1_bias_alignment_hint": _h1_bias_alignment_hint,
            "current_m15_close": float(_current_m15_close),
            "current_m15_bar_timestamp_jst": _m15_bar_timestamp_jst,
            "predicted_path": [],
            "predicted_path_type": _predicted_path_type,
            "predicted_path_source_horizons": _predicted_path_source_horizons,
            "reason_codes": ["MAIN_M15_PATH_SIGNAL_PATH_UNAVAILABLE"],
            "summary": "H1 predicted_path is unavailable.",
        }

    _absolute_max_gap_price = max(abs(_predicted_price - _current_m15_close) for _predicted_price in _predicted_path)
    _absolute_max_gap_pips = 0.0 if _pip_size <= 0.0 else float(_absolute_max_gap_price / _pip_size)

    if _env_direction == "LONG_ONLY":
        _signal_side = "LONG"
        _directional_gap_price = max(0.0, max(_predicted_path) - _current_m15_close)
        _is_direction_aligned = (_h1_bias_ready and _h1_bias_direction == "LONG_BIAS")
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_H2_LONG_ONLY")
    elif _env_direction == "SHORT_ONLY":
        _signal_side = "SHORT"
        _directional_gap_price = max(0.0, _current_m15_close - min(_predicted_path))
        _is_direction_aligned = (_h1_bias_ready and _h1_bias_direction == "SHORT_BIAS")
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_H2_SHORT_ONLY")
    else:
        _signal_side = "NONE"
        _directional_gap_price = 0.0
        _is_direction_aligned = False
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_H2_NO_TRADE")

    _directional_gap_pips = 0.0 if _pip_size <= 0.0 else float(_directional_gap_price / _pip_size)
    _gap_threshold_passed = _directional_gap_pips >= float(_gap_threshold_pips)
    _signal_ready = bool(_is_direction_aligned and _gap_threshold_passed)

    if _is_direction_aligned:
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_DIRECTION_ALIGNED")
    else:
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_DIRECTION_NOT_ALIGNED")

    if _gap_threshold_passed:
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_GAP_THRESHOLD_OK")
    else:
        _reason_codes.append("MAIN_M15_PATH_SIGNAL_GAP_THRESHOLD_LOW")

    _summary = "M15 path-gap signal is ready." if _signal_ready else "M15 path-gap signal is not ready."

    return {
        "module_name": "main_m15_path_signal",
        "status": "OK",
        "signal_ready": _signal_ready,
        "signal_side": _signal_side,
        "env_direction": _env_direction,
        "regime_direction": _h2_runtime_view["regime_direction"],
        "regime_score": _h2_runtime_view["regime_score"],
        "regime_quality": _h2_runtime_view["regime_quality"],
        "h1_forecast_role": _h1_forecast_role,
        "h1_forecast_status": _h1_forecast_status,
        "h1_net_direction": _h1_net_direction,
        "h1_bias_direction": _h1_bias_direction,
        "h1_bias_ready": _h1_bias_ready,
        "h1_bias_alignment_hint": _h1_bias_alignment_hint,
        "current_m15_close": float(_current_m15_close),
        "current_m15_bar_timestamp_jst": _m15_bar_timestamp_jst,
        "predicted_path": list(_predicted_path),
        "predicted_path_type": _predicted_path_type,
        "predicted_path_source_horizons": _predicted_path_source_horizons,
        "absolute_max_gap_price": float(_absolute_max_gap_price),
        "absolute_max_gap_pips": float(_absolute_max_gap_pips),
        "directional_gap_price": float(_directional_gap_price),
        "directional_gap_pips": float(_directional_gap_pips),
        "required_gap_pips": float(_gap_threshold_pips),
        "gap_threshold_passed": bool(_gap_threshold_passed),
        "reason_codes": _reason_codes,
        "summary": _summary,
    }


def apply_main_flow_gate(_base_final_decision_result, _m15_path_signal_result):
    _final_result = copy.deepcopy(_base_final_decision_result)
    _details = _final_result.get("details")
    if not isinstance(_details, dict):
        _details = {}
        _final_result["details"] = _details
    _details["main_m15_path_signal_result"] = _m15_path_signal_result

    _final_action = _final_result.get("final_action")
    _signal_ready = bool(_m15_path_signal_result.get("signal_ready"))
    _signal_side = _m15_path_signal_result.get("signal_side")

    if _final_action == "ENTER_LONG" and (not _signal_ready or _signal_side != "LONG"):
        _final_result["final_action"] = "WAIT"
        _final_result["approved"] = False
        _final_result["summary"] = "Main flow blocked LONG entry because the M15 path-gap signal is not ready."
        _final_result["reason_codes"] = list(_final_result.get("reason_codes", [])) + [
            "MAIN_FLOW_M15_PATH_SIGNAL_BLOCKED_LONG"
        ]
        return _final_result

    if _final_action == "ENTER_SHORT" and (not _signal_ready or _signal_side != "SHORT"):
        _final_result["final_action"] = "WAIT"
        _final_result["approved"] = False
        _final_result["summary"] = "Main flow blocked SHORT entry because the M15 path-gap signal is not ready."
        _final_result["reason_codes"] = list(_final_result.get("reason_codes", [])) + [
            "MAIN_FLOW_M15_PATH_SIGNAL_BLOCKED_SHORT"
        ]
        return _final_result

    if _signal_ready and _final_action in ["ENTER_LONG", "ENTER_SHORT"]:
        _final_result["reason_codes"] = list(_final_result.get("reason_codes", [])) + [
            "MAIN_FLOW_M15_PATH_SIGNAL_CONFIRMED"
        ]

    return _final_result


def build_main_flow_gated_decision(
    _market_data,
    _h2_environment_result,
    _h1_forecast_result,
    _base_final_decision_result,
    _gap_threshold_pips,
):
    _m15_path_signal_result = evaluate_main_m15_path_signal(
        _market_data=_market_data,
        _h2_environment_result=_h2_environment_result,
        _h1_forecast_result=_h1_forecast_result,
        _gap_threshold_pips=_gap_threshold_pips,
    )
    _final_decision_result = apply_main_flow_gate(
        _base_final_decision_result=_base_final_decision_result,
        _m15_path_signal_result=_m15_path_signal_result,
    )
    return _m15_path_signal_result, _final_decision_result
