import json
import os
import sys


_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.ROModule.main_flow_gate import (
    apply_main_flow_gate,
    evaluate_main_m15_path_signal,
)


def _build_base_market_data(_current_close):
    return {
        "M15": {
            "symbol": "USDJPY",
            "confirmed_bar_jst": "2026-04-09 12:15:00",
            "ohlc": [
                {"open": _current_close - 0.05, "high": _current_close + 0.05, "low": _current_close - 0.08, "close": _current_close}
            ],
        }
    }


def _build_h2_long():
    return {
        "env_direction": "LONG_ONLY",
        "regime_direction": "LONG_ONLY",
        "regime_score": 72,
        "regime_quality": "READY",
    }


def _build_h1_long(_predicted_path):
    return {
        "forecast_status": "SUCCESS",
        "net_direction": "LONG_BIAS",
        "confidence": 0.82,
        "predicted_path": list(_predicted_path),
    }


def _build_enter_long_final():
    return {
        "module_name": "final_decision",
        "status": "OK",
        "final_action": "ENTER_LONG",
        "decision_score": 78,
        "approved": True,
        "reason_codes": ["FINAL_DECISION_M15_ENTER_LONG"],
        "summary": "Long entry approved before main flow gate.",
        "details": {},
    }


def _run_case(_case_name, _current_close, _predicted_path, _required_gap_pips, _expected_signal_ready, _expected_final_action):
    _market_data = _build_base_market_data(_current_close)
    _signal_result = evaluate_main_m15_path_signal(
        _market_data=_market_data,
        _h2_environment_result=_build_h2_long(),
        _h1_forecast_result=_build_h1_long(_predicted_path),
        _gap_threshold_pips=_required_gap_pips,
    )
    _final_result = apply_main_flow_gate(
        _base_final_decision_result=_build_enter_long_final(),
        _m15_path_signal_result=_signal_result,
    )

    _passed = (
        bool(_signal_result.get("signal_ready")) == bool(_expected_signal_ready)
        and _final_result.get("final_action") == _expected_final_action
        and _signal_result.get("h1_bias_direction") == "LONG_BIAS"
        and _signal_result.get("predicted_path_type") == "LINEAR_INTERPOLATED_HORIZON_PATH"
    )

    print("")
    print("==================================================")
    print(f"CASE: {_case_name}")
    print("==================================================")
    print(f"expected_signal_ready: {_expected_signal_ready}")
    print(f"actual_signal_ready  : {_signal_result.get('signal_ready')}")
    print(f"expected_final_action: {_expected_final_action}")
    print(f"actual_final_action  : {_final_result.get('final_action')}")
    print(f"passed               : {_passed}")
    print(json.dumps({"signal_result": _signal_result, "final_result": _final_result}, ensure_ascii=False, indent=2))
    return _passed


def main():
    print("========== test_main_flow_path_gate_parity start ==========")

    _results = [
        _run_case(
            _case_name="LONG path confirmed",
            _current_close=150.00,
            _predicted_path=[150.10, 150.16, 150.24, 150.32, 150.40, 150.48],
            _required_gap_pips=30.0,
            _expected_signal_ready=True,
            _expected_final_action="ENTER_LONG",
        ),
        _run_case(
            _case_name="LONG blocked by low gap",
            _current_close=150.00,
            _predicted_path=[150.04, 150.05, 150.06, 150.07, 150.08, 150.09],
            _required_gap_pips=30.0,
            _expected_signal_ready=False,
            _expected_final_action="WAIT",
        ),
    ]

    _passed_count = sum(1 for _value in _results if _value)
    print("")
    print("========== Summary ==========")
    print(f"passed={_passed_count}/{len(_results)}")

    if _passed_count != len(_results):
        raise SystemExit(1)

    print("========== test_main_flow_path_gate_parity end ==========")


if __name__ == "__main__":
    main()
