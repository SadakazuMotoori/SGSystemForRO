import os
import sys

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import Framework.RealtimeFlow as realtime_flow


def _build_h1_result(_net_direction="LONG_BIAS", _confidence=0.78):
    return {
        "module_name": "h1_forecast",
        "timestamp_jst": "2026-04-09 10:00:00",
        "status": "OK",
        "forecast_status": "SUCCESS",
        "net_direction": _net_direction,
        "direction_score_long": 0.82 if _net_direction == "LONG_BIAS" else 0.18,
        "direction_score_short": 0.18 if _net_direction == "LONG_BIAS" else 0.82,
        "confidence": _confidence,
        "predicted_path": [150.2, 150.5],
        "reason_codes": [],
        "summary": "H1 test",
        "raw_features": {
            "active_model_id": "h1model_test",
            "dataset_id": "h1ds_test",
            "artifact_selection_source": "active_runtime",
            "sequence_length": 81,
        },
    }


def test_sync_h1_runtime_state():
    _state = {
        "bar_clock": {"H1": "2026-04-09 10:00:00"},
        "system_context": {"latest_update_jst": "2026-04-09 10:05:00"},
    }

    realtime_flow._sync_h1_runtime_state(_state, _build_h1_result())

    assert _state["h1_state"]["latest_forecast_bar_jst"] == "2026-04-09 10:00:00"
    assert _state["h1_state"]["latest_forecast_update_jst"] == "2026-04-09 10:05:00"
    assert _state["h1_runtime_view"]["active_model_id"] == "h1model_test"
    assert _state["h1_forecast_result"]["net_direction"] == "LONG_BIAS"


def test_update_h1_phase_refreshes_state():
    _state = {
        "bar_clock": {"H1": "2026-04-09 10:00:00"},
        "tracking": {"evaluated_bar_jst": {"H1": "2026-04-09 09:00:00"}},
        "market_data": {"H1": {"timestamp_jst": "2026-04-09 10:00:00", "ohlc": []}},
        "thresholds": {},
        "system_context": {"latest_update_jst": "2026-04-09 10:05:00"},
    }

    _original_predictor = realtime_flow.evaluate_h1_forecast
    _original_printer = realtime_flow._print_module_update

    try:
        realtime_flow.evaluate_h1_forecast = lambda _h1_data, _thresholds: _build_h1_result("SHORT_BIAS", 0.81)
        realtime_flow._print_module_update = lambda _title, _payload: None

        _updated = realtime_flow._update_h1_phase(_state)

        assert _updated is True
        assert _state["tracking"]["evaluated_bar_jst"]["H1"] == "2026-04-09 10:00:00"
        assert _state["h1_runtime_view"]["net_direction"] == "SHORT_BIAS"
        assert _state["h1_state"]["latest_forecast_update_jst"] == "2026-04-09 10:05:00"
    finally:
        realtime_flow.evaluate_h1_forecast = _original_predictor
        realtime_flow._print_module_update = _original_printer


def main():
    test_sync_h1_runtime_state()
    test_update_h1_phase_refreshes_state()
    print("[PASS] test_realtime_h1_state_management")


if __name__ == "__main__":
    main()
