import os
import sys

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

import Framework.ROModule.h1_forecast as h1_forecast


def _make_ohlc(_count, _start_price=150.0, _step=0.1):
    _ohlc = []

    for _index in range(_count):
        _close = _start_price + (_index * _step)
        _ohlc.append(
            {
                "open": _close - 0.05,
                "high": _close + 0.08,
                "low": _close - 0.09,
                "close": _close,
            }
        )

    return _ohlc


def _run_with_patches(_sequence_length, _predictor, _h1_data, _thresholds):
    _original_sequence_length = h1_forecast.GetForecastSequenceLength
    _original_predictor = h1_forecast.PredictMultiHorizonForecast

    try:
        h1_forecast.GetForecastSequenceLength = lambda: _sequence_length
        h1_forecast.PredictMultiHorizonForecast = _predictor
        return h1_forecast.evaluate_h1_forecast(_h1_data, _thresholds)
    finally:
        h1_forecast.GetForecastSequenceLength = _original_sequence_length
        h1_forecast.PredictMultiHorizonForecast = _original_predictor


def test_insufficient_data():
    _result = _run_with_patches(
        _sequence_length=5,
        _predictor=lambda **_: None,
        _h1_data={"timestamp_jst": "2026-04-09 10:00:00", "ohlc": _make_ohlc(4)},
        _thresholds={},
    )

    assert _result["status"] == "OK"
    assert _result["forecast_status"] == "INSUFFICIENT_DATA"
    assert _result["net_direction"] == "NEUTRAL"
    assert _result["reason_codes"] == ["H1_FORECAST_DATA_INSUFFICIENT"]


def test_successful_long_bias():
    def _predictor(**_):
        return {
            "net_direction": "LONG_BIAS",
            "direction_score_long": 0.81,
            "direction_score_short": 0.19,
            "confidence": 0.78,
            "predicted_path": [150.1, 150.3, 150.6],
            "artifact_role": "baseline",
            "active_model_id": "h1model_test",
            "artifact_selection_source": "active_runtime",
            "dataset_id": "h1ds_test",
            "sequence_length": 81,
            "history_end_timestamp_jst": "2026-04-09 10:00:00",
            "horizons": [6, 7, 8],
            "target_scale": 0.25,
            "signal_strength": 0.44,
            "direction_dominance": 0.62,
            "predicted_delta_by_horizon": {"6": 0.12},
            "predicted_close_by_horizon": {"6": 150.3},
            "drift_baseline": [150.0, 150.1],
        }

    _result = _run_with_patches(
        _sequence_length=5,
        _predictor=_predictor,
        _h1_data={"timestamp_jst": "2026-04-09 10:00:00", "ohlc": _make_ohlc(8)},
        _thresholds={"h1_confidence_min": 0.65},
    )

    assert _result["status"] == "OK"
    assert _result["forecast_status"] == "SUCCESS"
    assert _result["forecast_role"] == "TACTICAL_BIAS"
    assert _result["net_direction"] == "LONG_BIAS"
    assert _result["bias_direction"] == "LONG_BIAS"
    assert _result["bias_ready"] is True
    assert _result["bias_alignment_hint"] == "LONG_ONLY"
    assert _result["predicted_path_type"] == "LINEAR_INTERPOLATED_HORIZON_PATH"
    assert _result["predicted_path_source_horizons"] == [6, 7, 8]
    assert _result["reason_codes"] == [
        "H1_FORECAST_MODEL_LONG_BIAS",
        "H1_FORECAST_CONFIDENCE_OK",
    ]
    assert _result["raw_features"]["active_model_id"] == "h1model_test"
    assert _result["raw_features"]["dataset_id"] == "h1ds_test"


def test_low_confidence_becomes_neutral():
    def _predictor(**_):
        return {
            "net_direction": "SHORT_BIAS",
            "direction_score_long": 0.25,
            "direction_score_short": 0.75,
            "confidence": 0.49,
            "predicted_path": [149.9, 149.7],
        }

    _result = _run_with_patches(
        _sequence_length=5,
        _predictor=_predictor,
        _h1_data={"timestamp_jst": "2026-04-09 10:00:00", "ohlc": _make_ohlc(8)},
        _thresholds={"h1_confidence_min": 0.65},
    )

    assert _result["status"] == "OK"
    assert _result["forecast_status"] == "NEUTRAL"
    assert _result["net_direction"] == "NEUTRAL"
    assert _result["bias_direction"] == "NEUTRAL"
    assert _result["bias_ready"] is False
    assert _result["bias_alignment_hint"] == "NONE"
    assert _result["reason_codes"] == [
        "H1_FORECAST_MODEL_SHORT_BIAS",
        "H1_FORECAST_CONFIDENCE_LOW",
    ]
    assert _result["direction_score_short"] == 0.75


def test_predictor_error_returns_forecast_error():
    def _predictor(**_):
        raise RuntimeError("boom")

    _result = _run_with_patches(
        _sequence_length=5,
        _predictor=_predictor,
        _h1_data={"timestamp_jst": "2026-04-09 10:00:00", "ohlc": _make_ohlc(8)},
        _thresholds={},
    )

    assert _result["status"] == "ERROR"
    assert _result["forecast_status"] == "FORECAST_ERROR"
    assert _result["net_direction"] == "NEUTRAL"
    assert _result["bias_ready"] is False
    assert _result["reason_codes"] == ["H1_FORECAST_ERROR"]


def test_runtime_view_and_alignment_helpers():
    _forecast_result = {
        "forecast_status": "SUCCESS",
        "net_direction": "LONG_BIAS",
        "direction_score_long": 0.82,
        "direction_score_short": 0.18,
        "confidence": 0.77,
        "predicted_path": [150.2, 150.5],
        "summary": "ok",
        "raw_features": {
            "active_model_id": "h1model_test",
            "dataset_id": "h1ds_test",
            "artifact_selection_source": "active_runtime",
            "sequence_length": 81,
        },
    }

    _runtime_view = h1_forecast.build_h1_runtime_view(_forecast_result)
    _alignment_result = h1_forecast.evaluate_h1_alignment(
        _h1_forecast_result=_forecast_result,
        _env_direction="LONG_ONLY",
        _thresholds={"h1_confidence_min": 0.65},
    )

    assert _runtime_view["active_model_id"] == "h1model_test"
    assert _runtime_view["dataset_id"] == "h1ds_test"
    assert _runtime_view["forecast_role"] == "TACTICAL_BIAS"
    assert _runtime_view["bias_direction"] == "LONG_BIAS"
    assert _runtime_view["bias_ready"] is True
    assert _runtime_view["bias_alignment_hint"] == "LONG_ONLY"
    assert _alignment_result["alignment"] == "ALIGNED"
    assert _alignment_result["is_tradeable"] is True

    _neutral_alignment = h1_forecast.evaluate_h1_alignment(
        _h1_forecast_result={},
        _env_direction="LONG_ONLY",
        _thresholds={},
    )
    assert _neutral_alignment["alignment"] == "UNAVAILABLE"
    assert _neutral_alignment["is_available"] is False
    assert _neutral_alignment["forecast_role"] == "TACTICAL_BIAS"


def main():
    test_insufficient_data()
    test_successful_long_bias()
    test_low_confidence_becomes_neutral()
    test_predictor_error_returns_forecast_error()
    test_runtime_view_and_alignment_helpers()
    print("[PASS] test_h1_forecast_refactor")


if __name__ == "__main__":
    main()
