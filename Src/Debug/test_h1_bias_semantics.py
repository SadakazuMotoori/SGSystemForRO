import sys
from pathlib import Path


_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from Framework.ROModule.h1_forecast import build_h1_runtime_view, evaluate_h1_alignment


def test_runtime_view_derives_bias_fields_from_legacy_shape():
    _forecast_result = {
        "forecast_status": "SUCCESS",
        "net_direction": "LONG_BIAS",
        "confidence": 0.81,
        "predicted_path": [150.2, 150.4],
        "raw_features": {
            "horizons": [6, 7, 8],
        },
    }

    _runtime_view = build_h1_runtime_view(_forecast_result)

    assert _runtime_view["forecast_role"] == "TACTICAL_BIAS"
    assert _runtime_view["bias_direction"] == "LONG_BIAS"
    assert _runtime_view["bias_ready"] is True
    assert _runtime_view["bias_alignment_hint"] == "LONG_ONLY"
    assert _runtime_view["predicted_path_type"] == "LINEAR_INTERPOLATED_HORIZON_PATH"
    assert _runtime_view["predicted_path_source_horizons"] == [6, 7, 8]


def test_alignment_conflict_uses_shared_runtime_view_fields():
    _forecast_result = {
        "forecast_role": "TACTICAL_BIAS",
        "forecast_status": "SUCCESS",
        "net_direction": "SHORT_BIAS",
        "bias_direction": "SHORT_BIAS",
        "bias_ready": True,
        "bias_alignment_hint": "SHORT_ONLY",
        "confidence": 0.82,
        "predicted_path": [149.8, 149.6],
        "predicted_path_type": "LINEAR_INTERPOLATED_HORIZON_PATH",
        "predicted_path_source_horizons": [6, 7, 8],
        "raw_features": {},
    }

    _alignment_result = evaluate_h1_alignment(
        _h1_forecast_result=_forecast_result,
        _env_direction="LONG_ONLY",
        _thresholds={"h1_confidence_min": 0.65},
    )

    assert _alignment_result["alignment"] == "CONFLICT"
    assert _alignment_result["is_tradeable"] is False
    assert _alignment_result["bias_direction"] == "SHORT_BIAS"
    assert _alignment_result["bias_ready"] is True
    assert _alignment_result["predicted_path_type"] == "LINEAR_INTERPOLATED_HORIZON_PATH"


def test_unavailable_result_returns_neutral_runtime_view():
    _runtime_view = build_h1_runtime_view(None)
    _alignment_result = evaluate_h1_alignment(
        _h1_forecast_result=None,
        _env_direction="LONG_ONLY",
        _thresholds={},
    )

    assert _runtime_view["forecast_role"] == "TACTICAL_BIAS"
    assert _runtime_view["bias_direction"] == "NEUTRAL"
    assert _runtime_view["bias_ready"] is False
    assert _alignment_result["alignment"] == "UNAVAILABLE"
    assert _alignment_result["forecast_role"] == "TACTICAL_BIAS"
    assert _alignment_result["bias_direction"] == "NEUTRAL"


def main():
    test_runtime_view_derives_bias_fields_from_legacy_shape()
    test_alignment_conflict_uses_shared_runtime_view_fields()
    test_unavailable_result_returns_neutral_runtime_view()
    print("[PASS] test_h1_bias_semantics")


if __name__ == "__main__":
    main()
