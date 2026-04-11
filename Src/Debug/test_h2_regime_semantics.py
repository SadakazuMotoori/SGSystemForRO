# --------------------------------------------------
# test_h2_regime_semantics.py
# Purpose:
#   Verify the Phase 1 H2 regime additions without relying on MT5.
#   This checks:
#   - legacy threshold keys still work
#   - new regime threshold keys also work
#   - new regime-oriented output fields are present
# --------------------------------------------------

import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Framework.ROModule.h2_environment import evaluate_h2_environment
from Framework.ROModule.h2_environment_contract import build_h2_runtime_view


def _build_market_data_h2():
    return {
        "symbol": "USDJPY",
        "timeframe": "H2",
        "timestamp_jst": "2026-04-08 12:00:00",
        "ohlc": [],
        "indicators": {
            "ma_short": 158.40,
            "ma_long": 158.00,
            "ma_slope": 0.12,
            "adx": 28.0,
            "swing_structure": "HIGHER_HIGH",
        },
    }


def _build_external_filter_result():
    return {
        "can_trade": True,
    }


def _build_legacy_thresholds():
    return {
        "adx_min": 20.0,
        "trend_strength_min": 0.55,
    }


def _build_regime_thresholds():
    return {
        "h2_regime_adx_min": 20.0,
        "h2_regime_strength_min": 0.55,
        "h2_regime_score_min": 55,
    }


def _assert_required_keys(_result):
    _required_keys = [
        "env_direction",
        "env_score",
        "trend_strength",
        "reason_codes",
        "regime_direction",
        "regime_score",
        "regime_quality",
        "regime_components",
        "regime_reason_codes",
    ]

    for _key in _required_keys:
        if _key not in _result:
            raise AssertionError(f"Missing key: {_key}")


def _assert_runtime_view_keys(_runtime_view):
    _required_keys = [
        "status",
        "env_direction",
        "env_score",
        "trend_strength",
        "regime_direction",
        "regime_score",
        "regime_quality",
        "summary",
        "reason_codes",
        "regime_reason_codes",
    ]

    for _key in _required_keys:
        if _key not in _runtime_view:
            raise AssertionError(f"Missing runtime_view key: {_key}")


def main():
    _market_data_h2 = _build_market_data_h2()
    _external_filter_result = _build_external_filter_result()

    _legacy_result = evaluate_h2_environment(
        market_data_h2=_market_data_h2,
        external_filter_result=_external_filter_result,
        thresholds=_build_legacy_thresholds(),
    )
    _regime_result = evaluate_h2_environment(
        market_data_h2=_market_data_h2,
        external_filter_result=_external_filter_result,
        thresholds=_build_regime_thresholds(),
    )

    _assert_required_keys(_legacy_result)
    _assert_required_keys(_regime_result)

    _legacy_runtime_view = build_h2_runtime_view(_legacy_result)
    _regime_runtime_view = build_h2_runtime_view(_regime_result)

    _assert_runtime_view_keys(_legacy_runtime_view)
    _assert_runtime_view_keys(_regime_runtime_view)

    if _legacy_result["env_direction"] != _regime_result["env_direction"]:
        raise AssertionError("Legacy and regime thresholds produced different env_direction.")

    if _legacy_result["env_score"] != _regime_result["env_score"]:
        raise AssertionError("Legacy and regime thresholds produced different env_score.")

    if abs(_legacy_result["trend_strength"] - _regime_result["trend_strength"]) > 1.0e-9:
        raise AssertionError("Legacy and regime thresholds produced different trend_strength.")

    if _legacy_runtime_view["env_direction"] != _legacy_result["env_direction"]:
        raise AssertionError("Legacy runtime view env_direction mismatch.")

    if _regime_runtime_view["regime_score"] != _regime_result["regime_score"]:
        raise AssertionError("Regime runtime view regime_score mismatch.")

    print("[OK] h2 regime semantics test passed.")


if __name__ == "__main__":
    main()
