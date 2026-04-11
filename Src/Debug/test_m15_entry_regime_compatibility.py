# --------------------------------------------------
# test_m15_entry_regime_compatibility.py
# Purpose:
#   Verify that m15_entry can consume H2 regime-only fields
#   during the Phase 1 compatibility window.
# --------------------------------------------------

import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Framework.ROModule.m15_entry import evaluate_m15_entry


def main():
    _market_data_m15 = {
        "symbol": "USDJPY",
        "timeframe": "M15",
        "timestamp_jst": "2026-04-08 12:00:00",
        "ohlc": [],
        "indicators": {
            "momentum": 0.82,
            "pullback_state": "PULLBACK_LONG",
            "breakout": "BREAKOUT_UP",
            "noise": 0.10,
        },
        "spread": 0.01,
    }

    _h2_environment_result = {
        "module_name": "h2_environment",
        "timestamp_jst": "2026-04-08 12:00:00",
        "status": "OK",
        "regime_direction": "LONG_ONLY",
        "regime_score": 80,
        "regime_quality": "READY",
        "summary": "H2 regime allows long-only trades.",
        "reason_codes": ["H2_ENVIRONMENT_REGIME_DIRECTION_LONG_ONLY"],
        "raw_features": {},
    }

    _result = evaluate_m15_entry(
        market_data_m15=_market_data_m15,
        h2_environment_result=_h2_environment_result,
        h1_forecast_result=None,
        external_filter_result={"can_trade": True},
        thresholds={
            "spread_max": 0.02,
            "m15_noise_max": 0.40,
            "m15_entry_score_min": 70,
            "h1_confidence_min": 0.65,
        },
    )

    if _result.get("entry_action") != "ENTER":
        raise AssertionError(f"Unexpected entry_action: {_result.get('entry_action')}")

    if _result.get("entry_side") != "LONG":
        raise AssertionError(f"Unexpected entry_side: {_result.get('entry_side')}")

    _raw_features = _result.get("raw_features", {})
    if _raw_features.get("regime_direction") != "LONG_ONLY":
        raise AssertionError("Expected raw_features.regime_direction to be LONG_ONLY.")

    print("[OK] m15_entry regime compatibility test passed.")


if __name__ == "__main__":
    main()
