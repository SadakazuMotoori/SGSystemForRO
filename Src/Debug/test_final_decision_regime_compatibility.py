# --------------------------------------------------
# test_final_decision_regime_compatibility.py
# Purpose:
#   Verify that final_decision can consume H2 regime-only fields
#   during the Phase 1 compatibility window.
# --------------------------------------------------

import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parents[1]
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Framework.ROModule.final_decision import evaluate_final_decision


def main():
    _external_filter_result = {
        "module_name": "external_filter",
        "timestamp_jst": "TEST_EXT",
        "status": "OK",
        "filter_status": "OFF",
        "can_trade": True,
        "reason_codes": [],
        "summary": "external filter off",
        "raw_features": {},
    }

    _h2_environment_result = {
        "module_name": "h2_environment",
        "timestamp_jst": "TEST_H2",
        "status": "OK",
        "regime_direction": "LONG_ONLY",
        "regime_score": 80,
        "regime_quality": "READY",
        "reason_codes": ["H2_ENVIRONMENT_REGIME_DIRECTION_LONG_ONLY"],
        "summary": "H2 regime allows long-only trades.",
        "raw_features": {},
    }

    _m15_entry_result = {
        "module_name": "m15_entry",
        "timestamp_jst": "TEST_M15",
        "status": "OK",
        "entry_action": "ENTER",
        "entry_side": "LONG",
        "entry_score": 82,
        "timing_quality": 0.82,
        "risk_flag": False,
        "reason_codes": ["M15_ENTRY_MOMENTUM_ALIGNED_STRONG"],
        "summary": "M15 entry is ready.",
        "raw_features": {},
    }

    _result = evaluate_final_decision(
        external_filter_result=_external_filter_result,
        h2_environment_result=_h2_environment_result,
        h1_forecast_result=None,
        m15_entry_result=_m15_entry_result,
        thresholds={"h1_confidence_min": 0.65},
    )

    if _result.get("final_action") != "ENTER_LONG":
        raise AssertionError(f"Unexpected final_action: {_result.get('final_action')}")

    if _result.get("approved") is not True:
        raise AssertionError("Expected approved=True for regime-compatible long entry.")

    print("[OK] final_decision regime compatibility test passed.")


if __name__ == "__main__":
    main()
