import json
import os
import sys


_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.ROModule.final_decision import evaluate_final_decision
from Framework.ROModule.h1_forecast import build_h1_runtime_view


def main():
    print("========== test_phase3_h2_h1_m15_integration_conflict start ==========")

    _thresholds = {
        "h1_confidence_min": 0.65,
    }

    _externalFilterResult = {
        "module_name": "external_filter",
        "timestamp_jst": "TEST_EXT",
        "status": "OK",
        "filter_status": "OFF",
        "can_trade": True,
        "reason_codes": [],
        "summary": "External filter is off.",
        "raw_features": {},
    }

    _h2EnvironmentResult = {
        "module_name": "h2_environment",
        "timestamp_jst": "TEST_H2",
        "status": "OK",
        "env_direction": "LONG_ONLY",
        "env_score": 1,
        "trend_strength": 0.8,
        "reason_codes": ["H2_ENVIRONMENT_MA_BULLISH", "H2_ENVIRONMENT_ADX_STRONG"],
        "summary": "H2 environment is bullish.",
        "raw_features": {},
    }

    _h1ForecastResult = {
        "module_name": "h1_forecast",
        "timestamp_jst": "TEST_H1",
        "status": "OK",
        "forecast_status": "SUCCESS",
        "net_direction": "SHORT_BIAS",
        "direction_score_long": 0.2,
        "direction_score_short": 0.8,
        "confidence": 0.8,
        "predicted_path": [100.0, 99.8, 99.6, 99.4, 99.2, 99.0],
        "reason_codes": ["H1_FORECAST_DIRECTION_SHORT_DOMINANT", "H1_FORECAST_CONFIDENCE_OK"],
        "summary": "H1 forecast is short bias.",
        "raw_features": {},
    }
    _h1RuntimeView = build_h1_runtime_view(_h1ForecastResult)

    _m15EntryResult = {
        "module_name": "m15_entry",
        "timestamp_jst": "TEST_M15",
        "status": "OK",
        "entry_action": "ENTER",
        "entry_side": "LONG",
        "entry_score": 82,
        "timing_quality": 0.82,
        "risk_flag": False,
        "reason_codes": ["M15_ENTRY_MOMENTUM_ALIGNED_STRONG", "M15_ENTRY_PULLBACK_ALIGNED"],
        "summary": "M15 entry is long.",
        "raw_features": {},
    }

    _finalDecisionResult = evaluate_final_decision(
        external_filter_result=_externalFilterResult,
        h2_environment_result=_h2EnvironmentResult,
        h1_forecast_result=_h1ForecastResult,
        m15_entry_result=_m15EntryResult,
        thresholds=_thresholds,
    )

    print("----- Conflict Summary -----")
    print(f"H2 env_direction      : {_h2EnvironmentResult.get('env_direction')}")
    print(f"H1 net_direction      : {_h1ForecastResult.get('net_direction')}")
    print(f"H1 bias_direction     : {_h1RuntimeView.get('bias_direction')}")
    print(f"H1 bias_ready         : {_h1RuntimeView.get('bias_ready')}")
    print(f"M15 entry_action      : {_m15EntryResult.get('entry_action')}")
    print(f"Final final_action    : {_finalDecisionResult.get('final_action')}")
    print(f"Final approved        : {_finalDecisionResult.get('approved')}")
    print(f"Final reason_codes    : {_finalDecisionResult.get('reason_codes')}")

    print("----- Conflict Preview JSON -----")
    print(
        json.dumps(
            {
                "h1_forecast_result": _h1ForecastResult,
                "h1_runtime_view": _h1RuntimeView,
                "final_decision_result": _finalDecisionResult,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("========== test_phase3_h2_h1_m15_integration_conflict end ==========")


if __name__ == "__main__":
    main()
