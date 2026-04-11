# --------------------------------------------------
# test_phase1_flow.py
# 役割:
#   固定入力で判定フロー全体を確認するためのデバッグ用スクリプト
#
# テスト方針:
#   - 現行 M15 スキーマに合わせた固定データを使う
#   - external_filter / h2_environment / m15_entry / final_decision を通す
#   - 必要に応じて H1 固定結果も差し込めるようにする
#   - ケースごとに期待最終アクションとの一致を表示する
# --------------------------------------------------

import json
import sys
from pathlib import Path


_srcDir = Path(__file__).resolve().parents[1]
if str(_srcDir) not in sys.path:
    sys.path.insert(0, str(_srcDir))

from Framework.ROModule.external_filter import evaluate_external_filter
from Framework.ROModule.final_decision import evaluate_final_decision
from Framework.ROModule.h1_forecast import build_h1_runtime_view
from Framework.ROModule.h2_environment import evaluate_h2_environment
from Framework.ROModule.m15_entry import evaluate_m15_entry


def BuildTestThresholds():
    return {
        "spread_max": 0.02,
        "adx_min": 20.0,
        "trend_strength_min": 0.55,
        "m15_entry_score_min": 70,
        "m15_noise_max": 0.40,
        "h1_confidence_min": 0.65,
    }


def BuildTestSystemContext():
    return {
        "round_id": "20260404120000",
        "latest_update_jst": "2026-04-04 12:00:00",
        "last_decision": None,
        "last_entry_result": None,
        "position_state": "FLAT",
    }


def BuildBaseH2Long():
    return {
        "symbol": "USDJPY",
        "timeframe": "H2",
        "timestamp_jst": "2026-04-04 12:00:00",
        "ohlc": [],
        "indicators": {
            "ma_short": 158.40,
            "ma_long": 158.00,
            "ma_slope": 0.12,
            "adx": 28.0,
            "swing_structure": "HIGHER_HIGH",
        },
    }


def BuildBaseM15():
    return {
        "symbol": "USDJPY",
        "timeframe": "M15",
        "timestamp_jst": "2026-04-04 12:00:00",
        "ohlc": [],
        "indicators": {
            "momentum": 0.82,
            "pullback_state": "PULLBACK_LONG",
            "breakout": "BREAKOUT_UP",
            "noise": 0.10,
        },
        "spread": 0.01,
    }


def BuildBaseExternalContext():
    return {
        "high_impact_event_soon": False,
        "central_bank_speech": False,
        "geopolitical_alert": False,
        "data_feed_error": False,
        "abnormal_volatility": False,
    }


def BuildCaseEnterLong():
    return (
        BuildBaseH2Long(),
        BuildBaseM15(),
        BuildBaseExternalContext(),
        None,
        "ENTER_LONG",
    )


def BuildCaseExternalStop():
    _marketDataH2 = BuildBaseH2Long()
    _marketDataM15 = BuildBaseM15()
    _externalContext = BuildBaseExternalContext()
    _externalContext["high_impact_event_soon"] = True

    return _marketDataH2, _marketDataM15, _externalContext, None, "NO_TRADE"


def BuildCaseH2NoTrade():
    _marketDataH2 = BuildBaseH2Long()
    _marketDataM15 = BuildBaseM15()
    _externalContext = BuildBaseExternalContext()

    _marketDataH2["indicators"] = {
        "ma_short": 158.00,
        "ma_long": 158.00,
        "ma_slope": 0.00,
        "adx": 10.0,
        "swing_structure": "RANGE",
    }

    return _marketDataH2, _marketDataM15, _externalContext, None, "NO_TRADE"


def BuildCaseM15Wait():
    _marketDataH2 = BuildBaseH2Long()
    _marketDataM15 = BuildBaseM15()
    _externalContext = BuildBaseExternalContext()

    _marketDataM15["indicators"] = {
        "momentum": 0.08,
        "pullback_state": "NONE",
        "breakout": "BREAKOUT_UP",
        "noise": 0.25,
    }

    return _marketDataH2, _marketDataM15, _externalContext, None, "WAIT"


def BuildCaseM15Exit():
    _marketDataH2 = BuildBaseH2Long()
    _marketDataM15 = BuildBaseM15()
    _externalContext = BuildBaseExternalContext()

    _marketDataM15["indicators"] = {
        "momentum": 0.40,
        "pullback_state": "PULLBACK_LONG",
        "breakout": "BREAKOUT_UP",
        "noise": 0.70,
    }

    return _marketDataH2, _marketDataM15, _externalContext, None, "EXIT"


def BuildCaseH1ConflictWait():
    _marketDataH2 = BuildBaseH2Long()
    _marketDataM15 = BuildBaseM15()
    _externalContext = BuildBaseExternalContext()
    _h1ForecastResult = {
        "module_name": "h1_forecast",
        "timestamp_jst": "2026-04-04 12:00:00",
        "status": "OK",
        "forecast_status": "SUCCESS",
        "net_direction": "SHORT_BIAS",
        "direction_score_long": 0.2,
        "direction_score_short": 0.8,
        "confidence": 0.80,
        "predicted_path": [158.30, 158.20, 158.10, 158.00, 157.90, 157.80],
        "reason_codes": ["H1_FORECAST_DIRECTION_SHORT_DOMINANT", "H1_FORECAST_CONFIDENCE_OK"],
        "summary": "1時間足は下方向優位のため売りバイアス",
        "raw_features": {},
    }

    return _marketDataH2, _marketDataM15, _externalContext, _h1ForecastResult, "WAIT"


def RunCase(_caseName, _marketDataH2, _marketDataM15, _externalContext, _h1ForecastResult, _expectedFinalAction):
    _thresholds = BuildTestThresholds()
    _systemContext = BuildTestSystemContext()

    _externalFilterResult = evaluate_external_filter(
        market_data=_marketDataM15,
        external_context=_externalContext,
        system_context=_systemContext,
        thresholds=_thresholds,
    )

    _h2EnvironmentResult = evaluate_h2_environment(
        market_data_h2=_marketDataH2,
        external_filter_result=_externalFilterResult,
        thresholds=_thresholds,
    )

    _m15EntryResult = evaluate_m15_entry(
        market_data_m15=_marketDataM15,
        h2_environment_result=_h2EnvironmentResult,
        h1_forecast_result=_h1ForecastResult,
        external_filter_result=_externalFilterResult,
        thresholds=_thresholds,
    )

    _finalDecisionResult = evaluate_final_decision(
        external_filter_result=_externalFilterResult,
        h2_environment_result=_h2EnvironmentResult,
        h1_forecast_result=_h1ForecastResult,
        m15_entry_result=_m15EntryResult,
        thresholds=_thresholds,
    )

    _actualFinalAction = _finalDecisionResult.get("final_action")

    return {
        "case_name": _caseName,
        "expected_final_action": _expectedFinalAction,
        "actual_final_action": _actualFinalAction,
        "passed": _actualFinalAction == _expectedFinalAction,
        "external_filter_result": _externalFilterResult,
        "h2_environment_result": _h2EnvironmentResult,
        "h1_forecast_result": _h1ForecastResult,
        "h1_runtime_view": build_h1_runtime_view(_h1ForecastResult),
        "m15_entry_result": _m15EntryResult,
        "final_decision_result": _finalDecisionResult,
    }


def PrintCaseSummary(_result):
    print("")
    print("==================================================")
    print(f"CASE: {_result['case_name']}")
    print("==================================================")
    print(f"expected_final_action: {_result['expected_final_action']}")
    print(f"actual_final_action  : {_result['actual_final_action']}")
    print(f"passed               : {_result['passed']}")

    print("[external_filter]")
    print(json.dumps(_result["external_filter_result"], ensure_ascii=False, indent=2))

    print("[h2_environment]")
    print(json.dumps(_result["h2_environment_result"], ensure_ascii=False, indent=2))

    print("[h1_forecast]")
    print(json.dumps(_result["h1_forecast_result"], ensure_ascii=False, indent=2))

    print("[h1_runtime_view]")
    print(json.dumps(_result["h1_runtime_view"], ensure_ascii=False, indent=2))

    print("[m15_entry]")
    print(json.dumps(_result["m15_entry_result"], ensure_ascii=False, indent=2))

    print("[final_decision]")
    print(json.dumps(_result["final_decision_result"], ensure_ascii=False, indent=2))


def main():
    _builders = [
        ("ENTER_LONG正常系", BuildCaseEnterLong),
        ("外部停止条件ON", BuildCaseExternalStop),
        ("H2_NO_TRADE", BuildCaseH2NoTrade),
        ("M15_WAIT", BuildCaseM15Wait),
        ("M15_EXIT", BuildCaseM15Exit),
        ("H1_CONFLICT_WAIT", BuildCaseH1ConflictWait),
    ]

    _results = []
    for _caseName, _builder in _builders:
        _marketDataH2, _marketDataM15, _externalContext, _h1ForecastResult, _expectedFinalAction = _builder()
        _results.append(
            RunCase(
                _caseName,
                _marketDataH2,
                _marketDataM15,
                _externalContext,
                _h1ForecastResult,
                _expectedFinalAction,
            )
        )

    for _result in _results:
        PrintCaseSummary(_result)

    _failed_cases = [_result["case_name"] for _result in _results if not _result["passed"]]
    print("")
    print("==================================================")
    print("SUMMARY")
    print("==================================================")
    print(f"total : {len(_results)}")
    print(f"pass  : {len(_results) - len(_failed_cases)}")
    print(f"fail  : {len(_failed_cases)}")

    if len(_failed_cases) > 0:
        print(f"[FAIL] failed_cases={_failed_cases}")
    else:
        print("[OK] all fixed-flow cases passed")


if __name__ == "__main__":
    print("==========SGSystem Test Start==========")
    main()
    print("==========SGSystem Test End==========")
