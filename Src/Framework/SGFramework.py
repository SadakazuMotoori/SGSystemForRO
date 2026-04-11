from Framework.ROModule.external_filter import evaluate_external_filter
from Framework.ROModule.final_decision import evaluate_final_decision
from Framework.ROModule.h1_forecast import evaluate_h1_forecast
from Framework.ROModule.h2_environment import evaluate_h2_environment
from Framework.ROModule.main_flow_gate import (
    build_main_flow_gated_decision,
    resolve_m15_predicted_path_gap_threshold_pips,
)
from Framework.ROModule.m15_entry import evaluate_m15_entry


def RunDecisionPipeline(_market_data, _external_context, _system_context, _thresholds):
    _external_filter_result = evaluate_external_filter(
        market_data=_market_data["M15"],
        external_context=_external_context,
        system_context=_system_context,
        thresholds=_thresholds,
    )

    _h2_environment_result = evaluate_h2_environment(
        market_data_h2=_market_data["H2"],
        external_filter_result=_external_filter_result,
        thresholds=_thresholds,
    )

    _h1_forecast_result = evaluate_h1_forecast(
        _h1_data=_market_data["H1"],
        _thresholds=_thresholds,
    )

    _m15_entry_result = evaluate_m15_entry(
        market_data_m15=_market_data["M15"],
        h2_environment_result=_h2_environment_result,
        h1_forecast_result=_h1_forecast_result,
        external_filter_result=_external_filter_result,
        thresholds=_thresholds,
    )

    _base_final_decision_result = evaluate_final_decision(
        external_filter_result=_external_filter_result,
        h2_environment_result=_h2_environment_result,
        h1_forecast_result=_h1_forecast_result,
        m15_entry_result=_m15_entry_result,
        thresholds=_thresholds,
    )
    _gap_threshold_pips = resolve_m15_predicted_path_gap_threshold_pips(_thresholds)
    _m15_path_signal_result, _final_decision_result = build_main_flow_gated_decision(
        _market_data=_market_data,
        _h2_environment_result=_h2_environment_result,
        _h1_forecast_result=_h1_forecast_result,
        _base_final_decision_result=_base_final_decision_result,
        _gap_threshold_pips=_gap_threshold_pips,
    )

    return {
        "external_filter_result": _external_filter_result,
        "h2_environment_result": _h2_environment_result,
        "h2_regime_result": _h2_environment_result,
        "h1_forecast_result": _h1_forecast_result,
        "m15_path_signal_result": _m15_path_signal_result,
        "m15_entry_result": _m15_entry_result,
        "base_final_decision_result": _base_final_decision_result,
        "final_decision_result": _final_decision_result,
    }
