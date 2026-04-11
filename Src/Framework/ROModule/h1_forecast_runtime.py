from Framework.ROModule.h1_forecast_policy import (
    build_h1_forecast_decision,
    build_h1_insufficient_data_decision,
    build_h1_raw_features,
    classify_h1_model_decision,
    normalize_h1_model_payload,
)


def build_h1_module_result(_timestamp_jst, _status, _decision, _module_name="h1_forecast"):
    return {
        "module_name": _module_name,
        "timestamp_jst": _timestamp_jst,
        "status": _status,
        "forecast_role": _decision["forecast_role"],
        "forecast_status": _decision["forecast_status"],
        "net_direction": _decision["net_direction"],
        "bias_direction": _decision["bias_direction"],
        "bias_ready": _decision["bias_ready"],
        "bias_alignment_hint": _decision["bias_alignment_hint"],
        "direction_score_long": _decision["direction_score_long"],
        "direction_score_short": _decision["direction_score_short"],
        "confidence": _decision["confidence"],
        "predicted_path": _decision["predicted_path"],
        "predicted_path_type": _decision["predicted_path_type"],
        "predicted_path_source_horizons": _decision["predicted_path_source_horizons"],
        "reason_codes": _decision["reason_codes"],
        "summary": _decision["summary"],
        "raw_features": _decision["raw_features"],
    }


def evaluate_h1_runtime_direction(
    _close_list,
    _ohlc,
    _timestamp_jst,
    _thresholds,
    _required_sequence_length,
    _forecast_engine_name,
    _predictor,
    _recent_feature_builder,
):
    _recent_features = _recent_feature_builder(_close_list)

    if len(_close_list) < int(_required_sequence_length):
        return build_h1_insufficient_data_decision(
            _recent_features=_recent_features,
            _required_sequence_length=_required_sequence_length,
            _forecast_engine_name=_forecast_engine_name,
        )

    _model_payload = _predictor(_ohlc=_ohlc, _timestamp_jst=_timestamp_jst)
    _normalized_payload = normalize_h1_model_payload(_model_payload)
    _raw_features = build_h1_raw_features(
        _recent_features=_recent_features,
        _forecast_engine_name=_forecast_engine_name,
        _model_payload=_model_payload,
    )

    return classify_h1_model_decision(_normalized_payload, _thresholds, _raw_features)


def build_h1_error_result(
    _timestamp_jst,
    _error,
    _ohlc,
    _forecast_engine_name,
    _close_extractor,
    _recent_feature_builder,
    _module_name="h1_forecast",
):
    _recent_features = _recent_feature_builder(_close_extractor(_ohlc))
    _decision = build_h1_forecast_decision(
        _forecast_status="FORECAST_ERROR",
        _net_direction="NEUTRAL",
        _direction_score_long=0.0,
        _direction_score_short=0.0,
        _confidence=0.0,
        _predicted_path=[],
        _reason_codes=["H1_FORECAST_ERROR"],
        _summary=f"h1_forecast error: {_error}",
        _raw_features=build_h1_raw_features(_recent_features, _forecast_engine_name),
    )
    return build_h1_module_result(_timestamp_jst, "ERROR", _decision, _module_name=_module_name)


def evaluate_h1_forecast_runtime(
    _h1_data,
    _thresholds,
    _required_sequence_length_resolver,
    _forecast_engine_name_resolver,
    _predictor,
    _close_extractor,
    _recent_feature_builder,
    _module_name="h1_forecast",
):
    _timestamp_jst = _h1_data.get("timestamp_jst", "")
    _ohlc = _h1_data.get("ohlc", [])

    try:
        _close_list = _close_extractor(_ohlc)
        _decision = evaluate_h1_runtime_direction(
            _close_list=_close_list,
            _ohlc=_ohlc,
            _timestamp_jst=_timestamp_jst,
            _thresholds=_thresholds,
            _required_sequence_length=int(_required_sequence_length_resolver()),
            _forecast_engine_name=str(_forecast_engine_name_resolver()),
            _predictor=_predictor,
            _recent_feature_builder=_recent_feature_builder,
        )
        return build_h1_module_result(_timestamp_jst, "OK", _decision, _module_name=_module_name)
    except Exception as _error:
        try:
            _forecast_engine_name = str(_forecast_engine_name_resolver())
        except Exception:
            _forecast_engine_name = "unknown_h1_forecast_backend"

        return build_h1_error_result(
            _timestamp_jst=_timestamp_jst,
            _error=_error,
            _ohlc=_ohlc,
            _forecast_engine_name=_forecast_engine_name,
            _close_extractor=_close_extractor,
            _recent_feature_builder=_recent_feature_builder,
            _module_name=_module_name,
        )
