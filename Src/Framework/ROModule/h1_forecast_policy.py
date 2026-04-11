from Framework.ROModule.h1_forecast_contract import resolve_h1_confidence_min
from Framework.Utility.Utility import Clamp01 as _clamp_01, ToFloat as _to_float


_FORECAST_ROLE = "TACTICAL_BIAS"
_PREDICTED_PATH_TYPE = "LINEAR_INTERPOLATED_HORIZON_PATH"


def _rc(_suffix: str) -> str:
    return f"H1_FORECAST_{_suffix}"


def _resolve_bias_direction(_forecast_status, _net_direction):
    if _forecast_status == "SUCCESS" and _net_direction in ["LONG_BIAS", "SHORT_BIAS"]:
        return _net_direction

    return "NEUTRAL"


def _resolve_bias_ready(_forecast_status, _bias_direction):
    return bool(_forecast_status == "SUCCESS" and _bias_direction in ["LONG_BIAS", "SHORT_BIAS"])


def _resolve_bias_alignment_hint(_bias_direction):
    if _bias_direction == "LONG_BIAS":
        return "LONG_ONLY"
    if _bias_direction == "SHORT_BIAS":
        return "SHORT_ONLY"

    return "NONE"


def build_h1_forecast_decision(
    _forecast_status,
    _net_direction,
    _direction_score_long,
    _direction_score_short,
    _confidence,
    _predicted_path,
    _reason_codes,
    _summary,
    _raw_features,
):
    _bias_direction = _resolve_bias_direction(_forecast_status, _net_direction)
    _bias_ready = _resolve_bias_ready(_forecast_status, _bias_direction)

    return {
        "forecast_role": _FORECAST_ROLE,
        "forecast_status": _forecast_status,
        "net_direction": _net_direction,
        "bias_direction": _bias_direction,
        "bias_ready": _bias_ready,
        "bias_alignment_hint": _resolve_bias_alignment_hint(_bias_direction),
        "direction_score_long": float(_direction_score_long),
        "direction_score_short": float(_direction_score_short),
        "confidence": float(_confidence),
        "predicted_path": list(_predicted_path),
        "predicted_path_type": _raw_features.get("predicted_path_type", _PREDICTED_PATH_TYPE),
        "predicted_path_source_horizons": list(_raw_features.get("predicted_path_source_horizons", [])),
        "reason_codes": list(_reason_codes),
        "summary": _summary,
        "raw_features": _raw_features,
    }


def build_h1_raw_features(_recent_features, _forecast_engine_name, _model_payload=None):
    _raw_features = {
        "close_list": list(_recent_features["recent_close_list"]),
        "close_diff_list": list(_recent_features["recent_diff_list"]),
        "recent_momentum": float(_recent_features["recent_momentum"]),
        "trend_consistency": float(_recent_features["trend_consistency"]),
        "forecast_engine": _forecast_engine_name,
        "predicted_path_type": _PREDICTED_PATH_TYPE,
        "predicted_path_source_horizons": [],
    }

    if isinstance(_model_payload, dict):
        _raw_features.update(
            {
                "artifact_role": _model_payload.get("artifact_role"),
                "active_model_id": _model_payload.get("active_model_id"),
                "artifact_selection_source": _model_payload.get("artifact_selection_source"),
                "dataset_id": _model_payload.get("dataset_id"),
                "sequence_length": int(_model_payload.get("sequence_length", 0)),
                "history_end_timestamp_jst": _model_payload.get("history_end_timestamp_jst"),
                "horizons": list(_model_payload.get("horizons", [])),
                "predicted_path_source_horizons": list(_model_payload.get("horizons", [])),
                "target_scale": _to_float(_model_payload.get("target_scale")),
                "signal_strength": _to_float(_model_payload.get("signal_strength")),
                "direction_dominance": _to_float(_model_payload.get("direction_dominance")),
                "predicted_delta_by_horizon": dict(_model_payload.get("predicted_delta_by_horizon", {})),
                "predicted_close_by_horizon": dict(_model_payload.get("predicted_close_by_horizon", {})),
                "drift_baseline": list(_model_payload.get("drift_baseline", [])),
            }
        )

    return _raw_features


def build_h1_insufficient_data_decision(_recent_features, _required_sequence_length, _forecast_engine_name):
    return build_h1_forecast_decision(
        _forecast_status="INSUFFICIENT_DATA",
        _net_direction="NEUTRAL",
        _direction_score_long=0.0,
        _direction_score_short=0.0,
        _confidence=0.0,
        _predicted_path=[],
        _reason_codes=[_rc("DATA_INSUFFICIENT")],
        _summary=f"H1 requires at least {_required_sequence_length} bars.",
        _raw_features=build_h1_raw_features(_recent_features, _forecast_engine_name),
    )


def normalize_h1_model_payload(_model_payload):
    if not isinstance(_model_payload, dict):
        _model_payload = {}

    return {
        "net_direction": _model_payload.get("net_direction", "NEUTRAL"),
        "direction_score_long": _clamp_01(_to_float(_model_payload.get("direction_score_long"))),
        "direction_score_short": _clamp_01(_to_float(_model_payload.get("direction_score_short"))),
        "confidence": _clamp_01(_to_float(_model_payload.get("confidence"))),
        "predicted_path": list(_model_payload.get("predicted_path", [])),
    }


def _resolve_direction_reason(_net_direction):
    if _net_direction == "LONG_BIAS":
        return _rc("MODEL_LONG_BIAS")

    return _rc("MODEL_SHORT_BIAS")


def classify_h1_model_decision(_normalized_payload, _thresholds, _raw_features):
    _net_direction = _normalized_payload["net_direction"]
    _direction_score_long = _normalized_payload["direction_score_long"]
    _direction_score_short = _normalized_payload["direction_score_short"]
    _confidence = _normalized_payload["confidence"]
    _predicted_path = _normalized_payload["predicted_path"]
    _confidence_min = resolve_h1_confidence_min(_thresholds)

    if _net_direction == "NEUTRAL":
        return build_h1_forecast_decision(
            _forecast_status="NEUTRAL",
            _net_direction="NEUTRAL",
            _direction_score_long=_direction_score_long,
            _direction_score_short=_direction_score_short,
            _confidence=_confidence,
            _predicted_path=_predicted_path,
            _reason_codes=[_rc("MODEL_DIRECTION_NEUTRAL")],
            _summary="H1 model signal is mixed across horizons.",
            _raw_features=_raw_features,
        )

    _direction_reason = _resolve_direction_reason(_net_direction)

    if _confidence >= _confidence_min:
        return build_h1_forecast_decision(
            _forecast_status="SUCCESS",
            _net_direction=_net_direction,
            _direction_score_long=_direction_score_long,
            _direction_score_short=_direction_score_short,
            _confidence=_confidence,
            _predicted_path=_predicted_path,
            _reason_codes=[_direction_reason, _rc("CONFIDENCE_OK")],
            _summary="H1 model forecast has sufficient confidence.",
            _raw_features=_raw_features,
        )

    return build_h1_forecast_decision(
        _forecast_status="NEUTRAL",
        _net_direction="NEUTRAL",
        _direction_score_long=_direction_score_long,
        _direction_score_short=_direction_score_short,
        _confidence=_confidence,
        _predicted_path=_predicted_path,
        _reason_codes=[_direction_reason, _rc("CONFIDENCE_LOW")],
        _summary="H1 model forecast is below the confidence threshold.",
        _raw_features=_raw_features,
    )
