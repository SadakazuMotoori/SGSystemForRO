from Framework.Utility.Utility import ToFloat as _to_float


_DEFAULT_CONFIDENCE_MIN = 0.65
_NEUTRAL_FORECAST_STATUSES = {"NEUTRAL", "INSUFFICIENT_DATA", "FORECAST_ERROR"}
_FORECAST_ROLE = "TACTICAL_BIAS"
_PREDICTED_PATH_TYPE = "LINEAR_INTERPOLATED_HORIZON_PATH"


def resolve_h1_confidence_min(_thresholds):
    if not isinstance(_thresholds, dict):
        return float(_DEFAULT_CONFIDENCE_MIN)

    return _to_float(_thresholds.get("h1_confidence_min"), _DEFAULT_CONFIDENCE_MIN)


def has_h1_forecast_result(_h1_forecast_result):
    return isinstance(_h1_forecast_result, dict) and len(_h1_forecast_result) > 0


def build_h1_runtime_view(_h1_forecast_result):
    if not has_h1_forecast_result(_h1_forecast_result):
        return {
            "forecast_role": _FORECAST_ROLE,
            "forecast_status": None,
            "net_direction": None,
            "bias_direction": "NEUTRAL",
            "bias_ready": False,
            "bias_alignment_hint": "NONE",
            "direction_score_long": 0.0,
            "direction_score_short": 0.0,
            "confidence": 0.0,
            "predicted_path": [],
            "predicted_path_type": _PREDICTED_PATH_TYPE,
            "predicted_path_source_horizons": [],
            "summary": None,
            "active_model_id": None,
            "dataset_id": None,
            "artifact_selection_source": None,
            "sequence_length": 0,
        }

    _raw_features = _h1_forecast_result.get("raw_features", {})
    if not isinstance(_raw_features, dict):
        _raw_features = {}

    _forecast_status = _h1_forecast_result.get("forecast_status")
    _net_direction = _h1_forecast_result.get("net_direction")
    _bias_direction = _h1_forecast_result.get("bias_direction")
    if _bias_direction not in ["LONG_BIAS", "SHORT_BIAS", "NEUTRAL"]:
        if _forecast_status == "SUCCESS" and _net_direction in ["LONG_BIAS", "SHORT_BIAS"]:
            _bias_direction = _net_direction
        else:
            _bias_direction = "NEUTRAL"

    return {
        "forecast_role": _h1_forecast_result.get("forecast_role", _FORECAST_ROLE),
        "forecast_status": _forecast_status,
        "net_direction": _net_direction,
        "bias_direction": _bias_direction,
        "bias_ready": bool(
            _h1_forecast_result.get(
                "bias_ready",
                _forecast_status == "SUCCESS" and _bias_direction in ["LONG_BIAS", "SHORT_BIAS"],
            )
        ),
        "bias_alignment_hint": _h1_forecast_result.get(
            "bias_alignment_hint",
            "LONG_ONLY"
            if _bias_direction == "LONG_BIAS"
            else "SHORT_ONLY"
            if _bias_direction == "SHORT_BIAS"
            else "NONE",
        ),
        "direction_score_long": _to_float(_h1_forecast_result.get("direction_score_long")),
        "direction_score_short": _to_float(_h1_forecast_result.get("direction_score_short")),
        "confidence": _to_float(_h1_forecast_result.get("confidence")),
        "predicted_path": list(_h1_forecast_result.get("predicted_path", [])),
        "predicted_path_type": _h1_forecast_result.get(
            "predicted_path_type",
            _raw_features.get("predicted_path_type", _PREDICTED_PATH_TYPE),
        ),
        "predicted_path_source_horizons": list(
            _h1_forecast_result.get(
                "predicted_path_source_horizons",
                _raw_features.get("predicted_path_source_horizons", _raw_features.get("horizons", [])),
            )
        ),
        "summary": _h1_forecast_result.get("summary"),
        "active_model_id": _raw_features.get("active_model_id"),
        "dataset_id": _raw_features.get("dataset_id"),
        "artifact_selection_source": _raw_features.get("artifact_selection_source"),
        "sequence_length": int(_raw_features.get("sequence_length", 0)),
    }


def evaluate_h1_alignment(_h1_forecast_result, _env_direction, _thresholds):
    _runtime_view = build_h1_runtime_view(_h1_forecast_result)
    _confidence_min = resolve_h1_confidence_min(_thresholds)

    if not has_h1_forecast_result(_h1_forecast_result):
        return {
            "alignment": "UNAVAILABLE",
            "is_available": False,
            "is_aligned": False,
            "is_tradeable": False,
            "confidence_min": float(_confidence_min),
            **_runtime_view,
        }

    _forecast_status = _runtime_view["forecast_status"]
    _net_direction = _runtime_view["net_direction"]
    _confidence = _runtime_view["confidence"]

    if _env_direction not in ["LONG_ONLY", "SHORT_ONLY"]:
        return {
            "alignment": "UNAVAILABLE",
            "is_available": True,
            "is_aligned": False,
            "is_tradeable": False,
            "confidence_min": float(_confidence_min),
            **_runtime_view,
        }

    if _net_direction == "NEUTRAL" or _forecast_status in _NEUTRAL_FORECAST_STATUSES:
        return {
            "alignment": "NEUTRAL_OR_SKIPPED",
            "is_available": True,
            "is_aligned": False,
            "is_tradeable": False,
            "confidence_min": float(_confidence_min),
            **_runtime_view,
        }

    if _confidence < _confidence_min:
        return {
            "alignment": "LOW_CONFIDENCE",
            "is_available": True,
            "is_aligned": False,
            "is_tradeable": False,
            "confidence_min": float(_confidence_min),
            **_runtime_view,
        }

    if _env_direction == "LONG_ONLY" and _net_direction == "LONG_BIAS":
        return {
            "alignment": "ALIGNED",
            "is_available": True,
            "is_aligned": True,
            "is_tradeable": True,
            "confidence_min": float(_confidence_min),
            **_runtime_view,
        }

    if _env_direction == "SHORT_ONLY" and _net_direction == "SHORT_BIAS":
        return {
            "alignment": "ALIGNED",
            "is_available": True,
            "is_aligned": True,
            "is_tradeable": True,
            "confidence_min": float(_confidence_min),
            **_runtime_view,
        }

    return {
        "alignment": "CONFLICT",
        "is_available": True,
        "is_aligned": False,
        "is_tradeable": False,
        "confidence_min": float(_confidence_min),
        **_runtime_view,
    }
