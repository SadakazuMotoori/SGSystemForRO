def resolve_h2_direction(_h2_environment_result):
    if not isinstance(_h2_environment_result, dict):
        return "NO_TRADE"

    _regime_direction = _h2_environment_result.get("regime_direction")
    if _regime_direction in ["LONG_ONLY", "SHORT_ONLY", "NO_TRADE"]:
        return _regime_direction

    _env_direction = _h2_environment_result.get("env_direction")
    if _env_direction in ["LONG_ONLY", "SHORT_ONLY", "NO_TRADE"]:
        return _env_direction

    return "NO_TRADE"


def resolve_h2_trend_strength(_h2_environment_result):
    if not isinstance(_h2_environment_result, dict):
        return 0.0

    try:
        return float(_h2_environment_result.get("trend_strength", 0.0))
    except Exception:
        pass

    try:
        return max(
            0.0,
            min(1.0, float(_h2_environment_result.get("regime_score", 0.0)) / 100.0),
        )
    except Exception:
        return 0.0


def resolve_h2_regime_score(_h2_environment_result):
    if not isinstance(_h2_environment_result, dict):
        return 0

    try:
        return int(round(float(_h2_environment_result.get("regime_score", 0.0))))
    except Exception:
        return int(round(resolve_h2_trend_strength(_h2_environment_result) * 100.0))


def resolve_h2_regime_quality(_h2_environment_result):
    if not isinstance(_h2_environment_result, dict):
        return "UNKNOWN"

    _regime_quality = _h2_environment_result.get("regime_quality")
    if isinstance(_regime_quality, str) and len(_regime_quality) > 0:
        return _regime_quality

    _env_direction = resolve_h2_direction(_h2_environment_result)
    if _env_direction in ["LONG_ONLY", "SHORT_ONLY"]:
        return "READY"

    return "UNKNOWN"


def build_h2_runtime_view(_h2_environment_result):
    if not isinstance(_h2_environment_result, dict):
        return {
            "status": "UNAVAILABLE",
            "env_direction": "NO_TRADE",
            "env_score": 0,
            "trend_strength": 0.0,
            "regime_direction": "NO_TRADE",
            "regime_score": 0,
            "regime_quality": "UNKNOWN",
            "summary": "",
            "reason_codes": [],
            "regime_reason_codes": [],
        }

    _env_direction = resolve_h2_direction(_h2_environment_result)
    return {
        "status": _h2_environment_result.get("status", "OK"),
        "env_direction": _env_direction,
        "env_score": _h2_environment_result.get("env_score", 0),
        "trend_strength": resolve_h2_trend_strength(_h2_environment_result),
        "regime_direction": _h2_environment_result.get("regime_direction", _env_direction),
        "regime_score": resolve_h2_regime_score(_h2_environment_result),
        "regime_quality": resolve_h2_regime_quality(_h2_environment_result),
        "summary": _h2_environment_result.get("summary", ""),
        "reason_codes": list(_h2_environment_result.get("reason_codes", [])),
        "regime_reason_codes": list(_h2_environment_result.get("regime_reason_codes", [])),
    }
