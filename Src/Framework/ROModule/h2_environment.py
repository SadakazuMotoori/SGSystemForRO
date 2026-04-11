# --------------------------------------------------
# h2_environment.py
#
# 役割:
#   H2 の上位足環境を見て、売買を許可する方向帯
#   `LONG_ONLY / SHORT_ONLY / NO_TRADE` を決める。
#
# 互換方針:
#   既存の `env_*` 出力は維持しつつ、Phase 1 で追加した
#   `regime_*` 出力も並行して返す。
# --------------------------------------------------

from Framework.Utility.Utility import Clamp01 as _clamp_01, ToFloat as _to_float


def _rc(_suffix: str) -> str:
    return f"H2_ENVIRONMENT_{_suffix}"


# H2 固有のスコアは 0 - 100 に丸めて扱う。
def _clamp_score(_value):
    try:
        _value = int(round(float(_value)))
    except Exception:
        _value = 0

    if _value < 0:
        return 0
    if _value > 100:
        return 100
    return _value


# 数値方向をログ用ラベルへ変換する。
def _direction_to_label(_direction_value):
    if _direction_value > 0:
        return "BULLISH"
    if _direction_value < 0:
        return "BEARISH"
    return "NEUTRAL"


# 新旧の閾値キーをここで吸収する。
# Phase 1 の移行期間中は legacy key も読み続ける。
def _resolve_h2_thresholds(_thresholds):
    _legacy_adx_min = _to_float(_thresholds.get("adx_min"), 20.0)
    _legacy_trend_strength_min = _clamp_01(
        _to_float(_thresholds.get("trend_strength_min"), 0.55)
    )

    _regime_adx_min = _to_float(
        _thresholds.get("h2_regime_adx_min"),
        _legacy_adx_min,
    )
    _regime_strength_min = _clamp_01(
        _to_float(
            _thresholds.get("h2_regime_strength_min"),
            _legacy_trend_strength_min,
        )
    )
    _regime_score_min = _clamp_score(
        _thresholds.get("h2_regime_score_min", _regime_strength_min * 100.0)
    )

    return {
        "adx_min": float(_regime_adx_min),
        "trend_strength_min": float(_regime_strength_min),
        "regime_score_min": int(_regime_score_min),
        "legacy_adx_min": float(_legacy_adx_min),
        "legacy_trend_strength_min": float(_legacy_trend_strength_min),
    }


# 入力指標の生値はそのまま raw_features に残しておく。
def _build_raw_features(_indicators):
    return {
        "ma_short": _indicators.get("ma_short"),
        "ma_long": _indicators.get("ma_long"),
        "ma_slope": _indicators.get("ma_slope"),
        "adx": _indicators.get("adx"),
        "swing_structure": _indicators.get("swing_structure"),
    }


# H2 判定に使う中間材料を 1 か所で整理する。
# ここでは最終方向は決めず、方向票と強弱だけを作る。
def _build_regime_components(_indicators, _resolved_thresholds):
    _ma_short = _to_float(_indicators.get("ma_short"))
    _ma_long = _to_float(_indicators.get("ma_long"))
    _ma_slope = _to_float(_indicators.get("ma_slope"))
    _adx = _to_float(_indicators.get("adx"))
    _swing_structure = _indicators.get("swing_structure")

    _base_reason_codes = []

    if _ma_short > _ma_long:
        _base_reason_codes.append(_rc("MA_BULLISH"))
        _ma_direction = 1
    elif _ma_short < _ma_long:
        _base_reason_codes.append(_rc("MA_BEARISH"))
        _ma_direction = -1
    else:
        _ma_direction = 0

    if _ma_slope > 0:
        _base_reason_codes.append(_rc("MA_SLOPE_UP"))
        _slope_direction = 1
    elif _ma_slope < 0:
        _base_reason_codes.append(_rc("MA_SLOPE_DOWN"))
        _slope_direction = -1
    else:
        _slope_direction = 0

    _adx_strong = bool(_adx >= float(_resolved_thresholds["adx_min"]))
    if _adx_strong:
        _base_reason_codes.append(_rc("ADX_STRONG"))
    else:
        _base_reason_codes.append(_rc("ADX_WEAK"))

    if _swing_structure == "HIGHER_HIGH":
        _base_reason_codes.append(_rc("STRUCTURE_HIGHER_HIGH"))
        _structure_direction = 1
    elif _swing_structure == "LOWER_LOW":
        _base_reason_codes.append(_rc("STRUCTURE_LOWER_LOW"))
        _structure_direction = -1
    else:
        _structure_direction = 0

    _bullish_votes = 0
    _bearish_votes = 0

    for _direction in [_ma_direction, _slope_direction, _structure_direction]:
        if _direction == 1:
            _bullish_votes += 1
        elif _direction == -1:
            _bearish_votes += 1

    _max_votes = max(_bullish_votes, _bearish_votes)
    _trend_strength = _clamp_01(_max_votes / 3.0)
    _regime_score = _clamp_score(_trend_strength * 100.0)

    return {
        "ma_short": float(_ma_short),
        "ma_long": float(_ma_long),
        "ma_slope": float(_ma_slope),
        "adx": float(_adx),
        "swing_structure": _swing_structure,
        "ma_direction": _direction_to_label(_ma_direction),
        "slope_direction": _direction_to_label(_slope_direction),
        "structure_direction": _direction_to_label(_structure_direction),
        "adx_state": "STRONG" if _adx_strong else "WEAK",
        "adx_strong": bool(_adx_strong),
        "bullish_votes": int(_bullish_votes),
        "bearish_votes": int(_bearish_votes),
        "max_votes": int(_max_votes),
        "trend_strength": float(_trend_strength),
        "regime_score": int(_regime_score),
        "threshold_snapshot": {
            "adx_min": float(_resolved_thresholds["adx_min"]),
            "trend_strength_min": float(_resolved_thresholds["trend_strength_min"]),
            "regime_score_min": int(_resolved_thresholds["regime_score_min"]),
            "legacy_adx_min": float(_resolved_thresholds["legacy_adx_min"]),
            "legacy_trend_strength_min": float(
                _resolved_thresholds["legacy_trend_strength_min"]
            ),
        },
        "base_reason_codes": list(_base_reason_codes),
    }


# 既存互換の最終方向判定。
# ADX と方向票の合意度を見て、許可帯を 3 値で返す。
def _judge_h2_direction(_regime_components, _resolved_thresholds):
    _trend_strength = float(_regime_components["trend_strength"])
    _reason_codes = list(_regime_components["base_reason_codes"])

    if not _regime_components["adx_strong"]:
        return "NO_TRADE", 0, _trend_strength, _reason_codes

    if _trend_strength < float(_resolved_thresholds["trend_strength_min"]):
        return "NO_TRADE", 0, _trend_strength, _reason_codes

    if (
        _regime_components["bullish_votes"] >= 2
        and _regime_components["bearish_votes"] == 0
    ):
        return "LONG_ONLY", 1, _trend_strength, _reason_codes

    if (
        _regime_components["bearish_votes"] >= 2
        and _regime_components["bullish_votes"] == 0
    ):
        return "SHORT_ONLY", -1, _trend_strength, _reason_codes

    return "NO_TRADE", 0, _trend_strength, _reason_codes


# downstream が「なぜ NO_TRADE なのか」を読み取りやすいよう、
# regime の質を補助的に返す。
def _classify_regime_quality(_env_direction, _regime_components, _resolved_thresholds):
    if not _regime_components["adx_strong"]:
        return "WEAK_ADX"

    if _regime_components["trend_strength"] < float(
        _resolved_thresholds["trend_strength_min"]
    ):
        return "LOW_CONSENSUS"

    if _env_direction in ["LONG_ONLY", "SHORT_ONLY"]:
        return "READY"

    return "MIXED_SIGNALS"


# 既存の reason_codes に regime 用の補足 reason を足す。
def _build_regime_reason_codes(_reason_codes, _env_direction, _regime_quality):
    _regime_reason_codes = list(_reason_codes)

    if _env_direction == "LONG_ONLY":
        _regime_reason_codes.append(_rc("REGIME_DIRECTION_LONG_ONLY"))
    elif _env_direction == "SHORT_ONLY":
        _regime_reason_codes.append(_rc("REGIME_DIRECTION_SHORT_ONLY"))
    else:
        _regime_reason_codes.append(_rc("REGIME_DIRECTION_NO_TRADE"))

    _regime_reason_codes.append(_rc(f"REGIME_QUALITY_{_regime_quality}"))
    return _regime_reason_codes


# 返却 dict の形は 1 か所で組み立てる。
def _build_result(
    _timestamp_jst,
    _status,
    _env_direction,
    _env_score,
    _trend_strength,
    _reason_codes,
    _summary,
    _raw_features,
    _regime_direction,
    _regime_score,
    _regime_quality,
    _regime_components,
    _regime_reason_codes,
):
    return {
        "module_name": "h2_environment",
        "timestamp_jst": _timestamp_jst,
        "status": _status,
        "env_direction": _env_direction,
        "env_score": _env_score,
        "trend_strength": _trend_strength,
        "reason_codes": list(_reason_codes),
        "summary": _summary,
        "raw_features": _raw_features,
        "regime_direction": _regime_direction,
        "regime_score": _regime_score,
        "regime_quality": _regime_quality,
        "regime_components": _regime_components,
        "regime_reason_codes": list(_regime_reason_codes),
    }


# H2 環境評価の入口。
# ここでは外部停止条件を最優先し、その後に H2 判定を行う。
def evaluate_h2_environment(
    market_data_h2: dict,
    external_filter_result: dict,
    thresholds: dict,
) -> dict:
    _timestamp_jst = market_data_h2.get("timestamp_jst", "")
    _indicators = market_data_h2.get("indicators", {})
    _resolved_thresholds = _resolve_h2_thresholds(thresholds)
    _raw_features = _build_raw_features(_indicators)
    _regime_components = _build_regime_components(_indicators, _resolved_thresholds)

    try:
        if external_filter_result.get("can_trade") is False:
            _reason_codes = [_rc("EXTERNAL_FILTER_ON")]
            _regime_reason_codes = list(_reason_codes) + [
                _rc("REGIME_DIRECTION_NO_TRADE"),
                _rc("REGIME_QUALITY_BLOCKED_BY_EXTERNAL_FILTER"),
            ]

            return _build_result(
                _timestamp_jst=_timestamp_jst,
                _status="OK",
                _env_direction="NO_TRADE",
                _env_score=0,
                _trend_strength=0.0,
                _reason_codes=_reason_codes,
                _summary="External filter is active, so H2 trading is disabled.",
                _raw_features=_raw_features,
                _regime_direction="NO_TRADE",
                _regime_score=0,
                _regime_quality="BLOCKED_BY_EXTERNAL_FILTER",
                _regime_components=_regime_components,
                _regime_reason_codes=_regime_reason_codes,
            )

        _env_direction, _env_score, _trend_strength, _reason_codes = _judge_h2_direction(
            _regime_components,
            _resolved_thresholds,
        )
        _regime_quality = _classify_regime_quality(
            _env_direction,
            _regime_components,
            _resolved_thresholds,
        )
        _regime_reason_codes = _build_regime_reason_codes(
            _reason_codes,
            _env_direction,
            _regime_quality,
        )

        if _env_direction == "LONG_ONLY":
            _summary = "H2 regime allows long-only trades."
        elif _env_direction == "SHORT_ONLY":
            _summary = "H2 regime allows short-only trades."
        else:
            _summary = "H2 regime does not allow directional trades."

        return _build_result(
            _timestamp_jst=_timestamp_jst,
            _status="OK",
            _env_direction=_env_direction,
            _env_score=_env_score,
            _trend_strength=_trend_strength,
            _reason_codes=_reason_codes,
            _summary=_summary,
            _raw_features=_raw_features,
            _regime_direction=_env_direction,
            _regime_score=int(_regime_components["regime_score"]),
            _regime_quality=_regime_quality,
            _regime_components=_regime_components,
            _regime_reason_codes=_regime_reason_codes,
        )

    except Exception as _error:
        _reason_codes = [_rc("ERROR")]
        _regime_reason_codes = list(_reason_codes) + [
            _rc("REGIME_DIRECTION_NO_TRADE"),
            _rc("REGIME_QUALITY_ERROR"),
        ]

        return _build_result(
            _timestamp_jst=_timestamp_jst,
            _status="ERROR",
            _env_direction="NO_TRADE",
            _env_score=0,
            _trend_strength=0.0,
            _reason_codes=_reason_codes,
            _summary=f"h2_environment error: {_error}",
            _raw_features=_raw_features,
            _regime_direction="NO_TRADE",
            _regime_score=0,
            _regime_quality="ERROR",
            _regime_components=_regime_components,
            _regime_reason_codes=_regime_reason_codes,
        )
