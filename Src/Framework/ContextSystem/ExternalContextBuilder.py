from datetime import datetime, timedelta
from pathlib import Path

from Framework.Utility.Utility import (
    GetJSTNow,
    JST,
    LoadJsonSafe,
    ParseJSTDateTime,
    ToFloat as _to_float,
    UTC,
)

CONFIG_DIR = Path("Asset/Config")
EXTERNAL_EVENTS_PATH = CONFIG_DIR / "external_events.json"
MANUAL_RISK_FLAGS_PATH = CONFIG_DIR / "manual_risk_flags.json"


def _to_bool(_value):
    if isinstance(_value, bool):
        return _value

    if isinstance(_value, str):
        return _value.strip().lower() in ["true", "1", "yes", "on"]

    if isinstance(_value, (int, float)):
        return _value != 0

    return False


def _extract_rate_value(_row, _index, _field_name):
    if isinstance(_row, dict):
        return _row.get(_field_name)

    try:
        return _row[_field_name]
    except Exception:
        pass

    try:
        return _row[_index]
    except Exception:
        return None


def _extract_latest_bar_datetime_jst(_frame_data):
    _ohlc = _frame_data.get("ohlc", [])

    if _ohlc is None or len(_ohlc) == 0:
        return None

    _latest_row = _ohlc[-1]
    _unix_time = _extract_rate_value(_latest_row, 0, "time")

    try:
        return datetime.fromtimestamp(int(_unix_time), UTC).astimezone(JST)
    except Exception:
        return None


def _extract_bar_range(_row):
    _high = _to_float(_extract_rate_value(_row, 2, "high"))
    _low = _to_float(_extract_rate_value(_row, 3, "low"))
    return max(_high - _low, 0.0)


def _extract_target_currencies(_market_data):
    _symbol = _market_data.get("M15", {}).get("symbol", "")
    _symbol = "".join(_char for _char in str(_symbol).upper() if _char.isalpha())

    if len(_symbol) >= 6:
        return {_symbol[:3], _symbol[3:6]}

    return set()


def _load_manual_risk_flags(_path=MANUAL_RISK_FLAGS_PATH):
    _default = {
        "flags": {
            "high_impact_event_soon": False,
            "central_bank_speech": False,
            "geopolitical_alert": False,
            "data_feed_error": False,
            "abnormal_volatility": False,
        }
    }

    _loaded = LoadJsonSafe(_path, _default, _warn=True)
    _flags = _loaded.get("flags", {}) if isinstance(_loaded, dict) else {}

    return {
        "high_impact_event_soon": _to_bool(_flags.get("high_impact_event_soon")),
        "central_bank_speech": _to_bool(_flags.get("central_bank_speech")),
        "geopolitical_alert": _to_bool(_flags.get("geopolitical_alert")),
        "data_feed_error": _to_bool(_flags.get("data_feed_error")),
        "abnormal_volatility": _to_bool(_flags.get("abnormal_volatility")),
    }


def _load_external_events(_path=EXTERNAL_EVENTS_PATH):
    _loaded = LoadJsonSafe(_path, {"events": []}, _warn=True)

    if not isinstance(_loaded, dict):
        return []

    _events = _loaded.get("events", [])
    return _events if isinstance(_events, list) else []


def _is_target_currency_event(_event, _target_currencies):
    if len(_target_currencies) == 0:
        return False

    _currency = _event.get("currency")

    if isinstance(_currency, list):
        _currencies = {str(_item).upper() for _item in _currency}
    else:
        _currencies = {str(_currency).upper()}

    return len(_currencies.intersection(_target_currencies)) > 0


def _is_event_in_block_window(_event_time_jst, _now_jst, _before_minutes, _after_minutes):
    if _event_time_jst is None:
        return False

    _delta_minutes = (_event_time_jst - _now_jst).total_seconds() / 60.0
    return (-_after_minutes) <= _delta_minutes <= _before_minutes


def _has_high_impact_event_soon(_events, _target_currencies, _now_jst, _thresholds):
    _before_minutes = int(_to_float(_thresholds.get("high_impact_event_lookahead_minutes"), 60))
    _after_minutes = int(_to_float(_thresholds.get("high_impact_event_cooldown_minutes"), 30))

    for _event in _events:
        if not isinstance(_event, dict):
            continue

        if not _to_bool(_event.get("is_active", True)):
            continue

        if not _is_target_currency_event(_event, _target_currencies):
            continue

        _importance = str(_event.get("importance", "")).lower()
        if _importance not in ["high", "critical"]:
            continue

        _event_time_jst = ParseJSTDateTime(_event.get("event_time_jst"))
        if _is_event_in_block_window(_event_time_jst, _now_jst, _before_minutes, _after_minutes):
            return True

    return False


def _has_central_bank_speech_soon(_events, _target_currencies, _now_jst, _thresholds):
    _before_minutes = int(_to_float(_thresholds.get("central_bank_speech_lookahead_minutes"), 180))
    _after_minutes = int(_to_float(_thresholds.get("central_bank_speech_cooldown_minutes"), 60))
    _keywords = [
        "boj",
        "boj governor",
        "bank of japan",
        "fed",
        "fomc",
        "federal reserve",
        "powell",
        "ueda",
        "日銀",
        "植田",
        "パウエル",
    ]

    for _event in _events:
        if not isinstance(_event, dict):
            continue

        if not _to_bool(_event.get("is_active", True)):
            continue

        if not _is_target_currency_event(_event, _target_currencies):
            continue

        _category = str(_event.get("category", "")).lower()
        _title = str(_event.get("title", "")).lower()

        _is_speech_category = _category in [
            "central_bank_speech",
            "speech",
            "press_conference",
        ]

        _has_speech_keyword = any(_keyword in _title for _keyword in _keywords)

        if not (_is_speech_category or _has_speech_keyword):
            continue

        _event_time_jst = ParseJSTDateTime(_event.get("event_time_jst"))
        if _is_event_in_block_window(_event_time_jst, _now_jst, _before_minutes, _after_minutes):
            return True

    return False


def _has_data_feed_error(_market_data, _now_jst, _thresholds):
    _timeframe_settings = {
        "M15": int(_to_float(_thresholds.get("data_stale_minutes_m15"), 30)),
        "H1": int(_to_float(_thresholds.get("data_stale_minutes_h1"), 90)),
        "H2": int(_to_float(_thresholds.get("data_stale_minutes_h2"), 180)),
    }

    for _timeframe, _stale_minutes in _timeframe_settings.items():
        _frame_data = _market_data.get(_timeframe, {})
        _ohlc = _frame_data.get("ohlc", [])

        if _ohlc is None or len(_ohlc) == 0:
            return True

        _latest_bar_jst = _extract_latest_bar_datetime_jst(_frame_data)
        if _latest_bar_jst is None:
            return True

        if (_now_jst - _latest_bar_jst) > timedelta(minutes=_stale_minutes):
            return True

    _spread = _market_data.get("M15", {}).get("spread")

    try:
        float(_spread)
    except Exception:
        return True

    return False


def _has_abnormal_volatility(_market_data, _thresholds):
    _m15_data = _market_data.get("M15", {})
    _ohlc = _m15_data.get("ohlc", [])

    if _ohlc is None or len(_ohlc) < 21:
        return False

    _history_rows = _ohlc[-21:-1]
    _latest_row = _ohlc[-1]

    _historical_ranges = [
        _extract_bar_range(_row)
        for _row in _history_rows
        if _extract_bar_range(_row) > 0.0
    ]

    if len(_historical_ranges) == 0:
        return False

    _average_range = sum(_historical_ranges) / float(len(_historical_ranges))
    _latest_range = _extract_bar_range(_latest_row)

    _range_ratio_max = _to_float(_thresholds.get("abnormal_volatility_range_ratio_max"), 2.5)
    if _average_range > 0.0 and (_latest_range / _average_range) >= _range_ratio_max:
        return True

    _noise = _to_float(_m15_data.get("indicators", {}).get("noise"))
    _noise_max = _to_float(_thresholds.get("m15_noise_max"), 0.40)
    _noise_ratio_max = _to_float(_thresholds.get("abnormal_volatility_noise_ratio_max"), 1.5)

    if _noise > (_noise_max * _noise_ratio_max):
        return True

    return False


def BuildExternalContext(_marketData, _systemContext, _thresholds):
    _now_jst = ParseJSTDateTime(_systemContext.get("latest_update_jst")) or GetJSTNow()
    _manual_flags = _load_manual_risk_flags()
    _events = _load_external_events()
    _target_currencies = _extract_target_currencies(_marketData)

    _high_impact_event_soon = _has_high_impact_event_soon(
        _events=_events,
        _target_currencies=_target_currencies,
        _now_jst=_now_jst,
        _thresholds=_thresholds,
    )

    _central_bank_speech = _has_central_bank_speech_soon(
        _events=_events,
        _target_currencies=_target_currencies,
        _now_jst=_now_jst,
        _thresholds=_thresholds,
    )

    _data_feed_error = _has_data_feed_error(
        _market_data=_marketData,
        _now_jst=_now_jst,
        _thresholds=_thresholds,
    )

    _abnormal_volatility = _has_abnormal_volatility(
        _market_data=_marketData,
        _thresholds=_thresholds,
    )

    return {
        "high_impact_event_soon": _high_impact_event_soon or _manual_flags["high_impact_event_soon"],
        "central_bank_speech": _central_bank_speech or _manual_flags["central_bank_speech"],
        "geopolitical_alert": _manual_flags["geopolitical_alert"],
        "data_feed_error": _data_feed_error or _manual_flags["data_feed_error"],
        "abnormal_volatility": _abnormal_volatility or _manual_flags["abnormal_volatility"],
    }
