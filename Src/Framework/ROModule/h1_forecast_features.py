from Framework.Utility.Utility import Clamp01 as _clamp_01, ToFloat as _to_float


_RECENT_FEATURE_WINDOW = 6


def extract_h1_close_list(_ohlc):
    _close_list = []

    for _row in _ohlc:
        if isinstance(_row, dict):
            _close_list.append(_to_float(_row.get("close")))
        else:
            _close_list.append(_to_float(_row[4]))

    return _close_list


def build_h1_close_diff_list(_close_list):
    _diff_list = []

    for _index in range(1, len(_close_list)):
        _diff_list.append(float(_close_list[_index] - _close_list[_index - 1]))

    return _diff_list


def build_h1_recent_direction_features(_close_list, _window_size=_RECENT_FEATURE_WINDOW):
    if len(_close_list) == 0:
        return {
            "recent_close_list": [],
            "recent_diff_list": [],
            "recent_momentum": 0.0,
            "trend_consistency": 0.0,
        }

    _recent_close_list = _close_list[-min(len(_close_list), int(_window_size)) :]
    _recent_diff_list = build_h1_close_diff_list(_recent_close_list)

    if len(_recent_diff_list) == 0:
        return {
            "recent_close_list": list(_recent_close_list),
            "recent_diff_list": [],
            "recent_momentum": 0.0,
            "trend_consistency": 0.0,
        }

    _up_count = sum(1 for _value in _recent_diff_list if _value > 0.0)
    _down_count = sum(1 for _value in _recent_diff_list if _value < 0.0)
    _trend_consistency = _clamp_01(max(_up_count, _down_count) / float(len(_recent_diff_list)))
    _recent_momentum = float(sum(_recent_diff_list) / float(len(_recent_diff_list)))

    return {
        "recent_close_list": list(_recent_close_list),
        "recent_diff_list": list(_recent_diff_list),
        "recent_momentum": float(_recent_momentum),
        "trend_consistency": float(_trend_consistency),
    }
