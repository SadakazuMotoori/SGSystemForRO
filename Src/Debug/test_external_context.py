# --------------------------------------------------
# test_external_context.py
# 役割:
#   BuildExternalContext() の実データ連携部分を
#   固定市場データで確認するためのデバッグ用スクリプト
#
# 確認内容:
#   1. 正常な新鮮データでは停止フラグが立たないか
#   2. 古いM15データで data_feed_error が立つか
#   3. 異常値幅・高ノイズで abnormal_volatility が立つか
# --------------------------------------------------

import json
import os
import sys
from datetime import timedelta, timezone


_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from main import BuildExternalContext, BuildSystemContext, GetJSTNow, LoadThresholds


def _to_unix_seconds(_dt_jst):
    return int(_dt_jst.astimezone(timezone.utc).timestamp())


def _build_ohlc_rows(_latest_bar_jst, _timeframe_minutes, _count, _base_price, _normal_range, _latest_range=None):
    _rows = []

    for _index in range(_count):
        _bar_time = _latest_bar_jst - timedelta(minutes=_timeframe_minutes * (_count - 1 - _index))
        _open = _base_price + (_index * 0.01)
        _close = _open + 0.01
        _range = _latest_range if (_latest_range is not None and _index == (_count - 1)) else _normal_range
        _high = max(_open, _close) + (_range / 2.0)
        _low = min(_open, _close) - (_range / 2.0)

        _rows.append({
            "time": _to_unix_seconds(_bar_time),
            "open": _open,
            "high": _high,
            "low": _low,
            "close": _close,
        })

    return _rows


def _build_market_data(_now_jst, _m15_latest_offset_minutes=5, _m15_noise=0.10, _m15_latest_range=None):
    _m15_latest_bar = _now_jst - timedelta(minutes=_m15_latest_offset_minutes)
    _h1_latest_bar = _now_jst - timedelta(minutes=30)
    _h2_latest_bar = _now_jst - timedelta(minutes=60)

    return {
        "H2": {
            "symbol": "USDJPY",
            "timeframe": "H2",
            "timestamp_jst": _now_jst.strftime("%Y-%m-%d %H:%M:%S"),
            "ohlc": _build_ohlc_rows(_h2_latest_bar, 120, 30, 150.0, 0.30),
            "indicators": {},
        },
        "H1": {
            "symbol": "USDJPY",
            "timeframe": "H1",
            "timestamp_jst": _now_jst.strftime("%Y-%m-%d %H:%M:%S"),
            "ohlc": _build_ohlc_rows(_h1_latest_bar, 60, 30, 150.0, 0.20),
            "indicators": {},
        },
        "M15": {
            "symbol": "USDJPY",
            "timeframe": "M15",
            "timestamp_jst": _now_jst.strftime("%Y-%m-%d %H:%M:%S"),
            "ohlc": _build_ohlc_rows(_m15_latest_bar, 15, 30, 150.0, 0.08, _m15_latest_range),
            "indicators": {
                "noise": _m15_noise,
            },
            "spread": 0.01,
        },
    }


def _build_system_context(_now_jst):
    _system_context = BuildSystemContext()
    _system_context["latest_update_jst"] = _now_jst.strftime("%Y-%m-%d %H:%M:%S")
    return _system_context


def _run_case(_case_name, _market_data, _thresholds, _system_context):
    _context = BuildExternalContext(
        _marketData=_market_data,
        _systemContext=_system_context,
        _thresholds=_thresholds,
    )

    print("")
    print("==================================================")
    print(f"CASE: {_case_name}")
    print("==================================================")
    print(json.dumps(_context, ensure_ascii=False, indent=2))

    return _context


def main():
    print("========== test_external_context start ==========")

    _thresholds = LoadThresholds()
    _now_jst = GetJSTNow()
    _system_context = _build_system_context(_now_jst)

    _normal_context = _run_case(
        "normal",
        _build_market_data(_now_jst),
        _thresholds,
        _system_context,
    )

    _stale_context = _run_case(
        "stale_m15",
        _build_market_data(_now_jst, _m15_latest_offset_minutes=120),
        _thresholds,
        _system_context,
    )

    _volatility_context = _run_case(
        "abnormal_volatility",
        _build_market_data(_now_jst, _m15_noise=0.80, _m15_latest_range=0.40),
        _thresholds,
        _system_context,
    )

    if any(_normal_context.values()):
        print("[FAIL] normal case で停止フラグが立っています。")
        return

    if _stale_context.get("data_feed_error") is not True:
        print("[FAIL] stale_m15 case で data_feed_error が立っていません。")
        return

    if _volatility_context.get("abnormal_volatility") is not True:
        print("[FAIL] abnormal_volatility case で abnormal_volatility が立っていません。")
        return

    print("[OK] BuildExternalContext 実データ連携の基本確認成功")
    print("========== test_external_context end ==========")


if __name__ == "__main__":
    main()
