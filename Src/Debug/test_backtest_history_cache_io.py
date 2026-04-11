import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Backtest.Scripts.run_backtest import (
    JST,
    build_history_cache_file_map,
    has_history_cache,
    load_history_cache,
    save_history_cache,
)


def _build_history():
    _m15_ts = pd.date_range("2026-01-01 00:00:00", periods=4, freq="15min", tz=JST)
    _h1_ts = pd.date_range("2025-12-31 20:00:00", periods=4, freq="1h", tz=JST)
    _h2_ts = pd.date_range("2025-12-31 12:00:00", periods=4, freq="2h", tz=JST)

    def _build_df(_timestamps, _base_price, _include_spread):
        _rows = []
        for _index, _timestamp in enumerate(_timestamps):
            _close = _base_price + (_index * 0.1)
            _row = {
                "timestamp": _timestamp,
                "time": int(_timestamp.tz_convert("UTC").timestamp()),
                "open": _close - 0.05,
                "high": _close + 0.08,
                "low": _close - 0.09,
                "close": _close,
            }
            if _include_spread:
                _row["spread"] = 0.012
            _rows.append(_row)
        return pd.DataFrame(_rows)

    return {
        "M15": _build_df(_m15_ts, 150.0, True),
        "H1": _build_df(_h1_ts, 149.5, False),
        "H2": _build_df(_h2_ts, 149.0, False),
    }


def test_history_cache_roundtrip():
    _history = _build_history()
    _start_jst = _history["M15"].iloc[0]["timestamp"].to_pydatetime()
    _end_jst = _history["M15"].iloc[-1]["timestamp"].to_pydatetime()

    with tempfile.TemporaryDirectory() as _tmp_dir:
        save_history_cache(
            _history_cache_dir=_tmp_dir,
            _history=_history,
            _symbol="USDJPY",
            _start_jst=_start_jst,
            _end_jst=_end_jst,
        )

        assert has_history_cache(_tmp_dir)

        _paths = build_history_cache_file_map(_tmp_dir)
        assert _paths["metadata"].exists()

        _loaded = load_history_cache(_tmp_dir)

        for _timeframe in ["M15", "H1", "H2"]:
            assert len(_loaded[_timeframe]) == len(_history[_timeframe])
            assert list(_loaded[_timeframe]["timestamp"]) == list(_history[_timeframe]["timestamp"])
            assert list(_loaded[_timeframe]["close"]) == list(_history[_timeframe]["close"])

        assert "spread" in _loaded["M15"].columns
        assert abs(float(_loaded["M15"].iloc[-1]["spread"]) - 0.012) < 1e-9


def main():
    try:
        test_history_cache_roundtrip()
        print("[PASS] test_backtest_history_cache_io")
    except AssertionError as _error:
        print(f"[FAIL] test_backtest_history_cache_io: {_error}")
        raise


if __name__ == "__main__":
    main()
