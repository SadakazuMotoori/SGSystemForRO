import argparse
import json
import struct
from datetime import timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.MTSystem.MTManager import MTManager_Initialize
from Framework.SGFramework import RunDecisionPipeline
from Framework.Utility.Utility import (
    EnsureParentDirectory,
    FormatJSTDateTime,
    JST,
    LoadJson,
    ParseJSTDateTime,
    UTC,
)


# --------------------------------------------------
# CLI蠑墓焚繧定ｪｭ縺ｿ霎ｼ繧
# 蠖ｹ蜑ｲ:
#   繝舌ャ繧ｯ繝・せ繝亥ｮ溯｡後↓蠢・ｦ√↑蟇ｾ雎｡騾夊ｲｨ繝ｻ譛滄俣繝ｻ蜃ｺ蜉帛・縺ｪ縺ｩ繧貞女縺大叙繧・
# --------------------------------------------------
def parse_args():
    if len(sys.argv) == 1:
        return SimpleNamespace(
            symbol="USDJPY",
            start="2025-10-01 00:00:00",
            end="2026-03-07 23:59:59",
            output="Src/Backtest/Output/raw_signals/raw_signals_mt5.csv",
            thresholds="Asset/Config/thresholds_backtest_loose_01.json",
            future_hours=2,
            history_cache_dir=None,
            prefer_history_cache=False,
            save_history_cache=False,
            verbose=True,
        )

    _parser = argparse.ArgumentParser(description="Run backtest for SGSystemForRO")
    _parser.add_argument("--symbol", required=True, help="Symbol, for example USDJPY")
    _parser.add_argument("--start", required=True, help="Start timestamp in JST")
    _parser.add_argument("--end", required=True, help="End timestamp in JST")
    _parser.add_argument("--output", required=True, help="Output CSV path for raw_signals")
    _parser.add_argument(
        "--thresholds",
        default="Asset/Config/thresholds_backtest_loose_01.json",
        help="Threshold JSON path",
    )
    _parser.add_argument(
        "--future-hours",
        type=int,
        default=2,
        help="Future hours used for evaluation",
    )
    _parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose progress logs",
    )
    _parser.add_argument(
        "--history-cache-dir",
        default=None,
        help="Directory containing cached M15/H1/H2 history CSV files",
    )
    _parser.add_argument(
        "--prefer-history-cache",
        action="store_true",
        help="Use cached history first when cache files are available",
    )
    _parser.add_argument(
        "--save-history-cache",
        action="store_true",
        help="Save MT5-fetched history into the cache directory",
    )

    return _parser.parse_args()


# --------------------------------------------------
# JST譌･譎ゅｒUTC縺ｸ螟画鋤縺吶ｋ
# 蠖ｹ蜑ｲ:
#   MT5 API縺ｸ貂｡縺呎律譎ゅｒUTC縺ｸ謠・∴繧・
# --------------------------------------------------
def _to_utc(_dt_jst):
    if _dt_jst is None:
        return None

    if _dt_jst.tzinfo is None:
        _dt_jst = _dt_jst.replace(tzinfo=JST)

    return _dt_jst.astimezone(UTC)


# --------------------------------------------------
# MT5縺九ｉ謖・ｮ壽悄髢薙・螻･豁ｴ繧貞叙蠕励☆繧・
# 蠖ｹ蜑ｲ:
#   繝舌ャ繧ｯ繝・せ繝育畑縺ｫ timeframe 縺斐→縺ｮ rates 驟榊・繧偵∪縺ｨ繧√※蜿門ｾ励☆繧・
# --------------------------------------------------
def fetch_mt5_rates_range(_symbol, _timeframe, _start_jst, _end_jst):
    _start_utc = _to_utc(_start_jst)
    _end_utc = _to_utc(_end_jst)

    _rates = mt5.copy_rates_range(_symbol, _timeframe, _start_utc, _end_utc)

    if _rates is None:
        raise RuntimeError(
            f"MT5 history fetch failed: symbol={_symbol}, timeframe={_timeframe}, error={mt5.last_error()}"
        )

    if len(_rates) == 0:
        raise RuntimeError(
            f"MT5 history returned no rows: symbol={_symbol}, timeframe={_timeframe}, start={_start_jst}, end={_end_jst}"
        )

    return _rates

# --------------------------------------------------
# 繧ｷ繝ｳ繝懊Ν縺ｮpoint蛟､繧貞叙蠕励☆繧・
# 蠖ｹ蜑ｲ:
#   MT5縺ｮspread(points)繧剃ｾ｡譬ｼ蟾ｮ縺ｸ螟画鋤縺吶ｋ縺溘ａ縺ｫ菴ｿ縺・
# --------------------------------------------------
def get_symbol_point(_symbol):
    _symbol_info = mt5.symbol_info(_symbol)

    if _symbol_info is None:
        raise RuntimeError(f"symbol_info fetch failed: symbol={_symbol}")

    try:
        return float(_symbol_info.point)
    except Exception:
        raise RuntimeError(f"symbol point fetch failed: symbol={_symbol}")

# --------------------------------------------------
# MT5 rates 驟榊・繧奪ataFrame縺ｸ螟画鋤縺吶ｋ
# 蠖ｹ蜑ｲ:
#   莉･蠕後・譎らｳｻ蛻励せ繝ｩ繧､繧ｹ縺ｨ譛ｪ譚･萓｡譬ｼ蜿ら・繧偵＠繧・☆縺上☆繧・
# --------------------------------------------------
def convert_rates_to_dataframe(_rates, _include_spread, _symbol_point=0.0):
    _df = pd.DataFrame(_rates)

    if "time" not in _df.columns:
        raise RuntimeError("MT5 rates do not contain a time column.")

    _df["timestamp"] = pd.to_datetime(_df["time"], unit="s", utc=True).dt.tz_convert(JST)
    _df = _df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    _columns = ["timestamp", "time", "open", "high", "low", "close"]
    if _include_spread:
        if "spread" not in _df.columns:
            _df["spread"] = 0.0

        # --------------------------------------------------
        # MT5縺ｮspread縺ｯ繝昴う繝ｳ繝亥､縺ｪ縺ｮ縺ｧ縲・
        # 譛ｬ逡ｪMTManager縺ｨ蜷梧ｧ倥↓萓｡譬ｼ蟾ｮ縺ｸ豁｣隕丞喧縺吶ｋ
        # --------------------------------------------------
        _df["spread"] = _df["spread"].astype(float) * float(_symbol_point)
        _columns.append("spread")

    return _df[_columns].copy()


_MT5_HC_SECTION_BASE_OFFSET = 428


def _build_mt5_hc_section_offset_map(_count):
    _offsets = {}
    _offset = _MT5_HC_SECTION_BASE_OFFSET

    for _section_name in ["times", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
        _offsets[_section_name] = _offset
        _item_size = 4 if _section_name == "spread" else 8
        _offset += 4 + (_count * _item_size)

    return _offsets


def _read_mt5_hc_section_count(_buffer, _offset):
    if (_offset + 4) > len(_buffer):
        raise RuntimeError(f"MT5 terminal cache section header is truncated at offset={_offset}.")

    return int(struct.unpack_from("<I", _buffer, _offset)[0])


def load_mt5_hc_dataframe(_hc_path, _include_spread, _symbol_point=0.0):
    _hc_path = Path(_hc_path).resolve()
    if not _hc_path.exists():
        raise RuntimeError(f"MT5 terminal cache file was not found: {_hc_path}")

    _buffer = _hc_path.read_bytes()
    _count = _read_mt5_hc_section_count(_buffer, _MT5_HC_SECTION_BASE_OFFSET)
    if _count <= 0:
        raise RuntimeError(f"MT5 terminal cache row count is invalid: path={_hc_path}, count={_count}")

    _offsets = _build_mt5_hc_section_offset_map(_count)

    for _section_name, _section_offset in _offsets.items():
        _section_count = _read_mt5_hc_section_count(_buffer, _section_offset)
        if _section_count != _count:
            raise RuntimeError(
                "MT5 terminal cache section count mismatch: "
                f"path={_hc_path}, section={_section_name}, expected={_count}, actual={_section_count}"
            )

    _time_array = np.frombuffer(
        _buffer,
        dtype="<i8",
        count=_count,
        offset=_offsets["times"] + 4,
    ).copy()
    _open_array = np.frombuffer(
        _buffer,
        dtype="<f8",
        count=_count,
        offset=_offsets["open"] + 4,
    ).copy()
    _high_array = np.frombuffer(
        _buffer,
        dtype="<f8",
        count=_count,
        offset=_offsets["high"] + 4,
    ).copy()
    _low_array = np.frombuffer(
        _buffer,
        dtype="<f8",
        count=_count,
        offset=_offsets["low"] + 4,
    ).copy()
    _close_array = np.frombuffer(
        _buffer,
        dtype="<f8",
        count=_count,
        offset=_offsets["close"] + 4,
    ).copy()

    _df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(_time_array, unit="s", utc=True).tz_convert(JST),
            "time": _time_array.astype("int64"),
            "open": _open_array.astype(float),
            "high": _high_array.astype(float),
            "low": _low_array.astype(float),
            "close": _close_array.astype(float),
        }
    )

    if _include_spread:
        _spread_array = np.frombuffer(
            _buffer,
            dtype="<i4",
            count=_count,
            offset=_offsets["spread"] + 4,
        ).copy()
        _df["spread"] = _spread_array.astype(float) * float(_symbol_point)

    _columns = ["timestamp", "time", "open", "high", "low", "close"]
    if _include_spread:
        _columns.append("spread")

    return _df[_columns].sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def build_mt5_terminal_history_cache_dir_candidates(_symbol):
    _candidates = []

    try:
        _terminal_info = mt5.terminal_info()
    except Exception:
        _terminal_info = None

    try:
        _account_info = mt5.account_info()
    except Exception:
        _account_info = None

    if _terminal_info is not None:
        _data_path = str(getattr(_terminal_info, "data_path", "") or "").strip()
        if _data_path != "":
            _data_path = Path(_data_path)

            _server_name = ""
            if _account_info is not None:
                _server_name = str(getattr(_account_info, "server", "") or "").strip()

            if _server_name != "":
                _candidates.append(_data_path / "bases" / _server_name / "history" / _symbol / "cache")

            _candidates.append(_data_path / "bases" / "Default" / "History" / _symbol / "cache")

    return _candidates


def resolve_mt5_terminal_history_cache_dir(_symbol):
    for _candidate_dir in build_mt5_terminal_history_cache_dir_candidates(_symbol):
        if all((_candidate_dir / f"{_timeframe}.hc").exists() for _timeframe in _HISTORY_TIMEFRAMES):
            return _candidate_dir.resolve()

    return None


def load_mt5_terminal_history_cache(_symbol, _start_jst, _end_jst):
    _cache_dir = resolve_mt5_terminal_history_cache_dir(_symbol)
    if _cache_dir is None:
        raise RuntimeError(f"MT5 terminal history cache directory was not found: symbol={_symbol}")

    _m15_fetch_start = _start_jst - timedelta(days=5)
    _h1_fetch_start = _start_jst - timedelta(days=10)
    _h2_fetch_start = _start_jst - timedelta(days=20)
    _symbol_point = get_symbol_point(_symbol)

    _history = {
        "M15": load_mt5_hc_dataframe(_cache_dir / "M15.hc", _include_spread=True, _symbol_point=_symbol_point),
        "H1": load_mt5_hc_dataframe(_cache_dir / "H1.hc", _include_spread=False, _symbol_point=_symbol_point),
        "H2": load_mt5_hc_dataframe(_cache_dir / "H2.hc", _include_spread=False, _symbol_point=_symbol_point),
    }

    _range_map = {
        "M15": (_m15_fetch_start, _end_jst),
        "H1": (_h1_fetch_start, _end_jst),
        "H2": (_h2_fetch_start, _end_jst),
    }

    for _timeframe, (_range_start, _range_end) in _range_map.items():
        _filtered = _history[_timeframe][
            (_history[_timeframe]["timestamp"] >= _range_start) &
            (_history[_timeframe]["timestamp"] <= _range_end)
        ].copy()

        if len(_filtered) == 0:
            raise RuntimeError(
                "MT5 terminal history cache does not contain the required range: "
                f"symbol={_symbol}, timeframe={_timeframe}, start={_range_start}, end={_range_end}, cache_dir={_cache_dir}"
            )

        _history[_timeframe] = _filtered.reset_index(drop=True)

    return _history, _cache_dir


# --------------------------------------------------
# MT5縺九ｉ繝舌ャ繧ｯ繝・せ繝亥ｯｾ雎｡螻･豁ｴ繧偵∪縺ｨ繧√※蜿門ｾ励☆繧・
# 蠖ｹ蜑ｲ:
#   M15/H1/H2繧剃ｸ諡ｬ縺ｧ繝ｭ繝ｼ繝峨＠縲∽ｻ･蠕後・繝ｫ繝ｼ繝怜・逅・∈貂｡縺・
# --------------------------------------------------
def load_mt5_history(_symbol, _start_jst, _end_jst):
    # --------------------------------------------------
    # 繧ｦ繧ｩ繝ｼ繝繧｢繝・・逕ｨ縺ｫ縲・幕蟋区律譎ゅｈ繧雁燕縺ｮ螻･豁ｴ繧ゆｽ吝・縺ｫ蜿門ｾ励☆繧・
    # H2縺ｧMA50繧剃ｽｿ縺・◆繧√∝ｮ牙・蟇・ｊ縺ｫ菴呵｣輔ｒ謖√◆縺帙ｋ
    # --------------------------------------------------
    _m15_fetch_start = _start_jst - timedelta(days=5)
    _h1_fetch_start = _start_jst - timedelta(days=10)
    _h2_fetch_start = _start_jst - timedelta(days=20)

    _symbol_point = get_symbol_point(_symbol)

    try:
        _m15_rates = fetch_mt5_rates_range(_symbol, mt5.TIMEFRAME_M15, _m15_fetch_start, _end_jst)
        _h1_rates = fetch_mt5_rates_range(_symbol, mt5.TIMEFRAME_H1, _h1_fetch_start, _end_jst)
        _h2_rates = fetch_mt5_rates_range(_symbol, mt5.TIMEFRAME_H2, _h2_fetch_start, _end_jst)

        return {
            "M15": convert_rates_to_dataframe(_m15_rates, _include_spread=True, _symbol_point=_symbol_point),
            "H1": convert_rates_to_dataframe(_h1_rates, _include_spread=False, _symbol_point=_symbol_point),
            "H2": convert_rates_to_dataframe(_h2_rates, _include_spread=False, _symbol_point=_symbol_point),
        }
    except Exception as _mt5_api_error:
        print(f"[WARN] MT5 API history fetch failed. Trying terminal cache fallback: {_mt5_api_error}")
        _history, _terminal_cache_dir = load_mt5_terminal_history_cache(_symbol, _start_jst, _end_jst)
        print(f"[INFO] terminal_history_cache_dir={_terminal_cache_dir}")
        return _history


_HISTORY_TIMEFRAMES = ("M15", "H1", "H2")


def build_history_cache_file_map(_history_cache_dir):
    _cache_dir = Path(_history_cache_dir).resolve()
    return {
        "dir": _cache_dir,
        "metadata": _cache_dir / "metadata.json",
        "M15": _cache_dir / "M15.csv",
        "H1": _cache_dir / "H1.csv",
        "H2": _cache_dir / "H2.csv",
    }


def has_history_cache(_history_cache_dir):
    if not _history_cache_dir:
        return False

    _paths = build_history_cache_file_map(_history_cache_dir)
    return all(_paths[_timeframe].exists() for _timeframe in _HISTORY_TIMEFRAMES)


def _normalize_cached_history_dataframe(_df, _include_spread):
    _required_columns = ["timestamp", "time", "open", "high", "low", "close"]
    for _column_name in _required_columns:
        if _column_name not in _df.columns:
            raise RuntimeError(f"history cache is missing required column: {_column_name}")

    _timestamp_series = pd.to_datetime(_df["timestamp"], errors="coerce")
    if _timestamp_series.isna().any():
        raise RuntimeError("history cache timestamp could not be parsed.")

    if _timestamp_series.dt.tz is None:
        _timestamp_series = _timestamp_series.dt.tz_localize(JST)
    else:
        _timestamp_series = _timestamp_series.dt.tz_convert(JST)

    _normalized = _df.copy()
    _normalized["timestamp"] = _timestamp_series
    _normalized["time"] = pd.to_numeric(_normalized["time"], errors="coerce").astype("Int64")

    for _column_name in ["open", "high", "low", "close"]:
        _normalized[_column_name] = pd.to_numeric(_normalized[_column_name], errors="coerce")

    if _normalized["time"].isna().any():
        raise RuntimeError("history cache time column contains invalid values.")

    if _normalized[["open", "high", "low", "close"]].isna().any().any():
        raise RuntimeError("history cache OHLC columns contain invalid values.")

    _columns = ["timestamp", "time", "open", "high", "low", "close"]
    if _include_spread:
        if "spread" not in _normalized.columns:
            _normalized["spread"] = 0.0
        _normalized["spread"] = pd.to_numeric(_normalized["spread"], errors="coerce").fillna(0.0)
        _columns.append("spread")

    _normalized = _normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return _normalized[_columns].copy()


def load_history_cache(_history_cache_dir):
    _paths = build_history_cache_file_map(_history_cache_dir)
    _history = {}

    for _timeframe in _HISTORY_TIMEFRAMES:
        _csv_path = _paths[_timeframe]
        if not _csv_path.exists():
            raise RuntimeError(f"history cache file is missing: {_csv_path}")

        _df = pd.read_csv(_csv_path)
        _history[_timeframe] = _normalize_cached_history_dataframe(
            _df=_df,
            _include_spread=(_timeframe == "M15"),
        )

    return _history


def save_history_cache(_history_cache_dir, _history, _symbol, _start_jst, _end_jst):
    _paths = build_history_cache_file_map(_history_cache_dir)
    _paths["dir"].mkdir(parents=True, exist_ok=True)

    for _timeframe in _HISTORY_TIMEFRAMES:
        _df = _history[_timeframe].copy()
        _df["timestamp"] = _df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        _df.to_csv(_paths[_timeframe], index=False, encoding="utf-8")

    _metadata = {
        "symbol": _symbol,
        "start_jst": FormatJSTDateTime(_start_jst),
        "end_jst": FormatJSTDateTime(_end_jst),
        "source": "MT5",
        "generated_at_jst": FormatJSTDateTime(pd.Timestamp.now(tz=JST).to_pydatetime()),
        "rows": {
            _timeframe: int(len(_history[_timeframe]))
            for _timeframe in _HISTORY_TIMEFRAMES
        },
        "files": {
            _timeframe: _paths[_timeframe].name
            for _timeframe in _HISTORY_TIMEFRAMES
        },
    }

    with _paths["metadata"].open("w", encoding="utf-8") as _file:
        json.dump(_metadata, _file, ensure_ascii=False, indent=2)


def resolve_history_source(_args, _start_jst, _end_jst):
    _cache_available = has_history_cache(_args.history_cache_dir)

    if _args.prefer_history_cache and _cache_available:
        print(f"[INFO] history_source=HISTORY_CACHE_PREFERRED dir={Path(_args.history_cache_dir).resolve()}")
        return load_history_cache(_args.history_cache_dir), "HISTORY_CACHE_PREFERRED"

    try:
        if not MTManager_Initialize():
            raise RuntimeError("MT5 initialization failed.")

        _history = load_mt5_history(_args.symbol, _start_jst, _end_jst)

        if _args.history_cache_dir and _args.save_history_cache:
            save_history_cache(
                _history_cache_dir=_args.history_cache_dir,
                _history=_history,
                _symbol=_args.symbol,
                _start_jst=_start_jst,
                _end_jst=_end_jst,
            )
            print(f"[INFO] history_cache_saved={Path(_args.history_cache_dir).resolve()}")

        return _history, "MT5"

    except Exception as _mt5_error:
        if _cache_available:
            print(f"[WARN] MT5 history load failed. Falling back to cache: {_mt5_error}")
            print(f"[INFO] history_source=HISTORY_CACHE_FALLBACK dir={Path(_args.history_cache_dir).resolve()}")
            return load_history_cache(_args.history_cache_dir), "HISTORY_CACHE_FALLBACK"

        raise


# --------------------------------------------------
# DataFrame縺九ｉ謖・ｮ壽凾轤ｹ縺ｾ縺ｧ縺ｮOHLC陦後ｒdict驟榊・縺ｧ蛻・ｊ蜃ｺ縺・
# 蠖ｹ蜑ｲ:
#   蜷・ｩ穂ｾ｡譎らせ縺ｧ譛ｪ譚･繝・・繧ｿ繧呈ｷｷ縺懊★縺ｫ繝｢繧ｸ繝･繝ｼ繝ｫ蜈･蜉帙ｒ菴懊ｋ
# --------------------------------------------------
def extract_ohlc_records_until(_df, _timestamp_jst, _include_spread):
    _subset = _df[_df["timestamp"] <= _timestamp_jst].copy()

    if len(_subset) == 0:
        return []

    _records = []
    for _, _row in _subset.iterrows():
        _record = {
            "time": int(_row["timestamp"].astimezone(UTC).timestamp()),
            "open": float(_row["open"]),
            "high": float(_row["high"]),
            "low": float(_row["low"]),
            "close": float(_row["close"]),
        }

        if _include_spread:
            _record["spread"] = float(_row.get("spread", 0.0))

        _records.append(_record)

    return _records


# --------------------------------------------------
# M15繝｢繝｡繝ｳ繧ｿ繝險育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 momentum 繧堤函謌舌☆繧・
# --------------------------------------------------
def calc_m15_momentum(_rows, _lookback=4):
    if _rows is None or len(_rows) <= _lookback:
        return 0.0

    _current_close = _rows[-1]["close"]
    _past_close = _rows[-1 - _lookback]["close"]

    return float(_current_close - _past_close)


# --------------------------------------------------
# M15繝励Ν繝舌ャ繧ｯ迥ｶ諷句愛螳・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 pullback_state 繧堤函謌舌☆繧・
# --------------------------------------------------
def judge_m15_pullback_state(_rows):
    if _rows is None or len(_rows) < 20:
        return "NONE"

    _ma = sum(_row["close"] for _row in _rows[-20:]) / 20.0
    _close = _rows[-1]["close"]

    if _close < _ma:
        return "PULLBACK_LONG"

    if _close > _ma:
        return "PULLBACK_SHORT"

    return "NONE"


# --------------------------------------------------
# M15繝悶Ξ繧､繧ｯ繧｢繧ｦ繝亥愛螳・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 breakout 繧堤函謌舌☆繧・
# --------------------------------------------------
def judge_m15_breakout(_rows):
    if _rows is None or len(_rows) < 6:
        return "NONE"

    _current_high = _rows[-1]["high"]
    _current_low = _rows[-1]["low"]
    _past_high = max(_row["high"] for _row in _rows[-6:-1])
    _past_low = min(_row["low"] for _row in _rows[-6:-1])

    if _current_high > _past_high:
        return "BREAKOUT_UP"

    if _current_low < _past_low:
        return "BREAKOUT_DOWN"

    return "NONE"


# --------------------------------------------------
# M15繝弱う繧ｺ險育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 noise 繧堤函謌舌☆繧・
# --------------------------------------------------
def calc_m15_noise(_rows):
    if _rows is None or len(_rows) < 1:
        return 0.0

    _open = _rows[-1]["open"]
    _high = _rows[-1]["high"]
    _low = _rows[-1]["low"]
    _close = _rows[-1]["close"]

    _body = abs(_close - _open)
    _range = _high - _low
    _wick = _range - _body

    if _range <= 0:
        return 0.0

    return float(_wick / _range)


# --------------------------------------------------
# M15蟶ょｴ繝・・繧ｿ繧呈ｧ狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   main.py縺九ｉ蜷СOModule縺ｸ貂｡縺励※縺・ｋ蠖｢蠑上↓謠・∴繧・
# --------------------------------------------------
def build_m15_market_data(_rows, _timestamp_jst, _symbol):
    if len(_rows) == 0:
        return {
            "symbol": _symbol,
            "timeframe": "M15",
            "timestamp_jst": FormatJSTDateTime(_timestamp_jst),
            "ohlc": [],
            "indicators": {
                "momentum": 0.0,
                "pullback_state": "NONE",
                "breakout": "NONE",
                "noise": 0.0,
            },
            "spread": 0.0,
        }

    return {
        "symbol": _symbol,
        "timeframe": "M15",
        "timestamp_jst": FormatJSTDateTime(_timestamp_jst),
        "ohlc": _rows,
        "indicators": {
            "momentum": float(calc_m15_momentum(_rows, 4)),
            "pullback_state": judge_m15_pullback_state(_rows),
            "breakout": judge_m15_breakout(_rows),
            "noise": float(calc_m15_noise(_rows)),
        },
        "spread": float(_rows[-1].get("spread", 0.0)),
    }


# --------------------------------------------------
# H1邨ょ､繝ｪ繧ｹ繝域歓蜃ｺ
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 raw_features 繧剃ｽ懊ｋ
# --------------------------------------------------
def extract_h1_close_list(_rows):
    if _rows is None or len(_rows) == 0:
        return []

    return [float(_row["close"]) for _row in _rows]


# --------------------------------------------------
# H1邨ょ､蟾ｮ蛻・Μ繧ｹ繝育函謌・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 close_diff_list 繧剃ｽ懊ｋ
# --------------------------------------------------
def build_h1_close_diff_list(_close_list):
    if _close_list is None or len(_close_list) < 2:
        return []

    _diff_list = []
    for _index in range(1, len(_close_list)):
        _diff_list.append(float(_close_list[_index] - _close_list[_index - 1]))

    return _diff_list


# --------------------------------------------------
# H1邁｡譏薙Δ繝｡繝ｳ繧ｿ繝險育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 recent_momentum 繧剃ｽ懊ｋ
# --------------------------------------------------
def calc_h1_momentum(_close_list, _lookback=5):
    if _close_list is None or len(_close_list) <= _lookback:
        return 0.0

    _current_close = _close_list[-1]
    _past_close = _close_list[-1 - _lookback]

    return float(_current_close - _past_close)


# --------------------------------------------------
# H1繝医Ξ繝ｳ繝我ｸ雋ｫ諤ｧ險育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 trend_consistency 繧剃ｽ懊ｋ
# --------------------------------------------------
def calc_h1_trend_consistency(_diff_list):
    if _diff_list is None or len(_diff_list) == 0:
        return 0.0

    _up_count = sum(1 for _diff in _diff_list if _diff > 0.0)
    _down_count = sum(1 for _diff in _diff_list if _diff < 0.0)
    _dominant_count = max(_up_count, _down_count)

    return float(_dominant_count / len(_diff_list))


# --------------------------------------------------
# H1蟶ょｴ繝・・繧ｿ繧呈ｧ狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   main.py縺九ｉ蜷СOModule縺ｸ貂｡縺励※縺・ｋ蠖｢蠑上↓謠・∴繧・
# --------------------------------------------------
def build_h1_market_data(_rows, _timestamp_jst, _symbol):
    _close_list = extract_h1_close_list(_rows)
    _close_diff_list = build_h1_close_diff_list(_close_list)

    return {
        "symbol": _symbol,
        "timeframe": "H1",
        "timestamp_jst": FormatJSTDateTime(_timestamp_jst),
        "ohlc": _rows,
        "indicators": {
            "raw_features": {
                "close_list": _close_list,
                "close_diff_list": _close_diff_list,
                "recent_momentum": float(calc_h1_momentum(_close_list, 5)),
                "trend_consistency": float(calc_h1_trend_consistency(_close_diff_list)),
            }
        },
    }


# --------------------------------------------------
# H2遏ｭ譛・髟ｷ譛櫪A險育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 MA 繧剃ｽ懊ｋ
# --------------------------------------------------
def calc_h2_ma(_rows, _period):
    if _rows is None or len(_rows) < _period:
        return 0.0

    _closes = [_row["close"] for _row in _rows[-_period:]]
    return sum(_closes) / float(_period)


# --------------------------------------------------
# H2 MA蛯ｾ縺崎ｨ育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 ma_slope 繧剃ｽ懊ｋ
# --------------------------------------------------
def calc_h2_slope(_rows, _period, _lookback=3):
    if _rows is None or len(_rows) < (_period + _lookback):
        return 0.0

    _current_closes = [_row["close"] for _row in _rows[-_period:]]
    _past_closes = [_row["close"] for _row in _rows[-(_period + _lookback):-_lookback]]

    _current_ma = sum(_current_closes) / float(_period)
    _past_ma = sum(_past_closes) / float(_period)

    return _current_ma - _past_ma


# --------------------------------------------------
# H2 ADX險育ｮ・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 adx 繧剃ｽ懊ｋ
# --------------------------------------------------
def calc_h2_adx(_rows, _period=14):
    if _rows is None or len(_rows) < (_period * 2):
        return 0.0

    _trs = []
    _plus_dms = []
    _minus_dms = []

    for _index in range(1, len(_rows)):
        _prev_high = _rows[_index - 1]["high"]
        _prev_low = _rows[_index - 1]["low"]
        _prev_close = _rows[_index - 1]["close"]

        _high = _rows[_index]["high"]
        _low = _rows[_index]["low"]

        _up_move = _high - _prev_high
        _down_move = _prev_low - _low

        _plus_dm = _up_move if (_up_move > _down_move and _up_move > 0) else 0.0
        _minus_dm = _down_move if (_down_move > _up_move and _down_move > 0) else 0.0

        _tr = max(
            _high - _low,
            abs(_high - _prev_close),
            abs(_low - _prev_close),
        )

        _trs.append(_tr)
        _plus_dms.append(_plus_dm)
        _minus_dms.append(_minus_dm)

    if len(_trs) < _period:
        return 0.0

    _dxs = []

    for _index in range(_period - 1, len(_trs)):
        _tr_sum = sum(_trs[_index - _period + 1:_index + 1])
        _plus_dm_sum = sum(_plus_dms[_index - _period + 1:_index + 1])
        _minus_dm_sum = sum(_minus_dms[_index - _period + 1:_index + 1])

        if _tr_sum == 0:
            _dxs.append(0.0)
            continue

        _plus_di = (_plus_dm_sum / _tr_sum) * 100.0
        _minus_di = (_minus_dm_sum / _tr_sum) * 100.0

        _di_sum = _plus_di + _minus_di
        if _di_sum == 0:
            _dxs.append(0.0)
            continue

        _dx = (abs(_plus_di - _minus_di) / _di_sum) * 100.0
        _dxs.append(_dx)

    if len(_dxs) < _period:
        return 0.0

    return sum(_dxs[-_period:]) / float(_period)


# --------------------------------------------------
# H2繧ｹ繧､繝ｳ繧ｰ讒矩蛻､螳・
# 蠖ｹ蜑ｲ:
#   譌｢蟄弄TManager繝ｭ繧ｸ繝・け縺ｨ蜷後§蝓ｺ貅悶〒 swing_structure 繧剃ｽ懊ｋ
# --------------------------------------------------
def judge_h2_swing_structure(_rows):
    if _rows is None or len(_rows) < 4:
        return "RANGE"

    _recent_high = max(_row["high"] for _row in _rows[-2:])
    _past_high = max(_row["high"] for _row in _rows[-4:-2])

    _recent_low = min(_row["low"] for _row in _rows[-2:])
    _past_low = min(_row["low"] for _row in _rows[-4:-2])

    if _recent_high > _past_high and _recent_low >= _past_low:
        return "HIGHER_HIGH"

    if _recent_low < _past_low and _recent_high <= _past_high:
        return "LOWER_LOW"

    return "RANGE"


# --------------------------------------------------
# H2蟶ょｴ繝・・繧ｿ繧呈ｧ狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   main.py縺九ｉ蜷СOModule縺ｸ貂｡縺励※縺・ｋ蠖｢蠑上↓謠・∴繧・
# --------------------------------------------------
def build_h2_market_data(_rows, _timestamp_jst, _symbol):
    return {
        "symbol": _symbol,
        "timeframe": "H2",
        "timestamp_jst": FormatJSTDateTime(_timestamp_jst),
        "ohlc": _rows,
        "indicators": {
            "ma_short": float(calc_h2_ma(_rows, 20)),
            "ma_long": float(calc_h2_ma(_rows, 50)),
            "ma_slope": float(calc_h2_slope(_rows, 20, 3)),
            "adx": float(calc_h2_adx(_rows, 14)),
            "swing_structure": judge_h2_swing_structure(_rows),
        },
    }


# --------------------------------------------------
# 隧穂ｾ｡譎らせ縺ｮ蟶ょｴ繝・・繧ｿ繧呈ｧ狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   M15/H1/H2繧呈悽逡ｪ縺ｨ蜷後§繧ｭ繝ｼ讒矩縺ｸ縺ｾ縺ｨ繧√ｋ
# --------------------------------------------------
def build_market_data_at_time(_history, _timestamp_jst, _symbol):
    _m15_rows = extract_ohlc_records_until(_history["M15"], _timestamp_jst, _include_spread=True)
    _h1_rows = extract_ohlc_records_until(_history["H1"], _timestamp_jst, _include_spread=False)
    _h2_rows = extract_ohlc_records_until(_history["H2"], _timestamp_jst, _include_spread=False)

    return {
        "H2": build_h2_market_data(_h2_rows, _timestamp_jst, _symbol),
        "H1": build_h1_market_data(_h1_rows, _timestamp_jst, _symbol),
        "M15": build_m15_market_data(_m15_rows, _timestamp_jst, _symbol),
    }


# --------------------------------------------------
# system_context 繧呈ｧ狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   main.py縺ｨ蜷後§譛蟆乗ｧ矩繧偵ヰ繝・け繝・せ繝育畑縺ｫ菴懊ｋ
# --------------------------------------------------
def build_system_context_at_time(_timestamp_jst):
    _timestamp_str = FormatJSTDateTime(_timestamp_jst)

    return {
        "round_id": _timestamp_str.replace("-", "").replace(" ", "").replace(":", ""),
        "latest_update_jst": _timestamp_str,
        "last_decision": None,
        "last_entry_result": None,
        "position_state": "FLAT",
    }


# --------------------------------------------------
# external_context 繧呈ｧ狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   蛻晉沿縺ｧ縺ｯ螟夜Κ譁・ц繧貞・縺ｦFalse蝗ｺ螳壹↓縺吶ｋ
# --------------------------------------------------
def build_external_context_off():
    return {
        "high_impact_event_soon": False,
        "central_bank_speech": False,
        "geopolitical_alert": False,
        "data_feed_error": False,
        "abnormal_volatility": False,
    }


# --------------------------------------------------
# 譛ｪ譚･萓｡譬ｼ繧定ｧ｣豎ｺ縺吶ｋ
# 蠖ｹ蜑ｲ:
#   隧穂ｾ｡譎らせ縺九ｉ謖・ｮ壽凾髢灘ｾ後・M15遒ｺ螳夊ｶｳclose繧貞叙蠕励☆繧・
# --------------------------------------------------
def resolve_future_price(_m15_df, _timestamp_jst, _future_hours):
    _future_timestamp_jst = _timestamp_jst + timedelta(hours=_future_hours)
    _matched = _m15_df[_m15_df["timestamp"] == _future_timestamp_jst]

    if len(_matched) == 0:
        return None, _future_timestamp_jst

    return float(_matched.iloc[-1]["close"]), _future_timestamp_jst


# --------------------------------------------------
# final_action 繧剃ｺ域ｸｬ譁ｹ蜷代∈螟画鋤縺吶ｋ
# 蠖ｹ蜑ｲ:
#   ENTER_LONG/SHORT縺ｮ縺ｿ繧呈怏蜉ｹ繧ｷ繧ｰ繝翫Ν縺ｨ縺励※謇ｱ縺・
# --------------------------------------------------
def convert_final_action_to_predicted_direction(_final_action):
    if _final_action == "ENTER_LONG":
        return "UP"

    if _final_action == "ENTER_SHORT":
        return "DOWN"

    return "NO_SIGNAL"


# --------------------------------------------------
# entry_price 縺ｨ future_price 縺九ｉ螳滓婿蜷代ｒ菴懊ｋ
# 蠖ｹ蜑ｲ:
#   2譎る俣蠕後↓荳翫°荳九°縺縺代ｒ蛻､螳壹☆繧・
# --------------------------------------------------
def convert_actual_direction(_entry_price, _future_price):
    if _future_price > _entry_price:
        return "UP"

    if _future_price < _entry_price:
        return "DOWN"

    return "DRAW"


# --------------------------------------------------
# reason_codes 繧辰SV菫晏ｭ伜髄縺第枚蟄怜・縺ｸ螟画鋤縺吶ｋ
# 蠖ｹ蜑ｲ:
#   list[str] 繧・; 蛹ｺ蛻・ｊ縺ｧ菫晏ｭ倥〒縺阪ｋ繧医≧縺ｫ縺吶ｋ
# --------------------------------------------------
def join_reason_codes(_result):
    _codes = _result.get("reason_codes", [])

    if not isinstance(_codes, list):
        return ""

    return ";".join(str(_code) for _code in _codes)


# --------------------------------------------------
# 繝舌ャ繧ｯ繝・せ繝亥・蜉・陦後ｒ讒狗ｯ峨☆繧・
# 蠖ｹ蜑ｲ:
#   pipeline邨先棡縺ｨ譛ｪ譚･萓｡譬ｼ縺九ｉ raw_signals 逕ｨ繝ｬ繧ｳ繝ｼ繝峨ｒ菴懊ｋ
# --------------------------------------------------
def build_backtest_record(_timestamp_jst, _future_timestamp_jst, _entry_price, _future_price, _pipeline_result, _symbol):
    _external_filter_result = _pipeline_result["external_filter_result"]
    _h2_environment_result = _pipeline_result["h2_environment_result"]
    _h1_forecast_result = _pipeline_result["h1_forecast_result"]
    _m15_path_signal_result = _pipeline_result.get("m15_path_signal_result", {})
    _m15_entry_result = _pipeline_result["m15_entry_result"]
    _base_final_decision_result = _pipeline_result.get("base_final_decision_result", {})
    _final_decision_result = _pipeline_result["final_decision_result"]

    _final_action = _final_decision_result.get("final_action")
    _base_final_action = _base_final_decision_result.get("final_action")
    _predicted_direction = convert_final_action_to_predicted_direction(_final_action)
    _actual_direction = convert_actual_direction(_entry_price, _future_price)

    _is_draw = _actual_direction == "DRAW"
    _is_correct = (
        (_predicted_direction in ["UP", "DOWN"]) and
        (not _is_draw) and
        (_predicted_direction == _actual_direction)
    )

    _m15_raw = _m15_entry_result.get("raw_features", {})

    return {
        "timestamp_jst": FormatJSTDateTime(_timestamp_jst),
        "symbol": _symbol,
        "entry_price": float(_entry_price),
        "future_timestamp_jst": FormatJSTDateTime(_future_timestamp_jst),
        "future_price_2h": float(_future_price),
        "actual_direction": _actual_direction,
        "predicted_direction": _predicted_direction,
        "is_correct": bool(_is_correct),
        "is_draw": bool(_is_draw),
        "final_action": _final_action,
        "base_final_action": _base_final_action,
        "approved": bool(_final_decision_result.get("approved", False)),
        "decision_score": int(_final_decision_result.get("decision_score", 0)),
        "external_filter_status": _external_filter_result.get("filter_status"),
        "external_can_trade": _external_filter_result.get("can_trade"),
        "env_direction": _h2_environment_result.get("env_direction"),
        "env_score": _h2_environment_result.get("env_score"),
        "trend_strength": _h2_environment_result.get("trend_strength"),
        "h1_forecast_status": _h1_forecast_result.get("forecast_status"),
        "h1_net_direction": _h1_forecast_result.get("net_direction"),
        "h1_confidence": _h1_forecast_result.get("confidence"),
        "entry_action": _m15_entry_result.get("entry_action"),
        "entry_side": _m15_entry_result.get("entry_side"),
        "entry_score": _m15_entry_result.get("entry_score"),
        "timing_quality": _m15_entry_result.get("timing_quality"),
        "risk_flag": _m15_entry_result.get("risk_flag"),
        "m15_path_signal_ready": _m15_path_signal_result.get("signal_ready"),
        "m15_path_signal_side": _m15_path_signal_result.get("signal_side"),
        "m15_path_gap_threshold_passed": _m15_path_signal_result.get("gap_threshold_passed"),
        "m15_path_directional_gap_pips": _m15_path_signal_result.get("directional_gap_pips"),
        "m15_path_required_gap_pips": _m15_path_signal_result.get("required_gap_pips"),
        "m15_momentum": _m15_raw.get("momentum"),
        "m15_pullback_state": _m15_raw.get("pullback_state"),
        "m15_breakout": _m15_raw.get("breakout"),
        "m15_noise": _m15_raw.get("noise"),
        "spread": _m15_raw.get("spread"),
        "h1_alignment": _m15_raw.get("h1_alignment"),
        "external_reason_codes": join_reason_codes(_external_filter_result),
        "h2_reason_codes": join_reason_codes(_h2_environment_result),
        "h1_reason_codes": join_reason_codes(_h1_forecast_result),
        "m15_path_reason_codes": join_reason_codes(_m15_path_signal_result),
        "m15_reason_codes": join_reason_codes(_m15_entry_result),
        "final_reason_codes": join_reason_codes(_final_decision_result),
    }


# --------------------------------------------------
# 逕溘Ξ繧ｳ繝ｼ繝峨ｒCSV菫晏ｭ倥☆繧・
# 蠖ｹ蜑ｲ:
#   evaluate_signals.py縺ｸ貂｡縺吝次譚先侭繧貞・蜉帙☆繧・
# --------------------------------------------------
def save_raw_records(_output_path, _records):
    EnsureParentDirectory(_output_path)
    pd.DataFrame(_records).to_csv(_output_path, index=False, encoding="utf-8")


# --------------------------------------------------
# 繝舌ャ繧ｯ繝・せ繝亥ｯｾ雎｡譎らせ繧呈歓蜃ｺ縺吶ｋ
# 蠖ｹ蜑ｲ:
#   讀懆ｨｼ蟇ｾ雎｡譛滄俣縺ｮM15遒ｺ螳夊ｶｳ縺縺代ｒ隧穂ｾ｡蟇ｾ雎｡縺ｫ縺吶ｋ
# --------------------------------------------------
def build_target_timestamps(_m15_df, _start_jst, _end_jst):
    _target = _m15_df[
        (_m15_df["timestamp"] >= _start_jst) &
        (_m15_df["timestamp"] <= _end_jst)
    ].copy()

    return list(_target["timestamp"])


# --------------------------------------------------
# 隧穂ｾ｡譎らせ縺ｨ縺励※蜊∝・縺ｪ繧ｦ繧ｩ繝ｼ繝繧｢繝・・譛ｬ謨ｰ縺後≠繧九°遒ｺ隱阪☆繧・
# 蠖ｹ蜑ｲ:
#   謖・ｨ呵ｨ育ｮ励ｄ蛻､螳壼ｿ・ｦ∵悽謨ｰ縺御ｸ崎ｶｳ縺吶ｋ譎らせ繧帝勁螟悶☆繧・
# --------------------------------------------------
def has_sufficient_history(_history, _timestamp_jst):
    _m15_count = len(_history["M15"][_history["M15"]["timestamp"] <= _timestamp_jst])
    _h1_count = len(_history["H1"][_history["H1"]["timestamp"] <= _timestamp_jst])
    _h2_count = len(_history["H2"][_history["H2"]["timestamp"] <= _timestamp_jst])

    if _m15_count < 50:
        return False

    if _h1_count < 10:
        return False

    if _h2_count < 60:
        return False

    return True


# --------------------------------------------------
# 繝｡繧､繝ｳ蜃ｦ逅・
# 蠖ｹ蜑ｲ:
#   MT5螻･豁ｴ繧貞叙蠕励＠縲∵凾邉ｻ蛻励〒pipeline繧貞屓縺励※raw_signals繧剃ｿ晏ｭ倥☆繧・
# --------------------------------------------------
def main():
    _args = parse_args()

    _start_jst = ParseJSTDateTime(_args.start)
    _end_jst = ParseJSTDateTime(_args.end)

    if _start_jst is None:
        raise RuntimeError(f"failed to parse start timestamp: {_args.start}")

    if _end_jst is None:
        raise RuntimeError(f"failed to parse end timestamp: {_args.end}")

    if _start_jst > _end_jst:
        raise RuntimeError("start must be earlier than or equal to end.")

    print("========== Backtest Start ==========")
    print(f"[INFO] cwd={Path.cwd()}")
    _thresholds = LoadJson(_args.thresholds)
    _history, _history_source = resolve_history_source(_args, _start_jst, _end_jst)
    _target_timestamps = build_target_timestamps(_history["M15"], _start_jst, _end_jst)

    print(f"[INFO] symbol={_args.symbol}")
    print(f"[INFO] start={FormatJSTDateTime(_start_jst)}")
    print(f"[INFO] end={FormatJSTDateTime(_end_jst)}")
    print(f"[INFO] history_source={_history_source}")
    print(f"[INFO] target_count={len(_target_timestamps)}")

    _records = []
    _skip_count = 0
    _error_count = 0

    for _index, _timestamp_jst in enumerate(_target_timestamps, start=1):
        try:
            if not has_sufficient_history(_history, _timestamp_jst):
                _skip_count += 1
                continue

            _future_price, _future_timestamp_jst = resolve_future_price(
                _history["M15"],
                _timestamp_jst,
                _args.future_hours,
            )

            if _future_price is None:
                _skip_count += 1
                continue

            _market_data = build_market_data_at_time(
                _history=_history,
                _timestamp_jst=_timestamp_jst,
                _symbol=_args.symbol,
            )
            _system_context = build_system_context_at_time(_timestamp_jst)
            _external_context = build_external_context_off()

            _pipeline_result = RunDecisionPipeline(
                _market_data=_market_data,
                _external_context=_external_context,
                _system_context=_system_context,
                _thresholds=_thresholds,
            )

            _m15_ohlc = _market_data["M15"]["ohlc"]
            if len(_m15_ohlc) == 0:
                _skip_count += 1
                continue

            _entry_price = float(_m15_ohlc[-1]["close"])

            _record = build_backtest_record(
                _timestamp_jst=_timestamp_jst,
                _future_timestamp_jst=_future_timestamp_jst,
                _entry_price=_entry_price,
                _future_price=_future_price,
                _pipeline_result=_pipeline_result,
                _symbol=_args.symbol,
            )

            _records.append(_record)

            if _args.verbose and (_index % 100 == 0):
                print(f"[INFO] progress={_index}/{len(_target_timestamps)} records={len(_records)} skips={_skip_count}")

        except Exception as _error:
            _error_count += 1
            if _args.verbose:
                print(f"[WARN] failed at {FormatJSTDateTime(_timestamp_jst)}: {_error}")

    _output_path = Path(_args.output).resolve()

    print(f"[INFO] output_dir={_output_path.parent}")
    print(f"[INFO] output_file={_output_path}")

    save_raw_records(_args.output, _records)

    print(f"[INFO] saved={_args.output}")
    print(f"[INFO] record_count={len(_records)}")
    print(f"[INFO] skip_count={_skip_count}")
    print(f"[INFO] error_count={_error_count}")
    print("========== Backtest End ==========")


if __name__ == "__main__":
    main()


