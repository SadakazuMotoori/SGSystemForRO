import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import MetaTrader5 as mt5

_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Backtest.Scripts.run_backtest import (
    build_history_cache_file_map,
    has_history_cache,
    load_mt5_history,
    save_history_cache,
)
from Framework.MTSystem.MTManager import MTManager_Initialize
from Framework.Utility.Utility import FormatJSTDateTime, ParseJSTDateTime


DEFAULT_SYMBOL = "USDJPY"
DEFAULT_START = "2025-10-01 00:00:00"
DEFAULT_END = "2026-03-07 23:59:59"
DEFAULT_HISTORY_CACHE_DIR = "Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307"


def get_debug_args():
    return SimpleNamespace(
        symbol=DEFAULT_SYMBOL,
        start=DEFAULT_START,
        end=DEFAULT_END,
        history_cache_dir=DEFAULT_HISTORY_CACHE_DIR,
        force_refresh=True,
    )


def parse_args():
    if len(sys.argv) == 1:
        return get_debug_args()

    _parser = argparse.ArgumentParser(
        description="Fetch MT5 history and save the H1 Phase 2 backtest history cache.",
    )
    _parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    _parser.add_argument("--start", default=DEFAULT_START, help="Start timestamp in JST")
    _parser.add_argument("--end", default=DEFAULT_END, help="End timestamp in JST")
    _parser.add_argument(
        "--history-cache-dir",
        default=DEFAULT_HISTORY_CACHE_DIR,
        help="Directory where M15/H1/H2 cache CSV files will be saved",
    )
    _parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Overwrite an existing cache directory with freshly fetched MT5 data",
    )
    return _parser.parse_args()


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

    _cache_dir = Path(_args.history_cache_dir).resolve()
    _cache_paths = build_history_cache_file_map(_cache_dir)

    print("========== Build H1 Phase2 History Cache Start ==========")
    print(f"[INFO] symbol={_args.symbol}")
    print(f"[INFO] start={FormatJSTDateTime(_start_jst)}")
    print(f"[INFO] end={FormatJSTDateTime(_end_jst)}")
    print(f"[INFO] history_cache_dir={_cache_dir}")

    if has_history_cache(_cache_dir) and not _args.force_refresh:
        print("[INFO] history cache already exists. Reuse the existing files.")
        print(f"[INFO] metadata={_cache_paths['metadata']}")
        print("========== Build H1 Phase2 History Cache End ==========")
        return

    if not MTManager_Initialize():
        raise RuntimeError("MT5 initialization failed.")

    try:
        if not mt5.symbol_select(_args.symbol, True):
            raise RuntimeError(f"symbol_select failed: symbol={_args.symbol}, error={mt5.last_error()}")

        _history = load_mt5_history(_args.symbol, _start_jst, _end_jst)
        save_history_cache(
            _history_cache_dir=_cache_dir,
            _history=_history,
            _symbol=_args.symbol,
            _start_jst=_start_jst,
            _end_jst=_end_jst,
        )
    finally:
        mt5.shutdown()

    print(f"[INFO] metadata={_cache_paths['metadata']}")
    for _timeframe in ["M15", "H1", "H2"]:
        print(f"[INFO] saved_{_timeframe}={_cache_paths[_timeframe]}")
        print(f"[INFO] rows_{_timeframe}={len(_history[_timeframe])}")

    print("========== Build H1 Phase2 History Cache End ==========")


if __name__ == "__main__":
    main()
