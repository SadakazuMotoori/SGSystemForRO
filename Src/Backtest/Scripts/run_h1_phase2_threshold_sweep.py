import argparse
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import evaluate_signals_backtest as evaluator


THRESHOLD_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
DEFAULT_SYMBOL = "USDJPY"
DEFAULT_START = "2025-10-01 00:00:00"
DEFAULT_END = "2026-03-07 23:59:59"
DEFAULT_FUTURE_HOURS = 2
DEFAULT_THRESHOLDS_DIR = Path("Asset/Config/experiments/h1_phase2_threshold_sweep")
DEFAULT_OUTPUT_DIR = Path("Src/Backtest/Output/experiments/h1_phase2_threshold_sweep")
DEFAULT_COMPARISON_PATH = DEFAULT_OUTPUT_DIR / "phase2_threshold_comparison.md"
DEFAULT_HISTORY_CACHE_DIR = Path("Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307")
RUN_BACKTEST_SCRIPT = Path("Src/Backtest/Scripts/run_backtest.py")


def get_debug_args():
    return SimpleNamespace(
        symbol=DEFAULT_SYMBOL,
        start=DEFAULT_START,
        end=DEFAULT_END,
        future_hours=DEFAULT_FUTURE_HOURS,
        history_cache_dir=str(DEFAULT_HISTORY_CACHE_DIR),
        prefer_history_cache=True,
        save_history_cache=False,
        top_n_reason_codes=10,
        skip_backtest=False,
        resume=True,
        stop_on_error=True,
        python_exe=sys.executable,
    )


def parse_args():
    if len(sys.argv) == 1:
        return get_debug_args()

    _parser = argparse.ArgumentParser(
        description="Run the H1 Phase 2 threshold sweep and build aggregate outputs.",
    )
    _parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    _parser.add_argument("--start", default=DEFAULT_START)
    _parser.add_argument("--end", default=DEFAULT_END)
    _parser.add_argument("--future-hours", type=int, default=DEFAULT_FUTURE_HOURS)
    _parser.add_argument("--history-cache-dir", default=None)
    _parser.add_argument("--prefer-history-cache", action="store_true")
    _parser.add_argument("--save-history-cache", action="store_true")
    _parser.add_argument("--top-n-reason-codes", type=int, default=10)
    _parser.add_argument("--skip-backtest", action="store_true")
    _parser.add_argument("--resume", action="store_true")
    _parser.add_argument("--stop-on-error", action="store_true")
    _parser.add_argument("--python-exe", default=sys.executable)
    return _parser.parse_args()


def threshold_to_suffix(_threshold_value):
    return f"{int(round(float(_threshold_value) * 100)):03d}"


def build_run_paths(_threshold_value):
    _suffix = threshold_to_suffix(_threshold_value)
    _run_dir = DEFAULT_OUTPUT_DIR / f"h1conf_{_suffix}"
    return {
        "threshold": float(_threshold_value),
        "suffix": _suffix,
        "threshold_path": DEFAULT_THRESHOLDS_DIR / f"thresholds_h1conf_{_suffix}.json",
        "run_dir": _run_dir,
        "raw_signals_path": _run_dir / "raw_signals.csv",
        "eval_summary_path": _run_dir / "eval_summary.txt",
        "backtest_log_path": _run_dir / "backtest_log.txt",
    }


def ensure_parent_directory(_path):
    Path(_path).parent.mkdir(parents=True, exist_ok=True)


def has_reusable_run_artifacts(_paths):
    return Path(_paths["raw_signals_path"]).exists()


def run_backtest_for_threshold(_args, _paths):
    ensure_parent_directory(_paths["raw_signals_path"])

    _command = [
        _args.python_exe,
        str(RUN_BACKTEST_SCRIPT),
        "--symbol",
        _args.symbol,
        "--start",
        _args.start,
        "--end",
        _args.end,
        "--thresholds",
        str(_paths["threshold_path"]),
        "--output",
        str(_paths["raw_signals_path"]),
        "--future-hours",
        str(_args.future_hours),
        "--verbose",
    ]

    if _args.history_cache_dir:
        _command.extend(["--history-cache-dir", _args.history_cache_dir])

    if _args.prefer_history_cache:
        _command.append("--prefer-history-cache")

    if _args.save_history_cache:
        _command.append("--save-history-cache")

    _result = subprocess.run(
        _command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    _log_text = (_result.stdout or "") + (_result.stderr or "")
    _paths["backtest_log_path"].write_text(_log_text, encoding="utf-8")

    if _result.returncode != 0:
        raise RuntimeError(
            f"Backtest failed for threshold={_paths['threshold']:.2f}. "
            f"See {_paths['backtest_log_path']}"
        )


def build_eval_artifacts(_raw_signals_path, _top_n_reason_codes):
    _df, _input_path = evaluator.load_raw_signals(str(_raw_signals_path))

    _summary = evaluator.build_summary(_df)
    _evaluated_df = evaluator.build_evaluated_signals_df(_df)
    _direction_side_summary = evaluator.build_direction_side_summary(_evaluated_df)
    _decision_score_band_summary = evaluator.build_score_band_summary(_evaluated_df, "decision_score")
    _entry_score_band_summary = evaluator.build_score_band_summary(_evaluated_df, "entry_score")
    _decision_score_threshold_summary = evaluator.build_decision_score_threshold_summary(
        _evaluated_df,
        [60, 70, 80],
    )
    _candidate_strategy_summary = evaluator.build_candidate_strategy_summary(_evaluated_df)
    _main_flow_gate_summary = evaluator.build_main_flow_gate_summary(_df)

    _final_reason_counts = evaluator.build_reason_code_counts(_df, "final_reason_codes")
    _external_reason_counts = evaluator.build_reason_code_counts(_df, "external_reason_codes")
    _h2_reason_counts = evaluator.build_reason_code_counts(_df, "h2_reason_codes")
    _h1_reason_counts = evaluator.build_reason_code_counts(_df, "h1_reason_codes")
    _m15_path_reason_counts = evaluator.build_reason_code_counts(_df, "m15_path_reason_codes")
    _m15_reason_counts = evaluator.build_reason_code_counts(_df, "m15_reason_codes")

    with io.StringIO() as _buffer:
        with redirect_stdout(_buffer):
            print(f"[INFO] input_csv={_input_path}")
            print(f"[INFO] rows={len(_df)}")
            evaluator.print_summary(_summary)
            evaluator.print_dict_block("final_action_counts", _summary["final_action_counts"])
            evaluator.print_dict_block("base_final_action_counts", _summary["base_final_action_counts"])
            evaluator.print_dict_block("external_filter_status_counts", _summary["external_filter_status_counts"])
            evaluator.print_dict_block("env_direction_counts", _summary["env_direction_counts"])
            evaluator.print_dict_block("predicted_direction_counts", _summary["predicted_direction_counts"])
            evaluator.print_dict_block("main_flow_gate_summary", _main_flow_gate_summary)
            evaluator.print_top_reason_codes("top_final_reason_codes", _final_reason_counts, _top_n_reason_codes)
            evaluator.print_top_reason_codes("top_external_reason_codes", _external_reason_counts, _top_n_reason_codes)
            evaluator.print_top_reason_codes("top_h2_reason_codes", _h2_reason_counts, _top_n_reason_codes)
            evaluator.print_top_reason_codes("top_h1_reason_codes", _h1_reason_counts, _top_n_reason_codes)
            evaluator.print_top_reason_codes("top_m15_path_reason_codes", _m15_path_reason_counts, _top_n_reason_codes)
            evaluator.print_top_reason_codes("top_m15_reason_codes", _m15_reason_counts, _top_n_reason_codes)
            evaluator.print_count_hit_rate_block("direction_side_summary", _direction_side_summary)
            evaluator.print_count_hit_rate_block("decision_score_band_summary", _decision_score_band_summary)
            evaluator.print_count_hit_rate_block("entry_score_band_summary", _entry_score_band_summary)
            evaluator.print_threshold_summary_block(
                "decision_score_threshold_summary",
                _decision_score_threshold_summary,
            )
            evaluator.print_candidate_strategy_summary(
                "candidate_strategy_summary",
                _candidate_strategy_summary,
            )
        _summary_text = _buffer.getvalue()

    return {
        "summary": _summary,
        "direction_side_summary": _direction_side_summary,
        "main_flow_gate_summary": _main_flow_gate_summary,
        "summary_text": _summary_text,
    }


def _fmt_float(_value, _digits=4):
    if _value is None:
        return ""
    return f"{float(_value):.{_digits}f}"


def _fmt_int(_value):
    if _value is None:
        return ""
    return str(int(_value))


def build_comparison_row(_threshold_value, _metrics, _note=""):
    _summary = _metrics["summary"]
    _gate = _metrics["main_flow_gate_summary"]
    _side = _metrics["direction_side_summary"]

    _long = _side.get("ENTER_LONG", {})
    _short = _side.get("ENTER_SHORT", {})

    return [
        f"{float(_threshold_value):.2f}",
        _fmt_int(_summary.get("signal_count")),
        _fmt_int(_summary.get("evaluated_signal_count")),
        _fmt_int(_summary.get("no_signal_count")),
        _fmt_float(_summary.get("no_signal_ratio")),
        _fmt_float(_summary.get("enter_hit_rate")),
        _fmt_int(_summary.get("correct_count")),
        _fmt_int(_gate.get("final_signal_count")),
        _fmt_int(_gate.get("blocked_signal_count")),
        _fmt_float(_gate.get("final_enter_hit_rate")),
        _fmt_float(_gate.get("hit_rate_delta")),
        _fmt_int(_long.get("count")),
        _fmt_float(_long.get("hit_rate")),
        _fmt_int(_short.get("count")),
        _fmt_float(_short.get("hit_rate")),
        _note,
    ]


def write_comparison_markdown(_rows):
    ensure_parent_directory(DEFAULT_COMPARISON_PATH)

    _header = (
        "| threshold | signal_count | evaluated_signal_count | no_signal_count | "
        "no_signal_ratio | enter_hit_rate | correct_count | final_signal_count | "
        "blocked_signal_count | final_enter_hit_rate | hit_rate_delta | "
        "long_count | long_hit_rate | short_count | short_hit_rate | notes |"
    )
    _separator = (
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    _lines = [_header, _separator]
    for _row in _rows:
        _lines.append("| " + " | ".join(_row) + " |")

    DEFAULT_COMPARISON_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")


def main():
    _args = parse_args()
    _rows = []
    _errors = []

    for _threshold_value in THRESHOLD_VALUES:
        _paths = build_run_paths(_threshold_value)
        print(f"[INFO] threshold={_threshold_value:.2f} start")

        if not _paths["threshold_path"].exists():
            _message = f"threshold config is missing: {_paths['threshold_path']}"
            print(f"[ERROR] {_message}")
            _errors.append(_message)
            if _args.stop_on_error:
                break
            continue

        try:
            if _args.resume and has_reusable_run_artifacts(_paths):
                print(f"[INFO] threshold={_threshold_value:.2f} reuse_existing_raw_signals")
            elif not _args.skip_backtest:
                run_backtest_for_threshold(_args, _paths)

            _metrics = build_eval_artifacts(
                _raw_signals_path=_paths["raw_signals_path"],
                _top_n_reason_codes=_args.top_n_reason_codes,
            )
            _paths["eval_summary_path"].write_text(_metrics["summary_text"], encoding="utf-8")

            _note = "baseline" if abs(float(_threshold_value) - 0.65) < 1e-9 else ""
            _rows.append(build_comparison_row(_threshold_value, _metrics, _note))
            print(f"[INFO] threshold={_threshold_value:.2f} done")

        except Exception as _error:
            _message = f"threshold={_threshold_value:.2f} failed: {_error}"
            print(f"[ERROR] {_message}")
            _errors.append(_message)
            if _args.stop_on_error:
                break

    write_comparison_markdown(_rows)

    print(f"[INFO] comparison_markdown={DEFAULT_COMPARISON_PATH.resolve()}")
    print(f"[INFO] completed_runs={len(_rows)}")
    print(f"[INFO] error_count={len(_errors)}")

    if _errors:
        for _message in _errors:
            print(f"[ERROR] {_message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
