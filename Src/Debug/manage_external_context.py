# --------------------------------------------------
# manage_external_context.py
# 役割:
#   external_events.json と manual_risk_flags.json の
#   更新フローを安全に扱うための管理スクリプト
#
# 使い方例:
#   python Src\Debug\manage_external_context.py summary
#   python Src\Debug\manage_external_context.py add-event --title "US CPI" --currency USD --event-time-jst "2026-04-05 21:30:00" --importance high
#   python Src\Debug\manage_external_context.py set-flag --flag geopolitical_alert --value true
#   python Src\Debug\manage_external_context.py clear-expired-events
# --------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path


_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.Utility.Utility import (
    GetJSTNow,
    GetJSTNowStr,
    LoadJsonSafe,
    ParseJSTDateTime,
    SaveJsonPretty,
)

CONFIG_DIR = Path("Asset/Config")
EXTERNAL_EVENTS_PATH = CONFIG_DIR / "external_events.json"
MANUAL_RISK_FLAGS_PATH = CONFIG_DIR / "manual_risk_flags.json"

DEFAULT_EVENTS = {
    "schema_version": 1,
    "events": [],
}

DEFAULT_FLAGS = {
    "schema_version": 1,
    "updated_jst": "",
    "notes": "",
    "flags": {
        "high_impact_event_soon": False,
        "central_bank_speech": False,
        "geopolitical_alert": False,
        "data_feed_error": False,
        "abnormal_volatility": False,
    },
}

VALID_FLAGS = list(DEFAULT_FLAGS["flags"].keys())
VALID_IMPORTANCE = ["low", "medium", "high", "critical"]


def _to_bool(_value):
    return str(_value).strip().lower() in ["true", "1", "yes", "on"]


def _load_events():
    _loaded = LoadJsonSafe(EXTERNAL_EVENTS_PATH, DEFAULT_EVENTS)

    if not isinstance(_loaded, dict):
        return dict(DEFAULT_EVENTS)

    _events = _loaded.get("events", [])
    if not isinstance(_events, list):
        _events = []

    return {
        "schema_version": _loaded.get("schema_version", 1),
        "events": _events,
    }


def _load_flags():
    _loaded = LoadJsonSafe(MANUAL_RISK_FLAGS_PATH, DEFAULT_FLAGS)

    if not isinstance(_loaded, dict):
        return json.loads(json.dumps(DEFAULT_FLAGS))

    _flags = _loaded.get("flags", {})
    if not isinstance(_flags, dict):
        _flags = {}

    _merged_flags = dict(DEFAULT_FLAGS["flags"])
    for _key in _merged_flags.keys():
        _merged_flags[_key] = _to_bool(_flags.get(_key, _merged_flags[_key]))

    return {
        "schema_version": _loaded.get("schema_version", 1),
        "updated_jst": _loaded.get("updated_jst", ""),
        "notes": _loaded.get("notes", ""),
        "flags": _merged_flags,
    }


def _save_events(_events_data):
    SaveJsonPretty(EXTERNAL_EVENTS_PATH, _events_data)


def _save_flags(_flags_data):
    _flags_data["updated_jst"] = GetJSTNowStr()
    SaveJsonPretty(MANUAL_RISK_FLAGS_PATH, _flags_data)


def ensure_files():
    _save_events(_load_events())
    _save_flags(_load_flags())


def summary():
    _events_data = _load_events()
    _flags_data = _load_flags()

    print("----- external_events.json -----")
    print(f"event_count: {len(_events_data['events'])}")
    for _index, _event in enumerate(_events_data["events"], start=1):
        print(
            f"{_index}. title={_event.get('title')} "
            f"currency={_event.get('currency')} "
            f"event_time_jst={_event.get('event_time_jst')} "
            f"importance={_event.get('importance')} "
            f"is_active={_event.get('is_active', True)}"
        )

    print("")
    print("----- manual_risk_flags.json -----")
    print(json.dumps(_flags_data, ensure_ascii=False, indent=2))


def add_event(_args):
    _event_time = ParseJSTDateTime(_args.event_time_jst)
    if _event_time is None:
        raise ValueError("event_time_jst must be parseable as JST datetime")

    _importance = str(_args.importance).lower()
    if _importance not in VALID_IMPORTANCE:
        raise ValueError(f"importance must be one of: {', '.join(VALID_IMPORTANCE)}")

    _events_data = _load_events()
    _events_data["events"].append({
        "title": _args.title,
        "currency": _args.currency.upper(),
        "event_time_jst": _event_time.strftime("%Y-%m-%d %H:%M:%S"),
        "importance": _importance,
        "category": _args.category,
        "is_active": True,
        "notes": _args.notes,
    })
    _events_data["events"].sort(key=lambda _event: str(_event.get("event_time_jst", "")))
    _save_events(_events_data)
    print("[OK] event added")


def clear_expired_events():
    _events_data = _load_events()
    _now_jst = GetJSTNow()
    _kept_events = []
    _removed_count = 0

    for _event in _events_data["events"]:
        _event_time = ParseJSTDateTime(_event.get("event_time_jst"))
        if _event_time is None:
            _removed_count += 1
            continue

        if _event_time < _now_jst:
            _removed_count += 1
            continue

        _kept_events.append(_event)

    _events_data["events"] = _kept_events
    _save_events(_events_data)
    print(f"[OK] removed_events={_removed_count}")


def set_flag(_args):
    _flag_name = _args.flag
    if _flag_name not in VALID_FLAGS:
        raise ValueError(f"flag must be one of: {', '.join(VALID_FLAGS)}")

    _flags_data = _load_flags()
    _flags_data["flags"][_flag_name] = _to_bool(_args.value)
    if _args.notes is not None:
        _flags_data["notes"] = _args.notes
    _save_flags(_flags_data)
    print("[OK] flag updated")


def clear_flags():
    _flags_data = _load_flags()
    for _flag_name in VALID_FLAGS:
        _flags_data["flags"][_flag_name] = False
    _flags_data["notes"] = ""
    _save_flags(_flags_data)
    print("[OK] all flags cleared")


def _build_parser():
    _parser = argparse.ArgumentParser(description="Manage external context files")
    _subparsers = _parser.add_subparsers(dest="command", required=True)

    _subparsers.add_parser("summary")
    _subparsers.add_parser("ensure-files")
    _subparsers.add_parser("clear-expired-events")
    _subparsers.add_parser("clear-flags")

    _add_event = _subparsers.add_parser("add-event")
    _add_event.add_argument("--title", required=True)
    _add_event.add_argument("--currency", required=True)
    _add_event.add_argument("--event-time-jst", required=True)
    _add_event.add_argument("--importance", default="high")
    _add_event.add_argument("--category", default="economic_indicator")
    _add_event.add_argument("--notes", default="")

    _set_flag = _subparsers.add_parser("set-flag")
    _set_flag.add_argument("--flag", required=True)
    _set_flag.add_argument("--value", required=True)
    _set_flag.add_argument("--notes")

    return _parser


def main():
    _parser = _build_parser()
    _args = _parser.parse_args()

    if _args.command == "summary":
        summary()
        return

    if _args.command == "ensure-files":
        ensure_files()
        print("[OK] files ensured")
        return

    if _args.command == "add-event":
        add_event(_args)
        return

    if _args.command == "clear-expired-events":
        clear_expired_events()
        return

    if _args.command == "set-flag":
        set_flag(_args)
        return

    if _args.command == "clear-flags":
        clear_flags()
        return


if __name__ == "__main__":
    main()
