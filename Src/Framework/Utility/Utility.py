import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
JST_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def EnsureParentDirectory(_path):
    Path(_path).parent.mkdir(parents=True, exist_ok=True)


def LoadJson(_path):
    with open(_path, "r", encoding="utf-8") as _file:
        return json.load(_file)


def LoadJsonSafe(_path, _default, _warn=False):
    try:
        return LoadJson(_path)
    except FileNotFoundError:
        return _default
    except Exception as _error:
        if _warn:
            print(f"[WARN] failed to load json: path={_path}, error={_error}")
        return _default


def SaveJsonPretty(_path, _data):
    EnsureParentDirectory(_path)

    with open(_path, "w", encoding="utf-8") as _file:
        json.dump(_data, _file, ensure_ascii=False, indent=2)
        _file.write("\n")


# 文字列や None を含む入力を、安全に float へ寄せる。
def ToFloat(_value, _default=0.0):
    try:
        return float(_value)
    except Exception:
        return float(_default)


# 0.0 - 1.0 の範囲に正規化する。
def Clamp01(_value):
    _value = ToFloat(_value)

    if _value < 0.0:
        return 0.0
    if _value > 1.0:
        return 1.0
    return float(_value)


def ParseJSTDateTime(_value):
    if not _value:
        return None

    if isinstance(_value, datetime):
        if _value.tzinfo is None:
            return _value.replace(tzinfo=JST)
        return _value.astimezone(JST)

    try:
        _parsed = datetime.fromisoformat(str(_value))
    except Exception:
        _parsed = None

    if _parsed is None:
        for _fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
            try:
                _parsed = datetime.strptime(str(_value), _fmt)
                break
            except Exception:
                continue

    if _parsed is None:
        return None

    if _parsed.tzinfo is None:
        return _parsed.replace(tzinfo=JST)

    return _parsed.astimezone(JST)


def FormatJSTDateTime(_value):
    _dt = ParseJSTDateTime(_value)
    if _dt is None:
        return ""
    return _dt.strftime(JST_DATETIME_FORMAT)


def GetJSTNow():
    return datetime.now(JST)


def GetJSTNowStr():
    return FormatJSTDateTime(GetJSTNow())
