# ===================================================
# MTManager.py
# - MetaTrader5から為替データを取得・加工する中核モジュール
# ===================================================

import os
from datetime import datetime

import MetaTrader5 as mt5

from Framework.Utility.Utility import GetJSTNowStr, JST

# ---------------------------------------------------
# 使用する通貨ペア（MT5に接続して有効である必要がある）
# ---------------------------------------------------
symbol = "USDJPY"

# ===================================================
# MT5初期化＆ログイン
# - 環境変数からIDとパスを読み込み、OANDA MT5サーバへ接続
# ===================================================
def MTManager_Initialize():
    print("[INFO] MTManager Initialize 開始")

    loginIDRaw = os.getenv("MT_LOGIN_ID")
    loginPass = os.getenv("MT_LOGIN_PASS")
    serverName = os.getenv("MT_SERVER")

    if loginIDRaw is None or str(loginIDRaw).strip() == "":
        print("[ERROR] MT_LOGIN_ID が未設定です。")
        return False

    try:
        loginID = int(str(loginIDRaw).strip())
    except Exception:
        print(f"[ERROR] MT_LOGIN_ID が数値ではありません: {loginIDRaw}")
        return False

    if loginPass is None or str(loginPass).strip() == "":
        print("[ERROR] MT_LOGIN_PASS が未設定です。")
        return False

    try:
        _initialized = mt5.initialize(
            login=loginID,
            server=serverName,
            password=loginPass,
        )
    except Exception as e:
        print(f"[ERROR] MT5初期化中に例外発生: {e}")
        return False

    if not _initialized:
        print("[ERROR] MT5接続失敗：", mt5.last_error())
        return False

    print("[INFO] MT5接続成功")
    return True

def MTManager_GetRates(_timeframe, _count, _symbol=symbol):
    _rates = mt5.copy_rates_from_pos(_symbol, _timeframe, 0, _count)

    if _rates is None or len(_rates) == 0:
        print(f"[ERROR] レート取得失敗: symbol={_symbol}, timeframe={_timeframe}, count={_count}")
        return None

    return _rates


# ===================================================
# rates 1 行 value 抽出
# - numpy structured array / tuple のどちらでも同じ値を安全に読む
#
# 目的:
#   MT5 の戻り値形式差分を MTManager 内で吸収し、呼び出し側へ生の配列事情を漏らさない
# ===================================================
def MTManager_ExtractRateValue(_rate_row, _field_name, _field_index, _default=0.0):
    try:
        return float(_rate_row[_field_name])
    except Exception:
        try:
            return float(_rate_row[_field_index])
        except Exception:
            return float(_default)


# ===================================================
# rates 1 行 timestamp 変換
# - MT5 rates の unix time を JST datetime へ変換する
#
# 目的:
#   確定バー時刻の判定を main.py ではなく MTManager 側で扱えるようにする
# ===================================================
def MTManager_ExtractRateTimestampJST(_rate_row):
    try:
        _unix_time = int(_rate_row["time"])
    except Exception:
        try:
            _unix_time = int(_rate_row[0])
        except Exception:
            return None

    return datetime.fromtimestamp(_unix_time, JST)


# ===================================================
# JST datetime 文字列化
# - JST datetime をログ表示と比較用の文字列へ揃える
#
# 目的:
#   confirmed_bar_jst を全 timeframe で同じ書式に統一する
# ===================================================
def MTManager_FormatJSTDatetime(_dt):
    if _dt is None:
        return ""

    return _dt.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S")


# ===================================================
# 確定バー rates 取得
# - 形成中バーを除外し、直前の確定バーから count 本を取得する
#
# 目的:
#   realtime loop 側から MT5 API を直接呼ばずに、確定バー基準の取得を一元化する
# ===================================================
def MTManager_GetConfirmedRates(_timeframe, _count, _symbol=symbol, _timeframe_name=""):
    _rates = mt5.copy_rates_from_pos(_symbol, _timeframe, 1, _count)

    if _rates is None or len(_rates) == 0:
        raise RuntimeError(
            f"MT5 confirmed rate fetch failed: symbol={_symbol}, timeframe={_timeframe_name}, error={mt5.last_error()}"
        )

    return _rates

# ===================================================
# M15レート取得
# - MetaTrader5から15分足のOHLCデータを取得する
#
# 役割:
#   main.py など上位層から呼ばれ、
#   M15執行判定に必要な元データを供給する
#
# 入力:
#   _count  : 取得本数
#   _symbol : 通貨ペア（省略時はグローバルsymbolを使用）
#
# 戻り値:
#   正常時 : rates配列
#   異常時 : None
# ===================================================
def MTManager_GetM15Rates(_count, _symbol=symbol):
    _rates = mt5.copy_rates_from_pos(_symbol, mt5.TIMEFRAME_M15, 0, _count)

    # ---------------------------------------------------
    # データ取得失敗時は None を返す
    # ここでは例外を投げず、上位側で安全に扱える形に揃える
    # ---------------------------------------------------
    if _rates is None or len(_rates) == 0:
        print(f"[ERROR] M15レート取得失敗: symbol={_symbol}, count={_count}")
        return None

    return _rates


# ===================================================
# M15モメンタム計算
# - 直近終値と指定本数前終値の差分を返す
#
# 役割:
#   m15_entry へ渡す momentum を生成する
#
# 入力:
#   _rates    : MT5のrates配列
#   _lookback : 何本前と比較するか
#
# 戻り値:
#   正常時 : float
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcM15Momentum(_rates, _lookback=4):
    if _rates is None or len(_rates) <= _lookback:
        return 0.0

    _currentClose = _rates[-1][4]
    _pastClose = _rates[-1 - _lookback][4]

    return float(_currentClose - _pastClose)


# ===================================================
# M15プルバック状態判定
# - 直近価格位置を短期MAに対して簡易判定する
#
# 役割:
#   m15_entry へ渡す pullback_state を生成する
#
# 入力:
#   _rates : MT5のrates配列
#
# 戻り値:
#   "PULLBACK_LONG" | "PULLBACK_SHORT" | "NONE"
# ===================================================
def _MTManager_JudgeM15PullbackState(_rates):
    if _rates is None or len(_rates) < 20:
        return "NONE"

    _ma = sum(_row[4] for _row in _rates[-20:]) / 20.0
    _close = _rates[-1][4]

    if _close < _ma:
        return "PULLBACK_LONG"

    if _close > _ma:
        return "PULLBACK_SHORT"

    return "NONE"


# ===================================================
# M15ブレイクアウト判定
# - 直近高値/安値の更新を簡易判定する
#
# 役割:
#   m15_entry へ渡す breakout を生成する
#
# 入力:
#   _rates : MT5のrates配列
#
# 戻り値:
#   "BREAKOUT_UP" | "BREAKOUT_DOWN" | "NONE"
# ===================================================
def _MTManager_JudgeM15Breakout(_rates):
    if _rates is None or len(_rates) < 6:
        return "NONE"

    _currentHigh = _rates[-1][2]
    _currentLow = _rates[-1][3]
    _pastHigh = max(_row[2] for _row in _rates[-6:-1])
    _pastLow = min(_row[3] for _row in _rates[-6:-1])

    if _currentHigh > _pastHigh:
        return "BREAKOUT_UP"

    if _currentLow < _pastLow:
        return "BREAKOUT_DOWN"

    return "NONE"


# ===================================================
# M15ノイズ計算
# - 実体幅に対するヒゲ幅の比率を簡易ノイズとして返す
#
# 役割:
#   m15_entry へ渡す noise を生成する
#
# 入力:
#   _rates : MT5のrates配列
#
# 戻り値:
#   正常時 : float
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcM15Noise(_rates):
    if _rates is None or len(_rates) < 1:
        return 0.0

    _open = _rates[-1][1]
    _high = _rates[-1][2]
    _low = _rates[-1][3]
    _close = _rates[-1][4]

    _body = abs(_close - _open)
    _range = _high - _low
    _wick = _range - _body

    if _range <= 0:
        return 0.0

    return float(_wick / _range)


# ===================================================
# M15市場データ構築
# - M15レート取得と最小指標生成をまとめて行う
#
# 役割:
#   main.py から直接呼ばれ、
#   m15_entry に渡せるM15市場データ辞書を返す
#
# 入力:
#   _count  : 取得本数
#   _symbol : 通貨ペア（省略時はグローバルsymbolを使用）
#
# 戻り値:
#   {
#       "symbol": str,
#       "timeframe": "M15",
#       "ohlc": list,
#       "indicators": {
#           "momentum": float,
#           "pullback_state": str,
#           "breakout": str,
#           "noise": float,
#       },
#       "spread": float,
#   }
# ===================================================
def MTManager_BuildM15Data(_count, _symbol=symbol):
    _rates = MTManager_GetM15Rates(_count=_count, _symbol=_symbol)

    if _rates is None:
        return {
            "symbol": _symbol,
            "timeframe": "M15",
            "ohlc": [],
            "indicators": {
                "momentum": 0.0,
                "pullback_state": "NONE",
                "breakout": "NONE",
                "noise": 0.0,
            },
            "spread": 0.0,
        }

    _momentum = _MTManager_CalcM15Momentum(_rates, 4)
    _pullbackState = _MTManager_JudgeM15PullbackState(_rates)
    _breakout = _MTManager_JudgeM15Breakout(_rates)
    _noise = _MTManager_CalcM15Noise(_rates)

    # ---------------------------------------------------
    # spread を価格差ベースへ変換する
    # MT5 rates配列のspreadはポイント値のため、
    # symbol_info().point を掛けて閾値比較用の価格差へ揃える
    # ---------------------------------------------------
    _symbolInfo = mt5.symbol_info(_symbol)
    _point      = 0.0 if _symbolInfo is None else float(_symbolInfo.point)
    _spread     = float(_rates[-1][6]) * _point

    return {
        "symbol": _symbol,
        "timeframe": "M15",
        "ohlc": _rates,
        "indicators": {
            "momentum": float(_momentum),
            "pullback_state": _pullbackState,
            "breakout": _breakout,
            "noise": float(_noise),
        },
        "spread": _spread,
    }

# ===================================================
# H1レート取得
# - MetaTrader5から1時間足のOHLCデータを取得する
#
# 役割:
#   main.py など上位層から呼ばれ、
#   H1予測判定に必要な元データを供給する
#
# 入力:
#   _count  : 取得本数
#   _symbol : 通貨ペア（省略時はグローバルsymbolを使用）
#
# 戻り値:
#   正常時 : rates配列
#   異常時 : None
# ===================================================
def MTManager_GetH1Rates(_count, _symbol=symbol):
    _rates = mt5.copy_rates_from_pos(_symbol, mt5.TIMEFRAME_H1, 0, _count)

    # ---------------------------------------------------
    # データ取得失敗時は None を返す
    # ここでは例外を投げず、上位側で安全に扱える形に揃える
    # ---------------------------------------------------
    if _rates is None or len(_rates) == 0:
        print(f"[ERROR] H1レート取得失敗: symbol={_symbol}, count={_count}")
        return None

    return _rates


# ===================================================
# H1終値リスト抽出
# - rates配列から終値のみを取り出す
#
# 役割:
#   h1_forecast で使う raw_features の元データを生成する
#
# 入力:
#   _rates : MT5のrates配列
#
# 戻り値:
#   正常時 : list[float]
#   異常時 : []
# ===================================================
def _MTManager_ExtractH1CloseList(_rates):
    if _rates is None or len(_rates) == 0:
        return []

    return [float(_row[4]) for _row in _rates]


# ===================================================
# H1終値差分リスト生成
# - 終値リストから前本比の差分を生成する
#
# 役割:
#   h1_forecast で使う trend 判定素材を生成する
#
# 入力:
#   _close_list : 終値リスト
#
# 戻り値:
#   正常時 : list[float]
#   異常時 : []
# ===================================================
def _MTManager_BuildH1CloseDiffList(_close_list):
    if _close_list is None or len(_close_list) < 2:
        return []

    _diff_list = []

    for _index in range(1, len(_close_list)):
        _diff_list.append(float(_close_list[_index] - _close_list[_index - 1]))

    return _diff_list


# ===================================================
# H1簡易モメンタム計算
# - 直近終値と指定本数前終値の差分を返す
#
# 役割:
#   h1_forecast で参照する recent_momentum を生成する
#
# 入力:
#   _close_list : 終値リスト
#   _lookback   : 何本前と比較するか
#
# 戻り値:
#   正常時 : float
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcH1Momentum(_close_list, _lookback=5):
    if _close_list is None or len(_close_list) <= _lookback:
        return 0.0

    _current_close = _close_list[-1]
    _past_close = _close_list[-1 - _lookback]

    return float(_current_close - _past_close)


# ===================================================
# H1トレンド一貫性計算
# - 終値差分の符号一致率から一貫性を簡易計算する
#
# 役割:
#   h1_forecast で参照する trend_consistency を生成する
#
# 入力:
#   _diff_list : 終値差分リスト
#
# 戻り値:
#   正常時 : float(0.0 ~ 1.0)
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcH1TrendConsistency(_diff_list):
    if _diff_list is None or len(_diff_list) == 0:
        return 0.0

    _up_count = sum(1 for _diff in _diff_list if _diff > 0.0)
    _down_count = sum(1 for _diff in _diff_list if _diff < 0.0)
    _dominant_count = max(_up_count, _down_count)

    return float(_dominant_count / len(_diff_list))


# ===================================================
# H1市場データ構築
# - H1レート取得と最小特徴量生成をまとめて行う
#
# 役割:
#   main.py から直接呼ばれ、
#   h1_forecast に渡せるH1市場データ辞書を返す
#
# 入力:
#   _count  : 取得本数
#   _symbol : 通貨ペア（省略時はグローバルsymbolを使用）
#
# 戻り値:
#   {
#       "symbol": str,
#       "timeframe": "H1",
#       "ohlc": list,
#       "indicators": {
#           "raw_features": {
#               "close_list": list[float],
#               "close_diff_list": list[float],
#               "recent_momentum": float,
#               "trend_consistency": float,
#           }
#       }
#   }
# ===================================================
def MTManager_BuildH1Data(_count, _symbol=symbol):
    _rates = MTManager_GetH1Rates(_count=_count, _symbol=_symbol)

    if _rates is None:
        return {
            "symbol": _symbol,
            "timeframe": "H1",
            "ohlc": [],
            "indicators": {
                "raw_features": {
                    "close_list": [],
                    "close_diff_list": [],
                    "recent_momentum": 0.0,
                    "trend_consistency": 0.0,
                }
            },
        }

    _close_list = _MTManager_ExtractH1CloseList(_rates)
    _close_diff_list = _MTManager_BuildH1CloseDiffList(_close_list)
    _recent_momentum = _MTManager_CalcH1Momentum(_close_list, 5)
    _trend_consistency = _MTManager_CalcH1TrendConsistency(_close_diff_list)

    return {
        "symbol": _symbol,
        "timeframe": "H1",
        "ohlc": _rates,
        "indicators": {
            "raw_features": {
                "close_list": _close_list,
                "close_diff_list": _close_diff_list,
                "recent_momentum": float(_recent_momentum),
                "trend_consistency": float(_trend_consistency),
            }
        },
    }

# ===================================================
# H2レート取得
# - MetaTrader5から2時間足のOHLCデータを取得する
#
# 役割:
#   main.py など上位層から呼ばれ、
#   H2環境認識に必要な元データを供給する
#
# 入力:
#   _count  : 取得本数
#   _symbol : 通貨ペア（省略時はグローバルsymbolを使用）
#
# 戻り値:
#   正常時 : rates配列
#   異常時 : None
# ===================================================
def MTManager_GetH2Rates(_count, _symbol=symbol):
    _rates = mt5.copy_rates_from_pos(_symbol, mt5.TIMEFRAME_H2, 0, _count)

    # ---------------------------------------------------
    # データ取得失敗時は None を返す
    # ここでは例外を投げず、上位側で安全に扱える形に揃える
    # ---------------------------------------------------
    if _rates is None or len(_rates) == 0:
        print(f"[ERROR] H2レート取得失敗: symbol={_symbol}, count={_count}")
        return None

    return _rates


# ===================================================
# H2短期/長期MA計算
# - rates配列のcloseを用いて単純移動平均を計算する
#
# 役割:
#   h2_environment へ渡す ma_short / ma_long を生成する
#
# 入力:
#   _rates  : MT5のrates配列
#   _period : MA期間
#
# 戻り値:
#   正常時 : float
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcH2MA(_rates, _period):
    if _rates is None or len(_rates) < _period:
        return 0.0

    _closes = [_row[4] for _row in _rates[-_period:]]
    return sum(_closes) / float(_period)


# ===================================================
# H2 MA傾き計算
# - 短期MAの現在値と過去値の差分を返す
#
# 役割:
#   h2_environment へ渡す ma_slope を生成する
#
# 入力:
#   _rates      : MT5のrates配列
#   _period     : MA期間
#   _lookback   : 何本前と比較するか
#
# 戻り値:
#   正常時 : float
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcH2Slope(_rates, _period, _lookback=3):
    if _rates is None or len(_rates) < (_period + _lookback):
        return 0.0

    _currentCloses = [_row[4] for _row in _rates[-_period:]]
    _pastCloses = [_row[4] for _row in _rates[-(_period + _lookback):-_lookback]]

    _currentMA = sum(_currentCloses) / float(_period)
    _pastMA = sum(_pastCloses) / float(_period)

    return _currentMA - _pastMA


# ===================================================
# H2 ADX計算
# - rates配列から簡易ADXを計算する
#
# 役割:
#   h2_environment へ渡す adx を生成する
#
# 入力:
#   _rates  : MT5のrates配列
#   _period : ADX期間
#
# 戻り値:
#   正常時 : float
#   異常時 : 0.0
# ===================================================
def _MTManager_CalcH2ADX(_rates, _period=14):
    if _rates is None or len(_rates) < (_period * 2):
        return 0.0

    _trs = []
    _plusDMs = []
    _minusDMs = []

    for i in range(1, len(_rates)):
        _prevHigh = _rates[i - 1][2]
        _prevLow = _rates[i - 1][3]
        _prevClose = _rates[i - 1][4]

        _high = _rates[i][2]
        _low = _rates[i][3]

        _upMove = _high - _prevHigh
        _downMove = _prevLow - _low

        _plusDM = _upMove if (_upMove > _downMove and _upMove > 0) else 0.0
        _minusDM = _downMove if (_downMove > _upMove and _downMove > 0) else 0.0

        _tr = max(
            _high - _low,
            abs(_high - _prevClose),
            abs(_low - _prevClose),
        )

        _trs.append(_tr)
        _plusDMs.append(_plusDM)
        _minusDMs.append(_minusDM)

    if len(_trs) < _period:
        return 0.0

    _dxs = []

    for i in range(_period - 1, len(_trs)):
        _trSum = sum(_trs[i - _period + 1:i + 1])
        _plusDMSum = sum(_plusDMs[i - _period + 1:i + 1])
        _minusDMSum = sum(_minusDMs[i - _period + 1:i + 1])

        if _trSum == 0:
            _dxs.append(0.0)
            continue

        _plusDI = (_plusDMSum / _trSum) * 100.0
        _minusDI = (_minusDMSum / _trSum) * 100.0

        _diSum = _plusDI + _minusDI
        if _diSum == 0:
            _dxs.append(0.0)
            continue

        _dx = (abs(_plusDI - _minusDI) / _diSum) * 100.0
        _dxs.append(_dx)

    if len(_dxs) < _period:
        return 0.0

    return sum(_dxs[-_period:]) / float(_period)


# ===================================================
# H2スイング構造判定
# - 直近高値/安値の切り上げ・切り下げを簡易判定する
#
# 役割:
#   h2_environment へ渡す swing_structure を生成する
#
# 入力:
#   _rates : MT5のrates配列
#
# 戻り値:
#   "HIGHER_HIGH" | "LOWER_LOW" | "RANGE"
# ===================================================
def _MTManager_JudgeH2SwingStructure(_rates):
    if _rates is None or len(_rates) < 4:
        return "RANGE"

    _recentHigh = max(_row[2] for _row in _rates[-2:])
    _pastHigh = max(_row[2] for _row in _rates[-4:-2])

    _recentLow = min(_row[3] for _row in _rates[-2:])
    _pastLow = min(_row[3] for _row in _rates[-4:-2])

    if _recentHigh > _pastHigh and _recentLow >= _pastLow:
        return "HIGHER_HIGH"

    if _recentLow < _pastLow and _recentHigh <= _pastHigh:
        return "LOWER_LOW"

    return "RANGE"


# ===================================================
# H2市場データ構築
# - H2レート取得と最小指標生成をまとめて行う
#
# 役割:
#   main.py から直接呼ばれ、
#   h2_environment に渡せるH2市場データ辞書を返す
#
# 入力:
#   _count  : 取得本数
#   _symbol : 通貨ペア（省略時はグローバルsymbolを使用）
#
# 戻り値:
#   {
#       "symbol": str,
#       "timeframe": "H2",
#       "ohlc": list,
#       "indicators": {
#           "ma_short": float,
#           "ma_long": float,
#           "ma_slope": float,
#           "adx": float,
#           "swing_structure": str,
#       }
#   }
# ===================================================
def MTManager_BuildH2Data(_count, _symbol=symbol):
    _rates = MTManager_GetH2Rates(_count=_count, _symbol=_symbol)

    if _rates is None:
        return {
            "symbol": _symbol,
            "timeframe": "H2",
            "ohlc": [],
            "indicators": {
                "ma_short": 0.0,
                "ma_long": 0.0,
                "ma_slope": 0.0,
                "adx": 0.0,
                "swing_structure": "RANGE",
            },
        }

    _maShort        = _MTManager_CalcH2MA(_rates, 20)
    _maLong         = _MTManager_CalcH2MA(_rates, 50)
    _maSlope        = _MTManager_CalcH2Slope(_rates, 20, 3)
    _adx            = _MTManager_CalcH2ADX(_rates, 14)
    _swingStructure = _MTManager_JudgeH2SwingStructure(_rates)

    return {
        "symbol": _symbol,
        "timeframe": "H2",
        "ohlc": _rates,
        "indicators": {
            "ma_short": float(_maShort),
            "ma_long": float(_maLong),
            "ma_slope": float(_maSlope),
            "adx": float(_adx),
            "swing_structure": _swingStructure,
        },
    }


# ===================================================
# confirmed M15 データ構築
# - 直前の確定 M15 バーだけを前提に、entry 判定用の M15 データを構築する
#
# 目的:
#   realtime loop が MT5 依存処理を持たずに済むよう、confirmed M15 の組み立てを MTManager へ寄せる
# ===================================================
def MTManager_BuildConfirmedM15Data(_count, _symbol=symbol, _timestamp_jst=""):
    _rates = MTManager_GetConfirmedRates(
        _timeframe=mt5.TIMEFRAME_M15,
        _count=_count,
        _symbol=_symbol,
        _timeframe_name="M15",
    )
    _momentum = _MTManager_CalcM15Momentum(_rates, 4)
    _pullbackState = _MTManager_JudgeM15PullbackState(_rates)
    _breakout = _MTManager_JudgeM15Breakout(_rates)
    _noise = _MTManager_CalcM15Noise(_rates)
    _symbolInfo = mt5.symbol_info(_symbol)
    _point = 0.0 if _symbolInfo is None else float(_symbolInfo.point)
    _spread = MTManager_ExtractRateValue(_rates[-1], "spread", 6, 0.0) * _point
    _confirmed_bar_jst = MTManager_FormatJSTDatetime(MTManager_ExtractRateTimestampJST(_rates[-1]))

    return {
        "symbol": _symbol,
        "timeframe": "M15",
        "timestamp_jst": _timestamp_jst or GetJSTNowStr(),
        "confirmed_bar_jst": _confirmed_bar_jst,
        "ohlc": _rates,
        "indicators": {
            "momentum": float(_momentum),
            "pullback_state": _pullbackState,
            "breakout": _breakout,
            "noise": float(_noise),
        },
        "spread": float(_spread),
    }


# ===================================================
# confirmed H1 データ構築
# - 直前の確定 H1 バーだけを前提に、forecast 判定用の H1 データを構築する
#
# 目的:
#   runtime 予測で使う raw_features と confirmed H1 バー時刻を MTManager から返す
# ===================================================
def MTManager_BuildConfirmedH1Data(_count, _symbol=symbol, _timestamp_jst=""):
    _rates = MTManager_GetConfirmedRates(
        _timeframe=mt5.TIMEFRAME_H1,
        _count=_count,
        _symbol=_symbol,
        _timeframe_name="H1",
    )
    _close_list = _MTManager_ExtractH1CloseList(_rates)
    _close_diff_list = _MTManager_BuildH1CloseDiffList(_close_list)
    _confirmed_bar_jst = MTManager_FormatJSTDatetime(MTManager_ExtractRateTimestampJST(_rates[-1]))

    return {
        "symbol": _symbol,
        "timeframe": "H1",
        "timestamp_jst": _timestamp_jst or GetJSTNowStr(),
        "confirmed_bar_jst": _confirmed_bar_jst,
        "ohlc": _rates,
        "indicators": {
            "raw_features": {
                "close_list": _close_list,
                "close_diff_list": _close_diff_list,
                "recent_momentum": float(_MTManager_CalcH1Momentum(_close_list, 5)),
                "trend_consistency": float(_MTManager_CalcH1TrendConsistency(_close_diff_list)),
            }
        },
    }


# ===================================================
# confirmed H2 データ構築
# - 直前の確定 H2 バーだけを前提に、大局判定用の H2 データを構築する
#
# 目的:
#   realtime loop 側では confirmed H2 の完成データを受け取るだけにする
# ===================================================
def MTManager_BuildConfirmedH2Data(_count, _symbol=symbol, _timestamp_jst=""):
    _rates = MTManager_GetConfirmedRates(
        _timeframe=mt5.TIMEFRAME_H2,
        _count=_count,
        _symbol=_symbol,
        _timeframe_name="H2",
    )
    _confirmed_bar_jst = MTManager_FormatJSTDatetime(MTManager_ExtractRateTimestampJST(_rates[-1]))

    return {
        "symbol": _symbol,
        "timeframe": "H2",
        "timestamp_jst": _timestamp_jst or GetJSTNowStr(),
        "confirmed_bar_jst": _confirmed_bar_jst,
        "ohlc": _rates,
        "indicators": {
            "ma_short": float(_MTManager_CalcH2MA(_rates, 20)),
            "ma_long": float(_MTManager_CalcH2MA(_rates, 50)),
            "ma_slope": float(_MTManager_CalcH2Slope(_rates, 20, 3)),
            "adx": float(_MTManager_CalcH2ADX(_rates, 14)),
            "swing_structure": _MTManager_JudgeH2SwingStructure(_rates),
        },
    }


# ===================================================
# confirmed market data 一括構築
# - realtime loop が必要とする H2 / H1 / M15 の confirmed データをまとめて返す
#
# 目的:
#   main flow から timeframe 別の MT5 呼び出しを隠蔽し、呼び出し責務を整理する
# ===================================================
def MTManager_BuildConfirmedMarketData(
    _symbol=symbol,
    _timestamp_jst="",
    _m15_count=200,
    _h1_count=200,
    _h2_count=200,
):
    _evaluation_timestamp_jst = _timestamp_jst or GetJSTNowStr()

    return {
        "H2": MTManager_BuildConfirmedH2Data(
            _count=_h2_count,
            _symbol=_symbol,
            _timestamp_jst=_evaluation_timestamp_jst,
        ),
        "H1": MTManager_BuildConfirmedH1Data(
            _count=_h1_count,
            _symbol=_symbol,
            _timestamp_jst=_evaluation_timestamp_jst,
        ),
        "M15": MTManager_BuildConfirmedM15Data(
            _count=_m15_count,
            _symbol=_symbol,
            _timestamp_jst=_evaluation_timestamp_jst,
        ),
    }
