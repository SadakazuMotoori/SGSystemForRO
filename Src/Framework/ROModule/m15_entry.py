# --------------------------------------------------
# m15_entry.py
# 役割:
#   15分足を用いて、上位足が許可した方向に対して
#   今この瞬間にエントリーすべきかを判定する
#
# 設計方針:
#   - 15分足は執行の関所として扱う
#   - 上位足の結論を覆さない
#   - 外部停止条件がONなら基本は SKIP
#   - 戻り値は常に固定形式で返す
# --------------------------------------------------

from Framework.ROModule.h1_forecast import evaluate_h1_alignment
from Framework.ROModule.h2_environment_contract import (
    build_h2_runtime_view,
    resolve_h2_direction,
)
from Framework.Utility.Utility import Clamp01 as _clamp_01, ToFloat as _to_float


def _rc(_suffix: str) -> str:
    return f"M15_ENTRY_{_suffix}"


# --------------------------------------------------
# 内部関数:
#   値を安全に float へ変換する
#
# 役割:
#   None や不正値混入時でも落ちにくくする
# --------------------------------------------------
# --------------------------------------------------
# 内部関数:
#   値を 0.0 ~ 1.0 に丸める
#
# 役割:
#   timing_quality などの出力を安定させる
# --------------------------------------------------
# --------------------------------------------------
# 内部関数:
#   entry_score を 0 ~ 100 に丸める
#
# 役割:
#   スコア出力を固定範囲に収める
# --------------------------------------------------
# 共通の数値変換と 0.0 - 1.0 の丸めは Utility 側を使う。
# M15 固有の 0 - 100 スコア補正だけをこのモジュールに残す。
def _clamp_score(_value):
    _value = int(round(_value))

    if _value < 0:
        return 0
    if _value > 100:
        return 100
    return _value


def _build_h2_context_features(_h2_environment_result):
    _h2_runtime_view = build_h2_runtime_view(_h2_environment_result)
    return {
        "env_direction": _h2_runtime_view["env_direction"],
        "regime_direction": _h2_runtime_view["regime_direction"],
        "regime_score": _h2_runtime_view["regime_score"],
        "regime_quality": _h2_runtime_view["regime_quality"],
    }


# --------------------------------------------------
# 内部関数:
#   H2方向に対して momentum が整合しているか判定する
#
# 役割:
#   LONG_ONLY なら正方向、SHORT_ONLY なら負方向の
#   momentum を有利とみなす
# --------------------------------------------------
def _is_momentum_aligned(_momentum, _env_direction):
    if _env_direction == "LONG_ONLY":
        return _momentum > 0.0
    if _env_direction == "SHORT_ONLY":
        return _momentum < 0.0
    return False


# --------------------------------------------------
# 内部関数:
#   H2方向に対して pullback_state が整合しているか判定する
#
# 役割:
#   LONG_ONLY なら PULLBACK_LONG、
#   SHORT_ONLY なら PULLBACK_SHORT を有利とみなす
# --------------------------------------------------
def _is_pullback_aligned(_pullback_state, _env_direction):
    if _env_direction == "LONG_ONLY":
        return _pullback_state == "PULLBACK_LONG"
    if _env_direction == "SHORT_ONLY":
        return _pullback_state == "PULLBACK_SHORT"
    return False


# --------------------------------------------------
# 内部関数:
#   H2方向に対して breakout が整合しているか判定する
#
# 役割:
#   LONG_ONLY なら BREAKOUT_UP、
#   SHORT_ONLY なら BREAKOUT_DOWN を有利とみなす
# --------------------------------------------------
def _is_breakout_aligned(_breakout, _env_direction):
    if _env_direction == "LONG_ONLY":
        return _breakout == "BREAKOUT_UP"
    if _env_direction == "SHORT_ONLY":
        return _breakout == "BREAKOUT_DOWN"
    return False


# --------------------------------------------------
# 内部関数:
#   M15執行判定に対するH1バイアス文脈を整理する
#
# 役割:
#   H1はM15の主判定は覆さないが、
#   どの状態として扱っているかを reason_codes / raw_features へ残す
# 実装方針:
#   H1 の解釈は shared helper evaluate_h1_alignment() に寄せる
# --------------------------------------------------
def _evaluate_h1_bias_context(_h1_forecast_result, _env_direction, _thresholds):
    _alignment_result = evaluate_h1_alignment(
        _h1_forecast_result=_h1_forecast_result,
        _env_direction=_env_direction,
        _thresholds=_thresholds,
    )
    _features = {
        "h1_alignment": _alignment_result["alignment"],
        "h1_net_direction": _alignment_result["net_direction"],
        "h1_confidence": _alignment_result["confidence"],
        "h1_forecast_status": _alignment_result["forecast_status"],
    }
    _alignment = _alignment_result["alignment"]

    if _alignment == "UNAVAILABLE":
        return [], _features

    if _alignment == "NEUTRAL_OR_SKIPPED":
        return [_rc("H1_BIAS_NEUTRAL_OR_SKIPPED")], _features

    if _alignment == "LOW_CONFIDENCE":
        return [_rc("H1_BIAS_LOW_CONFIDENCE")], _features

    if _alignment == "ALIGNED" and _env_direction == "LONG_ONLY":
        return [_rc("H1_BIAS_ALIGNED_LONG")], _features

    if _alignment == "ALIGNED" and _env_direction == "SHORT_ONLY":
        return [_rc("H1_BIAS_ALIGNED_SHORT")], _features

    if _alignment == "CONFLICT":
        return [_rc("H1_BIAS_CONFLICT")], _features

    return [], _features


# --------------------------------------------------
# 内部関数:
#   15分足の簡易スコアを算出する
#
# 入力で参照する想定項目:
#   indicators["momentum"]
#   indicators["pullback_state"]
#   indicators["breakout"]
#   indicators["noise"]
#   market_data_m15["spread"]
#
# 役割:
#   最小実装として、執行タイミングの良し悪しを
#   score と quality に変換する
# --------------------------------------------------
def _calculate_entry_metrics(_market_data_m15, _env_direction, _thresholds):
    _indicators = _market_data_m15.get("indicators", {})

    _momentum = _to_float(_indicators.get("momentum"))
    _pullback_state = _indicators.get("pullback_state")
    _breakout = _indicators.get("breakout")
    _noise = _to_float(_indicators.get("noise"))
    _spread = _to_float(_market_data_m15.get("spread"))

    _spread_max = _to_float(_thresholds.get("spread_max"), 0.02)
    _noise_max = _to_float(_thresholds.get("m15_noise_max"), 0.40)

    _score = 0
    _reason_codes = []

    # --------------------------------------------------
    # ① モメンタム評価
    # 上位足方向と整合し、かつ値幅が大きいほど加点する
    # --------------------------------------------------
    if _is_momentum_aligned(_momentum, _env_direction):
        if abs(_momentum) >= 0.05:
            _score += 35
            _reason_codes.append(_rc("MOMENTUM_ALIGNED_STRONG"))
        else:
            _score += 20
            _reason_codes.append(_rc("MOMENTUM_ALIGNED_WEAK"))
    else:
        _reason_codes.append(_rc("MOMENTUM_MISALIGNED"))

    # --------------------------------------------------
    # ② プルバック状態評価
    # 上位足方向に対する押し・戻しとして整合していれば加点する
    # --------------------------------------------------
    if _is_pullback_aligned(_pullback_state, _env_direction):
        _score += 25
        _reason_codes.append(_rc("PULLBACK_ALIGNED"))
    elif _pullback_state == "NONE":
        _score += 5
        _reason_codes.append(_rc("PULLBACK_NONE"))
    else:
        _reason_codes.append(_rc("PULLBACK_MISALIGNED"))

    # --------------------------------------------------
    # ③ ブレイク有無評価
    # 上位足方向に沿うブレイクが出ていれば加点する
    # --------------------------------------------------
    if _is_breakout_aligned(_breakout, _env_direction):
        _score += 20
        _reason_codes.append(_rc("BREAKOUT_ALIGNED"))
    elif _breakout == "NONE":
        _reason_codes.append(_rc("BREAKOUT_NONE"))
    else:
        _reason_codes.append(_rc("BREAKOUT_MISALIGNED"))

    # --------------------------------------------------
    # ④ ノイズ評価
    # ノイズが大きいほど減点する
    # --------------------------------------------------
    if _noise <= (_noise_max * 0.5):
        _score += 15
        _reason_codes.append(_rc("NOISE_LOW"))
    elif _noise <= _noise_max:
        _score += 5
        _reason_codes.append(_rc("NOISE_ACCEPTABLE"))
    else:
        _score -= 20
        _reason_codes.append(_rc("NOISE_HIGH"))

    # --------------------------------------------------
    # ⑤ スプレッド評価
    # スプレッドが閾値超過なら減点する
    # --------------------------------------------------
    if _spread > _spread_max:
        _score -= 25
        _reason_codes.append(_rc("SPREAD_UNFAVORABLE"))
    else:
        _reason_codes.append(_rc("SPREAD_ACCEPTABLE"))

    _timing_quality = _clamp_01(_score / 100.0)
    _entry_score = _clamp_score(_score)

    return _entry_score, _timing_quality, _reason_codes


# --------------------------------------------------
# メイン関数:
#   15分足の執行判定を行う
#
# 入力:
#   market_data_m15:
#       15分足OHLC、指標、spreadなど
#   h2_environment_result:
#       2時間足の方向許可結果
#   h1_forecast_result:
#       1時間足予測結果（Phase 2では None 許容）
#   external_filter_result:
#       外部停止条件結果
#   thresholds:
#       m15_entry_score_min, spread_max, m15_noise_max など
#
# 出力:
#   {
#       "module_name": "m15_entry",
#       "timestamp_jst": str,
#       "status": "OK" | "ERROR",
#       "entry_action": "ENTER" | "WAIT" | "SKIP" | "EXIT",
#       "entry_side": "LONG" | "SHORT" | "NONE",
#       "entry_score": int,
#       "timing_quality": float,
#       "risk_flag": bool,
#       "reason_codes": list[str],
#       "summary": str,
#       "raw_features": dict,
#   }
# --------------------------------------------------
def evaluate_m15_entry(
    market_data_m15: dict,
    h2_environment_result: dict,
    h1_forecast_result: dict | None,
    external_filter_result: dict,
    thresholds: dict,
) -> dict:
    _timestamp_jst = market_data_m15.get("timestamp_jst", "")
    _indicators = market_data_m15.get("indicators", {})
    _env_direction = resolve_h2_direction(h2_environment_result)
    _h2_context_features = _build_h2_context_features(h2_environment_result)
    _h1_reason_codes, _h1_features = _evaluate_h1_bias_context(
        h1_forecast_result,
        _env_direction,
        thresholds,
    )

    try:
        # --------------------------------------------------
        # ① 外部停止条件が有効なら SKIP
        # 執行前段階で安全側に倒す
        # --------------------------------------------------
        if external_filter_result.get("can_trade") is False:
            return {
                "module_name": "m15_entry",
                "timestamp_jst": _timestamp_jst,
                "status": "OK",
                "entry_action": "SKIP",
                "entry_side": "NONE",
                "entry_score": 0,
                "timing_quality": 0.0,
                "risk_flag": True,
                "reason_codes": [_rc("EXTERNAL_FILTER_ON")],
                "summary": "外部停止条件が有効のため執行見送り",
                "raw_features": {
                    "momentum": _indicators.get("momentum"),
                    "pullback_state": _indicators.get("pullback_state"),
                    "breakout": _indicators.get("breakout"),
                    "noise": _indicators.get("noise"),
                    "spread": market_data_m15.get("spread"),
                    **_h2_context_features,
                    **_h1_features,
                },
            }

        # --------------------------------------------------
        # ② 2時間足が NO_TRADE なら SKIP
        # 上位足が方向を許可していないため入らない
        # --------------------------------------------------
        if _env_direction == "NO_TRADE":
            return {
                "module_name": "m15_entry",
                "timestamp_jst": _timestamp_jst,
                "status": "OK",
                "entry_action": "SKIP",
                "entry_side": "NONE",
                "entry_score": 0,
                "timing_quality": 0.0,
                "risk_flag": True,
                "reason_codes": [_rc("H2_NO_TRADE")],
                "summary": "2時間足が取引を許可していないため執行見送り",
                "raw_features": {
                    "momentum": _indicators.get("momentum"),
                    "pullback_state": _indicators.get("pullback_state"),
                    "breakout": _indicators.get("breakout"),
                    "noise": _indicators.get("noise"),
                    "spread": market_data_m15.get("spread"),
                    **_h2_context_features,
                    **_h1_features,
                },
            }

        # --------------------------------------------------
        # ③ 15分足の執行スコアを算出
        # 最小実装として momentum / pullback / breakout /
        # noise / spread で score を決める
        # --------------------------------------------------
        _entry_score, _timing_quality, _reason_codes = _calculate_entry_metrics(
            market_data_m15,
            _env_direction,
            thresholds,
        )
        _reason_codes.extend(_h1_reason_codes)

        _entry_score_min = _clamp_score(thresholds.get("m15_entry_score_min", 70))
        _entry_side = "LONG" if _env_direction == "LONG_ONLY" else "SHORT"

        _pullback_state = _indicators.get("pullback_state")
        _breakout = _indicators.get("breakout")
        _noise = _to_float(_indicators.get("noise"))
        _noise_max = _to_float(thresholds.get("m15_noise_max"), 0.40)

        # --------------------------------------------------
        # ④ EXIT 判定
        # 最小版では短期ノイズ極大時のみ EXIT 候補とする
        # --------------------------------------------------
        if _noise > (_noise_max * 1.5):
            _reason_codes.append(_rc("REVERSAL_DETECTED"))
            return {
                "module_name": "m15_entry",
                "timestamp_jst": _timestamp_jst,
                "status": "OK",
                "entry_action": "EXIT",
                "entry_side": "NONE",
                "entry_score": _entry_score,
                "timing_quality": _timing_quality,
                "risk_flag": True,
                "reason_codes": _reason_codes,
                "summary": "短期ノイズが大きくシナリオ崩れのため撤退判定",
                "raw_features": {
                    "momentum": _indicators.get("momentum"),
                    "pullback_state": _pullback_state,
                    "breakout": _breakout,
                    "noise": _indicators.get("noise"),
                    "spread": market_data_m15.get("spread"),
                    **_h2_context_features,
                    **_h1_features,
                },
            }

        # --------------------------------------------------
        # ⑤ ENTER / WAIT / SKIP 判定
        # score と pullback / breakout の整合をもとに簡易判定する
        # --------------------------------------------------
        _pullback_aligned = _is_pullback_aligned(_pullback_state, _env_direction)
        _breakout_aligned = _is_breakout_aligned(_breakout, _env_direction)

        if _entry_score >= _entry_score_min and (_pullback_aligned or _breakout_aligned):
            _entry_action = "ENTER"
            _summary = "上位足と整合し、15分足の執行タイミングが良好"
            _risk_flag = False
        elif _entry_score >= max(_entry_score_min - 20, 0):
            _entry_action = "WAIT"
            _summary = "方向性はあるが、15分足のタイミングがまだ弱いため待機"
            _risk_flag = False
        else:
            _entry_action = "SKIP"
            _summary = "15分足の執行条件が弱いため見送り"
            _risk_flag = True

        return {
            "module_name": "m15_entry",
            "timestamp_jst": _timestamp_jst,
            "status": "OK",
            "entry_action": _entry_action,
            "entry_side": _entry_side if _entry_action != "SKIP" else "NONE",
            "entry_score": _entry_score,
            "timing_quality": _timing_quality,
            "risk_flag": _risk_flag,
            "reason_codes": _reason_codes,
            "summary": _summary,
            "raw_features": {
                "momentum": _indicators.get("momentum"),
                "pullback_state": _pullback_state,
                "breakout": _breakout,
                "noise": _indicators.get("noise"),
                "spread": market_data_m15.get("spread"),
                **_h2_context_features,
                **_h1_features,
            },
        }

    except Exception as e:
        # --------------------------------------------------
        # 例外時の保険
        # 15分足判定で例外が出た場合は、
        # 安全側に倒して SKIP を返す
        # --------------------------------------------------
        return {
            "module_name": "m15_entry",
            "timestamp_jst": _timestamp_jst,
            "status": "ERROR",
            "entry_action": "SKIP",
            "entry_side": "NONE",
            "entry_score": 0,
            "timing_quality": 0.0,
            "risk_flag": True,
            "reason_codes": [_rc("ERROR")],
            "summary": f"m15_entryで例外発生: {e}",
            "raw_features": {
                "momentum": _indicators.get("momentum"),
                "pullback_state": _indicators.get("pullback_state"),
                "breakout": _indicators.get("breakout"),
                "noise": _indicators.get("noise"),
                "spread": market_data_m15.get("spread"),
                **_h2_context_features,
                **_h1_features,
            },
        }
