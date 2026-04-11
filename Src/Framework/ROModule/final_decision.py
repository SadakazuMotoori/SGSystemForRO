# --------------------------------------------------
# final_decision.py
# 役割:
#   external_filter / h2_environment / h1_forecast / m15_entry
#   の結果を統合し、最終的な売買候補を返す
#
# 設計方針:
#   - 統合優先順位は
#       ① 外部停止条件
#       ② 2時間足の許可帯
#       ③ 1時間足予測（Phase 1では未接続許容）
#       ④ 15分足執行判定
#   - 上位足の結論を下位足が覆さない
#   - 例外時は安全側に倒して NO_TRADE を返す
#   - 戻り値は常に固定形式で返す
# --------------------------------------------------

from Framework.ROModule.h1_forecast import evaluate_h1_alignment
from Framework.ROModule.h2_environment_contract import (
    resolve_h2_direction,
    resolve_h2_trend_strength,
)


def _rc(_suffix: str) -> str:
    return f"FINAL_DECISION_{_suffix}"


# --------------------------------------------------
# 内部関数:
#   decision_score を 0 ~ 100 に丸める
#
# 役割:
#   最終スコアの出力を固定範囲に収める
# --------------------------------------------------
def _clamp_score(_value):
    try:
        _value = int(round(_value))
    except Exception:
        _value = 0

    if _value < 0:
        return 0
    if _value > 100:
        return 100
    return _value


# --------------------------------------------------
# 内部関数:
#   H1予測とH2許可帯の整合性を確認する
#
# 役割:
#   shared helper evaluate_h1_alignment() を通して、
#   H1 tactical bias が上位足と整合するかを返す
#
# 戻り値:
#   True  = 整合
#   False = 不整合
#   None  = H1未接続または判定保留
# --------------------------------------------------
def _check_h1_alignment(_h2_environment_result, _h1_forecast_result, _thresholds):
    _envDirection = resolve_h2_direction(_h2_environment_result)
    _alignment_result = evaluate_h1_alignment(
        _h1_forecast_result=_h1_forecast_result,
        _env_direction=_envDirection,
        _thresholds=_thresholds,
    )
    _alignment = _alignment_result["alignment"]

    if _alignment == "UNAVAILABLE":
        return None, []

    if _alignment == "NEUTRAL_OR_SKIPPED":
        return None, [_rc("H1_ALIGNMENT_NEUTRAL")]

    if _alignment == "LOW_CONFIDENCE":
        return None, [_rc("H1_ALIGNMENT_LOW_CONFIDENCE")]

    if _alignment == "ALIGNED" and _envDirection == "LONG_ONLY":
        return True, [
            _rc("H1_ALIGNMENT_CONFIDENCE_OK"),
            _rc("H1_ALIGNMENT_ALIGNED_LONG"),
        ]

    if _alignment == "ALIGNED" and _envDirection == "SHORT_ONLY":
        return True, [
            _rc("H1_ALIGNMENT_CONFIDENCE_OK"),
            _rc("H1_ALIGNMENT_ALIGNED_SHORT"),
        ]

    if _alignment == "CONFLICT":
        return False, [
            _rc("H1_ALIGNMENT_CONFIDENCE_OK"),
            _rc("H1_ALIGNMENT_MISMATCH"),
        ]
    return None, []


# --------------------------------------------------
# メイン関数:
#   各モジュールの結果を統合し、最終判断を返す
#
# 入力:
#   external_filter_result:
#       外部停止条件の結果
#   h2_environment_result:
#       2時間足の方向許可結果
#   h1_forecast_result:
#       1時間足予測結果（Phase 1では None 許容）
#   m15_entry_result:
#       15分足執行判定結果
#   thresholds:
#       h1_confidence_min などの設定値
#
# 出力:
#   {
#       "module_name": "final_decision",
#       "timestamp_jst": str,
#       "status": "OK" | "ERROR",
#       "final_action": "ENTER_LONG" | "ENTER_SHORT" | "WAIT" | "SKIP" | "EXIT" | "NO_TRADE",
#       "decision_score": int,
#       "approved": bool,
#       "reason_codes": list[str],
#       "summary": str,
#       "details": dict,
#   }
# --------------------------------------------------
def evaluate_final_decision(
    external_filter_result: dict,
    h2_environment_result: dict,
    h1_forecast_result: dict | None,
    m15_entry_result: dict,
    thresholds: dict,
) -> dict:
    _timestampJST = (
        m15_entry_result.get("timestamp_jst")
        or h2_environment_result.get("timestamp_jst")
        or external_filter_result.get("timestamp_jst")
        or ""
    )

    try:
        _reasonCodes = []
        _decisionScore = 0

        _canTrade = external_filter_result.get("can_trade")
        _envDirection = resolve_h2_direction(h2_environment_result)
        _entryAction = m15_entry_result.get("entry_action")
        _entrySide = m15_entry_result.get("entry_side")
        _entryScore = m15_entry_result.get("entry_score", 0)
        _trendStrength = resolve_h2_trend_strength(h2_environment_result)

        try:
            _entryScore = int(_entryScore)
        except Exception:
            _entryScore = 0

        # --------------------------------------------------
        # ① 外部停止条件を最優先で確認
        # 外部停止条件ONなら final_action は NO_TRADE
        # --------------------------------------------------
        if _canTrade is False:
            _reasonCodes.append(_rc("EXTERNAL_FILTER_ON"))

            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "NO_TRADE",
                "decision_score": 0,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "外部停止条件が有効のため取引停止",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        _reasonCodes.append(_rc("EXTERNAL_FILTER_OFF"))

        # --------------------------------------------------
        # ② 2時間足の許可帯を確認
        # H2が NO_TRADE なら final_action は NO_TRADE
        # --------------------------------------------------
        if _envDirection == "NO_TRADE":
            _reasonCodes.append(_rc("H2_NO_TRADE"))

            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "NO_TRADE",
                "decision_score": 0,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "2時間足が方向を許可していないため取引停止",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        if _envDirection == "LONG_ONLY":
            _reasonCodes.append(_rc("H2_LONG_ONLY"))
        elif _envDirection == "SHORT_ONLY":
            _reasonCodes.append(_rc("H2_SHORT_ONLY"))

        # --------------------------------------------------
        # ③ 15分足の EXIT を優先確認
        # シナリオ崩れ・急変時は撤退判断を優先する
        # --------------------------------------------------
        if _entryAction == "EXIT":
            _reasonCodes.append(_rc("M15_EXIT"))

            _decisionScore = _clamp_score(_entryScore)

            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "EXIT",
                "decision_score": _decisionScore,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "15分足でシナリオ崩れを検出したため撤退",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        # --------------------------------------------------
        # ④ H1予測がある場合のみ整合確認
        # Phase 1 では未接続(None)を許容する
        # --------------------------------------------------
        _h1Aligned, _h1ReasonCodes = _check_h1_alignment(
            h2_environment_result,
            h1_forecast_result,
            thresholds,
        )
        _reasonCodes.extend(_h1ReasonCodes)

        if _h1Aligned is False:
            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "WAIT",
                "decision_score": 0,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "1時間足予測が上位足と整合しないため待機",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        if _h1Aligned is None:
            _reasonCodes.append(_rc("H1_ALIGNMENT_SKIPPED"))

        # --------------------------------------------------
        # ⑤ 15分足の執行判定を最終行動へ変換
        # ENTER / WAIT / SKIP を最終アクションへ落とし込む
        # --------------------------------------------------
        if _entryAction == "ENTER":
            # ----------------------------------------------
            # H2がLONG_ONLY かつ M15がLONG なら ENTER_LONG
            # H2がSHORT_ONLY かつ M15がSHORT なら ENTER_SHORT
            # それ以外は方向不整合として SKIP
            # ----------------------------------------------
            if _envDirection == "LONG_ONLY" and _entrySide == "LONG":
                _reasonCodes.append(_rc("M15_ENTER_LONG"))
                _decisionScore = _clamp_score((_entryScore * 0.7) + (_trendStrength * 30.0))

                return {
                    "module_name": "final_decision",
                    "timestamp_jst": _timestampJST,
                    "status": "OK",
                    "final_action": "ENTER_LONG",
                    "decision_score": _decisionScore,
                    "approved": True,
                    "reason_codes": _reasonCodes,
                    "summary": "買い条件が揃ったためロング候補を承認",
                    "details": {
                        "external_filter_result": external_filter_result,
                        "h2_environment_result": h2_environment_result,
                        "h1_forecast_result": h1_forecast_result,
                        "m15_entry_result": m15_entry_result,
                    },
                }

            if _envDirection == "SHORT_ONLY" and _entrySide == "SHORT":
                _reasonCodes.append(_rc("M15_ENTER_SHORT"))
                _decisionScore = _clamp_score((_entryScore * 0.7) + (_trendStrength * 30.0))

                return {
                    "module_name": "final_decision",
                    "timestamp_jst": _timestampJST,
                    "status": "OK",
                    "final_action": "ENTER_SHORT",
                    "decision_score": _decisionScore,
                    "approved": True,
                    "reason_codes": _reasonCodes,
                    "summary": "売り条件が揃ったためショート候補を承認",
                    "details": {
                        "external_filter_result": external_filter_result,
                        "h2_environment_result": h2_environment_result,
                        "h1_forecast_result": h1_forecast_result,
                        "m15_entry_result": m15_entry_result,
                    },
                }

            _reasonCodes.append(_rc("M15_ENTRY_SIDE_MISMATCH"))

            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "SKIP",
                "decision_score": 0,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "執行方向が上位足と不整合のため見送り",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        # --------------------------------------------------
        # ⑥ M15がWAITなら WAIT
        # --------------------------------------------------
        if _entryAction == "WAIT":
            _reasonCodes.append(_rc("M15_WAIT"))

            _decisionScore = _clamp_score(_entryScore * 0.5)

            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "WAIT",
                "decision_score": _decisionScore,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "方向性はあるが執行タイミング待ち",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        # --------------------------------------------------
        # ⑦ M15がSKIPなら SKIP
        # --------------------------------------------------
        if _entryAction == "SKIP":
            _reasonCodes.append(_rc("M15_SKIP"))

            return {
                "module_name": "final_decision",
                "timestamp_jst": _timestampJST,
                "status": "OK",
                "final_action": "SKIP",
                "decision_score": 0,
                "approved": False,
                "reason_codes": _reasonCodes,
                "summary": "15分足の執行条件が弱いため見送り",
                "details": {
                    "external_filter_result": external_filter_result,
                    "h2_environment_result": h2_environment_result,
                    "h1_forecast_result": h1_forecast_result,
                    "m15_entry_result": m15_entry_result,
                },
            }

        # --------------------------------------------------
        # ⑧ 想定外の entry_action は安全側で SKIP
        # --------------------------------------------------
        _reasonCodes.append(_rc("UNKNOWN_ENTRY_ACTION"))

        return {
            "module_name": "final_decision",
            "timestamp_jst": _timestampJST,
            "status": "OK",
            "final_action": "SKIP",
            "decision_score": 0,
            "approved": False,
            "reason_codes": _reasonCodes,
            "summary": "執行判定が不明のため安全側で見送り",
            "details": {
                "external_filter_result": external_filter_result,
                "h2_environment_result": h2_environment_result,
                "h1_forecast_result": h1_forecast_result,
                "m15_entry_result": m15_entry_result,
            },
        }

    except Exception as e:
        # --------------------------------------------------
        # 例外時の保険
        # 統合判定で例外が出た場合は、
        # 安全側に倒して NO_TRADE を返す
        # --------------------------------------------------
        return {
            "module_name": "final_decision",
            "timestamp_jst": _timestampJST,
            "status": "ERROR",
            "final_action": "NO_TRADE",
            "decision_score": 0,
            "approved": False,
            "reason_codes": [_rc("ERROR")],
            "summary": f"final_decisionで例外発生: {e}",
            "details": {
                "external_filter_result": external_filter_result,
                "h2_environment_result": h2_environment_result,
                "h1_forecast_result": h1_forecast_result,
                "m15_entry_result": m15_entry_result,
            },
        }
