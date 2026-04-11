# --------------------------------------------------
# external_filter.py
# 役割:
#   取引判定の最上流で、今この時点でそもそも取引を止めるべきかを判定する
#
# 設計方針:
#   - 外部要因は加点ではなく停止条件として扱う
#   - 戻り値は常に固定形式で返す
#   - 判定不能や例外時も status / can_trade / reason_codes を返す
# --------------------------------------------------

from typing import Any


def _rc(_suffix: str) -> str:
    return f"EXTERNAL_FILTER_{_suffix}"


# --------------------------------------------------
# 内部関数:
#   bool系の値を安全に真偽判定する
#
# 役割:
#   external_context 側の値が True/False 前提でも、
#   None や未定義混入時に落ちにくくする
# --------------------------------------------------
def _is_true(_value: Any) -> bool:
    return _value is True


# --------------------------------------------------
# 内部関数:
#   spread の異常判定を行う
#
# 入力:
#   _market_data["spread"]
#   thresholds["spread_max"]
#
# 役割:
#   スプレッドが閾値超過なら取引停止理由に加える
# --------------------------------------------------
def _check_spread_too_wide(_market_data: dict, _thresholds: dict) -> bool:
    _spread = _market_data.get("spread")
    _spreadMax = _thresholds.get("spread_max")

    if _spread is None:
        return False

    if _spreadMax is None:
        return False

    try:
        return float(_spread) > float(_spreadMax)
    except Exception:
        return False


# --------------------------------------------------
# 内部関数:
#   異常ボラの判定を行う
#
# 入力:
#   external_context["abnormal_volatility"]
#
# 役割:
#   現段階ではフラグ入力のみ対応
#   将来的には market_data 側の実測ボラとも接続可能
# --------------------------------------------------
def _check_abnormal_volatility(_external_context: dict) -> bool:
    return _is_true(_external_context.get("abnormal_volatility"))


# --------------------------------------------------
# メイン関数:
#   外部停止条件を判定する
#
# 入力:
#   market_data:
#       主に M15 データを想定
#   external_context:
#       経済指標、要人発言、地政学、通信異常などの外部文脈
#   system_context:
#       現在の回合やポジション状態などの内部文脈
#   thresholds:
#       spread_max などの設定値
#
# 出力:
#   {
#       "module_name": "external_filter",
#       "timestamp_jst": str,
#       "status": "OK" | "ERROR",
#       "filter_status": "ON" | "OFF",
#       "can_trade": bool,
#       "reason_codes": list[str],
#       "summary": str,
#       "raw_features": dict,
#   }
# --------------------------------------------------
def evaluate_external_filter(
    market_data: dict,
    external_context: dict,
    system_context: dict,
    thresholds: dict,
) -> dict:
    _timestampJST = market_data.get(
        "timestamp_jst",
        system_context.get("latest_update_jst", ""),
    )

    _reasonCodes = []

    try:
        # --------------------------------------------------
        # ① 高影響指標が近いか
        # 指標直前は停止条件として扱う
        # --------------------------------------------------
        if _is_true(external_context.get("high_impact_event_soon")):
            _reasonCodes.append(_rc("HIGH_IMPACT_EVENT_SOON"))

        # --------------------------------------------------
        # ② 中央銀行・要人発言があるか
        # 急変動の原因になりやすいため停止条件として扱う
        # --------------------------------------------------
        if _is_true(external_context.get("central_bank_speech")):
            _reasonCodes.append(_rc("CENTRAL_BANK_SPEECH"))

        # --------------------------------------------------
        # ③ 地政学アラートがあるか
        # 突発的変動要因として停止条件に加える
        # --------------------------------------------------
        if _is_true(external_context.get("geopolitical_alert")):
            _reasonCodes.append(_rc("GEOPOLITICAL_ALERT"))

        # --------------------------------------------------
        # ④ データ取得系に異常があるか
        # 通信・配信異常時は判定の前提が壊れるため停止する
        # --------------------------------------------------
        if _is_true(external_context.get("data_feed_error")):
            _reasonCodes.append(_rc("DATA_FEED_ERROR"))

        # --------------------------------------------------
        # ⑤ 異常ボラか
        # 現段階では external_context のフラグ入力のみ対応
        # --------------------------------------------------
        if _check_abnormal_volatility(external_context):
            _reasonCodes.append(_rc("ABNORMAL_VOLATILITY"))

        # --------------------------------------------------
        # ⑥ スプレッドが閾値超過か
        # 実取引コスト悪化・異常状態の兆候として停止条件に加える
        # --------------------------------------------------
        if _check_spread_too_wide(market_data, thresholds):
            _reasonCodes.append(_rc("SPREAD_TOO_WIDE"))

        # --------------------------------------------------
        # ⑦ 最終判定
        # 停止理由が1つでもあれば filter_status=ON / can_trade=False
        # --------------------------------------------------
        _canTrade = len(_reasonCodes) == 0
        _filterStatus = "OFF" if _canTrade else "ON"

        if _canTrade:
            _summary = "外部停止条件なし"
        else:
            _summary = "外部停止条件により取引停止"

        return {
            "module_name": "external_filter",
            "timestamp_jst": _timestampJST,
            "status": "OK",
            "filter_status": _filterStatus,
            "can_trade": _canTrade,
            "reason_codes": _reasonCodes,
            "summary": _summary,
            "raw_features": {
                "high_impact_event_soon": external_context.get("high_impact_event_soon"),
                "central_bank_speech": external_context.get("central_bank_speech"),
                "geopolitical_alert": external_context.get("geopolitical_alert"),
                "data_feed_error": external_context.get("data_feed_error"),
                "abnormal_volatility": external_context.get("abnormal_volatility"),
                "spread": market_data.get("spread"),
                "spread_max": thresholds.get("spread_max"),
                "round_id": system_context.get("round_id"),
                "position_state": system_context.get("position_state"),
            },
        }

    except Exception as e:
        # --------------------------------------------------
        # 例外時の保険
        # 外部停止条件モジュールで例外が出た場合は、
        # 安全側に倒して取引停止を返す
        # --------------------------------------------------
        return {
            "module_name": "external_filter",
            "timestamp_jst": _timestampJST,
            "status": "ERROR",
            "filter_status": "ON",
            "can_trade": False,
            "reason_codes": [_rc("ERROR")],
            "summary": f"external_filterで例外発生: {e}",
            "raw_features": {
                "high_impact_event_soon": external_context.get("high_impact_event_soon"),
                "central_bank_speech": external_context.get("central_bank_speech"),
                "geopolitical_alert": external_context.get("geopolitical_alert"),
                "data_feed_error": external_context.get("data_feed_error"),
                "abnormal_volatility": external_context.get("abnormal_volatility"),
                "spread": market_data.get("spread"),
                "spread_max": thresholds.get("spread_max"),
                "round_id": system_context.get("round_id"),
                "position_state": system_context.get("position_state"),
            },
        }
