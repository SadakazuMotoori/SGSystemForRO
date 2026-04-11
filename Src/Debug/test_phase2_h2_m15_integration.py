# --------------------------------------------------
# test_phase2_h2_m15_integration.py
# 役割:
#   Phase2-4 として、H2実データ結果を M15判定へ接続し、
#   H2 + M15 の統合フローが正しく流れるか確認する
#
# 確認内容:
#   1. BuildMarketData() で H2 / M15 の両方が構築されるか
#   2. evaluate_h2_environment() が実データで結果を返すか
#   3. evaluate_m15_entry() が H2結果を受けて結果を返すか
#   4. H2 と M15 の結果が構造的に矛盾していないか
#
# 方針:
#   - 数値の極端さ自体はこの段階では評価しない
#   - 今回は「接続」「受け渡し」「返却形式」の確認に集中する
# --------------------------------------------------

import sys
from pathlib import Path

# Debugファイル単体実行時でも Src 配下の Framework を参照できるようにする
_srcDir = Path(__file__).resolve().parents[1]
if str(_srcDir) not in sys.path:
    sys.path.insert(0, str(_srcDir))
    
from main import BuildMarketData, LoadThresholds
from Framework.MTSystem.MTManager import MTManager_Initialize
from Framework.ROModule.h2_environment import evaluate_h2_environment
from Framework.ROModule.m15_entry import evaluate_m15_entry


# --------------------------------------------------
# 外部停止条件ダミー
# 役割:
#   H2 / M15 統合確認用に、
#   外部停止なしの最小入力を用意する
# --------------------------------------------------
def _build_dummy_external_filter_result():
    return {
        "can_trade": True
    }


# --------------------------------------------------
# H2 + M15 統合確認テスト
# 役割:
#   実データで H2 → M15 の順に評価し、
#   結果が構造的に正しいか確認する
# --------------------------------------------------
def test_h2_m15_integration():
    print("========== test_h2_m15_integration ==========")

    if not MTManager_Initialize():
        print("[ERROR] MT5初期化失敗")
        return

    _market_data = BuildMarketData()
    _thresholds = LoadThresholds()
    _external_filter_result = _build_dummy_external_filter_result()

    _h2_result = evaluate_h2_environment(
        market_data_h2=_market_data["H2"],
        external_filter_result=_external_filter_result,
        thresholds=_thresholds,
    )

    _m15_result = evaluate_m15_entry(
        market_data_m15=_market_data["M15"],
        h2_environment_result=_h2_result,
        h1_forecast_result=None,
        external_filter_result=_external_filter_result,
        thresholds=_thresholds,
    )

    print("[INFO] h2_environment result:")
    print(_h2_result)
    print("[INFO] m15_entry result:")
    print(_m15_result)

    # --------------------------------------------------
    # H2戻り値の最低限構造確認
    # --------------------------------------------------
    _required_h2_keys = [
        "module_name",
        "timestamp_jst",
        "status",
        "env_direction",
        "env_score",
        "trend_strength",
        "reason_codes",
        "summary",
        "raw_features",
    ]

    for _key in _required_h2_keys:
        if _key not in _h2_result:
            print(f"[FAIL] h2_environment result 欠落: {_key}")
            return

    # --------------------------------------------------
    # M15戻り値の最低限構造確認
    # --------------------------------------------------
    _required_m15_keys = [
        "module_name",
        "timestamp_jst",
        "status",
        "entry_action",
        "entry_side",
        "entry_score",
        "timing_quality",
        "risk_flag",
        "reason_codes",
        "summary",
        "raw_features",
    ]

    for _key in _required_m15_keys:
        if _key not in _m15_result:
            print(f"[FAIL] m15_entry result 欠落: {_key}")
            return

    # --------------------------------------------------
    # H2 と M15 の構造整合確認
    # 役割:
    #   上位足の結論と下位足の返却が、
    #   少なくとも明らかに破綻していないかを見る
    # --------------------------------------------------
    _env_direction = _h2_result["env_direction"]
    _entry_action = _m15_result["entry_action"]
    _entry_side = _m15_result["entry_side"]

    # H2が NO_TRADE なら、M15 は ENTER してはいけない
    if _env_direction == "NO_TRADE" and _entry_action == "ENTER":
        print("[FAIL] H2がNO_TRADEなのにM15がENTERしている")
        return

    # H2が LONG_ONLY なら、M15 の ENTER side は LONG であるべき
    if _env_direction == "LONG_ONLY" and _entry_action == "ENTER" and _entry_side != "LONG":
        print("[FAIL] H2がLONG_ONLYなのにM15のENTER sideがLONGではない")
        return

    # H2が SHORT_ONLY なら、M15 の ENTER side は SHORT であるべき
    if _env_direction == "SHORT_ONLY" and _entry_action == "ENTER" and _entry_side != "SHORT":
        print("[FAIL] H2がSHORT_ONLYなのにM15のENTER sideがSHORTではない")
        return

    print("[OK] H2 + M15 統合確認成功")


# --------------------------------------------------
# テスト実行入口
# 役割:
#   単体実行時に H2 + M15 統合確認テストを流す
# --------------------------------------------------
if __name__ == "__main__":
    print("==========SGSystem Test Start==========")
    test_h2_m15_integration()
    print("==========SGSystem Test End==========")