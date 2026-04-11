# --------------------------------------------------
# test_phase2_h2_fetch.py
# 役割:
#   Phase2-1 で追加した H2実データ取得処理を単体確認する
#
# 確認内容:
#   1. MTManager_GetH2Rates() が正常にH2レートを返すか
#   2. BuildMarketData() に H2 OHLC が格納されるか
#
# 方針:
#   - 判定ロジックの正しさはまだ見ない
#   - H2指標の妥当性もまだ見ない
#   - 今回は「取得」と「受け渡し」のみ確認する
# --------------------------------------------------

import sys
from pathlib import Path

# Debugファイル単体実行時でも Src 配下の Framework を参照できるようにする
_srcDir = Path(__file__).resolve().parents[1]
if str(_srcDir) not in sys.path:
    sys.path.insert(0, str(_srcDir))

from main import BuildMarketData, LoadThresholds
from Framework.MTSystem.MTManager import (
    MTManager_Initialize,
    MTManager_GetH2Rates,
)
from Framework.ROModule.h2_environment import evaluate_h2_environment

# --------------------------------------------------
# 外部停止条件ダミー
# 役割:
#   h2_environment 単体確認用に、
#   外部停止なしの最小入力を用意する
# --------------------------------------------------
def _build_dummy_external_filter_result():
    return {
        "can_trade": True
    }

# --------------------------------------------------
# H2レート取得単体テスト
# 役割:
#   MT5から2時間足レートが直接取得できるか確認する
# --------------------------------------------------
def test_mtmanager_get_h2_rates():
    print("========== test_mtmanager_get_h2_rates ==========")

    if not MTManager_Initialize():
        print("[ERROR] MT5初期化失敗")
        return

    _rates = MTManager_GetH2Rates(_count=10)

    # --------------------------------------------------
    # 取得結果が None または空なら失敗
    # --------------------------------------------------
    if _rates is None:
        print("[FAIL] H2データ取得失敗: None が返却された")
        return

    if len(_rates) == 0:
        print("[FAIL] H2データ取得失敗: 0件")
        return

    print(f"[OK] H2取得件数: {len(_rates)}")
    print("[OK] H2先頭データ:", _rates[0])
    print("[OK] H2末尾データ:", _rates[-1])


# --------------------------------------------------
# BuildMarketData の H2格納確認テスト
# 役割:
#   main.py の市場データ構築において、
#   H2 OHLC が正しく格納されるか確認する
# --------------------------------------------------
def test_build_market_data_h2():
    print("========== test_build_market_data_h2 ==========")

    if not MTManager_Initialize():
        print("[ERROR] MT5初期化失敗")
        return

    _marketData = BuildMarketData()
    _h2Data = _marketData["H2"]
    _h2Ohlc = _h2Data["ohlc"]

    print("[INFO] symbol:", _h2Data["symbol"])
    print("[INFO] timeframe:", _h2Data["timeframe"])
    print("[INFO] timestamp_jst:", _h2Data["timestamp_jst"])
    print("[INFO] ohlc count:", len(_h2Ohlc))

    # --------------------------------------------------
    # H2 OHLC が空なら受け渡し失敗
    # --------------------------------------------------
    if len(_h2Ohlc) == 0:
        print("[FAIL] BuildMarketData H2格納失敗: ohlc が空")
        return

    print("[OK] BuildMarketData H2格納成功")
    print("[OK] H2末尾データ:", _h2Ohlc[-1])

# --------------------------------------------------
# h2_environment 接続確認テスト
# 役割:
#   BuildMarketData で構築したH2データを
#   h2_environment にそのまま流して結果を確認する
# --------------------------------------------------
def test_h2_environment_with_live_h2_data():
    print("========== test_h2_environment_with_live_h2_data ==========")

    if not MTManager_Initialize():
        print("[ERROR] MT5初期化失敗")
        return

    _marketData = BuildMarketData()
    _thresholds = LoadThresholds()

    _result = evaluate_h2_environment(
        market_data_h2=_marketData["H2"],
        external_filter_result=_build_dummy_external_filter_result(),
        thresholds=_thresholds,
    )

    print("[INFO] h2_environment result:")
    print(_result)

    # --------------------------------------------------
    # 戻り値の最低限の構造確認
    # --------------------------------------------------
    _requiredKeys = [
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

    for _key in _requiredKeys:
        if _key not in _result:
            print(f"[FAIL] h2_environment result 欠落: {_key}")
            return

    print("[OK] h2_environment 接続確認成功")

# --------------------------------------------------
# テスト実行入口
# 役割:
#   単体実行時に H2取得確認テストを順に流す
# --------------------------------------------------
if __name__ == "__main__":
    print("==========SGSystem Test Start==========")
    test_mtmanager_get_h2_rates()
    print()
    test_build_market_data_h2()
    print()
    test_h2_environment_with_live_h2_data()
    print("==========SGSystem Test End==========")