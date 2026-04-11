# --------------------------------------------------
# test_phase2_m15_fetch.py
# 役割:
#   Phase2-3 で追加した M15実データ取得処理と
#   M15執行判定への接続状態を単体確認する
#
# 確認内容:
#   1. BuildMarketData() に M15 OHLC / indicators / spread が格納されるか
#   2. evaluate_h2_environment() が実データから結果を返すか
#   3. evaluate_m15_entry() にそのまま渡して結果が返るか
#
# 方針:
#   - M15判定結果の妥当性を厳密評価する段階ではない
#   - 今回は「取得」「構築」「接続確認」までを見る
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
#   H2 / M15 単体確認用に、
#   外部停止なしの最小入力を用意する
# --------------------------------------------------
def _build_dummy_external_filter_result():
    return {
        "can_trade": True
    }


# --------------------------------------------------
# M15市場データ構築確認テスト
# 役割:
#   main.py の市場データ構築において、
#   M15 OHLC / indicators / spread が正しく格納されるか確認する
# --------------------------------------------------
def test_build_market_data_m15():
    print("========== test_build_market_data_m15 ==========")

    if not MTManager_Initialize():
        print("[ERROR] MT5初期化失敗")
        return

    _market_data = BuildMarketData()
    _m15_data = _market_data["M15"]
    _m15_ohlc = _m15_data["ohlc"]
    _m15_indicators = _m15_data["indicators"]

    print("[INFO] symbol:", _m15_data["symbol"])
    print("[INFO] timeframe:", _m15_data["timeframe"])
    print("[INFO] timestamp_jst:", _m15_data["timestamp_jst"])
    print("[INFO] ohlc count:", len(_m15_ohlc))
    print("[INFO] indicators:", _m15_indicators)
    print("[INFO] spread:", _m15_data["spread"])

    # --------------------------------------------------
    # M15 OHLC が空なら受け渡し失敗
    # --------------------------------------------------
    if len(_m15_ohlc) == 0:
        print("[FAIL] BuildMarketData M15格納失敗: ohlc が空")
        return

    print("[OK] BuildMarketData M15格納成功")
    print("[OK] M15末尾データ:", _m15_ohlc[-1])

    # --------------------------------------------------
    # indicators の主要キーが存在するか確認
    # --------------------------------------------------
    _required_keys = [
        "momentum",
        "pullback_state",
        "breakout",
        "noise",
    ]

    for _key in _required_keys:
        if _key not in _m15_indicators:
            print(f"[FAIL] indicators 欠落: {_key}")
            return

    print("[OK] M15 indicators 格納成功")

    # --------------------------------------------------
    # spread が数値として取得できているか確認
    # --------------------------------------------------
    try:
        float(_m15_data["spread"])
    except Exception:
        print("[FAIL] spread が数値変換できない")
        return

    print("[OK] M15 spread 格納成功")


# --------------------------------------------------
# M15執行判定接続確認テスト
# 役割:
#   BuildMarketData で構築した H2 / M15 データを
#   h2_environment と m15_entry にそのまま流して結果を確認する
# --------------------------------------------------
def test_m15_entry_with_live_data():
    print("========== test_m15_entry_with_live_data ==========")

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

    print("[INFO] h2_environment result:")
    print(_h2_result)

    _m15_result = evaluate_m15_entry(
        market_data_m15=_market_data["M15"],
        h2_environment_result=_h2_result,
        h1_forecast_result=None,
        external_filter_result=_external_filter_result,
        thresholds=_thresholds,
    )

    print("[INFO] m15_entry result:")
    print(_m15_result)

    # --------------------------------------------------
    # 戻り値の最低限の構造確認
    # --------------------------------------------------
    _required_keys = [
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

    for _key in _required_keys:
        if _key not in _m15_result:
            print(f"[FAIL] m15_entry result 欠落: {_key}")
            return

    print("[OK] m15_entry 接続確認成功")


# --------------------------------------------------
# テスト実行入口
# 役割:
#   単体実行時に M15取得確認テストを順に流す
# --------------------------------------------------
if __name__ == "__main__":
    print("==========SGSystem Test Start==========")
    test_build_market_data_m15()
    print()
    test_m15_entry_with_live_data()
    print("==========SGSystem Test End==========")