# --------------------------------------------------
# test_phase3_h1_fetch.py
# 役割:
#   MTManager_BuildH1Data() が返すH1市場データ構造を
#   単体で確認するためのデバッグ用スクリプト
#
# 確認内容:
#   - MT5初期化が通るか
#   - H1データ取得ができるか
#   - H1市場データ辞書の主要キーが揃うか
#   - raw_features の中身が想定通り入るか
# --------------------------------------------------

import json
import os
import sys

# --------------------------------------------------
# 実行位置に依存せず Framework を import できるよう、
# このデバッグファイル基準で Src ディレクトリを import path に追加する
# --------------------------------------------------
_CURRENT_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from Framework.MTSystem.MTManager import (
    MTManager_Initialize,
    MTManager_BuildH1Data,
)


def main():
    print("========== test_phase3_h1_fetch start ==========")

    # --------------------------------------------------
    # ① MT5初期化確認
    # H1取得前提として、まずMT5へ正常接続できるか確認する
    # --------------------------------------------------
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。")
        return

    # --------------------------------------------------
    # ② H1市場データ構築
    # Phase3-1で追加したH1取得関数が返す辞書構造を確認する
    # --------------------------------------------------
    _h1Data = MTManager_BuildH1Data(_count=200)

    print("----- H1 Data Summary -----")
    print(f"symbol    : {_h1Data.get('symbol')}")
    print(f"timeframe : {_h1Data.get('timeframe')}")
    print(f"ohlc_len  : {len(_h1Data.get('ohlc', []))}")

    _indicators = _h1Data.get("indicators", {})
    _rawFeatures = _indicators.get("raw_features", {})

    print("----- H1 Raw Features Summary -----")
    print(f"close_list_len      : {len(_rawFeatures.get('close_list', []))}")
    print(f"close_diff_list_len : {len(_rawFeatures.get('close_diff_list', []))}")
    print(f"recent_momentum     : {_rawFeatures.get('recent_momentum')}")
    print(f"trend_consistency   : {_rawFeatures.get('trend_consistency')}")

    # --------------------------------------------------
    # ③ 全体構造確認用にJSON整形出力
    # 内容を目視で追いやすくするため最小限の形で表示する
    # --------------------------------------------------
    _preview = {
        "symbol": _h1Data.get("symbol"),
        "timeframe": _h1Data.get("timeframe"),
        "ohlc_len": len(_h1Data.get("ohlc", [])),
        "raw_features": {
            "close_list_len": len(_rawFeatures.get("close_list", [])),
            "close_diff_list_len": len(_rawFeatures.get("close_diff_list", [])),
            "recent_momentum": _rawFeatures.get("recent_momentum"),
            "trend_consistency": _rawFeatures.get("trend_consistency"),
        },
    }

    print("----- H1 Preview JSON -----")
    print(json.dumps(_preview, ensure_ascii=False, indent=2))

    print("========== test_phase3_h1_fetch end ==========")


if __name__ == "__main__":
    main()