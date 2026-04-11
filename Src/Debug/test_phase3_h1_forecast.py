# --------------------------------------------------
# test_phase3_h1_forecast.py
# 役割:
#   evaluate_h1_forecast() が返すH1予測結果構造を
#   単体で確認するためのデバッグ用スクリプト
#
# 確認内容:
#   - MT5初期化が通るか
#   - H1市場データが取得できるか
#   - H1予測結果が固定形式で返るか
#   - net_direction / confidence / predicted_path / reason_codes が確認できるか
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
from Framework.ROModule.h1_forecast import build_h1_runtime_view, evaluate_h1_forecast


def main():
    print("========== test_phase3_h1_forecast start ==========")

    # --------------------------------------------------
    # ① MT5初期化確認
    # H1市場データ取得と予測判定の前提として接続を確認する
    # --------------------------------------------------
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。")
        return

    # --------------------------------------------------
    # ② H1市場データ構築
    # Phase3-1で追加したH1データ供給を予測モジュールへ渡す
    # --------------------------------------------------
    _h1Data = MTManager_BuildH1Data(_count=200)

    # --------------------------------------------------
    # ③ H1予測用閾値を最小構成で用意する
    # 現段階では confidence 閾値のみ使えば十分
    # --------------------------------------------------
    _thresholds = {
        "h1_confidence_min": 0.65
    }

    # --------------------------------------------------
    # ④ H1予測判定実行
    # H1市場データを使って方向優位性と信頼度を確認する
    # --------------------------------------------------
    _h1ForecastResult = evaluate_h1_forecast(
        _h1_data=_h1Data,
        _thresholds=_thresholds,
    )
    _h1RuntimeView = build_h1_runtime_view(_h1ForecastResult)

    print("----- H1 Forecast Summary -----")
    print(f"module_name      : {_h1ForecastResult.get('module_name')}")
    print(f"status           : {_h1ForecastResult.get('status')}")
    print(f"forecast_status  : {_h1ForecastResult.get('forecast_status')}")
    print(f"net_direction    : {_h1ForecastResult.get('net_direction')}")
    print(f"bias_direction   : {_h1RuntimeView.get('bias_direction')}")
    print(f"bias_ready       : {_h1RuntimeView.get('bias_ready')}")
    print(f"confidence       : {_h1ForecastResult.get('confidence')}")
    print(f"predicted_len    : {len(_h1ForecastResult.get('predicted_path', []))}")
    print(f"path_type        : {_h1RuntimeView.get('predicted_path_type')}")
    print(f"reason_codes     : {_h1ForecastResult.get('reason_codes')}")

    # --------------------------------------------------
    # ⑤ 返却構造の主要部をJSON整形して表示する
    # predicted_path 全量は長くなりすぎないようそのまま表示する
    # --------------------------------------------------
    _preview = {
        "module_name": _h1ForecastResult.get("module_name"),
        "status": _h1ForecastResult.get("status"),
        "forecast_status": _h1ForecastResult.get("forecast_status"),
        "net_direction": _h1ForecastResult.get("net_direction"),
        "forecast_role": _h1RuntimeView.get("forecast_role"),
        "bias_direction": _h1RuntimeView.get("bias_direction"),
        "bias_ready": _h1RuntimeView.get("bias_ready"),
        "direction_score_long": _h1ForecastResult.get("direction_score_long"),
        "direction_score_short": _h1ForecastResult.get("direction_score_short"),
        "confidence": _h1ForecastResult.get("confidence"),
        "predicted_path": _h1ForecastResult.get("predicted_path"),
        "predicted_path_type": _h1RuntimeView.get("predicted_path_type"),
        "predicted_path_source_horizons": _h1RuntimeView.get("predicted_path_source_horizons"),
        "reason_codes": _h1ForecastResult.get("reason_codes"),
        "summary": _h1ForecastResult.get("summary"),
        "raw_features": _h1ForecastResult.get("raw_features"),
        "runtime_view": _h1RuntimeView,
    }

    print("----- H1 Forecast Preview JSON -----")
    print(json.dumps(_preview, ensure_ascii=False, indent=2))

    print("========== test_phase3_h1_forecast end ==========")


if __name__ == "__main__":
    main()
