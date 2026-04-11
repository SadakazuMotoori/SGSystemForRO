# --------------------------------------------------
# test_phase3_h2_h1_m15_integration.py
# 役割:
#   H2 / H1 / M15 / final_decision の統合フローを
#   実データで確認するためのデバッグ用スクリプト
#
# 確認内容:
#   - H2環境認識が返るか
#   - H1予測が返るか
#   - M15執行判定が返るか
#   - final_decision が最終アクションへ落とし込めるか
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
    MTManager_BuildH2Data,
    MTManager_BuildH1Data,
    MTManager_BuildM15Data,
)
from Framework.ROModule.external_filter import evaluate_external_filter
from Framework.ROModule.h2_environment import evaluate_h2_environment
from Framework.ROModule.h1_forecast import build_h1_runtime_view, evaluate_h1_forecast
from Framework.ROModule.m15_entry import evaluate_m15_entry
from Framework.ROModule.final_decision import evaluate_final_decision


def main():
    print("========== test_phase3_h2_h1_m15_integration start ==========")

    # --------------------------------------------------
    # ① MT5初期化確認
    # 各時間足の実データ取得前提として接続確認を行う
    # --------------------------------------------------
    if not MTManager_Initialize():
        print("[ERROR] MT5初期化に失敗しました。")
        return

    # --------------------------------------------------
    # ② 閾値を最小構成で定義する
    # 現行 main.py 相当の判定に必要な値だけをここで持つ
    # --------------------------------------------------
    _thresholds = {
        "spread_max": 0.20,
        "adx_min": 20.0,
        "trend_strength_min": 0.55,
        "m15_entry_score_min": 70,
        "m15_noise_max": 0.40,
        "h1_confidence_min": 0.65,
    }

    # --------------------------------------------------
    # ③ 外部文脈は安全側に寄せず、通常状態の固定値で与える
    # 統合フロー確認が主目的のため停止条件は今回は切る
    # --------------------------------------------------
    _externalContext = {
        "high_impact_event_soon": False,
        "central_bank_speech": False,
        "geopolitical_alert": False,
        "data_feed_error": False,
        "abnormal_volatility": False,
    }

    _systemContext = {
        "round_id": "TEST_PHASE3_INTEGRATION",
        "latest_update_jst": "TEST",
        "last_decision": None,
        "last_entry_result": None,
        "position_state": "FLAT",
    }

    # --------------------------------------------------
    # ④ 各時間足の市場データを構築する
    # H2 / H1 / M15 をそれぞれ実データで取得する
    # --------------------------------------------------
    _h2Data = MTManager_BuildH2Data(_count=200)
    _h1Data = MTManager_BuildH1Data(_count=200)
    _m15Data = MTManager_BuildM15Data(_count=200)

    # --------------------------------------------------
    # ⑤ main.py と同じ形に近づけるため timestamp_jst を補う
    # MTManager返却値だけでは持っていないためテスト側で仮設定する
    # --------------------------------------------------
    _h2Data["timestamp_jst"] = "TEST_H2"
    _h1Data["timestamp_jst"] = "TEST_H1"
    _m15Data["timestamp_jst"] = "TEST_M15"

    # --------------------------------------------------
    # ⑥ external_filter 判定
    # 最終統合の最上流条件として先に評価する
    # --------------------------------------------------
    _externalFilterResult = evaluate_external_filter(
        market_data=_m15Data,
        external_context=_externalContext,
        system_context=_systemContext,
        thresholds=_thresholds,
    )

    # --------------------------------------------------
    # ⑦ H2環境認識
    # 2時間足の方向許可帯を判定する
    # --------------------------------------------------
    _h2EnvironmentResult = evaluate_h2_environment(
        market_data_h2=_h2Data,
        external_filter_result=_externalFilterResult,
        thresholds=_thresholds,
    )

    # --------------------------------------------------
    # ⑧ H1予測判定
    # 1時間足の方向優位性と信頼度を判定する
    # --------------------------------------------------
    _h1ForecastResult = evaluate_h1_forecast(
        _h1_data=_h1Data,
        _thresholds=_thresholds,
    )
    _h1RuntimeView = build_h1_runtime_view(_h1ForecastResult)

    # --------------------------------------------------
    # ⑨ M15執行判定
    # H2方向許可帯を前提に執行タイミングを判定する
    # --------------------------------------------------
    _m15EntryResult = evaluate_m15_entry(
        market_data_m15=_m15Data,
        h2_environment_result=_h2EnvironmentResult,
        h1_forecast_result=_h1ForecastResult,
        external_filter_result=_externalFilterResult,
        thresholds=_thresholds,
    )

    # --------------------------------------------------
    # ⑩ 最終統合判定
    # H2 / H1 / M15 の結果を統合して最終アクションを確認する
    # --------------------------------------------------
    _finalDecisionResult = evaluate_final_decision(
        external_filter_result=_externalFilterResult,
        h2_environment_result=_h2EnvironmentResult,
        h1_forecast_result=_h1ForecastResult,
        m15_entry_result=_m15EntryResult,
        thresholds=_thresholds,
    )

    print("----- Integration Summary -----")
    print(f"H2 env_direction      : {_h2EnvironmentResult.get('env_direction')}")
    print(f"H1 net_direction      : {_h1ForecastResult.get('net_direction')}")
    print(f"H1 bias_direction     : {_h1RuntimeView.get('bias_direction')}")
    print(f"H1 bias_ready         : {_h1RuntimeView.get('bias_ready')}")
    print(f"H1 confidence         : {_h1ForecastResult.get('confidence')}")
    print(f"H1 path_type          : {_h1RuntimeView.get('predicted_path_type')}")
    print(f"M15 entry_action      : {_m15EntryResult.get('entry_action')}")
    print(f"M15 entry_side        : {_m15EntryResult.get('entry_side')}")
    print(f"Final final_action    : {_finalDecisionResult.get('final_action')}")
    print(f"Final approved        : {_finalDecisionResult.get('approved')}")
    print(f"Final reason_codes    : {_finalDecisionResult.get('reason_codes')}")

    _preview = {
        "external_filter_result": _externalFilterResult,
        "h2_environment_result": _h2EnvironmentResult,
        "h1_forecast_result": _h1ForecastResult,
        "h1_runtime_view": _h1RuntimeView,
        "m15_entry_result": _m15EntryResult,
        "final_decision_result": _finalDecisionResult,
    }

    print("----- Integration Preview JSON -----")
    print(json.dumps(_preview, ensure_ascii=False, indent=2))

    print("========== test_phase3_h2_h1_m15_integration end ==========")


if __name__ == "__main__":
    main()
