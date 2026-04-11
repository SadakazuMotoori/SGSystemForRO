# SGSystem_引継ぎ資料

更新日: 2026-04-11 JST

## 1. 先に押さえるべき要点
現時点の SGSystem は、完成済みの自動売買システムではなく、

- 上位足の方向許可帯を決める
- 数時間先のバイアスを確認する
- 15 分足で執行タイミングを絞る
- 外部要因がある時は止める

という判断系の実装が中心である。

特に重要なのは次の 4 点。

1. realtime の本体は `main.py` ではなく `RealtimeFlow.py`
2. H2 は「中期予測」ではなく「regime filter」として整理済み
3. H1 runtime は multi-horizon patch mixer が主系統
4. execution / position management は未実装

## 2. 現在の主なファイル構成
まず見るべきファイルは以下。

### 2.1 realtime 本体
- [main.py](../Src/main.py)
  - 互換エントリポイント
  - 実体は薄い再エクスポート
- [RealtimeFlow.py](../Src/Framework/RealtimeFlow.py)
  - realtime loop 本体
  - H2 / H1 / M15 の更新制御
  - M15 `predicted_path` ゲート
  - `final_decision` 更新
- [MTManager.py](../Src/Framework/MTSystem/MTManager.py)
  - confirmed bar データ構築

### 2.2 判断モジュール
- [external_filter.py](../Src/Framework/ROModule/external_filter.py)
- [h2_environment.py](../Src/Framework/ROModule/h2_environment.py)
- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)
- [m15_entry.py](../Src/Framework/ROModule/m15_entry.py)
- [final_decision.py](../Src/Framework/ROModule/final_decision.py)

### 2.3 旧バッチ系・参照用
- [SGFramework.py](../Src/Framework/SGFramework.py)
  - 旧来の一括パイプライン
  - realtime 主経路ではないが、構成把握とテストでまだ有用
- [train_h1_lstm_regressor.py](../Src/Framework/ForecastSystem/train_h1_lstm_regressor.py)
- [analyze_h1_lstm_predictions.py](../Src/Framework/ForecastSystem/analyze_h1_lstm_predictions.py)
  - 旧 LSTM 系。runtime 主系統ではない

## 3. realtime の見方
`main.py` を追いかけるより、`RealtimeFlow.py` を起点に読む方が早い。

現在の loop は次の考え方で整理されている。

1. H2 / H1 / M15 の confirmed bar データを取る
2. `system_context` と `external_context` を更新する
3. `external_filter` を評価する
4. H2 confirmed bar が更新された時だけ H2 を再評価する
5. H1 confirmed bar が更新された時だけ H1 を再評価する
6. M15 confirmed bar が更新された時だけ M15 と `final_decision` を再評価する

補足:

- 1 秒 loop だが、全モジュールを毎秒全再計算する設計ではない
- `predicted_path` による M15 補助ゲートは M15 更新時だけ評価する

## 4. H2 の最新整理
今回の更新で H2 はかなり意味づけが整理された。

### 4.1 新しい役割
H2 の役割は「中期トレンドを当てる」ことではなく、

- 上位足としてどちら方向を許可するかを決める
- `LONG_ONLY / SHORT_ONLY / NO_TRADE` を返す

という regime filter である。

### 4.2 現在の入力
`h2_environment.py` は主に以下を使う。

- `ma_short`
- `ma_long`
- `ma_slope`
- `adx`
- `swing_structure`

### 4.3 出力
旧互換出力は維持されている。

- `env_direction`
- `env_score`
- `trend_strength`
- `reason_codes`

加えて、新しい regime 系出力が追加されている。

- `regime_direction`
- `regime_score`
- `regime_quality`
- `regime_components`
- `regime_reason_codes`

### 4.4 閾値互換
[thresholds.json](../Asset/Config/thresholds.json) は旧キーに加え、H2 regime 用キーを持つ。

- `adx_min`
- `trend_strength_min`
- `h2_regime_adx_min`
- `h2_regime_strength_min`
- `h2_regime_score_min`

`h2_environment.py` は Phase 1 の互換として、新旧どちらのキーでも同じ意味で読めるようにしてある。

### 4.5 downstream 反映状況
以下は `regime_*` を優先して読む準備が入っている。

- [RealtimeFlow.py](../Src/Framework/RealtimeFlow.py)
- [final_decision.py](../Src/Framework/ROModule/final_decision.py)
- [m15_entry.py](../Src/Framework/ROModule/m15_entry.py)
- [SGFramework.py](../Src/Framework/SGFramework.py)

`h2_regime_result` の alias も付与済みで、今後の完全移行を見据えた形になっている。

## 5. H1 の最新整理
### 5.1 主系統
H1 runtime は multi-horizon patch mixer が主系統である。

主要ファイル:

- [train_h1_multi_horizon_forecaster.py](../Src/Framework/ForecastSystem/train_h1_multi_horizon_forecaster.py)
- [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py)
- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)

### 5.2 現在の役割
H1 は「中期予測器」というより、現状は

- `t+6 / t+7 / t+8` の数時間先を推論する
- `LONG_BIAS / SHORT_BIAS / NEUTRAL`
- `confidence`
- `predicted_path`

を返す tactical bias 層として見るのが自然である。

### 5.3 artifact 管理と runtime 境界
2026-04-09 時点で、H1 artifact は baseline freeze と active pointer を持つ形に整理済みである。

- [active_runtime.json](../Asset/Models/h1/active_runtime.json)
- [promotion_log.jsonl](../Asset/Models/h1/promotion_log.jsonl)
- [h1model_tactical_usdjpy_h1_seq32_tg6-7-8_baseline_v1_20260408T232845JST.manifest.json](../Asset/Models/h1/h1model_tactical_usdjpy_h1_seq32_tg6-7-8_baseline_v1_20260408T232845JST.manifest.json)
- [h1ds_usdjpy_h1_20251001_20260307_0600_seq32_legacycompat_baseline_v1.manifest.json](../Src/Backtest/Output/datasets/h1/h1ds_usdjpy_h1_20251001_20260307_0600_seq32_legacycompat_baseline_v1.manifest.json)

`h1_forecast.py` には、downstream 側が参照する共通境界を追加してある。

- `build_h1_runtime_view`
- `evaluate_h1_alignment`
- `has_h1_forecast_result`

このため、`m15_entry.py`、`final_decision.py`、`main_flow_gate.py` は H1 生 dict をばらばらに読むより、この境界を通す前提で追うとよい。
`h1_forecast.py` 単体の役割整理は [SGSystem_H1Forecast_RuntimeGuide.md](./SGSystem_H1Forecast_RuntimeGuide.md) を参照。

また、`RealtimeFlow.py` は `h1_state` を持つ形へ整理済みで、以下をまとめて保持する。

- `result`
- `runtime_view`
- `latest_forecast_bar_jst`
- `latest_forecast_update_jst`

### 5.4 旧 LSTM 系
以下は残っているが、runtime 本体では使っていない。

- [train_h1_lstm_regressor.py](../Src/Framework/ForecastSystem/train_h1_lstm_regressor.py)
- [analyze_h1_lstm_predictions.py](../Src/Framework/ForecastSystem/analyze_h1_lstm_predictions.py)

## 6. M15 の最新整理
`m15_entry.py` は、H2 方向に沿った執行タイミング判定を担う。

現在の主な入力:

- `momentum`
- `pullback_state`
- `breakout`
- `noise`
- `spread`
- H2 の `regime_direction`
- H1 の `net_direction / confidence`

現在の出力:

- `ENTER`
- `WAIT`
- `SKIP`
- `EXIT`

raw_features には H2 の `regime_direction / regime_score / regime_quality` も残すようにしてあり、後でデバッグしやすい。

## 7. `predicted_path` ゲート
`predicted_path` 補助ゲートは [main_flow_gate.py](../Src/Framework/ROModule/main_flow_gate.py) に shared 化済みであり、`RealtimeFlow.py` と `SGFramework.py` の両方から使う。

現行仕様:

- H2 方向に沿う `predicted_path` の最大方向乖離を測る
- 乖離が小さい場合は `final_decision` の `ENTER_LONG / ENTER_SHORT` を `WAIT` に落とす
- 閾値は `m15_predicted_path_gap_threshold_pips`
- 未指定時の runtime 既定値は `30.0`

補足:

- backtest 側でも同じ gate policy を通す前提に整理済み
- [run_backtest.py](../Src/Backtest/Scripts/run_backtest.py) には history cache I/O を追加済み
- [evaluate_signals_backtest.py](../Src/Backtest/Scripts/evaluate_signals_backtest.py) は `base_final_action` と `final_action` の差分、および `main_flow_gate_summary` を出力できる

## 8. 外部停止条件
外部停止条件は [ExternalContextBuilder.py](../Src/Framework/ContextSystem/ExternalContextBuilder.py) と [external_filter.py](../Src/Framework/ROModule/external_filter.py) が担う。

現時点の主要な停止条件:

- 高インパクトイベント接近
- 要人発言
- 地政学的アラート
- データ鮮度異常
- 異常ボラティリティ

入力ファイル:

- [external_events.json](../Asset/Config/external_events.json)
- [manual_risk_flags.json](../Asset/Config/manual_risk_flags.json)

管理スクリプト:

- [manage_external_context.py](../Src/Debug/manage_external_context.py)

## 9. Utility 共通化
今回、数値変換の一部を Utility 側へ共通化した。

- [Utility.py](../Src/Framework/Utility/Utility.py)
  - `ToFloat`
  - `Clamp01`

これにより以下の重複実装を整理している。

- `h2_environment.py`
- `h1_forecast.py`
- `m15_entry.py`
- `RealtimeFlow.py`
- `ExternalContextBuilder.py`
- `LSTMModel.py`

ただし `0 - 100` スコア用の `_clamp_score` は、まだ各モジュールの文脈依存としてローカルに残している。

## 10. 現在の確認用テスト
今回までの整理で、以下のテストが追加または継続利用されている。

- [test_h2_regime_semantics.py](../Src/Debug/test_h2_regime_semantics.py)
- [test_final_decision_regime_compatibility.py](../Src/Debug/test_final_decision_regime_compatibility.py)
- [test_m15_entry_regime_compatibility.py](../Src/Debug/test_m15_entry_regime_compatibility.py)
- [test_numeric_utility_helpers.py](../Src/Debug/test_numeric_utility_helpers.py)
- [test_main_flow_path_gate_parity.py](../Src/Debug/test_main_flow_path_gate_parity.py)
- [test_backtest_history_cache_io.py](../Src/Debug/test_backtest_history_cache_io.py)
- [test_h1_forecast_refactor.py](../Src/Debug/test_h1_forecast_refactor.py)
- [test_realtime_h1_state_management.py](../Src/Debug/test_realtime_h1_state_management.py)
- [test_phase1_flow.py](../Src/Debug/test_phase1_flow.py)
- [test_phase3_h2_h1_m15_integration_conflict.py](../Src/Debug/test_phase3_h2_h1_m15_integration_conflict.py)

## 11. ここまでの変更で把握しておくこと
### 11.1 完了済み
- realtime 本体を `RealtimeFlow.py` 中心に整理
- H2 を regime filter として整理
- H2 の `regime_*` 出力を追加
- downstream を `regime_*` 優先読取に対応
- Utility に `ToFloat` / `Clamp01` を共通化
- H1 dataset / model の versioning 方針を整理し、baseline artifact を freeze
- [active_runtime.json](../Asset/Models/h1/active_runtime.json) による active model 選択を導入
- H1 consumer 境界を `h1_forecast.py` に寄せ、`m15_entry.py` / `final_decision.py` / `main_flow_gate.py` の読取経路を整理
- `RealtimeFlow.py` に `h1_state` を導入し、H1 state 保持を一元化
- `predicted_path` 補助ゲートを shared `main_flow_gate.py` に集約
- `run_backtest.py` に history cache I/O、`evaluate_signals_backtest.py` に gate 差分集計を追加

### 11.2 まだ残っているもの
- execution
- position management
- 外部イベントの自動取得
- H1 tactical bias の昇格条件定義と walk-forward 再評価
- parity 反映後の raw signals 再生成と gate 効果の実測
- M15 ゲートのボラティリティ正規化

## 12. 次に触る時のおすすめ順
次に開発を進めるなら、優先順は次が自然。

1. execution / position management の設計
2. 外部イベント自動取得
3. parity 反映後の raw signals 再生成と gate 効果の定量確認
4. H1 を tactical bias として昇格条件まで含めて明文化
5. M15 の固定 30 pips 閾値を ATR 等へ置き換える検討

## 13. 運用上の補足
- [h1_training_dataset.csv](../Src/Backtest/Output/datasets/h1_training_dataset.csv) は 2026-04-09 時点で約 25.2MB
- 現段階の refactor とテストに MT5 起動は必須ではない
- MT5 が必要なのは parity backtest 用 raw signals の再生成、または live fetch / live 推論確認の段階
- より詳細な設計変更方針は [SGSystem_TrendRefactor_ImplementationPlan.md](./SGSystem_TrendRefactor_ImplementationPlan.md) と [SGSystem_H1Dataset_Versioning_Promotion_Design.md](./SGSystem_H1Dataset_Versioning_Promotion_Design.md) を参照

## 14. 起動・確認コマンド
### 14.1 realtime 起動
```powershell
python Src/main.py
```

### 14.2 学習データ生成
```powershell
python Src/Backtest/Scripts/build_h1_training_dataset.py
```

### 14.3 現行 H1 モデル学習
```powershell
python Src/Framework/ForecastSystem/train_h1_multi_horizon_forecaster.py
```

### 14.4 外部イベント確認
```powershell
python Src/Debug/manage_external_context.py summary
```

### 14.5 H2 regime 互換テスト
```powershell
python Src/Debug/test_h2_regime_semantics.py
```

### 14.6 H1 境界整理テスト
```powershell
python Src/Debug/test_h1_forecast_refactor.py
python Src/Debug/test_realtime_h1_state_management.py
```

### 14.7 main flow parity テスト
```powershell
python Src/Debug/test_main_flow_path_gate_parity.py
python Src/Debug/test_backtest_history_cache_io.py
```

## 15. 2026-04-11 時点の最新状況
H1 Performance 改善の直近作業は、再学習ではなく Phase 2 の `h1_confidence_min` threshold sweep である。

ここでの目的は、active H1 artifact を固定したまま `h1_confidence_min` だけを変えた時に、

- signal 数がどう変わるか
- no signal 比率がどう変わるか
- `enter_hit_rate` が改善するか
- `predicted_path` gate や conflict 抑制がどの程度効いているか

を比較し、今の H1 を tactical bias 層としてどう使うのが最も妥当かを定量確認することである。

## 16. 今回追加した運用コード
今回、Phase 2 を止めずに進めるため、以下のコードを追加または更新している。

### 16.1 MT5 terminal cache fallback
- [run_backtest.py](../Src/Backtest/Scripts/run_backtest.py)
  - MT5 Python API の `copy_rates_*` が `Terminal: Call failed` になった場合でも、
    MT5 terminal 内に残っている `M15.hc / H1.hc / H2.hc` から履歴を復元できるようにした

### 16.2 history cache 収集用スクリプト
- [build_h1_phase2_history_cache.py](../Src/Backtest/Scripts/build_h1_phase2_history_cache.py)
  - Phase 2 検証で必要な history cache を workspace 配下へ保存する専用スクリプト
  - MT5 を起動した状態で、このスクリプトを VSCode から `F5` 実行すればよい

### 16.3 threshold sweep 実行用スクリプト
- [run_h1_phase2_threshold_sweep.py](../Src/Backtest/Scripts/run_h1_phase2_threshold_sweep.py)
  - threshold sweep 実行
  - per-threshold の `raw_signals.csv` / `eval_summary.txt` / `backtest_log.txt` 生成
  - 比較表更新
  - `--resume` による中断後再開

## 17. 現在の進捗
2026-04-11 JST 時点の Phase 2 状況は以下。

### 17.1 生成済み
- history cache:
  - `Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307/`
- threshold 設定:
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_050.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_055.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_060.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_065.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_070.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_075.json`
- sweep 完了 threshold:
  - `0.50`
  - `0.55`
  - `0.60`

### 17.2 未完了
- `0.65`
- `0.70`
- `0.75`

比較表は [phase2_threshold_comparison.md](../Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/phase2_threshold_comparison.md) に集約する前提である。

## 18. あなたが F5 実行するコード
私の検証を助けるために、あなたが明示的に `F5` 実行するコードは以下。

- [build_h1_phase2_history_cache.py](../Src/Backtest/Scripts/build_h1_phase2_history_cache.py)

このスクリプトの役割は、MT5 を起動した状態で検証用履歴を収集し、workspace 配下に保存することである。

生成対象:

- `M15.csv`
- `H1.csv`
- `H2.csv`
- `metadata.json`

保存先:

- `Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307/`

補足:

- この段階で必要なのは学習データ更新ではない
- `build_h1_training_dataset.py` はまだ使わない
- Phase 2 で必要なのは history cache と raw signals の生成である

## 19. 直近の次アクション
今すぐの次アクションは、Phase 2 sweep の残り 3 本を `resume` 付きで完了させること。

手順:

1. 古い Debug 実行や古い Python sweep プロセスが残っていれば停止する
2. VSCode Run and Debug で `H1 Phase2 Threshold Sweep (History Cache)` を選ぶ
3. `F5` で再開する
4. `0.65 / 0.70 / 0.75` の出力を揃える
5. 比較表を埋めて採用 threshold を決める

## 20. 再開コマンド
### 20.1 history cache 再生成
```powershell
python Src/Backtest/Scripts/build_h1_phase2_history_cache.py
```

VSCode では `Prepare H1 Phase2 History Cache (MT5)` を `F5` 実行する。

### 20.2 threshold sweep 再開
```powershell
python Src/Backtest/Scripts/run_h1_phase2_threshold_sweep.py `
  --history-cache-dir Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307 `
  --prefer-history-cache `
  --resume `
  --stop-on-error
```

VSCode では `H1 Phase2 Threshold Sweep (History Cache)` を `F5` 実行する。

