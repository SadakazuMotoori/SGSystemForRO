# SGSystem_Overview

更新日: 2026-04-11 JST

## 1. プロジェクト概要
`SGSystemForRO` は、USDJPY を対象に MetaTrader 5 の confirmed bar データを用いて判断する FX システムである。  
現在の中心思想は、短期売買の方向を直接当てにいくのではなく、

- H2 で上位足の売買許可帯を決める
- H1 で数時間先の tactical bias を確認する
- M15 で執行タイミングを整える
- 外部要因がある場合は最優先で停止する

という multi-timeframe 構成である。

現時点では execution と position management は未実装であり、現在の主戦場は「方向判定と執行可否の判断ロジック」である。

## 2. 現在の実行構成
現在の realtime 実行は `main.py` が実体ではなく、`RealtimeFlow.py` を薄く再エクスポートする構成になっている。

- [main.py](../Src/main.py)
  - 互換エントリポイント
  - realtime 本体は保持しない
- [RealtimeFlow.py](../Src/Framework/RealtimeFlow.py)
  - realtime loop 本体
  - H2 / H1 / M15 の更新制御
  - `predicted_path` を使った M15 補助ゲート
  - `final_decision` の更新
- [MTManager.py](../Src/Framework/MTSystem/MTManager.py)
  - MT5 から confirmed bar ベースの market data を構築
- [SGFramework.py](../Src/Framework/SGFramework.py)
  - 旧来の一括実行パイプライン
  - realtime の主経路ではないが、参照用・テスト用として残置

補足:

- `main.py` は現在 `BuildMarketData` や `main` などを再エクスポートする薄いラッパーである。
- realtime の責務分離後、実質的な運用コードは `RealtimeFlow.py` に集約されている。

## 3. realtime フロー
realtime loop は概ね次の順で動作する。

1. MT5 から confirmed H2 / H1 / M15 データを取得する
2. `system_context` を更新する
3. `external_context` を構築する
4. `external_filter` を評価する
5. H2 confirmed bar が更新された時だけ H2 を再評価する
6. H1 confirmed bar が更新された時だけ H1 を再評価する
7. M15 confirmed bar が更新された時だけ M15 と `final_decision` を再評価する
8. 1 秒 sleep して次の loop へ進む

重要なポイント:

- confirmed bar ベースでのみ判断する
- H2 / H1 / M15 は必要な timeframe だけ再計算する
- M15 更新時に `predicted_path` を使った補助ゲートが掛かる

## 4. 現在の判断モジュール
runtime 中で利用している主要モジュールは以下の通り。

- [external_filter.py](../Src/Framework/ROModule/external_filter.py)
- [h2_environment.py](../Src/Framework/ROModule/h2_environment.py)
- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)
- [m15_entry.py](../Src/Framework/ROModule/m15_entry.py)
- [final_decision.py](../Src/Framework/ROModule/final_decision.py)

役割整理:

- `external_filter`
  - 高インパクトイベント、要人発言、異常ボラティリティ、データ鮮度などの外部停止条件を判定する
- `h2_environment`
  - H2 の上位足 regime filter
  - `LONG_ONLY / SHORT_ONLY / NO_TRADE` の許可帯を決める
- `h1_forecast`
  - H1 multi-horizon patch mixer により `t+6 / t+7 / t+8` を予測する
  - `LONG_BIAS / SHORT_BIAS / NEUTRAL` と `confidence`、`predicted_path` を返す
- `m15_entry`
  - M15 の momentum / pullback / breakout / noise / spread を見て執行タイミングを判定する
- `final_decision`
  - 各モジュール結果を集約して `ENTER_LONG / ENTER_SHORT / WAIT / NO_TRADE / EXIT` を返す

## 5. H2 の現行設計
2026-04-08 時点で、H2 の役割は「中期トレンド予測器」ではなく「上位足の regime filter」として整理済みである。

### 5.1 出力仕様
`h2_environment.py` は旧来の互換出力を維持しつつ、regime 用出力を追加している。

旧互換出力:

- `env_direction`
- `env_score`
- `trend_strength`
- `reason_codes`

新規出力:

- `regime_direction`
- `regime_score`
- `regime_quality`
- `regime_components`
- `regime_reason_codes`

### 5.2 判定材料
現時点の H2 判定材料は以下。

- `ma_short`
- `ma_long`
- `ma_slope`
- `adx`
- `swing_structure`

ロジックは次の流れ。

1. MA の上下関係を見る
2. MA slope の向きを見る
3. swing structure を見る
4. ADX が閾値以上かを確認する
5. 方向票の一致度から `trend_strength` と `regime_score` を作る
6. 最終的に `LONG_ONLY / SHORT_ONLY / NO_TRADE` を返す

### 5.3 閾値互換
[thresholds.json](../Asset/Config/thresholds.json) には旧キーに加え、H2 regime 用の互換キーを追加済みである。

- `adx_min`
- `trend_strength_min`
- `h2_regime_adx_min`
- `h2_regime_strength_min`
- `h2_regime_score_min`

`h2_environment.py` は Phase 1 の互換期間として、新旧どちらのキーでも動作する。

### 5.4 downstream 互換
以下のモジュールは `regime_*` を優先し、無い場合は旧 `env_*` にフォールバックする。

- [RealtimeFlow.py](../Src/Framework/RealtimeFlow.py)
- [final_decision.py](../Src/Framework/ROModule/final_decision.py)
- [m15_entry.py](../Src/Framework/ROModule/m15_entry.py)
- [SGFramework.py](../Src/Framework/SGFramework.py)

また、`SGFramework.py` と `RealtimeFlow.py` は `h2_regime_result` の alias も保持している。

## 6. H1 予測系
H1 runtime は、旧 deterministic fallback や旧 LSTM runtime ではなく、multi-horizon patch mixer を使う構成に整理済みである。

### 6.1 学習データ
[build_h1_training_dataset.py](../Src/Backtest/Scripts/build_h1_training_dataset.py) が H1 学習データを構築する。

- 32 本の H1 履歴を利用
- `future_hours=2` を基準に `t+6 / t+7 / t+8` をターゲット化
- OHLC に加え、MA / RSI / MACD / wick / return / time features を利用
- 出力先:
  - [h1_training_dataset.csv](../Src/Backtest/Output/datasets/h1_training_dataset.csv)

2026-04-09 時点では、refactor 用 baseline として versioned freeze も導入済みである。

- [h1ds_usdjpy_h1_20251001_20260307_0600_seq32_legacycompat_baseline_v1.csv](../Src/Backtest/Output/datasets/h1/h1ds_usdjpy_h1_20251001_20260307_0600_seq32_legacycompat_baseline_v1.csv)
- [h1ds_usdjpy_h1_20251001_20260307_0600_seq32_legacycompat_baseline_v1.manifest.json](../Src/Backtest/Output/datasets/h1/h1ds_usdjpy_h1_20251001_20260307_0600_seq32_legacycompat_baseline_v1.manifest.json)

### 6.2 現行学習スクリプト
runtime の主系統は以下。

- [train_h1_multi_horizon_forecaster.py](../Src/Framework/ForecastSystem/train_h1_multi_horizon_forecaster.py)
- [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py)
- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)

現行 runtime の要点:

- [active_runtime.json](../Asset/Models/h1/active_runtime.json) を優先し、active な versioned model artifact を選択する
- promotion 履歴は [promotion_log.jsonl](../Asset/Models/h1/promotion_log.jsonl) に残す
- `t+6 / t+7 / t+8` の delta を推論する
- `LONG_BIAS / SHORT_BIAS / NEUTRAL`
- `confidence`
- `predicted_path`

を runtime の downstream へ渡す。

また、baseline freeze 済みの model artifact は以下で管理している。

- [h1model_tactical_usdjpy_h1_seq32_tg6-7-8_baseline_v1_20260408T232845JST.manifest.json](../Asset/Models/h1/h1model_tactical_usdjpy_h1_seq32_tg6-7-8_baseline_v1_20260408T232845JST.manifest.json)

### 6.3 H1 runtime 境界整理
2026-04-09 時点で、`h1_forecast.py` は H1 の共通参照面を持つ形に再整理済みである。

- `build_h1_runtime_view`
- `evaluate_h1_alignment`
- `has_h1_forecast_result`

`m15_entry.py`、`final_decision.py`、`main_flow_gate.py` はこの境界を経由して H1 を読む。
`h1_forecast.py` 単体の役割整理は [SGSystem_H1Forecast_RuntimeGuide.md](./SGSystem_H1Forecast_RuntimeGuide.md) を参照。

`RealtimeFlow.py` 側も `h1_state` を持ち、以下を一元保持する。

- `result`
- `runtime_view`
- `latest_forecast_bar_jst`
- `latest_forecast_update_jst`

### 6.4 旧 LSTM 系
以下は旧来の学習・分析用ラインであり、runtime 本体では使っていない。

- [train_h1_lstm_regressor.py](../Src/Framework/ForecastSystem/train_h1_lstm_regressor.py)
- [analyze_h1_lstm_predictions.py](../Src/Framework/ForecastSystem/analyze_h1_lstm_predictions.py)

## 7. M15 補助ゲート
`predicted_path` 補助ゲートは [main_flow_gate.py](../Src/Framework/ROModule/main_flow_gate.py) に shared 化され、`RealtimeFlow.py` と `SGFramework.py` の両方から使う構成に整理済みである。

現行仕様:

- `predicted_path` と現在値の最大方向乖離を測る
- H2 の方向に沿う乖離だけを有効とみなす
- 閾値未満なら `final_decision` の `ENTER_LONG / ENTER_SHORT` を `WAIT` に落とす
- runtime / backtest の main flow parity を取りやすい形に整理済み

現在の閾値:

- `m15_predicted_path_gap_threshold_pips`
  - 未指定時は runtime 既定値 `30.0`

また、backtest 側では以下も追加済みである。

- [run_backtest.py](../Src/Backtest/Scripts/run_backtest.py)
  - history cache の読込 / 保存 / fallback
  - MT5 Python API の `copy_rates_*` が失敗した場合でも、terminal 内の `M15.hc / H1.hc / H2.hc` を読む fallback を持つ
- [evaluate_signals_backtest.py](../Src/Backtest/Scripts/evaluate_signals_backtest.py)
  - `base_final_action` と `final_action` の差分集計
  - `main_flow_gate_summary` の出力
- [build_h1_phase2_history_cache.py](../Src/Backtest/Scripts/build_h1_phase2_history_cache.py)
  - Phase 2 用の workspace history cache を生成する専用スクリプト
  - MT5 接続成功後、API 取得または terminal cache fallback を使って `M15.csv / H1.csv / H2.csv / metadata.json` を保存する
- [run_h1_phase2_threshold_sweep.py](../Src/Backtest/Scripts/run_h1_phase2_threshold_sweep.py)
  - `h1_confidence_min` sweep を一括実行する補助スクリプト
  - `raw_signals.csv` と `eval_summary.txt` の生成、および比較表更新を担う
  - `--resume` で既存 run を再利用できる

## 8. 外部停止条件
外部停止条件は [ExternalContextBuilder.py](../Src/Framework/ContextSystem/ExternalContextBuilder.py) と [external_filter.py](../Src/Framework/ROModule/external_filter.py) で扱う。

現在の主な停止対象:

- `high_impact_event_soon`
- `central_bank_speech`
- `geopolitical_alert`
- `data_feed_error`
- `abnormal_volatility`

入力ソース:

- [external_events.json](../Asset/Config/external_events.json)
- [manual_risk_flags.json](../Asset/Config/manual_risk_flags.json)

補助スクリプト:

- [manage_external_context.py](../Src/Debug/manage_external_context.py)

## 9. Utility 共通化
2026-04-08 時点で、複数モジュールに散っていた数値変換の一部を Utility に共通化済みである。

- [Utility.py](../Src/Framework/Utility/Utility.py)
  - `ToFloat`
  - `Clamp01`

これにより、以下の重複実装を整理した。

- `h2_environment.py`
- `h1_forecast.py`
- `m15_entry.py`
- `RealtimeFlow.py`
- `ExternalContextBuilder.py`
- `LSTMModel.py`

`_clamp_score` はまだモジュール文脈依存のため、各モジュールに残している。

## 10. テスト
H2 regime 移行と互換性確認のため、以下のテストを追加済み。

- [test_h2_regime_semantics.py](../Src/Debug/test_h2_regime_semantics.py)
- [test_final_decision_regime_compatibility.py](../Src/Debug/test_final_decision_regime_compatibility.py)
- [test_m15_entry_regime_compatibility.py](../Src/Debug/test_m15_entry_regime_compatibility.py)
- [test_numeric_utility_helpers.py](../Src/Debug/test_numeric_utility_helpers.py)
- [test_main_flow_path_gate_parity.py](../Src/Debug/test_main_flow_path_gate_parity.py)
- [test_backtest_history_cache_io.py](../Src/Debug/test_backtest_history_cache_io.py)
- [test_h1_forecast_refactor.py](../Src/Debug/test_h1_forecast_refactor.py)
- [test_realtime_h1_state_management.py](../Src/Debug/test_realtime_h1_state_management.py)

既存の固定フローテスト:

- [test_phase1_flow.py](../Src/Debug/test_phase1_flow.py)
- [test_phase3_h2_h1_m15_integration_conflict.py](../Src/Debug/test_phase3_h2_h1_m15_integration_conflict.py)

## 11. 2026-04-09 時点の追加整理
本日のリファクタリングで、以下を整理済みである。

1. H1 dataset / model の versioning 方針を文書化し、baseline dataset / model を versioned artifact として freeze
2. [active_runtime.json](../Asset/Models/h1/active_runtime.json) と [promotion_log.jsonl](../Asset/Models/h1/promotion_log.jsonl) による active artifact 管理を導入
3. `h1_forecast.py` に H1 の共通境界を集約し、`m15_entry.py` / `final_decision.py` / `main_flow_gate.py` をその境界経由へ寄せた
4. `RealtimeFlow.py` に `h1_state` を導入し、H1 result / runtime view / latest forecast timestamp の保持を一元化
5. `predicted_path` 補助ゲートを [main_flow_gate.py](../Src/Framework/ROModule/main_flow_gate.py) に shared 化し、runtime / backtest parity の土台を作成
6. `run_backtest.py` に history cache I/O を追加し、`evaluate_signals_backtest.py` に gate summary 集計を追加

設計方針の詳細は [SGSystem_H1Dataset_Versioning_Promotion_Design.md](./SGSystem_H1Dataset_Versioning_Promotion_Design.md) にまとめている。

## 12. 現時点の未実装・次段候補
現時点で残っている主な論点は以下。

1. 外部イベントの自動取得
2. execution / position management の実装
3. H1 tactical bias の昇格条件定義と walk-forward 再評価
4. parity 反映後の raw signals 再生成と gate 効果の実測
5. M15 の固定 30 pips ゲートをボラティリティ正規化へ寄せる検討

## 13. 補足
[h1_training_dataset.csv](../Src/Backtest/Output/datasets/h1_training_dataset.csv) は 2026-04-09 時点で約 25.2MB である。  
現段階の refactor とテスト実行に MT5 起動は必須ではなく、MT5 が必要になるのは parity backtest 用の実データ再生成または live fetch / live 推論確認の段階である。  
また、設計変更の詳細方針は [SGSystem_TrendRefactor_ImplementationPlan.md](./SGSystem_TrendRefactor_ImplementationPlan.md) と [SGSystem_H1Dataset_Versioning_Promotion_Design.md](./SGSystem_H1Dataset_Versioning_Promotion_Design.md) にまとめている。

## 14. 2026-04-11 Latest Status
2026-04-11 JST 時点で、H1 Performance Phase 2 を止めずに進めるための運用経路を追加した。

追加済みの実務上のポイントは以下。

1. [run_backtest.py](../Src/Backtest/Scripts/run_backtest.py) に terminal cache fallback を追加し、MT5 Python API の `copy_rates_*` が `Terminal: Call failed` になっても、MT5 terminal 内の `M15.hc / H1.hc / H2.hc` から backtest 用履歴を復元できるようにした
2. [build_h1_phase2_history_cache.py](../Src/Backtest/Scripts/build_h1_phase2_history_cache.py) を追加し、Phase 2 検証用の history cache を workspace 配下へ保存できるようにした
3. [run_h1_phase2_threshold_sweep.py](../Src/Backtest/Scripts/run_h1_phase2_threshold_sweep.py) を追加し、threshold sweep の実行、集計、比較表更新を 1 本化した
4. 同スクリプトに `--resume` を追加し、中断後も既存の `raw_signals.csv` を再利用しながら続きを回せるようにした
5. VSCode Run and Debug から再現しやすいように、`.vscode/launch.json` に Phase 2 用の構成を追加した

## 15. 現在の進捗
2026-04-11 JST 時点の Phase 2 実行状況は以下。

- history cache 生成済み:
  - `Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307/`
- threshold 設定ファイル生成済み:
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_050.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_055.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_060.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_065.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_070.json`
  - `Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_075.json`
- sweep 完了済み threshold:
  - `0.50`
  - `0.55`
  - `0.60`
- 未完了 threshold:
  - `0.65`
  - `0.70`
  - `0.75`

比較表は [phase2_threshold_comparison.md](../Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/phase2_threshold_comparison.md) を使う前提である。

## 16. Immediate Next Action
現時点の次アクションは、Phase 2 sweep の残り 3 本を `resume` 付きで完了させることである。

優先手順:

1. 古い Debug 実行や古い Python sweep プロセスが残っていれば停止する
2. VSCode Run and Debug で `H1 Phase2 Threshold Sweep (History Cache)` を起動する
3. `0.65 / 0.70 / 0.75` の `raw_signals.csv` と `eval_summary.txt` を生成する
4. [phase2_threshold_comparison.md](../Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/phase2_threshold_comparison.md) を埋め、採用 threshold または暫定 operating band を決める

## 17. Quick Resume Point
### 17.1 データ収集
history cache を再生成したい場合は以下。

```powershell
python Src/Backtest/Scripts/build_h1_phase2_history_cache.py
```

VSCode では `Prepare H1 Phase2 History Cache (MT5)` を `F5` 実行する。

### 17.2 sweep 再開
threshold sweep の再開は以下。

```powershell
python Src/Backtest/Scripts/run_h1_phase2_threshold_sweep.py `
  --history-cache-dir Src/Backtest/Output/history_cache/h1_phase2_usdjpy_20251001_20260307 `
  --prefer-history-cache `
  --resume `
  --stop-on-error
```

VSCode では `H1 Phase2 Threshold Sweep (History Cache)` を `F5` 実行する。

### 17.3 補足
- Phase 2 では学習データ再生成や再学習はまだ不要
- この段階で必要なのは `build_h1_training_dataset.py` ではなく、Phase 2 検証用の history cache と raw signals の生成である

