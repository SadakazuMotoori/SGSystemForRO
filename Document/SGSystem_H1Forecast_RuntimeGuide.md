# SGSystem H1 Forecast Runtime Guide

更新日: 2026-04-09 JST

## 1. この資料の目的
この資料は、[h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py) が現在の SGSystem において何を担っているかを、runtime 観点で整理するためのものである。

特に重要なのは、`h1_forecast.py` を「H1 の学習器そのもの」や「中期方向予測器」として見るのではなく、

- H1 モデルの runtime 呼び出しを包む
- tactical bias として採用可能かを判定する
- downstream が共通形式で読むための境界を提供する

という `runtime adapter / policy boundary` として理解することである。

2026-04-09 時点の refactor では、内部実装の責務分離も進めており、

- downstream contract: [h1_forecast_contract.py](../Src/Framework/ROModule/h1_forecast_contract.py)
- policy layer: [h1_forecast_policy.py](../Src/Framework/ROModule/h1_forecast_policy.py)
- runtime orchestration: [h1_forecast_runtime.py](../Src/Framework/ROModule/h1_forecast_runtime.py)

に順次切り出している。
ただし public facade は引き続き [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py) で維持している。

## 2. 先に結論
`h1_forecast.py` の現在の役割は、H1 の生 OHLC を受け取り、H1 backend の推論結果を SGSystem 全体で使える共通形式へ整形し、

- `SUCCESS / NEUTRAL / INSUFFICIENT_DATA / FORECAST_ERROR`
- `LONG_BIAS / SHORT_BIAS / NEUTRAL`
- `confidence`
- `predicted_path`

を安定した contract として返すことである。

このモジュール自体は、

- 学習データ生成
- モデル学習
- artifact promotion
- execution
- position management

を担当しない。

## 3. アーキテクチャ上の位置づけ
現行の多時間足構成における位置づけは次の通りである。

- `H2`: regime permission
- `H1`: tactical bias / confirmation
- `M15`: execution timing
- `final_decision`: policy combiner

このため H1 は、H2 が許可した方向に対して

- 数時間先のモデル出力が同方向を支持しているか
- その支持を runtime で採用してよいだけの confidence があるか

を返す confirmation layer と見るのが自然である。

## 4. このモジュールの責務
現在の責務は大きく 6 つである。

### 4.1 H1 入力の受け取り
入口は `evaluate_h1_forecast(_h1_data, _thresholds)` であり、主に以下を受け取る。

- `_h1_data["timestamp_jst"]`
- `_h1_data["ohlc"]`
- `_thresholds["h1_confidence_min"]`

参照:

- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)

### 4.2 必要本数の確認
backend が要求する H1 本数に足りない場合、推論を試みず `INSUFFICIENT_DATA` を返す。

ここで注意すべきなのは、active model metadata の `sequence_length` がそのまま必要バー数ではない点である。
runtime 側では indicator warmup 分も必要になるため、必要本数は backend の `GetForecastSequenceLength()` を通して解決している。

関連:

- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)
- [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py)

### 4.3 backend 推論の呼び出し
十分な本数がある場合、`PredictMultiHorizonForecast()` を呼び出して H1 tactical forecast を取得する。

backend は現時点で主に以下を返す。

- `net_direction`
- `direction_score_long`
- `direction_score_short`
- `confidence`
- `predicted_path`
- artifact 関連 metadata

backend 本体:

- [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py)

### 4.4 model payload の正規化
backend 返却値は `_normalize_model_payload()` で最小限の整形を行う。

ここでは主に

- score を `0.0 - 1.0` に clamp
- `predicted_path` を list 化
- `net_direction` を既定値付きで読む

という正規化を行う。

### 4.5 runtime policy の適用
このモジュールの本質はここにある。

モデルが `LONG_BIAS` や `SHORT_BIAS` を返しても、そのまま採用はしない。
`_classify_model_decision()` によって、

1. `NEUTRAL` なら `forecast_status="NEUTRAL"`
2. directional bias があり、`confidence >= h1_confidence_min` なら `forecast_status="SUCCESS"`
3. directional bias があっても confidence 不足なら `NEUTRAL` に落とす

という runtime 採用判定を行う。

つまり `h1_forecast.py` は「モデル出力をそのまま返すモジュール」ではなく、
「モデル出力を SGSystem の H1 tactical bias として採用するかを決めるモジュール」である。

### 4.6 downstream 用の共通境界提供
このファイルには、推論本体以外に downstream 向け helper がある。

- `has_h1_forecast_result`
- `build_h1_runtime_view`
- `evaluate_h1_alignment`

これにより、下流モジュールが H1 の生 dict をそれぞれ独自解釈せず、共通の読み方に寄せられるようになっている。

## 5. 処理フロー
`evaluate_h1_forecast()` の流れは概ね次の通りである。

1. `_h1_data` から `timestamp_jst` と `ohlc` を取得する
2. `_extract_close_list()` で close の並びを作る
3. `_evaluate_h1_direction()` で必要本数を確認する
4. 必要本数不足なら `INSUFFICIENT_DATA` を返す
5. backend の `PredictMultiHorizonForecast()` を呼ぶ
6. payload を正規化する
7. `confidence` 閾値を用いて `SUCCESS` か `NEUTRAL` かを分類する
8. `raw_features` と reason codes を付与して module result を返す
9. 例外時は `FORECAST_ERROR` を返す

重要なのは、失敗時や曖昧時も固定形式で返す点である。
このため realtime loop や downstream は「H1 が無いかもしれない世界」を安全に扱える。

## 6. 返り値の意味
主な返り値項目は以下の通りである。

| 項目 | 意味 |
| --- | --- |
| `status` | モジュール実行成否。`OK` または `ERROR` |
| `forecast_role` | H1 出力の役割。現在は `TACTICAL_BIAS` |
| `forecast_status` | H1 tactical bias の採用状態。`SUCCESS / NEUTRAL / INSUFFICIENT_DATA / FORECAST_ERROR` |
| `net_direction` | 現時点で採用される H1 方向。`LONG_BIAS / SHORT_BIAS / NEUTRAL` |
| `bias_direction` | tactical bias として採用される方向。未採用時は `NEUTRAL` |
| `bias_ready` | H1 tactical bias を confirmation として使えるか |
| `bias_alignment_hint` | この H1 がどちらの H2 許可帯を補強しうるか |
| `direction_score_long` | ロング側の方向強度 |
| `direction_score_short` | ショート側の方向強度 |
| `confidence` | runtime 採用判定に使う信頼度 |
| `predicted_path` | 疎な horizon 予測から補間した価格経路 |
| `predicted_path_type` | 現在の path 契約種別。現状は線形補間 path |
| `predicted_path_source_horizons` | 現在 path を構成する sparse horizon 群 |
| `reason_codes` | どの分岐でその結論になったかの識別子 |
| `summary` | 人がログを読むための要約 |
| `raw_features` | 最近の H1 断面と active artifact 情報 |

補足:

- `status` は Python 実行の成功・失敗
- `forecast_status` は H1 tactical bias としての意味付け結果

であり、似て見えるが役割が異なる。

## 7. helper 関数の意味
### 7.1 `has_h1_forecast_result`
H1 結果が存在するかだけを判定する。
Phase 1 / refactor 互換のため、空 dict や未接続ケースを吸収しやすくしている。

### 7.2 `build_h1_runtime_view`
H1 生 result から、runtime やログに必要な最低限の view を取り出す。

現在ここで見せている主なものは以下。

- `forecast_role`
- `forecast_status`
- `net_direction`
- `bias_direction`
- `bias_ready`
- `bias_alignment_hint`
- `confidence`
- `predicted_path`
- `predicted_path_type`
- `predicted_path_source_horizons`
- `summary`
- `active_model_id`
- `dataset_id`
- `artifact_selection_source`
- `sequence_length`

`RealtimeFlow.py` の `h1_state["runtime_view"]` はこの helper を通して作られる。

### 7.3 `evaluate_h1_alignment`
H2 の許可方向と H1 tactical bias の整合を共通ルールで判定する。

返り値の主な分類は以下。

- `UNAVAILABLE`
- `NEUTRAL_OR_SKIPPED`
- `LOW_CONFIDENCE`
- `ALIGNED`
- `CONFLICT`

この helper の狙いは、`m15_entry.py`、`final_decision.py`、`main_flow_gate.py` で H1 の解釈を揃えることにある。

## 8. 周辺モジュールとの関係
### 8.1 `RealtimeFlow.py`
realtime では H1 confirmed bar が更新された時だけ `evaluate_h1_forecast()` を再実行し、結果を `h1_state` に保持する。

保持項目:

- `result`
- `runtime_view`
- `latest_forecast_bar_jst`
- `latest_forecast_update_jst`

これにより H1 は「最新の生 result」と「運用上見やすい view」の両方を持つ。

### 8.2 `m15_entry.py`
M15 は H1 を主判定にはしないが、`evaluate_h1_alignment()` を通じて

- H2 と H1 が整合しているか
- 低 confidence か
- conflict か

を raw_features / reason_codes に反映する。

つまり H1 は M15 の主役ではなく、timing 判定の文脈情報として使われている。

### 8.3 `final_decision.py`
`final_decision.py` では H2 許可帯と H1 tactical bias の整合を確認し、

- `ALIGNED` なら通す
- `CONFLICT` なら `WAIT`
- `NEUTRAL_OR_SKIPPED` や `LOW_CONFIDENCE` なら H1 判定を保留

という policy を採る。

### 8.4 `main_flow_gate.py`
`main_flow_gate.py` では、H1 direction の採用判定を `forecast_status` の直読みではなく、`build_h1_runtime_view()` が返す `bias_ready` / `bias_direction` ベースへ寄せている。

あわせて `predicted_path_type` と `predicted_path_source_horizons` も signal result に載せることで、`predicted_path` 契約の可観測性を main flow 側でも保っている。
ここでは `build_h1_runtime_view()` を通して `predicted_path` を読み、
M15 現在値との gap が十分かを測る。

このため `predicted_path` は H1 表示用情報ではなく、realtime / backtest parity に関わる main flow の入力でもある。

## 9. backend / artifact との関係
`h1_forecast.py` 自体は model file を直接解決しない。
実際の artifact 解決は backend 側の [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py) が担当している。

現行 runtime は [active_runtime.json](../Asset/Models/h1/active_runtime.json) を優先し、
active model と metadata をそこから選ぶ。

そのうえで `h1_forecast.py` は backend payload から以下を取り込み、`raw_features` や `runtime_view` に残す。

- `active_model_id`
- `dataset_id`
- `artifact_selection_source`
- `sequence_length`
- `horizons`
- `history_end_timestamp_jst`
- `predicted_delta_by_horizon`
- `predicted_close_by_horizon`

この設計により、「今どの H1 artifact で判断したか」を runtime ログや debug 出力から追いやすくしている。

## 10. 誤解しやすい点
### 10.1 `h1_forecast.py` はモデル本体ではない
モデル推論本体は [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py) にある。
`h1_forecast.py` はその上に載る runtime 判断層である。

### 10.2 `predicted_path` は密な将来予測列ではない
現在の `predicted_path` は `t+6 / t+7 / t+8` の疎な horizon 予測から線形補間している。
したがって、真の意味で「各1時間先を個別予測した path」ではない。

### 10.3 `confidence` は calibrated probability ではない
現在の `confidence` は backend が返す heuristic であり、
direction dominance と magnitude を合成した runtime 用 score である。
統計的に厳密な確率とはみなさない方がよい。

### 10.4 H1 は現時点では confirmation layer
H1 は現状、H2 の regime permission を補強する tactical bias 層であり、
単独で方向を主導する設計にはなっていない。

## 11. 今後の拡張方針
現在のコードと設計方針を踏まえると、次の方向が自然である。

責務分離の段階計画は [SGSystem_H1Forecast_ResponsibilitySplitPlan.md](./SGSystem_H1Forecast_ResponsibilitySplitPlan.md) にまとめている。

### 11.1 runtime 境界としての責務を維持する
`h1_forecast.py` には今後も

- dataset 生成
- model training
- auto promotion
- live execution

を入れない方がよい。

runtime でやるべきことは、

- active artifact の読取
- H1 tactical bias の採用判定
- downstream へ渡す共通 contract の維持

までに留めるのが見通しがよい。

### 11.2 downstream の読取経路をさらに一本化する
`m15_entry.py`、`final_decision.py`、`main_flow_gate.py` の 3 箇所は、すでに shared helper / shared field ベースの読取へ揃い始めている。

そのため今後の主眼は、「一本化を始めること」よりも「field 名と意味の重複解釈を downstream 側で増やさないこと」に移っている。
現在すでに `evaluate_h1_alignment()` へ寄せているが、
今後は H1 解釈をこの helper と `build_h1_runtime_view()` にさらに集約していくのが望ましい。

これにより、

- `m15_entry.py`
- `final_decision.py`
- `main_flow_gate.py`

での重複ロジックや古い互換コードを減らせる。

### 11.3 role-oriented field の維持と拡張
`forecast_role`、`bias_direction`、`bias_ready`、`bias_alignment_hint` は、現在すでに `runtime_view` と consumer 側の一部で利用している。

ここから先は field を増やすこと自体より、昇格条件や confirmation policy とどう結びつけるかを慎重に定義する段階である。
現在は、H1 の意味付けを明確にするため、すでに以下を result / runtime view に追加している。

- `forecast_role = "TACTICAL_BIAS"`
- `bias_direction`
- `bias_ready`
- `bias_alignment_hint`

これにより downstream が `forecast_status` の意味を推測せずに済むようになっている。
今後は必要に応じて、昇格条件や confirmation policy と結びつく field へ拡張する余地がある。

### 11.4 confidence の昇格条件を見直す
今の `confidence` は heuristic なので、

- walk-forward 再評価
- promotion 基準との接続
- calibrated score への改善

が進むと、H1 tactical bias の昇格条件をより明示しやすくなる。

### 11.5 `predicted_path` の扱いをさらに明示する
`predicted_path_type` と `predicted_path_source_horizons` は、現在すでに `main_flow_gate.py` の signal result と `RealtimeFlow.py` の観測ログへ反映されている。

今後は metadata を返すだけでなく、runtime / backtest parity の確認軸としてどう使うかを決めるのが次の論点になる。
現在は `predicted_path_type` と `predicted_path_source_horizons` を返すことで、
少なくとも「何由来の path か」は result から読めるようにしている。

そのうえで、`predicted_path` の重要度がさらに増すなら、将来的には

- 補間 path のまま使う
- dense horizon 予測へ変える
- `path_type` などの metadata を追加する

のどれかを明示した方がよい。

特に main flow gate の根拠として使うなら、runtime と backtest の parity 文脈でも定義を固定する必要がある。

## 12. 現時点での読み方
このモジュールを読む時は、次の順で追うと理解しやすい。

1. [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py) の `evaluate_h1_forecast()`
2. [LSTMModel.py](../Src/Framework/ForecastSystem/LSTMModel.py) の `PredictMultiHorizonForecast()`
3. [RealtimeFlow.py](../Src/Framework/RealtimeFlow.py) の `h1_state` 更新箇所
4. [final_decision.py](../Src/Framework/ROModule/final_decision.py) の H1 alignment 判定
5. [main_flow_gate.py](../Src/Framework/ROModule/main_flow_gate.py) の `predicted_path` 利用箇所

## 13. まとめ
`h1_forecast.py` は現在、

- H1 tactical bias の runtime 採用判定
- downstream 共通境界
- active artifact の可観測性確保

を担うモジュールである。

言い換えると、

- 「予測器そのもの」ではなく
- 「予測器を SGSystem の判断ロジックへ接続するための翻訳層」

として理解すると、現在の責務も今後の拡張方針も掴みやすい。

## 14. Structural Refactor Closure (2026-04-09)
This document is now scoped to the H1 runtime structural refactor only.
Performance tuning, promotion criteria redesign, confidence calibration, and walk-forward evaluation are intentionally out of scope here and should be reviewed in a separate performance discussion after the structural work is closed.

### 14.1 Done Criteria
The H1 runtime structural refactor is considered complete when all of the following are true:

- `h1_forecast.py` is a thin facade that only wires the runtime entry point and public exports.
- preprocessing helpers are extracted from `h1_forecast.py` into dedicated helper modules.
- downstream consumers read shared H1 semantics through `build_h1_runtime_view()` and `evaluate_h1_alignment()`.
- phase/debug scripts use the shared H1 runtime view instead of reinterpreting H1 ad hoc.
- dedicated semantic tests cover `forecast_role`, `bias_direction`, `bias_ready`, and `predicted_path` metadata semantics.
- the remaining work for H1 performance is tracked separately from the runtime structure.

### 14.2 Closure Status
As of 2026-04-09, the structural scope above is complete.

- facade role: completed
- contract layer extraction: completed
- policy layer extraction: completed
- runtime orchestration extraction: completed
- preprocessing helper extraction: completed
- downstream shared-field adoption: completed
- phase/debug shared-contract adoption: completed
- H1 semantic regression coverage: completed

### 14.3 Deferred Topics
The following items are intentionally deferred until the next discussion:

- confidence redesign and calibration
- tactical-bias threshold redesign
- promotion criteria review
- raw signal regeneration after parity changes
- walk-forward evaluation and candidate comparison
- possible redesign of sparse-horizon `predicted_path`
