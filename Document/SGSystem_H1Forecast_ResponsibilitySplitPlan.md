# SGSystem H1 Forecast Responsibility Split Plan

更新日: 2026-04-09 JST

## 1. 目的
この資料は、[h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py) の責務を段階的に分離するための実装計画をまとめたものである。

現在の `h1_forecast.py` は runtime 上の重要な境界として機能している一方で、

- backend 呼び出し
- runtime policy
- downstream contract
- result 組み立て
- error fallback

が 1 ファイルに同居している。

この状態でも動作はするが、今後

- H1 tactical bias の昇格条件整理
- confidence policy の見直し
- `predicted_path` の意味付け強化
- downstream 読取経路の完全一本化

を進めるには、責務を分けておいた方が安全である。

## 2. 現状の責務棚卸し
2026-04-09 時点の [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py) には、主に次の責務が入っている。

### 2.1 入力前処理
- close 抽出
- recent diff 計算
- recent momentum / trend consistency 作成

該当関数:

- `_extract_close_list`
- `_build_close_diff_list`
- `_build_recent_direction_features`

### 2.2 runtime contract
- H1 result の有無確認
- runtime view 生成
- H2 と H1 の alignment 判定

該当関数:

- `has_h1_forecast_result`
- `build_h1_runtime_view`
- `evaluate_h1_alignment`

### 2.3 model payload policy
- confidence 閾値解決
- backend payload の normalize
- raw_features 構築
- `SUCCESS / NEUTRAL / INSUFFICIENT_DATA` 分類

該当関数:

- `_resolve_confidence_min`
- `_build_raw_features`
- `_normalize_model_payload`
- `_classify_model_decision`

### 2.4 result assembly
- forecast decision dict 組み立て
- module result dict 組み立て
- error fallback 組み立て

該当関数:

- `_build_forecast_decision`
- `_build_module_result`
- `_build_insufficient_data_decision`
- `_build_error_result`

### 2.5 orchestration
- backend 必要本数確認
- backend 推論呼び出し
- top-level try/except

該当関数:

- `_evaluate_h1_direction`
- `evaluate_h1_forecast`

## 3. 目標形
責務分離後の理想形は、「public facade は維持しつつ、中身を 3 層に分ける」構成である。

### 3.1 facade layer
ファイル:

- [h1_forecast.py](../Src/Framework/ROModule/h1_forecast.py)

責務:

- 既存 import path を維持する
- public API を re-export する
- top-level orchestration を持つ

ここは downstream 互換維持のため、Phase 1 では残す。

### 3.2 contract layer
想定ファイル:

- `Src/Framework/ROModule/h1_forecast_contract.py`

責務:

- `has_h1_forecast_result`
- `build_h1_runtime_view`
- `evaluate_h1_alignment`
- H1 downstream contract の固定化

この層は backend に依存せず、pure に近い helper として保つ。

### 3.3 policy layer
想定ファイル:

- `Src/Framework/ROModule/h1_forecast_policy.py`

責務:

- confidence 閾値解決
- model payload normalize
- raw_features 組み立て
- `forecast_status` 分類
- reason code / summary policy

ここは「H1 model output を tactical bias として採用するか」という意味付けを担う。

### 3.4 runtime orchestration layer
想定ファイル:

- `Src/Framework/ROModule/h1_forecast_runtime.py`

責務:

- backend 必要本数確認
- backend 呼び出し
- decision / module result の組み立て
- error fallback

必要なら将来的に backend 関数を引数注入しやすい形へ寄せる。

## 4. 分離順序
一気に分けるのではなく、下流互換を壊しにくい順で切る。

### 現在の進捗
- Phase 1: `downstream contract` 切り出し済み
- Phase 2: `policy layer` の初期切り出し済み
- Phase 3: `runtime orchestration` の初期切り出し済み
- Phase 4: downstream consumer の shared field 活用を開始

### Phase 1
まず `downstream contract` を切り出す。

対象:

- `has_h1_forecast_result`
- `build_h1_runtime_view`
- `evaluate_h1_alignment`

理由:

- `RealtimeFlow.py`
- `m15_entry.py`
- `final_decision.py`
- `main_flow_gate.py`

が依存しているのは主にここだからである。

先に contract を固定すると、その後で H1 core の中身を動かしても downstream の読み方を崩しにくい。

### Phase 2
次に `policy layer` を切り出す。

対象:

- `_resolve_confidence_min`
- `_build_raw_features`
- `_normalize_model_payload`
- `_classify_model_decision`

この段階で `h1_forecast.py` の意味付けロジックを独立させる。

現時点では、以下を [h1_forecast_policy.py](../Src/Framework/ROModule/h1_forecast_policy.py) へ移している。

- raw_features 構築
- model payload normalize
- confidence を用いた分類
- insufficient data decision の組み立て

一方で、top-level orchestration と module result assembly はまだ `h1_forecast.py` 側に残している。

### Phase 3
最後に orchestration を切り出す。

対象:

- `_evaluate_h1_direction`
- `_build_error_result`
- `evaluate_h1_forecast`

この時点で `h1_forecast.py` は薄い facade に近づく。

現時点では、以下を [h1_forecast_runtime.py](../Src/Framework/ROModule/h1_forecast_runtime.py) へ移している。

- H1 runtime evaluation orchestration
- insufficient data / normal path の dispatch
- error fallback
- module result assembly

一方で、入力前処理 helper はまだ `h1_forecast.py` 側に残している。

## 5. 互換ルール
分離時に守るべき互換条件は次の通り。

1. public import path は当面 `Framework.ROModule.h1_forecast` を維持する
2. `evaluate_h1_forecast()` の返り値構造を変えない
3. `build_h1_runtime_view()` の返り値構造を変えない
4. `evaluate_h1_alignment()` の返り値構造を変えない
5. `test_h1_forecast_refactor.py` の monkeypatch 前提をすぐには壊さない

特に 5 が重要で、現行テストは `h1_forecast.py` 上の `GetForecastSequenceLength` と `PredictMultiHorizonForecast` を patch している。
そのため orchestration や backend dependency の分離は、Phase 1 では行わない方が安全である。

## 6. 最初の実装スライス
今回の開始作業では、Phase 1 の contract layer 切り出しを最初の実装スライスとする。

### 6.1 切り出すもの
- `has_h1_forecast_result`
- `build_h1_runtime_view`
- `evaluate_h1_alignment`
- alignment 判定に必要な最小限の定数と confidence 閾値解決

### 6.2 まだ切り出さないもの
- `evaluate_h1_forecast`
- `_evaluate_h1_direction`
- `PredictMultiHorizonForecast` 呼び出し
- `_build_raw_features`
- error handling

### 6.3 期待効果
- H1 downstream contract が独立ファイルで見える
- consumer 観点の責務が分かりやすくなる
- 次の Phase で policy を切り出しやすくなる

## 7. 次の候補作業
補足として、当初候補だった `role-oriented field` の consumer 活用は、`main_flow_gate.py` と `RealtimeFlow.py` で初期反映まで完了している。

そのため次の候補作業は、「field を返す」段階ではなく、「その field をどの責務で読むか」をさらに整理する段階に入っている。
現在までに、role-oriented field と `predicted_path` metadata の初期追加も完了している。
この前提での次候補は以下である。

1. 入力前処理 helper をさらに切り出すか判断する
2. role-oriented field を consumer 側でどこまで活用するか決める
3. `predicted_path` metadata を main flow / backtest parity 文脈へ接続する
4. 入力前処理 helper を shared 化するか判断する
5. H1 confidence policy と promotion criteria を接続する

## 8. 完了条件
責務分離の各段階で、最低限以下を満たすことを完了条件とする。

- public API 互換維持
- downstream import 互換維持
- H1 helper の意味が docs と一致
- 既存 refactor テスト通過

## 9. まとめ
`h1_forecast.py` の責務分離は、

- まず contract
- 次に policy
- 最後に orchestration

の順で進めるのが最も安全である。

今回の開始作業では、そのうち最初の `downstream contract` 分離を着手点とする。

## 10. Closure Status (2026-04-09)
This plan is now closed for the H1 runtime structural refactor.
The remaining H1 performance topics stay out of scope for this plan and should be discussed separately after the structure is accepted.

### 10.1 Completed Items
- thin facade in `h1_forecast.py`
- extracted contract layer
- extracted policy layer
- extracted runtime orchestration layer
- extracted preprocessing helpers
- downstream adoption of shared H1 semantics
- phase/debug adoption of shared H1 runtime view
- dedicated H1 semantic regression test

### 10.2 Out of Scope After Closure
- confidence calibration
- threshold redesign for tactical bias
- promotion report and candidate comparison
- raw signal regeneration after parity changes
- walk-forward re-evaluation
- redesign of sparse-horizon `predicted_path`
