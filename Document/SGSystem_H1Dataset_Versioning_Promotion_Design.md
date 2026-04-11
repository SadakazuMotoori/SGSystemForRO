# SGSystem H1 Dataset / Versioning / Promotion Design

Updated: 2026-04-09 JST

## 1. Purpose

This document defines how the SGSystem `H1` dataset and model artifacts should be created, versioned, evaluated, and promoted.

The design goal is not only to make research easier.
It is also to protect runtime stability and refactor reproducibility while the architecture is still being cleaned up.

This document assumes the current intended module roles are:

- `H2`: regime permission
- `H1`: tactical bias / confirmation
- `M15`: execution timing
- `final_decision`: policy combiner

## 2. Current Findings

- Runtime `H1` reevaluation does not generate training data and does not retrain a model.
- Runtime uses the saved `H1` model only through `evaluate_h1_forecast()` and `PredictMultiHorizonForecast()`.
- Current `H1` horizons are `6 / 7 / 8` hours, which are tactical horizons, not medium-horizon trend forecasts.
- Current `predicted_path` is not a direct per-hour forecast path.
- It is a linearly interpolated path built from `last_close` and sparse horizon predictions.
- Current `confidence` is a heuristic built from directional dominance and magnitude, not a calibrated probability.
- Current training defaults keep `direction_loss_weight = 0.0`, so the model is still primarily a regression model.
- Current saved dataset and current saved summary are not fully self-consistent.
- Existing backtest flow and realtime flow are not fully aligned because realtime adds an additional `predicted_path` gate.

These findings imply that the present `H1` component should be managed as a tactical-bias artifact, not as an always-current self-refreshing forecaster.

## 3. Design Principles

- Reproducibility first during refactor.
- Research changes must not silently mutate the runtime baseline.
- No dataset or model should become active by filename overwrite alone.
- Artifact identity must be explicit and reviewable.
- Promotion must be based on end-to-end evidence, not model loss alone.
- Runtime and backtest logic must be brought into parity before strong model claims are trusted.
- `H1` should remain a confirmation layer until its training objective and evaluation protocol are upgraded.

## 4. Recommended Operating Model

The recommended strategy is a dual-lane operating model.

### 4.1 Baseline lane

- Purpose: protect refactor work and regression testing.
- Dataset window: fixed and explicit.
- Model artifact: frozen and versioned.
- Thresholds: fixed unless a deliberate baseline revision is approved.
- Usage: code refactor, compatibility tests, parity checks, architecture cleanup.

### 4.2 Research lane

- Purpose: improve the tactical-bias model and related policy logic.
- Dataset window: rolling and explicitly regenerated.
- Model artifact: versioned candidate artifacts only.
- Usage: experiments, walk-forward tests, alternative windows, objective redesign.
- Promotion: manual only after review.

This split gives the project two capabilities at once:

- stable refactor verification
- safe research iteration

## 5. Strategic Decision

The project should not choose between:

- always freezing the current fixed-period dataset forever
- always regenerating the latest dataset and immediately replacing runtime artifacts

Instead, it should do the following:

1. Freeze the current fixed-period dataset as the baseline reference line.
2. Keep all refactor work and regression tests pinned to that baseline.
3. Build a separate rolling research pipeline for new candidate datasets and models.
4. Promote only reviewed candidates into runtime.

This is the highest-effect strategy because it preserves comparability while still allowing actual model improvement.

## 6. Dataset Policy

### 6.1 Baseline dataset

The current fixed range should be kept, but only as a baseline artifact.

- Baseline start: `2025-10-01 00:00:00`
- Baseline end: `2026-03-07 06:00:00` for the current saved CSV
- Expected role: refactor baseline, not eternal ground truth

The baseline dataset should be copied into a versioned immutable path.

Recommended baseline dataset ID:

`h1ds_usdjpy_h1_20251001_20260307_seq32_tg6-7-8_baseline_v1`

Recommended baseline storage:

- `Src/Backtest/Output/datasets/h1/h1ds_usdjpy_h1_20251001_20260307_seq32_tg6-7-8_baseline_v1.csv`
- `Src/Backtest/Output/datasets/h1/h1ds_usdjpy_h1_20251001_20260307_seq32_tg6-7-8_baseline_v1.manifest.json`

### 6.2 Research datasets

Research datasets should be generated explicitly and versioned explicitly.

Recommended default research window:

- primary window: rolling `365` days
- secondary sensitivity window: rolling `540` days

Rationale:

- `365` days is recent enough for a non-stationary FX tactical model
- `365` days is already close to the builder default intent
- `540` days is useful only as a comparison line, not as the default active window
- an unbounded expanding window should not be the default because it dilutes recent market structure

### 6.3 No runtime generation

Dataset generation must never happen inside runtime reevaluation.

- no dataset generation in `RealtimeFlow.py`
- no model training in `RealtimeFlow.py`
- no automatic pre-forecast dataset refresh

Runtime must consume only a previously promoted artifact set.

### 6.4 Schema policy

Every dataset must declare its schema explicitly.

Minimum required manifest fields:

| Field | Meaning |
| --- | --- |
| `dataset_id` | Stable artifact identity |
| `schema_version` | Dataset schema version |
| `symbol` | Trading symbol |
| `timeframe` | Expected source timeframe |
| `sequence_length` | Input H1 bar count |
| `target_hours` | Forecast target hours |
| `start_jst` | Requested start timestamp |
| `end_jst` | Requested end timestamp |
| `row_count_raw` | Raw loaded row count |
| `row_count_eligible` | Eligible record count before save |
| `row_count_saved` | Saved CSV row count |
| `builder_script` | Builder file path |
| `builder_git_commit` | Git commit or revision marker |
| `generated_at_jst` | Generation timestamp |
| `source_type` | `mt5` or `csv` |
| `source_descriptor` | MT5 symbol or source CSV path |
| `notes` | Human-readable notes |

Recommended `schema_version` start:

`2`

Reason:

- current saved CSV appears to be a mixed-generation artifact
- future explicit manifests should mark a clean post-refactor schema

## 7. Artifact Versioning Policy

### 7.1 Never overwrite active artifacts silently

The following files should stop acting as silent mutable truth:

- `Src/Backtest/Output/datasets/h1_training_dataset.csv`
- `Asset/Models/h1_multi_horizon_patch_mixer.pt`
- `Asset/Models/h1_multi_horizon_patch_mixer_metadata.json`
- `Asset/Models/h1_multi_horizon_patch_mixer_summary.json`

They may remain as compatibility aliases for a transition period, but they should no longer be treated as authoritative storage.

### 7.2 Recommended model ID format

Recommended model ID:

`h1model_tactical_usdjpy_h1_seq32_tg6-7-8_<dataset_id>_<trainstamp>`

Example:

`h1model_tactical_usdjpy_h1_seq32_tg6-7-8_h1ds_usdjpy_h1_20251001_20260307_seq32_tg6-7-8_baseline_v1_20260409T210000JST`

### 7.3 Recommended model storage

- `Asset/Models/h1/<model_id>.pt`
- `Asset/Models/h1/<model_id>.metadata.json`
- `Asset/Models/h1/<model_id>.summary.json`
- `Asset/Models/h1/<model_id>.predictions.csv`

### 7.4 Active pointer

Runtime should eventually resolve the active `H1` artifact through an explicit pointer file.

Recommended pointer file:

- `Asset/Models/h1/active_runtime.json`

Recommended pointer content:

```json
{
  "role": "H1_TACTICAL_BIAS",
  "active_model_id": "h1model_tactical_usdjpy_h1_seq32_tg6-7-8_example",
  "model_path": "Asset/Models/h1/h1model_tactical_usdjpy_h1_seq32_tg6-7-8_example.pt",
  "metadata_path": "Asset/Models/h1/h1model_tactical_usdjpy_h1_seq32_tg6-7-8_example.metadata.json",
  "summary_path": "Asset/Models/h1/h1model_tactical_usdjpy_h1_seq32_tg6-7-8_example.summary.json",
  "dataset_id": "h1ds_usdjpy_h1_20251001_20260307_seq32_tg6-7-8_baseline_v1",
  "promoted_at_jst": "2026-04-09 21:00:00",
  "promotion_stage": "active",
  "approved_by": "manual"
}
```

This avoids the dangerous pattern of replacing a runtime file in place without preserving identity.

## 8. Promotion Stages

Every `H1` model artifact should move through explicit stages.

Recommended stages:

- `draft`
- `candidate`
- `active`
- `retired`

### 8.1 Draft

- training finished
- artifact saved
- metadata complete
- no promotion claim yet

### 8.2 Candidate

- dataset manifest complete
- training summary complete
- backtest parity check completed
- walk-forward or rolling split evaluation completed
- benchmark comparison recorded

### 8.3 Active

- manual approval completed
- active pointer updated
- promotion log written

### 8.4 Retired

- no longer active
- kept for traceability

## 9. Promotion Criteria

Promotion criteria should be layered.

### 9.1 Data integrity gate

- dataset manifest exists
- schema version recognized
- row counts internally consistent
- target-hour columns and inferred targets are consistent
- no unexplained mismatch between dataset, metadata, and summary

### 9.2 Model-quality gate

- compare against constant baseline
- compare against drift baseline
- review per-horizon direction accuracy
- review aggregate direction accuracy
- review correlation

Model-quality gate alone is not enough for promotion.

### 9.3 System-quality gate

Promotion should be based mainly on system behavior under runtime-parity logic.

Required comparisons:

- `H2` only baseline
- `H2 + M15` without `H1`
- `H2 + H1 + M15`
- `H2 + H1 + M15 + realtime path gate`

Required outputs:

- enter count
- no-signal ratio
- hit rate
- decision-score stratification
- stability across folds

### 9.4 Operational gate

- active model path resolves correctly
- metadata can be loaded by runtime
- `test_phase3_h1_forecast.py` or equivalent smoke test passes
- runtime logging remains readable

## 10. Promotion Decision Rule

Until execution and position-management layers exist, the project should use a provisional promotion rule.

Recommended provisional rule:

- never promote on model MAE alone
- require non-trivial improvement on runtime-parity system evaluation
- reject candidates that reduce trade count heavily without clear quality gain
- reject candidates that improve a single fold but degrade consistency

For the current system state, a practical rule is:

- do not accept a candidate only because `H1` hard-gates more trades
- require evidence that reduced trade count buys a meaningful gain in hit quality
- treat small hit-rate gains with large trade-count loss as non-promotion outcomes

This is especially important because current evidence suggests that hard `H1` gating may remove many trades while providing only modest quality improvement.

## 11. Backtest Parity Requirement

Backtest and realtime should evaluate the same effective decision path before `H1` promotion is taken seriously.

Current gap:

- `RealtimeFlow.py` applies an additional `predicted_path` gate
- `SGFramework.py` currently stops before that extra realtime gate

Required design rule:

- shared policy logic must live in a reusable location
- realtime and backtest must both call the same gate implementation

Until this parity is fixed, `H1` promotions should be labeled provisional.

## 12. Runtime Role of H1

`H1` should remain a tactical bias layer for now.

The project should not currently treat `H1` as:

- a medium-term oracle
- a fully calibrated directional probability model
- a hard trade creator on its own

Recommended runtime use:

- allow `H1` to confirm an `H2`-permitted side
- allow high-confidence conflict to suppress entries
- allow neutral `H1` states to reduce confidence, not necessarily kill all entries

This is the most honest use of the current model design.

## 13. Recommended Implementation Order

### Phase A: Freeze baseline artifacts

- create a versioned baseline dataset copy
- create a baseline dataset manifest
- create a versioned baseline model copy
- create a baseline model manifest or pointer

### Phase B: Stop silent overwrite behavior

- introduce explicit versioned paths for new datasets
- introduce explicit versioned paths for new model artifacts
- keep compatibility aliases only as transitional outputs

### Phase C: Add active runtime pointer

- update runtime loader to resolve the active model through a pointer file
- keep current default path as fallback only during migration

### Phase D: Fix runtime and backtest parity

- move the realtime-only `predicted_path` gate into shared decision logic
- make backtest consume the same gate path

### Phase E: Add promotion report generation

- produce a machine-readable promotion report
- summarize dataset identity, model identity, benchmark comparison, and decision outcome

### Phase F: Upgrade research protocol

- rolling dataset generation
- walk-forward evaluation
- candidate comparison against baseline

### Phase G: Optional model redesign

- add direct directional objective
- add path-aware objective only if the project keeps using `predicted_path`
- optionally separate tactical-bias and longer-horizon models into different artifacts

## 14. Recommended File Layout

Recommended long-term layout:

- `Src/Backtest/Output/datasets/h1/`
- `Asset/Models/h1/`
- `Asset/Models/h1/active_runtime.json`
- `Asset/Models/h1/promotion_log.jsonl`
- `Asset/Reports/h1/`

Recommended artifact ownership:

- dataset builder owns dataset manifests
- trainer owns model metadata and training summary
- promotion step owns active pointer and promotion log
- runtime owns only artifact consumption

## 15. Immediate Decisions For This Repository

The following decisions are recommended now:

1. Treat the current fixed-period dataset as a baseline-only artifact.
2. Do not regenerate dataset automatically before `H1` runtime reevaluation.
3. Do not overwrite `h1_training_dataset.csv` and claim it is the new truth.
4. Do not overwrite the current `pt / metadata / summary` triplet in place after experiments.
5. Keep `H1` semantics as tactical confirmation during the current refactor.
6. Fix runtime and backtest parity before trusting `H1` promotion decisions.

## 16. Summary

The strongest strategy for SGSystem is:

- freeze one explicit baseline line for refactor safety
- run a separate rolling research line for `H1`
- version every dataset and every model artifact
- activate models by promotion pointer, not overwrite
- judge `H1` by runtime-parity system behavior, not regression loss alone

This gives the project a stable foundation for refactoring now and a credible path to stronger model research later.
