# SGSystem Trend Refactor Implementation Plan

Updated: 2026-04-08 JST

## 1. Purpose

This document turns the literature-based review into an implementation plan for the current codebase.

The main design goal is to make the system stronger both research-wise and operationally without breaking the current runtime contract all at once.

The target interpretation is:

- `H2`: regime filter
- `H1`: tactical bias / short-horizon confirmation
- `M15`: execution timing
- `final_decision`: policy combiner

This is intentionally different from treating the whole system as a single "medium-term trend predictor".

## 2. Current Gaps

### 2.1 Role mismatch

- `h2_environment.py` is already close to a regime filter, but it is still described as a directional judge.
- `h1_forecast.py` predicts `t+6 / t+7 / t+8` hours and is therefore tactical, not medium-term.
- `RealtimeFlow.py` adds an M15 path-gap gate, which is execution logic rather than trend estimation.

### 2.2 Feature overlap in H2

Current H2 voting is based on:

- MA short vs MA long
- MA slope
- swing structure
- ADX filter

These are all price-derived features, and three of them overlap heavily in meaning. The current 3-vote structure looks diversified, but most evidence comes from the same underlying price path.

### 2.3 Time horizon mismatch

Current H2 indicators in `MTManager.py` are:

- `ma_short = MA(20)` on H2
- `ma_long = MA(50)` on H2
- `ma_slope = slope of MA(20)` with lookback `3`
- `adx = ADX(14)` on H2
- `swing_structure` from recent 4 H2 bars

This is closer to "multi-day directional filter" than "medium-term trend model".

### 2.4 Threshold design is too heuristic

Current thresholds such as `adx_min`, `trend_strength_min`, and fixed `30 pips` path-gap are hand-tuned operational values. They may be acceptable as temporary defaults, but they are not yet a research-grade configuration strategy.

## 3. Target Architecture

### 3.1 Module responsibilities

#### `external_filter.py`

- Remains the hard kill-switch layer.
- No change in responsibility.

#### `h2_environment.py`

- Becomes an explicit regime filter.
- It should answer:
  - Is the market in an acceptable directional regime?
  - If yes, which side is permitted?
  - How strong and how clean is the regime?

#### `h1_forecast.py`

- Becomes an explicit tactical bias layer.
- It should answer:
  - Does short-horizon model output support the currently permitted H2 side?
  - Is confidence sufficient to use it as confirmation?

#### `m15_entry.py`

- Becomes an execution timing layer only.
- It should answer:
  - Is the current M15 setup attractive enough to enter now?
  - It must not redefine the directional regime.

#### `final_decision.py`

- Becomes a policy combiner only.
- It should not reinterpret raw indicators.
- It should combine:
  - external safety
  - H2 regime permission
  - H1 tactical confirmation
  - M15 execution readiness

#### `RealtimeFlow.py`

- Keeps orchestration responsibility.
- Must make the phase boundaries visible in logs and state.

## 4. File-by-File Change Plan

### 4.1 `Src/Framework/MTSystem/MTManager.py`

#### Goal

Provide H2 regime features that are less redundant and better aligned with a regime filter.

#### Changes

- Keep current outputs for backward compatibility in phase 1.
- Add new H2 indicator fields under `market_data["H2"]["indicators"]`.

#### New fields to add

- `regime_ma_fast`
- `regime_ma_slow`
- `regime_ma_gap`
- `regime_slope_fast`
- `regime_adx`
- `regime_atr`
- `close_position_regime_window`
- `breakout_distance_atr`
- `volatility_state`

#### Notes

- Phase 1 should stay within existing H2 data.
- Optional phase 3 may add higher-timeframe context such as H4 or D1, but this should not be the first migration step.

### 4.2 `Src/Framework/ROModule/h2_environment.py`

#### Goal

Replace the current vote-based interpretation with an explicit regime-scoring model while preserving the existing output contract during migration.

#### New internal structure

- `_build_regime_components(_indicators, _thresholds)`
- `_score_regime_direction(_components, _thresholds)`
- `_classify_regime_quality(_components, _thresholds)`

#### New output fields

- `regime_direction`
- `regime_score`
- `regime_quality`
- `regime_components`
- `regime_reason_codes`

#### Backward-compatible fields to keep

- `env_direction`
- `trend_strength`

#### Compatibility rule

In phase 1:

- `env_direction` remains the primary field consumed by downstream modules.
- `trend_strength` remains available.
- New regime fields are added in parallel.

In phase 2:

- downstream modules start consuming `regime_direction` and `regime_score`
- `env_direction` becomes a compatibility alias

### 4.3 `Src/Framework/ROModule/h1_forecast.py`

#### Goal

Make the module semantics explicit: this is a tactical bias layer, not a medium-term trend oracle.

#### Changes

- Keep current model call unchanged in phase 1.
- Add role-oriented output fields:
  - `forecast_role = "TACTICAL_BIAS"`
  - `bias_direction`
  - `bias_ready`
  - `bias_alignment_hint`

#### Compatibility rule

- Keep `forecast_status`, `net_direction`, `confidence`, and `predicted_path`.
- Do not break existing `final_decision.py`, `m15_entry.py`, or debug scripts in phase 1.

### 4.4 `Src/Framework/ForecastSystem/LSTMModel.py`

#### Goal

Preserve current runtime behavior in phase 1, and prepare for a longer-horizon retraining path in phase 3.

#### Phase 1

- No behavioral change.
- Improve naming and payload metadata so runtime consumers can tell that current horizons are tactical.

#### Phase 3

- Introduce a separate model artifact for longer-horizon bias if the project decides to truly forecast medium-term direction.
- Do not silently overwrite the current model path.

#### Recommended artifact strategy

- Keep current model:
  - `h1_multi_horizon_patch_mixer.pt`
- Add future optional model:
  - `h1_tactical_bias_patch_mixer.pt`
  - or `h1_medium_horizon_bias_model.pt`

This avoids semantic confusion during migration.

### 4.5 `Src/Framework/ROModule/m15_entry.py`

#### Goal

Constrain the module to execution timing.

#### Changes

- Keep the current interface.
- Reframe internal scoring so that M15 never creates a new side by itself.
- It should only say whether the H2-permitted side is executable now.

#### Scoring adjustments

- Replace pure fixed-threshold logic with volatility-normalized logic where possible.
- Add optional ATR-normalized gates for:
  - spread penalty
  - breakout distance
  - path-gap readiness

#### Important migration note

Current fixed thresholds should remain available in phase 1:

- `m15_entry_score_min`
- `m15_noise_max`
- `m15_predicted_path_gap_threshold_pips`

New normalized thresholds should be introduced beside them, not instead of them.

### 4.6 `Src/Framework/ROModule/final_decision.py`

#### Goal

Make the combiner logic reflect the target architecture.

#### Changes

- Treat H2 strictly as permission.
- Treat H1 strictly as confirmation.
- Treat M15 strictly as execution readiness.

#### Decision policy target

- `external_filter` says no:
  - `NO_TRADE`
- H2 regime says no:
  - `NO_TRADE`
- H2 regime says yes, H1 unavailable:
  - allow configurable degraded path such as `WAIT` or lower-confidence entry policy
- H2 regime says yes, H1 aligned:
  - rely on M15 readiness
- M15 not ready:
  - `WAIT`

#### Compatibility rule

- Preserve current `final_action`, `decision_score`, `approved`, and `reason_codes`.
- Expand `details` first, then migrate policy semantics.

### 4.7 `Src/Framework/RealtimeFlow.py`

#### Goal

Reflect the new phase semantics in orchestration and logging.

#### Changes

- Rename logging language around H2/H1/M15 so their roles are obvious:
  - H2 regime update
  - H1 tactical bias update
  - M15 execution update
- Keep current update timing by confirmed bars.
- Move fixed-pips path-gap handling toward dual-mode support:
  - legacy fixed pips
  - normalized threshold mode

#### State additions

- `h2_regime_result`
- `h1_bias_result`
- `m15_execution_result`

In phase 1 these may point to the same underlying module outputs as the current state keys.

### 4.8 `Src/Framework/SGFramework.py`

#### Goal

Keep this file as the batch/static pipeline mirror of `RealtimeFlow.py`.

#### Changes

- Keep function name `RunDecisionPipeline`.
- Update returned dictionary keys only after realtime and debug paths are ready.
- In phase 1, add new semantic aliases while preserving old keys.

### 4.9 `Asset/Config/thresholds.json`

#### Goal

Separate legacy operational thresholds from new semantic thresholds.

#### Additions

- `h2_regime_score_min`
- `h2_regime_adx_min`
- `h2_regime_breakout_atr_min`
- `h2_regime_close_position_window`
- `m15_gap_threshold_mode`
- `m15_gap_threshold_pips`
- `m15_gap_threshold_atr_multiple`
- `h1_bias_confidence_min`

#### Compatibility rule

- Keep old names active in phase 1.
- Add translation logic so old and new keys can coexist.

## 5. Migration Phases

### Phase 1: Safe semantic refactor

#### Goal

Add structure and fields without changing runtime behavior.

#### Tasks

- Add new output fields to H2, H1, and realtime state.
- Keep old function names and return fields.
- Keep fixed-pips gate as default.
- Add alias thresholds.
- Update logs and summaries to reflect module roles.

#### Acceptance criteria

- Existing debug tests continue to pass.
- Existing realtime flow still behaves the same under current thresholds.

### Phase 2: Policy migration

#### Goal

Switch downstream logic to consume the new semantics.

#### Tasks

- `final_decision.py` consumes regime/bias/execution fields directly.
- `RealtimeFlow.py` and `SGFramework.py` use new aliases as first-class fields.
- Old names remain as compatibility fields for one migration cycle.

#### Acceptance criteria

- Old and new outputs are both present.
- Decision parity is measurable for the legacy threshold mode.

### Phase 3: Research upgrade

#### Goal

Improve actual predictive design rather than only renaming semantics.

#### Tasks

- Revisit H2 regime features with longer windows.
- Optionally add higher-timeframe context.
- Optionally retrain H1 with either:
  - longer horizons, or
  - explicit tactical-bias naming and artifact split
- Introduce walk-forward threshold calibration.

#### Acceptance criteria

- Out-of-sample evaluation against benchmarks
- cost-adjusted comparison
- regime-split analysis

## 6. Test Plan

### 6.1 Keep existing tests working

- `Src/Debug/test_phase1_flow.py`
- `Src/Debug/test_phase3_h2_h1_m15_integration.py`
- `Src/Debug/test_phase3_h2_h1_m15_integration_conflict.py`

### 6.2 Add new tests

- `Src/Debug/test_h2_regime_semantics.py`
- `Src/Debug/test_h1_bias_semantics.py`
- `Src/Debug/test_m15_execution_semantics.py`
- `Src/Debug/test_threshold_alias_compatibility.py`

### 6.3 Research validation tasks

- Compare against:
  - random walk sign benchmark
  - simple H2 MA regime baseline
  - H2-only filter
  - H2 + H1
  - H2 + H1 + M15
- Evaluate with:
  - walk-forward splits
  - cost-adjusted returns
  - max drawdown
  - turnover
  - hit rate by regime

## 7. Recommended Implementation Order

1. `thresholds.json` alias design
2. `MTManager.py` feature expansion
3. `h2_environment.py` regime fields
4. `h1_forecast.py` tactical bias fields
5. `m15_entry.py` execution semantics cleanup
6. `final_decision.py` policy cleanup
7. `RealtimeFlow.py` logging and state cleanup
8. `SGFramework.py` alignment
9. new debug tests

## 8. Out of Scope for the First Refactor

- MT5 connection redesign
- execution engine / position management implementation
- external event automation redesign
- replacing the current H1 model immediately

## 9. Summary

The first implementation step should not try to "make the strategy smarter" in one shot.

It should first make the architecture honest:

- H2 decides permission
- H1 provides tactical confirmation
- M15 decides timing

Once this semantic split is reflected in code and tests, the project can improve features, thresholds, and model horizons with much lower implementation risk.
