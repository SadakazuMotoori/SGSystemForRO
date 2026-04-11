# SGSystem H1 Performance Improvement Roadmap

Updated: 2026-04-10 JST

## 1. Purpose

This document defines the recommended next steps for improving `H1` performance after the structural refactor has been completed.

The goal is not to rush into retraining.
The goal is to improve `H1` as a tactical-bias layer in a way that is measurable, reviewable, and safe for runtime promotion.

## 2. Current View

My current view is:

- `H1` should still be treated as a tactical-bias / confirmation layer, not a medium-horizon trend oracle.
- the biggest immediate uncertainty is not model architecture, but the meaning of `confidence` and the runtime policy built on it
- the current `confidence` is a heuristic, not a calibrated probability
- the current `predicted_path` is a sparse-horizon interpolation, not a true dense path forecast
- the current training default still keeps `direction_loss_weight = 0.0`, so the model is still primarily regression-oriented

Because of that, the recommended order is:

1. fix evaluation and policy understanding first
2. measure what actually helps system behavior
3. calibrate or redesign confidence
4. only then spend time on retraining and promotion candidates

## 3. Performance Questions We Need To Answer

Before changing the model, we should answer these questions:

1. What `h1_confidence_min` actually improves end-to-end system quality?
2. Is `H1` helping mainly by blocking conflicts, or also by confirming aligned entries?
3. How much of the observed effect comes from `predicted_path` gating rather than `H1` direction itself?
4. Does `confidence` have a monotonic relationship with real decision quality?
5. After parity-related refactor changes, do regenerated raw signals still support the current policy?
6. Is there enough evidence to justify objective redesign and new candidate training?

## 4. Recommended Roadmap

### Phase 1: Freeze the evaluation line

Goal:
- keep the current baseline dataset and active runtime artifact as the evaluation reference line
- do not mix policy research and training changes in the same step

Actions:
- keep using the current baseline model artifact as the fixed comparison line
- keep performance experiments separate from runtime promotion
- treat any performance result before re-generated parity signals as provisional

Deliverable:
- one explicit baseline line for all following H1 comparisons

### Phase 2: Threshold sweep on the current model

Goal:
- find out whether the current model becomes useful simply by changing policy thresholds

Actions:
- sweep `h1_confidence_min` across a practical range
- evaluate at least:
  - entry count
  - no-trade ratio
  - hit rate
  - quality by decision bucket
  - consistency across periods or folds

Recommended first sweep:
- `0.50`
- `0.55`
- `0.60`
- `0.65`
- `0.70`
- `0.75`

Key decision rule:
- reject settings that improve hit rate only by killing too many trades

Deliverable:
- threshold comparison table for the current baseline artifact

Detailed execution plan:
- `Document/SGSystem_H1Performance_Phase2_ThresholdSweep_ExecutionPlan.md`

### Phase 3: H1 ablation study

Goal:
- isolate where H1 value actually comes from

Actions:
- compare these policy modes:
  - `H1 off`
  - `H1 conflict only suppress`
  - `H1 align + conflict`
  - `H1 align + conflict + predicted_path gate`
- compare trade count, hit rate, and stability for each mode

My expectation:
- the most durable value is likely to come from high-confidence conflict suppression
- neutral states should probably degrade confidence rather than hard-kill every entry

Deliverable:
- one ablation summary that explains whether `H1` is mainly a blocker, confirmer, or path-gate companion

### Phase 4: Regenerate parity-era raw signals and measure gate effect

Goal:
- verify the current policy using data generated after the parity-related refactor changes

Actions:
- regenerate raw signals after the shared `main_flow_gate.py` path-gate integration
- measure the actual effect of the gate on:
  - accepted entries
  - blocked entries
  - blocked-trade quality
  - approved-trade quality

Why this matters:
- the logic base has been shared, but the effect should still be re-measured on regenerated signals
- promotion decisions should rely on actual post-parity evidence

Deliverable:
- a post-parity measurement report for `predicted_path` gate contribution

### Phase 5: Confidence calibration

Goal:
- make `confidence` interpretable enough to support promotion and policy design

Actions:
- bucket `confidence` and compare expected vs actual outcome quality
- confirm whether higher `confidence` really maps to better tactical-bias usefulness
- decide whether to keep heuristic confidence with calibration, or redesign confidence itself

Recommended first step:
- start with reliability-style bin analysis before adding more complex calibration methods

Deliverable:
- a confidence calibration note with recommended operating bands

### Phase 6: Candidate training experiments

Goal:
- only after policy and evaluation are understood, test whether training changes improve the tactical-bias role

Candidate directions:
- introduce non-zero `direction_loss_weight`
- compare baseline regression-heavy training vs direction-aware candidates
- test whether alternative objectives improve conflict suppression and aligned confirmation quality
- only add path-aware objectives if the project decides to keep relying on `predicted_path`

Important rule:
- do not judge candidates on model loss alone
- judge candidates on runtime-parity system behavior

Deliverable:
- candidate vs baseline comparison report

### Phase 7: Promotion decision

Goal:
- decide whether a candidate is good enough to become the next active H1 artifact

Promotion rule:
- do not promote on MAE alone
- do not promote only because H1 blocks more trades
- require meaningful quality gain relative to trade-count loss
- reject unstable candidates even if one fold looks good

Deliverable:
- promotion report
- explicit accept / reject decision

## 5. What I Would Do First

If we start now, I would do these in order:

1. define the exact evaluation matrix for threshold sweep and ablation
2. run `h1_confidence_min` sweep on the current baseline artifact
3. run `H1 off / conflict only / align+conflict / path-gate on` ablation
4. regenerate post-parity raw signals and measure the gate effect
5. only then decide whether confidence calibration alone is enough, or whether retraining is justified

## 6. What I Would Not Do Yet

I would not do these first:

- immediate architecture redesign
- immediate horizon redesign
- immediate dense-path redesign
- immediate auto-promotion logic
- candidate promotion based only on offline regression metrics

## 7. Expected Outcome

If we follow this roadmap, we should get:

- a clear answer on whether the current H1 model is already usable with better policy
- a clear estimate of how much value comes from conflict suppression vs path gating
- an evidence-based threshold for tactical-bias adoption
- a cleaner basis for deciding whether retraining is worth the effort
- a safer promotion process for future H1 candidates

## 8. Bottom Line

My recommendation is:

- do not start with retraining
- start with threshold sweep, ablation, and post-parity measurement
- treat confidence design as the main performance bottleneck
- retrain only after the system-level evaluation says what kind of improvement is actually needed
