# SGSystem H1 Performance Phase 2 Threshold Sweep Execution Plan

Updated: 2026-04-10 JST

## 1. Objective

This plan turns Phase 2 of the H1 performance roadmap into an executable evaluation procedure.

The objective is:

- evaluate the current active H1 artifact without retraining
- measure whether policy-only threshold changes improve end-to-end system behavior
- choose the most defensible operating range for `h1_confidence_min`

This phase is intentionally limited to threshold research on the current baseline artifact.

## 2. In Scope

- sweep `h1_confidence_min`
- keep the current active H1 model fixed
- run backtests under the current shared `main_flow_gate.py` logic
- collect raw-signals outputs and summary metrics
- compare trade-count loss vs hit-quality gain

## 3. Out of Scope

- model retraining
- objective redesign
- `predicted_path` redesign
- H1 ablation modes from Phase 3
- automatic promotion
- replacing the active runtime artifact

## 4. Fixed Evaluation Conditions

To keep the sweep interpretable, all of the following must stay fixed across runs:

- symbol: `USDJPY`
- backtest window: `2025-10-01 00:00:00` to `2026-03-07 23:59:59`
- active H1 artifact: unchanged from current `active_runtime.json`
- `future_hours`: `2`
- H2 thresholds: unchanged
- M15 thresholds: unchanged
- `m15_predicted_path_gap_threshold_pips`: unchanged
- code revision: unchanged during the full sweep

Only this field should change across sweep runs:

- `h1_confidence_min`

## 5. Sweep Matrix

Recommended first-pass sweep values:

- `0.50`
- `0.55`
- `0.60`
- `0.65`
- `0.70`
- `0.75`

`0.65` is the current reference line and should be treated as the comparison baseline.

## 6. File And Output Convention

Recommended threshold config directory:

- `Asset/Config/experiments/h1_phase2_threshold_sweep/`

Recommended output directory:

- `Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/`

For each threshold, create:

- one threshold JSON file
- one raw-signals CSV
- one evaluator text summary
- one row in the aggregate comparison table

Recommended threshold config filenames:

- `thresholds_h1conf_050.json`
- `thresholds_h1conf_055.json`
- `thresholds_h1conf_060.json`
- `thresholds_h1conf_065.json`
- `thresholds_h1conf_070.json`
- `thresholds_h1conf_075.json`

Recommended per-run output layout:

- `Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/h1conf_050/raw_signals.csv`
- `Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/h1conf_050/eval_summary.txt`
- `Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/h1conf_055/raw_signals.csv`
- `...`

Recommended aggregate comparison file:

- `Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/phase2_threshold_comparison.md`

## 7. Threshold Config Rule

Each threshold JSON should start from the same base file:

- `Asset/Config/thresholds_backtest_loose_01.json`

Only one field should change per file:

- `"h1_confidence_min"`

No other threshold should be changed during this phase.

## 8. Execution Steps

### Step 1: Prepare threshold JSON files

Create six copies of the base backtest threshold file and change only:

- `"h1_confidence_min": 0.50`
- `"h1_confidence_min": 0.55`
- `"h1_confidence_min": 0.60`
- `"h1_confidence_min": 0.65`
- `"h1_confidence_min": 0.70`
- `"h1_confidence_min": 0.75`

### Step 2: Run backtest for each threshold

Use this command template:

```powershell
python Src/Backtest/Scripts/run_backtest.py `
  --symbol USDJPY `
  --start "2025-10-01 00:00:00" `
  --end "2026-03-07 23:59:59" `
  --thresholds Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_050.json `
  --output Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/h1conf_050/raw_signals.csv `
  --future-hours 2 `
  --verbose
```

If history cache is available, prefer cache-backed runs to reduce MT5 dependence:

```powershell
python Src/Backtest/Scripts/run_backtest.py `
  --symbol USDJPY `
  --start "2025-10-01 00:00:00" `
  --end "2026-03-07 23:59:59" `
  --thresholds Asset/Config/experiments/h1_phase2_threshold_sweep/thresholds_h1conf_050.json `
  --output Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/h1conf_050/raw_signals.csv `
  --future-hours 2 `
  --history-cache-dir <CACHE_DIR> `
  --prefer-history-cache `
  --verbose
```

Repeat the same command for all six threshold files.

### Step 3: Summarize each raw-signals CSV

Use this command template:

```powershell
python Src/Backtest/Scripts/evaluate_signals_backtest.py `
  Src/Backtest/Output/experiments/h1_phase2_threshold_sweep/h1conf_050/raw_signals.csv
```

Capture the printed summary into a sibling text file for each threshold.

### Step 4: Build the aggregate comparison table

For each threshold, record the fields below into one markdown table or CSV.

## 9. Required Metrics

### 9.1 Primary metrics

From `evaluate_signals_backtest.py`, record:

- `signal_count`
- `evaluated_signal_count`
- `no_signal_count`
- `no_signal_ratio`
- `enter_hit_rate`
- `correct_count`

### 9.2 Main-flow gate metrics

From `main_flow_gate_summary`, record:

- `gate_changed_count`
- `gate_changed_ratio`
- `base_signal_count`
- `final_signal_count`
- `blocked_signal_count`
- `retained_signal_count`
- `base_enter_hit_rate`
- `final_enter_hit_rate`
- `hit_rate_delta`
- `path_signal_ready_count`
- `path_signal_ready_ratio`
- `path_gap_threshold_passed_count`

### 9.3 Side-specific metrics

From direction-side summary, record:

- `ENTER_LONG.count`
- `ENTER_LONG.hit_rate`
- `ENTER_SHORT.count`
- `ENTER_SHORT.hit_rate`

### 9.4 Score-band metrics

Capture score-band hit-rate summaries for:

- `decision_score`
- `entry_score`

### 9.5 Reason-code review

Capture top counts for:

- `top_h1_reason_codes`
- `top_final_reason_codes`
- `top_m15_path_reason_codes`

This helps explain whether threshold changes are mainly:

- suppressing conflicts
- creating more neutral H1 states
- shifting path-gate behavior

## 10. Aggregate Table Template

Recommended comparison columns:

| threshold | signal_count | no_signal_ratio | enter_hit_rate | final_signal_count | blocked_signal_count | final_enter_hit_rate | hit_rate_delta | long_count | long_hit_rate | short_count | short_hit_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.55 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.60 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.65 |  |  |  |  |  |  |  |  |  |  |  | baseline |
| 0.70 |  |  |  |  |  |  |  |  |  |  |  |  |
| 0.75 |  |  |  |  |  |  |  |  |  |  |  |  |

## 11. Decision Rules

### 11.1 Red flags

Treat a threshold as unattractive if:

- `final_signal_count` drops sharply and `final_enter_hit_rate` barely improves
- one side improves while the other side becomes unstable
- `no_signal_ratio` rises materially without a meaningful quality gain
- reason codes show mostly additional blocking with weak quality evidence

Recommended red-flag heuristic:

- signal-count drop greater than `25%`
- hit-rate gain smaller than `2 percentage points`

### 11.2 Green flags

Treat a threshold as promising if:

- `final_enter_hit_rate` improves meaningfully
- signal-count loss stays moderate
- long and short sides remain reasonably stable
- improvements are not driven by only one small subset

Recommended green-flag heuristic:

- hit-rate gain at least `2 percentage points`
- signal-count drop no worse than `15%`

### 11.3 Tie-break rule

If two thresholds are close, prefer the one with:

- lower `no_signal_ratio`
- better side-balance
- smaller trade-count sacrifice
- clearer reason-code interpretation

## 12. Completion Criteria

Phase 2 is complete when all of the following are true:

- six threshold runs are completed
- six evaluator summaries are saved
- one aggregate comparison table is filled
- one recommended threshold or operating band is chosen
- one short conclusion note explains why that threshold was chosen

## 13. Expected Conclusions

This phase should answer one of these outcomes:

1. the current H1 model is already usable with a better threshold
2. the current H1 model remains weak even after threshold tuning
3. the current H1 model helps mainly as a conflict suppressor
4. the threshold effect is unclear until Phase 3 ablation and Phase 4 post-parity measurement

## 14. Recommended Next Step After Phase 2

After this sweep is complete:

- if one threshold clearly dominates, carry it into Phase 3 ablation as the new research default
- if results are ambiguous, keep `0.65` as the neutral reference line and proceed to Phase 3
- if every threshold looks weak, do not retrain immediately; run Phase 3 and Phase 4 first to isolate where H1 value is actually coming from
