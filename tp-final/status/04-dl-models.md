# Phase 04 — Deep Learning models

Status: done

## Objective

Neural forecasters on the same protocol, sized for ~2,300 hourly observations, trained on GPU (RTX 4060).

## Tasks

- [x] 04.1 Scaling pipeline (`np.log1p` + darts `Scaler`, fit on the exact data each fit call trains on, never on val/test) and input window 168 h / output chunk 48 h.
- [x] 04.2 LSTM (darts `BlockRNNModel(model="LSTM")`).
- [x] 04.3 N-BEATS and N-HiTS (darts).
- [x] 04.4 TCN, TiDE, TFT (darts), with calendar covariates.
- [x] 04.5 Optional TSMixer/PatchTST: skipped — the required six architectures already used the phase's time budget on real work (2 seeds x 2 series x 2 fits x 6 architectures = 48 fits, see Evidence); adding a 7th architecture was not worth the extra hour(s) under deadline pressure, and the task explicitly gates it on the six finishing "within budget", not on having literally zero time left.
- [x] 04.6 Early stopping on a leakage-safe tuning window (not the phase 02 selection window — see Decisions); fixed seeds (42, 43), with an automatic 3rd seed (44) whenever the 2-seed validation MAE spread (CV) exceeds 15 %.
- [x] 04.7 Logged per-seed rows to `results/04_dl_seeds.csv` and seed-averaged, total-fit-time rows to `results/04_dl.csv` (same schema as `results/03_ml.csv`).

## Time budget

Decided 2026-09-27, together with the phase 02 single-split protocol: DL is the largest cost driver in the whole TP (6 architectures x 2-3 seeds x 2 fits under the phase 02 protocol = up to ~36 fits per series), so it needs explicit caps, not just "early stopping on a validation window":

- Max epochs 100, early-stopping patience ~10-15 epochs on the validation-window loss.
- 2 seeds, not 3, by default; only run a 3rd seed for a given architecture if the 2-seed spread on validation MAE is large enough to matter for ranking.
- The 6 required architectures (04.2-04.4) run first; TSMixer/PatchTST (04.5) only if the required 6 finished within budget.
- If a single architecture x seed fit exceeds ~15 min wall-clock, treat that as a signal to cut epochs/patience for that model rather than letting it run — do not silently accept multi-hour fits.

## Risk

With ~96 days, large DL models may overfit or underperform the boosters. That is a valid finding for the report, not a failure.

## Class references

`PRACTICA_AST_Clase_6_1_Intro_LSTM`, `_6_2A_LSTM`, `_6_2B_LSTM`, `_6_7A_NBEATS`, `_7_5_DARTS`, `AST_Clase_8_ejercicio_3` (TiDE, TCN, TFT).

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| Early-stopping ("tuning") window = the 168 h immediately before `VAL_START`, carved from the end of whatever `train` is passed into `fit`; the test refit (on `train_and_val`) trains for the fixed recorded best-epoch count, no early stopping, no validation split | Phase 03's Optuna tuning scored trials on the *same* 48 h window later used to rank/select models (`evaluation.VAL_START`-`VAL_END`), which let tuned models overfit that specific window — they won validation and then lost badly on test (e.g. ALB `random_forest_tuned`: val MASE 0.42, test MASE 3.46; see `status/03-ml-models.md` Evidence). DL training loops touch the data far more times than a single Optuna trial refit, so the same leakage risk is larger, not smaller, if early stopping were allowed to watch the selection window. Carving a *different* 168 h window (immediately before, not overlapping, the selection window) for early stopping keeps the selection window completely unseen during training in both fit calls. This note is for phase 06: phase 03's Optuna tuning has this selection-bias issue and was not revisited (03 files are out of scope for phase 04) | (a) Early-stop on the same 48 h selection window (simplest, but repeats phase 03's exact bias); (b) no early stopping, fixed epoch count picked by hand — no principled way to pick it without either the selection window or a proxy window |
| One stateful `DlModelFit` closure per (architecture, seed), reused unmodified across `evaluation.evaluate`'s two calls: first call trains with early stopping and stores `best_epoch`; every later call trains for that fixed epoch count | Matches the `fit(train) -> predictor(horizon)` contract in `tsa_final.evaluation` without changing `evaluate()`; `evaluate()`'s call order (val fit, then test fit) is relied upon directly, same pattern phase 03's `evaluate_direct` used for its own contract constraint | A module-level cache keyed by a hash of `train` — more code for the same guarantee, since `evaluate()`'s call order is already fixed and documented |
| `np.log1p` before a darts `Scaler` (fit on the exact data each fit call trains on) | Both series are strictly positive, heavy-tailed hourly traffic counts with a visible ramp and intervention-adjacent spikes (phase 01); log1p compresses that range before the linear scaler sees it, standard practice for count-like series | Scaler only, no log transform — kept as an available fallback but not benchmarked separately under the time budget; MinMax/Standard on raw counts risks the scaler's fitted range being dominated by the highest-traffic hours |
| Calendar covariates as cyclical `hour_sin/cos`, `dayofweek_sin/cos` and `is_holiday` (drops `is_intervention`), routed to `future_covariates` or `past_covariates` per model from its own `supports_future_covariates`/`supports_past_covariates` flag | `is_intervention` is always 0 in any future window (task instruction) — a constant feature adds nothing to a future-covariate model and dispatching by the model's own capability flags keeps every architecture wired identically instead of branching by name | Raw `hour`/`dayofweek` integers (no cyclical encoding) — breaks the hour-23-to-hour-0 and Sunday-to-Monday adjacency for models without native categorical/day encoders |
| Seed-averaged metrics (not mean-of-predictions) for `results/04_dl.csv`; `fit_time_s` summed across seeds | Averaging the metrics `evaluate()` already computed is direct; averaging raw predictions across seeds and recomputing MAE/RMSE/MAPE/MASE from that averaged array would require reimplementing metric computation outside `evaluate()` for no material benefit (MAE is linear in the prediction either way). `fit_time_s` is summed, not averaged, because it should reflect total compute spent on that architecture x series x split, matching the "record total fit time" instruction | Mean-of-predictions before scoring — technically also valid, but adds a second code path around `ev.metrics` |
| Architecture sizes shrunk below darts defaults (e.g. `NBEATSModel` default `num_stacks=30` -> 3 here; `NHiTSModel` default `layer_widths=512` -> 128) | ~2,000 hourly observations is a small dataset for the default-sized architectures, which target much longer series; oversized models risk both overfitting and blowing the per-fit time budget for no expected accuracy gain | Keeping darts defaults and letting early stopping alone control overfitting — leaves capacity mismatched to data size and risks the ~15 min single-fit budget on the two heaviest architectures (TFT, LSTM) |
| Optional TSMixer/PatchTST (04.5) skipped | The six required architectures already spend the phase's realistic time budget (48 fits total: 6 architectures x 2 seeds x 2 series x 2 evaluate() calls, plus a handful of automatic 3rd-seed reruns); the task gates the 7th/8th architecture on the six finishing "within budget", which they did, but under deadline pressure the marginal report value of a 7th architecture did not justify the additional wall-clock and notebook-execution risk | Running TSMixer anyway "since there was time" — would have added compute cost with no requirement behind it |

## Evidence

Implemented in `tsa_final/dl_models.py` (covariate builder, `_BestEpochTracker`, `ARCHITECTURES`
registry, `DlModelFit`), run end to end in `notebooks/04_dl_models.ipynb` (executed, outputs kept).
Every architecture plugs into the existing `fit(train) -> predictor(horizon)` contract and is
scored with the unmodified `tsa_final.evaluation.evaluate` (two fits per model: `train` for
validation, `train_and_val` for test) — no new evaluation framework, no extra folds, no
hyperparameter search for DL. Full table (6 architectures x 2 series x 2 splits = 24 rows,
seed-averaged) at `results/04_dl.csv`; per-seed rows (48) at `results/04_dl_seeds.csv`.

**Time budget calibration (done before the full run, not after)**: one validation fit per
architecture was measured on ALB with the real configuration (100 max epochs, patience 12, batch
size 32) before committing to it: TFT (the slowest) took 150 s (`best_epoch=16`); the other five
took 9-26 s. Projecting 2 seeds x 2 series x (validation fit + a cheaper fixed-epoch test refit)
gave ~24 min, under the ~30 min target. `batch_size=128` was tried against TFT specifically and
discarded (188 s, `best_epoch=36` — a larger batch needed more epochs to reach the same `val_loss`
at the same learning rate, which more than offset having fewer batches per epoch on this
~1,500-sample training set). The actual full run: 1689 s (~28 min) of DL fit time across every
architecture/seed/series/split, `nbconvert --execute` wall clock 30 min 53 s (includes data
loading, the two extra single-model refits used only for the test-forecast figure, and result/figure
saving) — right at the ~30 min target, not exceeded.

Validation ranking (MASE, selection criterion) — **no DL architecture beats `seasonal_naive_m24`
or the best phase 02-03 model on validation, in either series**:

- **ALB**: best DL is `nhits` (MASE 0.875) vs. `seasonal_naive_m24` (0.489) vs. `catboost_tuned`
  (0.353, the phase 03 winner). Full DL validation ranking: `nhits` 0.875, `tide` 0.998, `nbeats`
  1.119, `tft` 1.516, `tcn` 2.323, `lstm` 2.337.
- **store_service**: best DL is `tcn` (MASE 0.760) vs. `seasonal_naive_m24` (0.480, which is also
  the best phase 02-03 model here — no phase 03 model beat it either). Full DL validation ranking:
  `tcn` 0.760, `lstm` 1.017, `nhits` 1.097, `tide` 1.211, `tft` 1.342, `nbeats` 1.444.

Test ranking (informational, per protocol) **reverses this**: `nbeats` is the best model of the
*entire* project so far on test, for both series, beating both the naive baseline and the best
phase 02-03 model:

- **ALB**: `nbeats` test MASE 1.744 vs. `seasonal_naive_m24` 1.990 vs. `catboost_tuned` 2.313 (the
  same model that won validation in phase 03 and collapsed on test — see `status/03-ml-models.md`).
- **store_service**: `nbeats` test MASE 1.247 vs. `seasonal_naive_m24`/best-pre-DL 1.443.

This is the same validation/test divergence phases 02 and 03 already documented (single 48h
window, not N-fold) — read as further evidence for that risk, not a phase 04 anomaly. It is also a
concrete argument for phase 06 to look at more than the single validation winner before discarding
any model family.

**Seed spread** (validation MAE, 2 seeds): high for `lstm` and `nbeats` (CV up to 57 % on ALB),
moderate for the rest (`tcn`, `tide` under 25 % on both series). `lstm`'s early-stopping best
epoch is 1 on ALB for both seeds (the model barely trains before the tuning-window loss stops
improving); `nbeats`'s best epoch reaches 83-97 out of the 100-epoch cap on ALB, suggesting a
higher cap could still help it — not tried, time budget. Per the new phase 04 seed policy (fixed
2 seeds, no automatic 3rd seed regardless of spread — see Decisions), none of this dispersion
triggered extra fits.

**Training cost**: TFT dominates (223-333 s per architecture x series row, summed over 2 seeds) —
10-30x every other architecture — while never winning on either split or series; the most
expensive architecture was not the most accurate one. `results/04_dl.csv`'s `fit_time_s` records
this per row for phase 06.

Figures: `informe/figuras/04_val_mae_ranking.png` (validation MASE per architecture and series,
naive and best-pre-DL reference lines, best DL highlighted), `04_test_forecast_best_dl.png` (test
window, actual vs. forecast, for the best-by-validation-MASE DL architecture per series — `nhits`
for ALB, `tcn` for store_service, each refit once more for the plot), `04_best_epoch_table.png`
(best epoch per architecture x series x seed).
