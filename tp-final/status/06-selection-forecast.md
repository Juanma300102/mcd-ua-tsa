# Phase 06 — Comparison, selection and final forecast

Status: done

## Objective

Select one model per series from validation evidence, confirm on the held-out test week, and produce the final forecast.

## Selection rule (decided with the user, revises the original 06.2 sketch)

- Pool: all 32 models per series from phases 02-05 (naive/classical/SARIMA baselines, ML, DL, Prophet/hybrid/AutoML/foundation).
- Ranking metric: validation MAE (primary, per the phase 02 protocol). Test metrics are reported for every model but never used to select.
- **Validation contamination**: a model is flagged (not excluded from the tables/figures, excluded only from eligibility) when its validation score was measured on the exact window used to tune its own hyperparameters or choose its own members. Verified against code/status, not assumed — see Decisions.
- **Selected model per series = lowest validation MAE among eligible models. No parsimony tie-break** — the user explicitly wants the top performer, not the simplest model near the top.
- Worst performer per series = highest validation MAE, also restricted to eligible models (a contaminated model's score is optimistic, not pessimistic, so it is never honestly "the worst").

## Tasks

- [x] 06.1 Consolidated table per series: all 32 models, validation and test metrics, fit time, contamination flag/reason, val/test rank — `tsa_final.selection.load_pool`/`add_contamination_flags`/`add_rank_columns`, persisted at `results/06_all_models.csv`.
- [x] 06.2 Selection rule: best validation MAE among eligible models, **no parsimony/cost tie-break** (revised from the original sketch — see Decisions).
- [x] 06.3 Validation vs. test consistency: Spearman rank correlation (all models and eligible-only) per series, plus the models with the largest rank divergence.
- [x] 06.4 Comparison against the TP1 SARIMA refit ("foundational" in the user's vocabulary) and against Chronos-Bolt ("foundation model"): % improvement of the selected model on validation and test.
- [x] 06.5 Refit the selected/worst/TP1-foundational/foundation-model roles on all available clean data (`data.load_clean`) and forecast the next 48h, with prediction intervals (native where available, empirical otherwise).
- [x] 06.6 Figures: best-per-family bar chart, validation-vs-test MASE scatter (family-colored, contamination and selection/worst marked), test-window overlay (selected/worst/TP1/Chronos vs. actual), final 48h forecast with interval.
- [x] Sensitivity table (informational only, explicitly not the selection rule): adopted rule vs. including contaminated scores vs. test-MAE oracle.

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| Selected model = lowest validation MAE among eligible models, no parsimony tie-break (revises the original 06.2 sketch, which proposed an information-criterion/parsimony tie-break) | Explicit user decision for this phase: the top performer is wanted, not the simplest model near the top. With no tie ever this close in either series' eligible ranking, a tie-break rule would not even have been invoked, but the rule itself is recorded as the user's choice, not silently dropped | TP1-style tie-break by information criteria/parsimony among near-tied validation scores |
| Contamination list fixed to exactly: `*_tuned` (phase 03, all of them, both series), `stacking_top3` (phase 03), `ensemble_equal_weight` (phase 05) — `prophet_tuned` explicitly excluded from the list | Verified against code and status docs, not assumed: `tsa_final.ml_models.tune_hyperparameters` scores Optuna trials with a `TimeSeriesFold` cut exactly at the train/val boundary — the selection window itself (`ml_models.py:239-282`); `notebooks/03_ml_models.ipynb` builds `stacking_top3` from the top-3 `*_tuned` models by validation MAE (inherits their leakage, plus its own member-selection leakage); `status/05-hybrid-automl-foundation.md` states `ensemble_equal_weight`'s members are "best-by-validation-MAE per phase". `prophet_tuned` tunes on a separate 168h leakage-safe window immediately before the selection window (`status/05-hybrid-automl-foundation.md` task 05.1, the same pattern phase 04's DL early stopping used) — never the selection window itself, so it is not flagged | Flagging every `_tuned`/AutoML/DL model regardless of which window it was scored on (over-broad, would have wrongly excluded `prophet_tuned` and the phase 04/05 leakage-safe-tuned models); flagging nothing (under-broad, would let `catboost_tuned`'s val MASE 0.353 — which collapses to 2.313 on test, phase 03 — win the ALB selection) |
| Worst performer restricted to eligible models (never a contaminated one) | A contaminated model's validation score is optimistic (tuned/selected on that exact window), so it can never honestly be reported as "the worst" — the true worst-elegible score is still a fair, uncontaminated number | Reporting the literal highest val MAE regardless of contamination — would occasionally have picked a contaminated model as "worst" purely because tuning happened to make it worse, which misrepresents what contamination means |
| Empirical intervals (naive/seasonal-naive/average/drift roles) built from the pooled phase 02 validation-window residuals (n=48), added as constant offsets to the final point forecast — not time-varying, not centered/bias-corrected | Simplest honest method available with the data at hand (a single 48h validation window, the same one phases 02-05 already flagged as noisy); the task explicitly asks to "keep it simple and honest" rather than build a calibrated interval machine for one phase | Bootstrapping/block-bootstrapping the residuals for a smoother interval — more code for a single 48h sample that would not materially change the honesty of the result; a full backtesting-based interval — reopens the expanding-window-backtesting cost tradeoff phase 02 already closed |
| Chronos-Bolt's final-forecast interval reports only the native 80% level (`quantile_levels=[0.1, 0.9]`); the 95% column is left `NaN`, not fabricated | Verified with a smoke test: Chronos-Bolt was trained on quantile levels in `[0.1, 0.9]` only (library warning); requesting `0.025`/`0.975` silently clips to `0.1`/`0.9`, so a naive "95% interval" would be numerically identical to the 80% one — reporting that as 95% would misrepresent the model's actual uncertainty | Requesting 0.025/0.975 anyway and labeling it "95%" — technically returns a number, but that number is not what a 95% interval means here, which fails the "honest" half of the instruction |
| Test-window overlay figure (06.6) and the final 48h forecast (06.5) both reuse the exact, unmodified `Fit` closures from `tsa_final.baselines`/`tsa_final.hybrid_models` (`auto_arima_fit`, `make_sarima_fit`, `chronos_zero_shot_fit`, `seasonal_naive_fit`, `average_fit`, `drift_fit`) | Matches the task's "reuse refits sensibly, keep runtime low" instruction and the phase's "import, don't modify" constraint on earlier-phase modules; no new model code needed since the four roles per series (selected/worst/TP1/Chronos) all already have a `fit(train) -> predictor(horizon)` closure from phases 02/05 | Reimplementing SARIMA/AutoARIMA/Chronos fitting logic inside `tsa_final/selection.py` — pure duplication for models phase 02/05 already implement correctly |

## Evidence

Implemented in `tsa_final/selection.py` (pool loading/concatenation, contamination flags, ranking,
selection, Spearman consistency, sensitivity table, native/empirical interval helpers, final-forecast
and test-overlay builders), run end to end in `notebooks/06_selection_forecast.ipynb`
(`nbconvert --execute --inplace`, exit 0). Consolidated table (32 models x 2 series x 2 splits = 128
rows, with `val_contaminated`/`contamination_reason`/`val_rank`/`test_rank` columns) at
`results/06_all_models.csv`; final 48h forecast (4 roles x 48h x 2 series = 384 rows, `lower_80`/`upper_80`/`lower_95`/`upper_95`) at `results/06_final_forecast.csv`.

**Contamination**: 5 of 32 models per series are flagged — `catboost_tuned`, `random_forest_tuned`,
`lightgbm_tuned`, `stacking_top3`, `ensemble_equal_weight` (ALB); `ridge_tuned`, `lightgbm_tuned`,
`xgboost_tuned`, `stacking_top3`, `ensemble_equal_weight` (store_service). 27 eligible models remain
per series.

**Selection result**:

- **ALB**: selected = `auto_arima` (val MAE 18,807.5, val MASE 0.475, test MASE 2.058). Runner-up =
  `seasonal_naive_m24` (val MAE 19,369.0, val MASE 0.489) — a near-tie, `autogluon_timeseries` third
  (val MAE 19,748.5). Worst eligible = `average` (val MAE 149,951.0, val MASE 3.788). TP1 foundational
  = `SARIMA(1,1,1)x(1,1,1,24)` (val MASE 0.923, test MASE 1.526). Foundation model = `chronos_bolt_base`
  (val MASE 0.598, test MASE 2.007). Selected vs. TP1 SARIMA: **+48.5% on validation, -34.9% on test**
  (auto_arima is worse on test). Selected vs. `seasonal_naive_m24`: +2.9% on validation, -3.4% on test.
  If validation contamination were allowed, `catboost_tuned` would win (val MAE 13,994.4) — but its
  test MASE is 2.313 (phase 03), no better than the honest selection.
- **store_service**: selected = `seasonal_naive_m24` itself (val MAE 2,281.9, val MASE 0.480, test MASE
  1.443) — no non-naive model beats it without contamination. Runner-up = `prophet_default` (val MAE
  2,405.9, val MASE 0.506). Worst eligible = `drift` (val MAE 17,022.7, val MASE 3.582). TP1 foundational
  = `SARIMA(1,1,1)x(1,0,1,24)` (val MASE 0.647, test MASE 1.426). Foundation model = `chronos_bolt_base`
  (val MASE 0.816, test MASE 1.673). Selected vs. TP1 SARIMA: **+25.8% on validation, -1.2% on test**
  (SARIMA edges it out on test, within noise). Selected vs. itself: 0% by construction.
  `ensemble_equal_weight` is a statistical tie with the naive (val MAE 2,281.6 vs. 2,281.9) but is
  excluded by construction (its members were chosen by that same validation MAE).

**Best per family** (validation MASE, contaminated allowed but flagged) — ALB: naive
`seasonal_naive_m24` 0.489, classical `auto_arima` 0.475, SARIMA (TP1 refit) 0.923, ML `catboost_tuned`
0.353 (contaminated), ML (direct) `catboost_direct` 0.534, DL `nhits` 0.556, Prophet `prophet_default`
0.532, Hybrid `mstl_lightgbm` 0.716, AutoML `autogluon_timeseries` 0.499, Foundation `chronos_bolt_base`
0.598, Ensemble `ensemble_equal_weight` 0.404 (contaminated). store_service: naive/classical/SARIMA
`seasonal_naive_m24` 0.480 (best overall), `auto_arima` 0.532, SARIMA refit 0.647, ML `lightgbm_tuned`
0.573 (contaminated), ML (direct) `ridge_direct` 0.611, DL `nbeats` 0.638, Prophet `prophet_default`
0.506, Hybrid `mstl_lightgbm` 1.848 (worst family here), AutoML `autots` 0.508, Foundation
`chronos_bolt_base` 0.816, Ensemble `ensemble_equal_weight` 0.480 (contaminated, ties naive).

**Validation vs. test consistency**: Spearman rho (32 models, p-value) — ALB 0.237 (p=0.192),
store_service 0.654 (p<0.001); restricted to eligible models — ALB 0.346 (p=0.077), store_service
0.692 (p<0.001). Both are low-to-moderate (and, for ALB, not even statistically significant at
p<0.05), consistent with every earlier phase's finding that the single 48h validation window is
noisy (see `status/02-evaluation-baselines.md`). Largest val/test rank divergences (of 32): ALB —
`random_forest_tuned` (rank 3 -> 28, contaminated), the TP1 SARIMA refit (rank 23 -> 1, i.e. one of
the worst on val, the best on test), `nbeats` (rank 24 -> 4), `ridge` (rank 22 -> 3); store_service —
`random_forest` (rank 26 -> 5), `ridge_direct` (rank 8 -> 23), `ensemble_equal_weight` (rank 1 -> 15,
contaminated — the model that "wins" validation most decisively if contamination were ignored falls
to 15th on test), `lightgbm_tuned` (rank 7 -> 21, contaminated), `neuralprophet` (rank 14 -> 1).

**Sensitivity table** (informational, not the selection rule) — ALB: (a) adopted rule ->
`auto_arima`; (b) including contaminated -> `catboost_tuned`; (c) test-MAE oracle -> the TP1 SARIMA
refit. store_service: (a) `seasonal_naive_m24`; (b) `ensemble_equal_weight`; (c) `neuralprophet`. Every
criterion picks a *different* winner in both series — concrete evidence that the selection criterion
matters, and that (b)/(c) are shown for discussion only, never as an alternative rule.

**Final 48h forecast** (2026-09-26 23:00 -> 2026-09-28 22:00 UTC, `results/06_final_forecast.csv`):
`auto_arima` (ALB, native statsforecast 80/95% intervals) and `seasonal_naive_m24` (store_service,
empirical 80/95% intervals from the 48h validation residuals) as the selected roles; `average` (ALB)
and `drift` (store_service) as the worst roles (also empirical intervals — both show intervals offset
away from the point forecast, itself a symptom of how badly biased these baselines are on recent data,
kept as reported evidence rather than corrected); the refit TP1 SARIMA spec per series (native
statsmodels 80/95% intervals); `chronos_bolt_base` per series (native 80% interval only — Chronos-Bolt's
trained quantile range is `[0.1, 0.9]`, so a fabricated 95% would just duplicate the 80% bounds, see
Decisions).

Figures: `informe/figuras/06_best_per_family.png`, `06_val_vs_test_scatter.png`,
`06_test_overlay_forecast.png`, `06_final_forecast.png` (all dpi 200, Spanish labels).

**Runtime**: kept low as instructed — the phase mostly reads `results/02-05_*.csv`; the only fits
performed are the 4 refits per series for the test-window overlay (06.6) and 4 refits per series for
the final forecast (06.5), all reusing existing, already-benchmarked `Fit` closures (auto_arima ~1-2
min per fit per phase 02's own measurement, everything else sub-second to a few seconds; see the
notebook's own timing for the exact figures).
