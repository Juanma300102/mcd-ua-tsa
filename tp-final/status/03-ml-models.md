# Phase 03 — Machine Learning models

Status: done

## Objective

Tree-based and linear ML forecasters on lag/calendar features, evaluated with the phase 02 protocol.

## Time budget (hard constraint, deadline pressure)

Target: phase 03 end to end (feature engineering + every fit + Optuna + stacking + SHAP,
notebook run included) fits in a few hours, not days. Two sub-tasks are the actual risk
(everything else is seconds-to-minutes on ~2,000 rows of tree models) and get an explicit
time-box; if a time-box is blown, cut scope and note the gap as a reported limitation
rather than letting it run unbounded:

| Task | Risk | Mitigation | Time-box |
|---|---|---|---|
| 03.3 direct strategy | `ForecasterDirect` fits one sub-model per horizon step by default (up to 48 for a 48h horizon) — up to ~48x a recursive fit | `n_jobs=-1`; if still too slow, drop to the single best booster only (already the plan) and accept a coarser comparison | ≤ 20 min total (2 series x 2 fits) |
| 03.4 Optuna search | Open-ended trial count | Fixed `n_trials` (≈20) **or** a wall-clock timeout (≈3 min/model), whichever hits first; only run for models that already beat `seasonal_naive_m24` with default hyperparameters (03.2 ranking gates this) | ≤ 18 min total (≤3 models x 2 series x 3 min) |
| 03.5 stacking | `sklearn.StackingRegressor` default `cv=5` refits every base learner ~6x to build meta-features | Set `cv` explicitly (small integer, e.g. 3) instead of leaving the default | ≤ 5 min total |

Everything else (03.1 feature engineering, 03.2 recursive sweep with default hyperparameters,
03.6 SHAP on the single final winner, 03.7 logging) has no time-box because it is not expected
to be the bottleneck — same asymmetry phase 02 found with `auto_arima`.

## Sequencing (strict, not parallel — later tasks depend on earlier rankings)

```
03.1 features
  -> 03.2 recursive, all 5 (LightGBM/XGBoost/CatBoost/RandomForest/Ridge), default hyperparams
    -> rank by validation MAE
      -> 03.3 direct, ONLY for the #1 recursive booster (recursive vs direct comparison)
      -> 03.4 Optuna, ONLY for models that already beat seasonal_naive_m24 on validation
        -> 03.5 stacking of the top-3 tuned models (not all 5 — the 4th/5th place only adds
           noise to the meta-learner)
          -> pick the single overall winner across 03.2-03.5 (validation MAE)
            -> 03.6 SHAP on that one winner only (not on every model)
              -> 03.7 log every row (winners and losers) to results/03_ml.csv, same schema
                 as results/02_baselines.csv, family="ML" — full-table logging is the
                 reportable evidence, not just the winner (same convention as phase 02)
```

## Tasks

- [x] 03.1 Feature set: lags 1–24, 48, 168, 336; rolling means/std (24, 168); hour, day of week, weekend, holidays (`tsa_final.data.calendar_features`, computed on the future index inside each `fit()` closure so the phase 02 `Predictor(horizon)` contract does not change). No ALB exogenous for store-service — see Decisions.
- [x] 03.2 skforecast `ForecasterRecursive`: LightGBM, XGBoost, CatBoost, RandomForest, Ridge, default hyperparameters. Produces the validation ranking that gates every later task.
- [x] 03.3 skforecast `ForecasterDirect` for the #1 recursive booster only (compare recursive vs direct strategy). Time-boxed, see above.
- [x] 03.4 Hyperparameter search with Optuna / skforecast `bayesian_search_forecaster`, top-3 recursive models by default-hyperparameter validation MAE (revised gate, see Decisions). Time-boxed, see above.
- [x] 03.5 Stacking ensemble (`sklearn.StackingRegressor`, explicit small `cv`) of the top-3 tuned ML forecasters, wrapped as the `estimator=` of a `ForecasterRecursive` (not fit flat/one-shot like the class notebook — flat fitting cannot produce a 48-step recursive forecast; see Decisions).
- [x] 03.6 Feature importance / SHAP for the single best model across 03.2-03.5 (report material: which lags matter, confirms 24/168 seasonality).
- [x] 03.7 Log every row (winners and losers) into `results/03_ml.csv` via the existing `tsa_final.evaluation.ResultsTable`/`evaluate()`, family="ML".

## Class references

`PRACTICA_AST_Clase_5_1_Intro`, `_5_2B(XByLGBM)`, `practica_ast_clase_5_2a(randycat).py`, `_5_3_Stacking`, `_5_4_Intro_recursive`, `_5_5_Skfore`. Confirmed `_5_5_Skfore` already uses `ForecasterRecursive` (current skforecast 0.25.0 API, not the deprecated `ForecasterAutoreg`), so no API-compatibility gap with the class material.

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| No ALB exogenous for store_service | Phase 01 CCF found the ALB/store_service correlation peaks at lag 0 (r=0.852) with no lead-lag advantage — ALB's future value is unknown at forecast time, and even lagged ALB is not shown to add signal beyond store_service's own lags + calendar. Spending fit/Optuna budget on a weakly-motivated feature does not fit the time budget | (a) Forecast ALB first and chain it as a known-future exogenous — couples two models and propagates ALB's error; (b) use lagged ALB anyway "just in case" — no evidence it helps, costs time to test |
| `StackingRegressor` (sklearn) wrapped as the `regressor=` of `ForecasterRecursive`, not fit flat/one-shot as in class 5.3's notebook | Horizon is 48h: lags 1-24 of a late step depend on the model's own earlier predictions, which only a recursive forecaster provides. The class notebook fits/evaluates on a single fixed split with a pre-built lag matrix — correct for that exercise, wrong for a 48-step-ahead forecast | Copying the class's flat fit/predict pattern — breaks past a 1-step horizon |
| Optuna search for the top-3 recursive models by default-hyperparameter validation MAE (revised from "only models already beating `seasonal_naive_m24`" — see Evidence), capped at ~20 trials or ~3 min/model | The original naive-beating gate was run first and came back empty for both series (no recursive model, including the direct-strategy winner, beats the naive baseline with default hyperparameters) — a gate that always evaluates empty defeats the purpose of the task. Same time-consciousness principle as phase 02 otherwise: cap stays fixed regardless of how many models pass | Grid/Bayesian search over all 5 boosters regardless of ranking (too slow); keeping the original empty gate and skipping tuning/stacking entirely (leaves the report unable to say whether ML could compete with tuning) |
| Direct-strategy comparison (03.3) limited to the single best recursive booster, with `n_jobs=-1`, and a hard 20 min time-box | `ForecasterDirect` trains one sub-model per horizon step by default (up to 48 for this horizon) — the single largest hidden multiplier in the whole phase. If still too slow, cut to a coarser comparison and record the gap rather than let it run unbounded | Direct strategy for all 5 boosters — multiplies an already expensive strategy by 5x for a secondary comparison |
| Results logged to a new `results/03_ml.csv` (same schema as `results/02_baselines.csv`) via a new `tsa_final/ml_models.py` module, reusing `tsa_final.evaluation.ResultsTable`/`evaluate()` unmodified | Matches phase 02's own stated plan ("results/02_baselines.csv for phases 03-06 to concatenate"); keeps phase boundaries in separate files/modules instead of one growing file | Appending directly into `02_baselines.csv` — mixes phases in one file and couples phase 03 code changes to phase 02's module |

## Evidence

Implemented in `tsa_final/ml_models.py` (feature config, recursive/direct/tuned/stacking `Fit`
builders, SHAP helper), run end to end in `notebooks/03_ml_models.ipynb` (executed, outputs kept).
Full table (10 models x 2 series x 2 splits = 40 rows) persisted at `results/03_ml.csv`. Total
wall-clock for the whole phase: ~200 s (~3.3 min) for both series combined — well inside the "few
hours" budget; none of the individual time-boxes (20 min direct-strategy, 18 min Optuna, 5 min
stacking) were approached, let alone hit.

**Key finding, discovered during implementation**: with default hyperparameters, no ML model —
recursive or direct — beats `seasonal_naive_m24` on validation MASE for either series (ALB naive
0.489 vs. best default `catboost` 0.598; store_service naive 0.480 vs. best default `ridge` 1.201).
This is why the original 03.4 gate ("only tune models that already beat the naive") came back empty
and was revised to "top-3 by default MAE" (see Decisions) — otherwise phases 03.4/03.5 would have
produced nothing at all.

Validation ranking (MAE, primary selection criterion) per series, best to worst:

- **ALB**: `catboost_tuned` wins (MAE 13,994, MASE 0.353) — the only ML model in the whole sweep that
  beats `seasonal_naive_m24` (0.489) in validation. `random_forest_tuned` (MASE 0.416) and
  `lightgbm_tuned` (MASE 0.421) also clear the naive bar. `catboost_direct` (MASE 0.534) does not.
  None of the 5 default-hyperparameter recursive models beat the naive baseline untuned.
- **store_service**: `lightgbm_tuned` wins (MAE 2,724, MASE 0.573) but does **not** beat
  `seasonal_naive_m24` (0.480) — no ML variant does, tuned or not, on this series. `ridge_direct`
  (MASE 0.611) and `xgboost_tuned` (MASE 0.690) are the next closest.

Test-window ranking (informational only, per the phase 02 protocol) diverges from the validation
ranking, consistent with phase 02's own noted risk of a single 48h validation window:

- **ALB**: `catboost_tuned` (the validation winner) is actually one of the *worst* on test (MASE
  2.313) — no better than the untuned defaults, and `random_forest_tuned` is the single worst model
  in the entire table on test (MASE 3.455, worse than its own untuned default's 0.607). Tuning to a
  48h validation window improved validation metrics substantially but did not transfer to test —
  read as overfitting to that specific window, not a code defect (every row, including this one, is
  logged in `results/03_ml.csv` per the "log winners and losers" requirement).
- **store_service**: similarly, `lightgbm_tuned`'s validation win does not hold up as cleanly on
  test; the untuned/tuned test MASEs cluster closer together than the validation ranking would
  suggest.

**Stacking underperformed in both series** — `stacking_top3` is the *worst* model in the entire
phase-03 table for both ALB (MASE 2.559) and store_service (MASE 1.963), well behind even the
untuned defaults. Diagnosed directly (predictions are not degenerate/constant — they track the
right level and range) — the ensemble's linear meta-combination of 3 tuned tree models compounds
error faster than any single model once fed back recursively over a 48-step horizon. Kept as
reported evidence per the "log winners and losers" requirement, not treated as a bug or excluded.

SHAP (`informe/figuras/03_shap_alb.png`, `03_shap_store_service.png`) on each series' single overall
winner confirms `lag_1`, `lag_2`, `lag_23`, `lag_24` and `lag_48` dominate feature importance —
consistent with phase 01's daily-seasonality (period 24) finding; `lag_168`-range features rank far
lower, consistent with phase 01's weekly-seasonality-negligible finding.

Figures: `informe/figuras/03_val_mae_ranking.png` (validation MASE bar chart per model/series,
winner highlighted, seasonal-naive reference line), `03_shap_alb.png`, `03_shap_store_service.png`.

**Implementation gotchas worth recording**: (1) skforecast 0.25.0 renamed the forecaster constructor
argument from `regressor=` to `estimator=`; (2) `bayesian_search_forecaster`'s `search_space`
callable must return *only* the keys produced by `trial.suggest_*` — fixed, non-tuned kwargs
(`random_state`, `verbose`) must live on the estimator template passed into the forecaster instead,
or the search raises `ValueError` on the very first trial.
