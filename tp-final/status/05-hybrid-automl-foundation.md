# Phase 05 — Prophet family, AutoML, hybrids and foundation models

Status: done

## Objective

Cover the remaining families required by the consigna (hybrids, AutoML, Prophet/NeuralProphet) plus zero-shot foundation models as an extra reference.

## Tasks

- [x] 05.1 Prophet: daily seasonality and Argentine holidays; weekly seasonality dropped (phase 01: negligible after go-live), yearly seasonality dropped (~13 weeks of history, cannot estimate a yearly Fourier term). Reports both an untuned reference (`prophet_default`) and an Optuna-tuned variant (`prophet_tuned`, `changepoint_prior_scale`/`seasonality_prior_scale`, 18 trials) scored on the 168h leakage-safe tuning window (never the 48h selection window).
- [x] 05.2 NeuralProphet with `n_lags=24` AR lags and daily seasonality. `n_forecasts=105` (the largest horizon this phase needs) so one trained model serves both the 48h validation and 105h test calls by slicing its direct multi-output forecast; documented in `tsa_final/hybrid_models.py` and the notebook.
- [x] 05.3 Hybrid: `statsmodels.tsa.seasonal.MSTL` (period 24 only) + LightGBM (default hyperparameters) on the deseasonalized series; the deterministic seasonal-24 component is tiled back onto LightGBM's forecast.
- [x] 05.4 Ensemble: equal-weight average of the best-by-validation-MAE model from phases 02 (baselines), 03 (untuned recursive ML only — see Decisions) and 04 (DL). Weights fixed, never fitted.
- [x] 05.5 AutoML: AutoGluon TimeSeries (300s val fit + 300s test fit, `WeightedEnsemble` selected in all 4 runs) and AutoTS (`model_list="superfast"`, `max_generations=3`, `num_validations=1`, fixed seed). PyCaret and FEDOT excluded: incompatible with the shared env (phase 00), stated here per the task list.
- [x] 05.6 Chronos-Bolt-base zero-shot on GPU; `fit` only stores context, `predict` reads the 0.5 quantile via `predict_df`.
- [x] 05.7 TimeGPT: **skipped**. Requires sending the series to an external API (Nixtla) with a key; no explicit user approval was granted for this project (see `status/index.md` "Open decisions"). No code, no API call.
- [x] 05.8 Logged every row (winners and losers) into `results/05_hybrid.csv` via the unmodified `tsa_final.evaluation.evaluate`/`ResultsTable`.

## Execution gate

A background job (`tp-final/experiments/04b_restore_best_weights.py`) was rewriting `results/04_dl.csv` on the GPU when this phase started. Module code, the notebook and smoke tests (tiny data slices, a few seconds, CPU/CPU-loaded Chronos) were prepared while the gate was active; the full notebook (fit-time measurement, GPU Chronos, AutoGluon) was executed only after confirming `pgrep -f 04b_restore_best_weights` returned nothing and `results/04_dl.csv` held its corrected values (e.g. `store_service`/`nbeats` val MASE 0.638, `alb`/`nhits` val MASE 0.556 — both improved from the pre-fix numbers phase 04 originally reported).

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| Ensemble's ML member restricted to the 5 untuned recursive models (`ml_models.RECURSIVE_MODELS`), not the overall best row of `results/03_ml.csv` (which is often a `_tuned` variant) | Reproducing a `_tuned` member for the ensemble would require re-running `ml_models.tune_hyperparameters`, which scores Optuna trials on the validation window itself (a phase 03 limitation, not revisited there) — exactly the tuning-on-the-selection-window leakage this phase's protocol bans for everything new. Restricting to untuned defaults needs no re-tuning and keeps the refit cheap | Re-running phase 03's Optuna search to reproduce the literal best row — reintroduces the leakage this phase explicitly prohibits, and is not "cheap" per the task's own instruction |
| NeuralProphet built once with `n_forecasts=105` (the test horizon) instead of one model per horizon | `n_forecasts` is fixed at construction (a direct multi-output AR model, like `ForecasterDirect`); building two models (48 and 105) would double training cost for no benefit, since the 48-step values are the *first* 48 of the 105-step direct output — predicting more targets does not change what the model needs to fit for the first 48 | A `evaluate_direct`-style mirror with two separately-sized models (the phase 03 pattern) — unnecessary here because, unlike skforecast, NeuralProphet's multi-output layer for step k does not depend on the total number of steps requested |
| MSTL+LightGBM hybrid fits LightGBM on the deseasonalized series (`train - seasonal`), not on the MSTL residual alone | The deseasonalized series still carries the trend, which the existing lag/rolling feature set (same as `ml_models`) can track directly; fitting on the bare residual would need a separate trend-forecasting step (e.g. a second ARIMA/naive-drift model) that the task's parenthetical alternative ("or LightGBM on the deseasonalized series") does not require | Fitting on the MSTL residual and reconstructing trend + seasonal separately — more moving parts for a family whose task wording explicitly allows the simpler deseasonalized-series alternative |
| AutoGluon TimeSeries split into two fits (300s val, 300s test), `prediction_length` matching each horizon | Gives AutoGluon a validation row comparable to every other model (its own internal validation does not otherwise produce one phase 06 could rank against) while respecting the teammate's 600s/series budget | A single fit on `train_and_val` with no validation row — cheaper, but leaves phase 06 unable to compare AutoGluon's validation ranking to everything else |
| AutoTS plugs into the shared `Fit`/`Predictor` contract directly (no mirror), constructed once with `forecast_length=48` and its `predict(forecast_length=horizon)` overridden per call | Verified experimentally that `AutoTS.predict(forecast_length=...)` accepts any horizon after a single `fit()`; `forecast_length` at construction only sizes AutoTS's own internal validation slicing, which is carved from `train` alone (never `val`/`test`), so no mirror or leakage risk | A `ForecasterDirect`-style mirror per horizon — unnecessary extra fitting cost for a library that already supports arbitrary-horizon prediction from one fit |
| `NeuralProphet(..., collect_metrics=False)` | NeuralProphet's default metrics logger drives PyTorch Lightning's `TensorBoardLogger`, which writes a `lightning_logs/` directory (tfevents + hparams files) to the process's current working directory on every fit; no training-time metrics are consumed here (only the final forecast), so disabling the logger removes the stray directory with no loss of information | Leaving `collect_metrics` at its default and deleting `lightning_logs/` after each run — fragile (depends on remembering to clean up) and leaves the directory present during execution |

## Evidence

Implemented in `tsa_final/hybrid_models.py` (Prophet default/tuned, NeuralProphet, MSTL+LightGBM
hybrid, equal-weight ensemble composer, AutoGluon two-fit mirror, AutoTS, Chronos-Bolt zero-shot),
run end to end in `notebooks/05_hybrid_automl_foundation.ipynb` (`nbconvert --execute`, exit 0,
outputs kept). Full table (8 models × 2 series × 2 splits = 32 rows) at `results/05_hybrid.csv`,
same schema as `results/03_ml.csv`/`results/04_dl.csv`; no NaNs.

**Runtime**: total fit time across every model/series/split: 1043.6 s (~17.4 min). Full notebook
wall clock (`nbconvert --execute --inplace`, `/usr/bin/time -v`): **21 min 42.21 s** (user 4213.5 s,
system 614.6 s, 370 % CPU — reflects AutoGluon's and NeuralProphet's own multi-process/-thread use),
well inside the ~40-minute target for both series, despite covering more model families than any
earlier phase. Per-series total: ALB 709.8 s, store_service 342.6 s. AutoGluon dominates the fixed
cost (119.7–161.8 s per fit call, 4 calls total ≈ 9 min) without being the most accurate model in
either series — the same "priciest model isn't the most accurate" pattern phase 04 found with TFT.

Validation ranking (MASE, primary selection criterion) per series, phase-05 models only, best to worst:

- **ALB**: `ensemble_equal_weight` 0.404 (beats `seasonal_naive_m24` 0.489, does not beat phase 03's
  `catboost_tuned` 0.353), `autogluon_timeseries` 0.499 (`WeightedEnsemble` selected both fits),
  `prophet_default` 0.532, `chronos_bolt_base` 0.598 (zero-shot, no tuning at all — notable),
  `prophet_tuned` 0.659 (worse than `prophet_default` — see Decisions/Conclusions in the notebook:
  tuning on the pre-selection window does not guarantee it transfers to the selection window
  itself), `mstl_lightgbm` 0.716, `neuralprophet` 0.744, `autots` 0.756.
- **store_service**: `ensemble_equal_weight` 0.480126 and `seasonal_naive_m24` 0.480171 are
  effectively tied (not a real improvement), `prophet_default` 0.506, `autots` 0.508,
  `prophet_tuned` 0.512, `neuralprophet` 0.694, `autogluon_timeseries` 0.801, `chronos_bolt_base`
  0.816, `mstl_lightgbm` 1.848 (worst).

Test-window ranking (informational, per protocol) diverges from validation again, consistent with
every earlier phase's finding of a noisy single 48h validation window:

- **ALB**: `prophet_tuned` is the best phase-05 model on test (MASE 1.578), beating both
  `seasonal_naive_m24` (1.990) and `catboost_tuned` (2.313) — the same model that was the *worst*
  Prophet variant on validation. `mstl_lightgbm` is the worst model in the *entire project* on test
  so far (MASE 5.720): extrapolating the deseasonalized LightGBM forecast 105h ahead compounds error
  far more than the 48h validation case (0.716) suggested.
- **store_service**: `neuralprophet` wins on test (MASE 1.071, beats naive/best-pre-05 at 1.443)
  despite being the second-worst Prophet-family model on validation (0.694) — another
  validation/test reversal. `mstl_lightgbm` is again the worst model (test MASE 3.100).

**Ensemble members selected** (best-by-validation-MAE per phase, read from `results/02_baselines.csv`,
`03_ml.csv`, `04_dl.csv` after the gate cleared):

- ALB: baseline=`auto_arima`, ML=`catboost` (untuned), DL=`nhits`.
- store_service: baseline=`seasonal_naive_m24`, ML=`ridge` (untuned), DL=`nbeats`.
- No substitution was needed (neither series' best-by-validation DL architecture was `tft`, the
  expensive one flagged as a possible substitution case).

AutoGluon selected `WeightedEnsemble` as `model_best` in all 4 fits (2 series × 2 splits).

**Implementation gotcha worth recording**: `CatBoostRegressor` (used unmodified via
`ml_models.RECURSIVE_MODELS["catboost"]` for ALB's ensemble ML member) writes a `catboost_info/`
directory to the process's current working directory regardless of `verbose=0` (that flag only
throttles stdout, not disk writes — `allow_writing_files=False` would be needed to suppress it, and
`ml_models.py` is intentionally not modified in this phase). It appeared once at
`tp-final/notebooks/catboost_info/` after the executed run and was deleted manually; a future rerun
of this notebook will regenerate it and it should be deleted again (or `ml_models.py` could add
`allow_writing_files=False` in a later phase if this becomes a recurring nuisance).

Figures: `informe/figuras/05_val_mae_ranking.png` (validation MASE per phase-05 model and series,
`seasonal_naive_m24` reference line, best-pre-05 reference line where different from the naive,
winner highlighted), `05_test_forecast_best_model.png` (test window, actual vs. forecast, for the
best-by-validation-MASE phase-05 model per series — `ensemble_equal_weight` for both).
