# Phase 02 — Evaluation protocol and baselines

Status: done

## Objective

One evaluation protocol shared by every model, so the comparison across ~15 models is fair, and baselines that represent TP1.

## Protocol (confirmed after phase 01 EDA)

- Test window: 2026-09-22 14:00 → 2026-09-26 22:00 UTC (105 h, `TEST_START`/`TEST_END` in `tsa_final.data`). This is observed, post-rollback data only — it starts right after the known intervention ends (2026-09-22 13:00 UTC) so the held-out set is not contaminated by the deployment/rollback anomaly, and it runs to the last available hour (2026-09-26 22:00 UTC).
- Horizon: 48 h (same as TP1, direct comparison). The 168 h secondary horizon proposed earlier is dropped: the test window only has 105 h, so a 168 h horizon cannot be evaluated on held-out data, and weekly seasonality was found negligible (phase 01), reducing the value of a week-long horizon anyway.
- Split: test = the 105 h window above (held out until phase 06). Previous data (2026-07-03 → 2026-09-22 13:00 UTC, intervention imputed) = train + validation.
- Validation: a single held-out window, not expanding-window backtesting. Length matches the test horizon (48 h), taken from the end of train+val, ending strictly before `INTERVENTION_START` (2026-09-17 18:00 UTC) so its scored horizon never touches the intervention window — concretely, the 48 h immediately preceding `INTERVENTION_START`. Each model fits once on train for the validation forecast (selection) and once more on train+val for the test forecast (final table): two fits per model, not N folds.
- Metrics, original scale: MAE, RMSE, MAPE, plus MASE. Weekly seasonality is negligible (phase 01: weekday means differ <3%, MSTL seasonal-168 strength well below seasonal-24), so **seasonal naive with m=24 is the main naive baseline and MASE uses m=24** (not m=168).
- Ranking metric: MAE in validation (primary), test used only for the final table.
- Seeds fixed; record fit time per model (cost is part of the comparison).

## Tasks

- [x] 02.1 Implement a split helper (train / single validation window / test) and a metrics function (reuse TP1 manual MAE/RMSE/MAPE, add MASE m=24). No fold generator, no dependency on `skforecast.TimeSeriesFold`: each model exposes a plain `fit(train) -> predict(horizon)` and the helper only scores it, so the same two functions work across statsmodels, skforecast, darts, neuralforecast, prophet and chronos in phases 02-06.
- [x] 02.2 Naive baselines: seasonal naive m=24 (primary; weekly seasonality is negligible per phase 01, so m=168 is not used as a baseline).
- [x] 02.3 TP1 baseline refit: `SARIMA(1,1,1)x(1,1,1,24)` for ALB and `SARIMA(1,1,1)x(1,0,1,24)` spec for store-service (TP1 POS API spec), refit on the new data. Optionally a small re-grid.
- [x] 02.4 Classical extras as reference: Holt-Winters / ETS, MSTL + ARIMA (handles 24 and 168), AutoARIMA.
- [x] 02.5 Results table skeleton: model, family, series, val MAE/RMSE/MAPE/MASE, test metrics, fit time.

## Note on TP1 comparison

TP1 test MAPE (ALB 0.79 %, POS API 3.90 %) came from a one-month series and a different target group. Quote them in the report only as context; the valid comparison is the refit baseline on the same test window.

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| Single held-out validation window (48 h, ending before `INTERVENTION_START`) instead of expanding-window backtesting; two fits per model (validation, then refit on train+val for test), not N folds | Time. With ~15 models across ML/DL/AutoML/foundation families per series, the dominant cost is phase 04 (DL, ~12-18 fits: 6 architectures x 2-3 seeds) and phase 05 (AutoGluon/AutoTS, whose defaults can run unbounded). Expanding-window backtesting multiplies exactly those expensive fits by the number of folds; a single split keeps the multiplier at 2x regardless of model family. Neither the consigna nor TP1 (`tp-1/notebooks/resolucion_consigna_series_bodegaai.ipynb`, section 6) required backtesting — TP1 used one 48 h train/test split with MAE/RMSE/MAPE, no rolling folds | (a) Full expanding-window backtesting (original protocol above, ~10-12 folds/week over the ~12-week train+val range) — estimated 4-10+ h per series once DL/AutoML are included; (b) middle ground of 2-3 folds — still ~2-3x the DL/AutoML cost for marginal robustness gain, rejected for the same reason |

## Evidence

Implemented in `tsa_final/evaluation.py` (split, metrics, results table) and `tsa_final/baselines.py` (naive
family, SARIMA refit, classical extras), run end to end in `notebooks/02_evaluation_baselines.ipynb`. Full
table (8 models x 2 series x 2 splits = 32 rows) persisted at `results/02_baselines.csv` for phases 03-06 to
concatenate. Total fit time for the whole sweep: 314.9 s (~5.3 min) — the entire bottleneck is `auto_arima`
(statsforecast, season_length=24), 46-98 s per fit; every other model fits in under 3 s.

Validation ranking (MAE, primary selection criterion) per series, best to worst:

- **ALB**: `auto_arima` (MAPE 0.80 %, MASE 0.475) and `seasonal_naive_m24` (MAPE 0.80 %, MASE 0.489) are
  effectively tied at the top; then `mstl_arima` (MASE 0.593), `holt_winters` (MASE 0.798), the SARIMA
  refit (MASE 0.923), and finally naive/drift/average (MASE 3.5-3.8, far worse).
- **store_service**: `seasonal_naive_m24` wins outright (MAPE 1.94 %, MASE 0.480), `auto_arima` close
  second (MASE 0.532), then `mstl_arima` (MASE 0.656), the SARIMA refit (MASE 0.647), `holt_winters`
  (MASE 1.439), and naive/drift/average (MASE 2.5-3.6).

Test-window ranking (informational only, per protocol) does **not** match the validation ranking:

- **ALB**: the SARIMA refit takes the lead on test (MAPE 2.42 %, MASE 1.526) even though it ranked behind
  `auto_arima` and `seasonal_naive_m24` on validation; `auto_arima`'s test MAPE (3.25 %) is worse than
  `seasonal_naive_m24`'s (3.15 %) despite winning validation.
- **store_service**: `auto_arima` takes the test lead by MAE (6693 vs. `mstl_arima`'s 6738 and
  `seasonal_naive_m24`'s 6858), consistent with its validation ranking, though by MAPE `mstl_arima` (6.73 %)
  edges it out (6.78 %) — the two are within noise of each other.

This divergence between the validation and test rankings (especially ALB, where the model that wins
validation is not the model that wins test) is the risk flagged in `design.md` when the single-split
protocol was chosen over multi-fold backtesting: a single 48 h validation window is noisier than an
N-fold ranking. It is treated as expected, reportable evidence, not a bug — every model's test metrics
are logged regardless of the validation winner (see `results/02_baselines.csv`), so this is visible rather
than hidden.

`seasonal_naive_m24` is a genuinely strong baseline on both series (MASE < 1 on both val and test, for
every series), confirming phase 01's finding that daily seasonality (period 24) is the dominant, well
behaved component. Any ML/DL/hybrid model in phases 03-05 that cannot beat it (MASE < seasonal_naive's) on
both splits has not earned its added complexity.

Figures: `informe/figuras/02_split_train_val_test.png` (train/val/test regions over both raw series),
`02_val_mape_ranking.png` (validation MAPE bar chart per model/series, seasonal-naive reference line),
`02_test_forecast_winners.png` (test-window forecast vs. actual for each series' validation winner).
