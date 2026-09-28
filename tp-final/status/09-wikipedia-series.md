# Phase 09 — Third series: public es.wikipedia baseline

Status: done

## Objective

Satisfy the consigna's "three series" requirement (point 1) with a public, hourly series on the same calendar as ALB and store-service, evaluated as a fast baseline contrast (the deadline left minutes, not hours).

## Series

- `wikipedia_es`: hourly user pageviews of `es.wikipedia` (`all-access`, agent `user`), Wikimedia Analytics REST API, CC0. Downloaded by `experiments/00_download_wikipedia_es.py` into `data/wikipedia-es-pageviews-hourly-since-2026-05-01.csv` (same columns as the AWS exports). 2,063 clean hours, 2026-07-03 00:00 → 2026-09-26 22:00 UTC, no gaps, same index as ALB/store-service.
- Registered in `tsa_final.data.SERIES_REGISTRY` with `"intervention": "no"`, so `load_clean` skips `impute_intervention` (the rollback did not affect it).

## Tasks

- [x] 09.1 Register the series and download script (CSV under `data/`).
- [x] 09.2 `notebooks/07_wikipedia_baseline.ipynb`: six default-hyperparameter models (`seasonal_naive_m24`, `holt_winters`, `SARIMA(1,1,1)x(1,1,1,24)`, `ridge`, `lightgbm`, `prophet_default`), phase 02 windows (val 48h, test 105h), same selection rule, 48h forecast. Runs in ~16 s.
- [x] 09.3 Add the same section (section 3) to the deliverable notebook without re-executing other cells.
- [x] 09.4 Report: Resumen, Introducción, Análisis (new subsection), Conclusiones, Apéndice A, bibliography entry; remove every "PENDIENTE" marker and the `\pendiente` macro; recompile.

## Decisions

| Decision | Reason | Discarded alternative | Feeds report section |
|---|---|---|---|
| Public es.wikipedia series | Same kind of phenomenon (hourly web requests by people), identical calendar so the phase 02 protocol applies unchanged, open and reproducible (CC0, license checked in Wikimedia's dataset readme), domain contrast (no deploy/rollback) | UAV altitude (TP1's third series: different calendar/sampling, 457 points at 1 Hz); NAB/ELB trace (2014, 5-minute, AGPL); waiting for the teammate | Introducción |
| Six default models, no DL/AutoML/Chronos/tuning | ~16 s total vs. >8 min just for one DL fit during exploration; time-box per the index's cross-cutting rule | Full phase 03-06 battery on the new series | Análisis, Límites |
| Not added to `sel.SERIES_NAMES` | The 32-model pool, contamination flags and integrity assertions assume two series | Extending the pool to 3 series (would need contamination review and DL/AutoML runs) | Análisis |
| SARIMA spec borrowed from ALB | No TP1 spec exists for this series; stated as a limitation | Fitting a new SARIMA order (extra time, breaks "TP1 refit" meaning) | Análisis, Límites |
| Table in the report generated from `results/07_wikipedia.csv` | Avoid transcription errors | Typing numbers by hand | Análisis, Apéndice A |
| Deliverable notebook patched, not regenerated | The generator emits cells without outputs; regenerating would wipe every executed output | Regenerate and re-execute (~1h with `REENTRENAR = True`, minutes with False but outputs would change) | — |

## Evidence

- Validation MAE / MASE: seasonal naive 21,146.77 / 0.438 (selected), Prophet 0.655, LightGBM 0.727, SARIMA 1.023, Holt-Winters 1.143, Ridge 1.351.
- Test MASE: Ridge 0.784 (best), seasonal naive 0.924, Prophet 1.119, LightGBM 1.338, Holt-Winters 2.264, SARIMA 2.513 (worst). Spearman between val and test rankings over the six models: 0.09.
- Structure vs BodegaAI (log series): autocorrelation at lag 24/168 = 0.838/0.614 (ALB), 0.863/0.809 (store-service), 0.958/0.956 (Wikipedia). Correlation of levels 0.41 (ALB) and 0.49 (store-service); of 24h changes -0.01 and 0.01, so the relation is contextual, not predictive.
- Report: `xelatex → biber → xelatex → xelatex`, 41 pages, 0 LaTeX errors, 0 undefined references, no "PENDIENTE" text in the PDF.
- Deliverable notebook: 92 cells (was 85); cells 0-82 outputs and execution counts unchanged; section 3 executed with 3 image outputs; both the "read results" branch and the "compute" branch (fresh results dir) were run and gave identical numbers.

## Known limitations

- Weekly seasonality as the explanation for the SARIMA's poor test result is plausible but not tested.
- Single validation and test window; six models only; no prediction intervals for the Wikipedia forecast.
- `calendar_features` emits `is_intervention` (1 on 2026-09-17 18:00 → 2026-09-22 13:00) and Ridge/LightGBM consume it, although this series was not affected. Measured effect (dummy forced to 0): validation unchanged (the dummy is constant 0 in the pre-intervention training set), LightGBM test unchanged, Ridge test MASE 0.784 → 0.756; ranking unchanged. Reported numbers keep the dummy; the effect is disclosed in the report, the notebooks and here. A clean fix would make `calendar_features` series-aware and rerun 07 and its consumers.
