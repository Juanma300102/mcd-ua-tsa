# Phase 02 — Evaluation protocol and baselines

Status: pending

## Objective

One evaluation protocol shared by every model, so the comparison across ~15 models is fair, and baselines that represent TP1.

## Protocol (confirmed after phase 01 EDA)

- Test window: 2026-09-22 14:00 → 2026-09-26 22:00 UTC (105 h, `TEST_START`/`TEST_END` in `tsa_final.data`). This is observed, post-rollback data only — it starts right after the known intervention ends (2026-09-22 13:00 UTC) so the held-out set is not contaminated by the deployment/rollback anomaly, and it runs to the last available hour (2026-09-26 22:00 UTC).
- Horizon: 48 h (same as TP1, direct comparison). The 168 h secondary horizon proposed earlier is dropped: the test window only has 105 h, so a 168 h horizon cannot be evaluated on held-out data, and weekly seasonality was found negligible (phase 01), reducing the value of a week-long horizon anyway.
- Split: test = the 105 h window above (held out until phase 06). Previous data (2026-07-03 → 2026-09-22 13:00 UTC, intervention imputed) = train + validation.
- Validation: expanding-window backtesting (rolling origin) on the weeks before the test window, step 24 h or 48 h, refit per fold where cost allows. Validation folds must end before `INTERVENTION_START` (2026-09-17 18:00 UTC) — no fold may include the intervention window, imputed or not, as part of its scored horizon, to keep validation representative of organic behaviour.
- Metrics, original scale: MAE, RMSE, MAPE, plus MASE. Weekly seasonality is negligible (phase 01: weekday means differ <3%, MSTL seasonal-168 strength well below seasonal-24), so **seasonal naive with m=24 is the main naive baseline and MASE uses m=24** (not m=168).
- Ranking metric: MAE in validation (primary), test used only for the final table.
- Seeds fixed; record fit time per model (cost is part of the comparison).

## Tasks

- [ ] 02.1 Implement split + backtesting helper and a metrics function (reuse TP1 manual MAE/RMSE/MAPE).
- [ ] 02.2 Naive baselines: seasonal naive m=24 (primary; weekly seasonality is negligible per phase 01, so m=168 is not used as a baseline).
- [ ] 02.3 TP1 baseline refit: `SARIMA(1,1,1)x(1,1,1,24)` for ALB and `SARIMA(1,1,1)x(1,0,1,24)` spec for store-service (TP1 POS API spec), refit on the new data. Optionally a small re-grid.
- [ ] 02.4 Classical extras as reference: Holt-Winters / ETS, MSTL + ARIMA (handles 24 and 168), AutoARIMA.
- [ ] 02.5 Results table skeleton: model, family, series, val MAE/RMSE/MAPE/MASE, test metrics, fit time.

## Note on TP1 comparison

TP1 test MAPE (ALB 0.79 %, POS API 3.90 %) came from a one-month series and a different target group. Quote them in the report only as context; the valid comparison is the refit baseline on the same test window.

## Evidence

_(fill in: final protocol, baseline metrics)_
