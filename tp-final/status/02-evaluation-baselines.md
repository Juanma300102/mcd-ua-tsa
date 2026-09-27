# Phase 02 — Evaluation protocol and baselines

Status: pending

## Objective

One evaluation protocol shared by every model, so the comparison across ~15 models is fair, and baselines that represent TP1.

## Protocol (proposal, confirm before phase 03)

- Horizon: 48 h (same as TP1, direct comparison). Optionally report 168 h as a secondary horizon since weekly seasonality is now modeled.
- Split: last 7 days = test (held out until phase 06). Previous data = train + validation.
- Validation: expanding-window backtesting (rolling origin) on the last ~3–4 weeks before test, step 24 h or 48 h, refit per fold where cost allows (as in the example report).
- Metrics, original scale: MAE, RMSE, MAPE, plus MASE (seasonal naive, m=168 or 24) because MAPE is unstable near zeros (interior zeros exist).
- Ranking metric: MAE in validation (primary), test used only for the final table.
- Seeds fixed; record fit time per model (cost is part of the comparison).

## Tasks

- [ ] 02.1 Implement split + backtesting helper and a metrics function (reuse TP1 manual MAE/RMSE/MAPE).
- [ ] 02.2 Naive baselines: seasonal naive 24 h and 168 h.
- [ ] 02.3 TP1 baseline refit: `SARIMA(1,1,1)x(1,1,1,24)` for ALB and `SARIMA(1,1,1)x(1,0,1,24)` spec for store-service (TP1 POS API spec), refit on the new data. Optionally a small re-grid.
- [ ] 02.4 Classical extras as reference: Holt-Winters / ETS, MSTL + ARIMA (handles 24 and 168), AutoARIMA.
- [ ] 02.5 Results table skeleton: model, family, series, val MAE/RMSE/MAPE/MASE, test metrics, fit time.

## Note on TP1 comparison

TP1 test MAPE (ALB 0.79 %, POS API 3.90 %) came from a one-month series and a different target group. Quote them in the report only as context; the valid comparison is the refit baseline on the same test window.

## Evidence

_(fill in: final protocol, baseline metrics)_
