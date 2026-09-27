# Phase 05 — Prophet family, AutoML, hybrids and foundation models

Status: pending

## Objective

Cover the remaining families required by the consigna (hybrids, AutoML, Prophet/NeuralProphet) plus zero-shot foundation models as an extra reference.

## Tasks

- [ ] 05.1 Prophet with daily + weekly seasonality and Argentine holidays; tune changepoint/seasonality priors (Optuna, class 6.5A).
- [ ] 05.2 NeuralProphet with AR lags (e.g. `n_lags=168`) and seasonalities.
- [ ] 05.3 Hybrid 1: SARIMA/MSTL + LightGBM on residuals.
- [ ] 05.4 Hybrid 2: simple weighted ensemble of the best model per family (weights from validation).
- [ ] 05.5 AutoML: AutoGluon TimeSeries (example report), plus AutoTS (optionally MLForecast). PyCaret and FEDOT are out: incompatible with the shared env (see phase 00); mention it in the report.
- [ ] 05.6 Foundation model zero-shot: Chronos (local, class 8).
- [ ] 05.7 TimeGPT only with explicit approval (sends data to Nixtla API).
- [ ] 05.8 Log results into the shared table.

## Class references

`PRACTICA_AST_Clase_6_5A_FPROPHET`, `_6_5B_NPROPHET`, `_7_1_AML`, `_7_2_PyCaret`, `_7_3_ANDET`, `_7_4_AUTOTS`, `_7_6_MLForecast_FEDOT`, `AST_Clase_8_ejercicio_1`, `AST_Clase_8_ejercicio_2_Extra_API_KEY`.

## Evidence

_(fill in)_
