# Phase 03 — Machine Learning models

Status: pending

## Objective

Tree-based and linear ML forecasters on lag/calendar features, evaluated with the phase 02 protocol.

## Tasks

- [ ] 03.1 Feature set: lags 1–24, 48, 168, 336; rolling means/std (24, 168); hour, day of week, weekend, holidays; optional ALB exogenous for store-service.
- [ ] 03.2 skforecast recursive: LightGBM, XGBoost, CatBoost, RandomForest, Ridge.
- [ ] 03.3 skforecast direct multi-step for the best booster (compare recursive vs direct strategy).
- [ ] 03.4 Hyperparameter search with Optuna / skforecast `bayesian_search_forecaster`, bounded budget per model.
- [ ] 03.5 Stacking ensemble of the best ML forecasters (class 5.3).
- [ ] 03.6 Feature importance / SHAP for the best model (report material: which lags matter, confirms 24/168 seasonality).
- [ ] 03.7 Log results into the shared table.

## Class references

`PRACTICA_AST_Clase_5_1_Intro`, `_5_2B(XByLGBM)`, `practica_ast_clase_5_2a(randycat).py`, `_5_3_Stacking`, `_5_4_Intro_recursive`, `_5_5_Skfore`.

## Evidence

_(fill in)_
