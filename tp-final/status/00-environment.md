# Phase 00 — Environment and tooling

Status: done (2026-09-27, branch `feat/tp-final-environment`)

## Objective

A reproducible Python environment (uv, root `pyproject.toml`) that runs every model family used in the TP.

## Tasks

- [x] 00.1 Core forecasting deps: scikit-learn, lightgbm, xgboost (via `xgboost-cpu`), catboost, skforecast, optuna, statsforecast, mlforecast, shap.
- [x] 00.2 DL deps on a single backend (torch): darts, neuralforecast. No TensorFlow: darts `RNNModel(model="LSTM")` covers the class 6 LSTM.
- [x] 00.3 prophet, neuralprophet.
- [x] 00.4 Foundation model: chronos-forecasting (local).
- [x] 00.5 AutoML: autogluon.timeseries and autots. PyCaret and FEDOT dropped (see below).
- [x] 00.6 Smoke test: `uv run python tp-final/check_env.py` (exit code 0 = all OK).

## Setup for teammates

```bash
uv sync
uv run python tp-final/check_env.py
```

Python 3.12 is required (`requires-python = ">=3.12,<3.13"`, imposed by neuralprophet 0.9).

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| pandas 2.3.3 / numpy 1.26.4 instead of pandas 3 / numpy 2 | The latest skforecast (0.25.0), statsforecast, mlforecast, autogluon.timeseries and neuralprophet all require `pandas<3`; neuralprophet also requires `numpy<2`. With pandas 3, uv silently fell back to skforecast 0.19.1 and a broken neuralprophet 0.4.2. TP1 backward compatibility is not needed (already delivered). | Keep pandas 3 and a second uv env for AutoGluon / NeuralProphet (two kernels to maintain) |
| Drop PyCaret and FEDOT | Only resolvable jointly with PyCaret 2.2.2 (2020) and broken neuralprophet; PyCaret 3.3.2 needs `pandas<2.2`, `numpy<1.27`, `scipy<=1.11.4`; FEDOT 0.7.5 needs `scikit-learn<1.6`, `scipy<1.13`. AutoML stays covered by AutoGluon and AutoTS. | Downgrade scipy/sklearn for the whole env |
| `xgboost-cpu` instead of `xgboost` | autogluon-tabular depends on `xgboost-cpu`; both packages install the same `xgboost` module and overwrite each other. GPU XGBoost is unnecessary for ~2,300 rows. | Keep both (fragile: uninstalling one deletes the other's files) |
| torch 2.10.0+cu128 | autogluon.timeseries requires `torch<2.14`. CUDA works (RTX 4060 8 GB). | torch 2.14 without AutoGluon |

## Evidence

Final versions: numpy 1.26.4, pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.3.0, catboost 1.2.10, skforecast 0.25.0, optuna 5.0.0, statsforecast 2.0.1, mlforecast 0.14.0, shap 0.49.1, torch 2.10.0+cu128, darts 0.47.0, neuralforecast 3.2.2, prophet 1.4.0, neuralprophet 0.9.0, chronos-forecasting 2.3.2, autots 1.0.4, autogluon.timeseries 1.6.3.

`tp-final/check_env.py` on a venv rebuilt from the lockfile: 19/19 imports OK, CUDA available, and fit (or predict) OK for skforecast recursive + direct, statsforecast MSTL(24,168)+AutoARIMA, mlforecast, darts TiDE/TCN/N-BEATS/TFT/LSTM, neuralforecast N-HiTS/N-BEATS, Prophet, NeuralProphet with AR lags, Chronos-Bolt zero-shot, AutoTS, AutoGluon `fast_training`. 0 failures.

Gotcha: switching versions in place left stale packages (e.g. `nvidia-nccl-cu13` sharing `libnccl.so.2` with `nvidia-nccl-cu12`); removing one broke torch. If imports fail after dependency changes, rebuild with `rm -rf .venv && uv sync`.
