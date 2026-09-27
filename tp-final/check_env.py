"""Smoke test for the TP Final environment.

Imports every forecasting library and runs a tiny fit on a synthetic hourly
series for each model family used in the project.

Run from the repo root: uv run python tp-final/check_env.py
"""

import importlib
import logging
import tempfile
import warnings

import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

LIBRARIES = [
    "numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "skforecast",
    "optuna", "statsforecast", "mlforecast", "shap", "torch", "darts",
    "neuralforecast", "prophet", "neuralprophet", "chronos", "autots",
    "autogluon.timeseries",
]
PL_TRAINER = {"enable_progress_bar": False, "logger": False}

idx = pd.date_range("2026-01-01", periods=24 * 21, freq="h")
y = pd.Series(100 + 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + np.random.rand(len(idx)), index=idx)
long_df = pd.DataFrame({"unique_id": "a", "ds": idx, "y": y.values})

failures = []


def check(name, fn):
    try:
        fn()
        print(f"OK    {name}")
    except Exception as exc:
        failures.append(name)
        print(f"FAIL  {name}: {type(exc).__name__}: {str(exc)[:200]}")


def fit_skforecast():
    from lightgbm import LGBMRegressor
    from skforecast.direct import ForecasterDirect
    from skforecast.recursive import ForecasterRecursive
    from xgboost import XGBRegressor

    ForecasterRecursive(LGBMRegressor(verbose=-1), lags=24).fit(y)
    ForecasterDirect(XGBRegressor(n_estimators=10), lags=24, steps=24).fit(y)


def fit_statsforecast():
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL, AutoARIMA, SeasonalNaive

    models = [MSTL(season_length=[24, 168], trend_forecaster=AutoARIMA()), SeasonalNaive(24)]
    StatsForecast(models=models, freq="h").forecast(df=long_df, h=24)


def fit_mlforecast():
    from lightgbm import LGBMRegressor
    from mlforecast import MLForecast

    MLForecast(models=[LGBMRegressor(verbose=-1)], freq="h", lags=[1, 24]).fit(long_df)


def fit_darts():
    from darts import TimeSeries
    from darts.models import NBEATSModel, RNNModel, TCNModel, TFTModel, TiDEModel

    series = TimeSeries.from_series(y.astype("float32"))
    for model_cls, kwargs in [(TiDEModel, {}), (TCNModel, {}), (NBEATSModel, {}), (TFTModel, {"add_relative_index": True})]:
        model_cls(input_chunk_length=48, output_chunk_length=24, n_epochs=1, pl_trainer_kwargs=PL_TRAINER, **kwargs).fit(series)
    RNNModel(model="LSTM", input_chunk_length=48, training_length=72, n_epochs=1, pl_trainer_kwargs=PL_TRAINER).fit(series)


def fit_neuralforecast():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS

    models = [NHITS(h=24, input_size=48, max_steps=5, enable_progress_bar=False),
              NBEATS(h=24, input_size=48, max_steps=5, enable_progress_bar=False)]
    NeuralForecast(models=models, freq="h").fit(long_df)


def fit_prophet():
    from prophet import Prophet

    Prophet().fit(long_df[["ds", "y"]])


def fit_neuralprophet():
    from neuralprophet import NeuralProphet

    NeuralProphet(epochs=2, n_lags=24).fit(long_df[["ds", "y"]], freq="h", progress=None)


def predict_chronos():
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu")
    pipeline.predict_quantiles(torch.tensor(y.values), prediction_length=24)


def fit_autots():
    from autots import AutoTS

    AutoTS(forecast_length=24, frequency="h", max_generations=1, num_validations=1,
           model_list="superfast", verbose=0).fit(pd.DataFrame({"y": y}))


def fit_autogluon():
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    data = TimeSeriesDataFrame.from_data_frame(long_df, id_column="unique_id", timestamp_column="ds")
    predictor = TimeSeriesPredictor(prediction_length=24, freq="h", target="y", path=tempfile.mkdtemp(), verbosity=0)
    predictor.fit(data, presets="fast_training", time_limit=90).predict(data)


if __name__ == "__main__":
    for lib in LIBRARIES:
        check(f"import {lib}", lambda lib=lib: print(f"      {lib} {getattr(importlib.import_module(lib), '__version__', '')}"))

    import torch

    print(f"      CUDA available: {torch.cuda.is_available()}")

    check("skforecast recursive + direct (LightGBM, XGBoost)", fit_skforecast)
    check("statsforecast MSTL(24, 168) + AutoARIMA", fit_statsforecast)
    check("mlforecast", fit_mlforecast)
    check("darts TiDE / TCN / N-BEATS / TFT / LSTM", fit_darts)
    check("neuralforecast N-HiTS / N-BEATS", fit_neuralforecast)
    check("prophet", fit_prophet)
    check("neuralprophet with AR lags", fit_neuralprophet)
    check("chronos-bolt-tiny zero-shot", predict_chronos)
    check("autots", fit_autots)
    check("autogluon fast_training", fit_autogluon)

    print(f"\n{len(failures)} failure(s)" + (f": {', '.join(failures)}" if failures else ""))
    raise SystemExit(1 if failures else 0)
