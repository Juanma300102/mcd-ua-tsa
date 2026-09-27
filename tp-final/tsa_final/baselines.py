"""Baseline forecasters for the TP Final phase 02 sweep.

Naive family (naive, seasonal-naive, drift, average), the TP1 SARIMA refit,
and classical extras (Holt-Winters/ETS, MSTL+ARIMA, AutoARIMA) — each a
``fit(train) -> predictor(horizon)`` pair per ``tsa_final.evaluation``'s
model-agnostic contract, so all of them are scored by the same
``evaluation.evaluate`` loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA as _SFAutoARIMA
from statsforecast.models import MSTL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .evaluation import SEASONAL_PERIOD, Fit, Predictor

# ---------------------------------------------------------------------------
# Naive family
# ---------------------------------------------------------------------------


def naive_fit(train: pd.Series) -> Predictor:
    """All forecasts equal the last observed value."""
    last = float(train.iloc[-1])

    def predict(horizon: int) -> np.ndarray:
        return np.full(horizon, last)

    return predict


def seasonal_naive_fit(train: pd.Series, m: int = SEASONAL_PERIOD) -> Predictor:
    """Each forecast equals the value from the same hour one season (m) ago; the primary naive baseline."""
    season = train.iloc[-m:].to_numpy()

    def predict(horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / m))
        return np.tile(season, reps)[:horizon]

    return predict


def drift_fit(train: pd.Series) -> Predictor:
    """Extrapolates the straight line from the first to the last training observation."""
    y1 = float(train.iloc[0])
    y_t = float(train.iloc[-1])
    slope = (y_t - y1) / (len(train) - 1)

    def predict(horizon: int) -> np.ndarray:
        return y_t + slope * np.arange(1, horizon + 1)

    return predict


def average_fit(train: pd.Series) -> Predictor:
    """All forecasts equal the training mean."""
    mean = float(train.mean())

    def predict(horizon: int) -> np.ndarray:
        return np.full(horizon, mean)

    return predict


NAIVE_MODELS: dict[str, Fit] = {
    "naive": naive_fit,
    "seasonal_naive_m24": seasonal_naive_fit,
    "drift": drift_fit,
    "average": average_fit,
}


# ---------------------------------------------------------------------------
# TP1 SARIMA refit
# ---------------------------------------------------------------------------


def make_sarima_fit(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> Fit:
    """Build a `fit` function for a fixed SARIMA(order)x(seasonal_order) spec."""

    def fit(train: pd.Series) -> Predictor:
        model = sm.tsa.SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

        def predict(horizon: int) -> np.ndarray:
            return result.get_forecast(steps=horizon).predicted_mean.to_numpy()

        return predict

    return fit


# TP1 specs, refit on the TP Final data (status/02-evaluation-baselines.md, task 02.3):
# ALB reuses TP1's own ALB spec; store_service reuses TP1's POS-API spec (no
# TP1 store_service model exists, POS API is the closest TP1 comparison).
SARIMA_SPECS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int, int]]] = {
    "alb": ((1, 1, 1), (1, 1, 1, 24)),
    "store_service": ((1, 1, 1), (1, 0, 1, 24)),
}


# ---------------------------------------------------------------------------
# Classical extras
# ---------------------------------------------------------------------------


def holt_winters_fit(train: pd.Series, m: int = SEASONAL_PERIOD) -> Predictor:
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=m, initialization_method="estimated"
    )
    result = model.fit()

    def predict(horizon: int) -> np.ndarray:
        return result.forecast(horizon).to_numpy()

    return predict


def _to_statsforecast_frame(train: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": "series", "ds": train.index.tz_convert(None), "y": train.to_numpy()})


def mstl_arima_fit(train: pd.Series) -> Predictor:
    """MSTL decomposition (periods 24, 168) with an AutoARIMA trend forecaster."""
    df = _to_statsforecast_frame(train)
    model = MSTL(season_length=[24, 168], trend_forecaster=_SFAutoARIMA())
    sf = StatsForecast(models=[model], freq="h")
    sf.fit(df)

    def predict(horizon: int) -> np.ndarray:
        forecast = sf.predict(h=horizon)
        return forecast["MSTL"].to_numpy()

    return predict


def auto_arima_fit(train: pd.Series, m: int = SEASONAL_PERIOD) -> Predictor:
    df = _to_statsforecast_frame(train)
    model = _SFAutoARIMA(season_length=m)
    sf = StatsForecast(models=[model], freq="h")
    sf.fit(df)

    def predict(horizon: int) -> np.ndarray:
        forecast = sf.predict(h=horizon)
        return forecast["AutoARIMA"].to_numpy()

    return predict


CLASSICAL_MODELS: dict[str, Fit] = {
    "holt_winters": holt_winters_fit,
    "mstl_arima": mstl_arima_fit,
    "auto_arima": auto_arima_fit,
}
