"""Shared evaluation protocol for the TP Final forecasting models.

Implements the phase 02 protocol from `status/02-evaluation-baselines.md`: a
single train/validation/test split (not expanding-window backtesting, which
was dropped 2026-09-27 to keep total training time in hours rather than days
across the ~15 models planned for phases 03-06), a metrics function (MAE,
RMSE, MAPE, MASE m=24), and a shared results table every phase appends to.

Every forecaster, from phase 02's baselines through phase 06's final
selection, conforms to a single calling convention so this module never
branches on which library produced the model::

    predictor = fit(train)          # fit(train: pd.Series) -> Predictor
    forecast = predictor(horizon)   # Predictor(horizon: int) -> np.ndarray, len == horizon

``evaluate`` fits a model twice (once on ``train`` for the validation
forecast, once on the full pre-test history for the test forecast) and logs
both rows into a ``ResultsTable``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from . import data as _data

VALIDATION_HOURS = 48
SEASONAL_PERIOD = 24

# Single validation window: 48h ending immediately before INTERVENTION_START,
# so its scored horizon never overlaps the imputed intervention (see
# status/02-evaluation-baselines.md).
VAL_START = _data.INTERVENTION_START - pd.Timedelta(hours=VALIDATION_HOURS)
VAL_END = _data.INTERVENTION_START - pd.Timedelta(hours=1)


class Predictor(Protocol):
    def __call__(self, horizon: int) -> np.ndarray: ...


Fit = Callable[[pd.Series], Predictor]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a clean series into (train, val, test) per the phase 02 protocol.

    ``val`` is the single 48h window ``[VAL_START, VAL_END]`` (never
    expanding-window folds). ``test`` is the fixed ``[TEST_START, TEST_END]``
    window. ``train`` is everything strictly before ``val``. Raises
    ``ValueError`` if ``series`` does not cover the required range.

    Use `train_and_val(series)` (not `pd.concat([train, val])`) when refitting
    for the test forecast: it includes the imputed intervention window that
    sits between ``val`` and ``test``, which `val` deliberately excludes from
    its *scored* horizon but which is still real history the model should train on.
    """
    if series.index.min() > VAL_START or series.index.max() < _data.TEST_END:
        raise ValueError(
            f"series range [{series.index.min()}, {series.index.max()}] does not cover "
            f"the required [{VAL_START}, {_data.TEST_END}] train/val/test range"
        )

    train = series.loc[: VAL_START - pd.Timedelta(hours=1)]
    val = series.loc[VAL_START:VAL_END]
    test = series.loc[_data.TEST_START : _data.TEST_END]

    if len(val) != VALIDATION_HOURS:
        raise ValueError(f"validation window has {len(val)} hours, expected {VALIDATION_HOURS}")
    expected_test_hours = int((_data.TEST_END - _data.TEST_START) / pd.Timedelta(hours=1)) + 1
    if len(test) != expected_test_hours:
        raise ValueError(f"test window has {len(test)} hours, expected {expected_test_hours}")

    return train, val, test


def train_and_val(series: pd.Series) -> pd.Series:
    """Continuous history up to (not including) the test window.

    Covers train + val + the imputed intervention window in between, with no
    gap in the hourly index — the data a model should be refit on before
    forecasting the test window.
    """
    return series.loc[: _data.TEST_START - pd.Timedelta(hours=1)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def _seasonal_naive_in_sample_mae(train: pd.Series, m: int = SEASONAL_PERIOD) -> float:
    """In-sample MAE of the seasonal-naive (period m) forecast on `train` — MASE's scale."""
    values = train.to_numpy(dtype=float)
    return float(np.mean(np.abs(values[m:] - values[:-m])))


def mase(y_true: np.ndarray, y_pred: np.ndarray, train: pd.Series, m: int = SEASONAL_PERIOD) -> float:
    """Mean Absolute Scaled Error (Hyndman-Koehler), scaled by seasonal-naive in-sample MAE.

    ``train`` is always the phase 02 ``train`` split (not ``train_and_val``),
    for both the validation and the test row of a given model, so the two
    rows share the same scale and are comparable to each other.
    """
    scale = _seasonal_naive_in_sample_mae(train, m=m)
    return mae(y_true, y_pred) / scale


def metrics(y_true: np.ndarray, y_pred: np.ndarray, train: pd.Series, m: int = SEASONAL_PERIOD) -> dict[str, float]:
    """MAE, RMSE, MAPE (%) and MASE (m) for one forecast, in the original scale."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, train, m=m),
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


@dataclass
class ResultsTable:
    """Accumulates one row per (model, family, series, split); shared across phases 02-06."""

    rows: list[dict] = field(default_factory=list)

    def append_result(
        self,
        *,
        model: str,
        family: str,
        series: str,
        split_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mase_train: pd.Series,
        fit_time_s: float,
        m: int = SEASONAL_PERIOD,
    ) -> None:
        row = {
            "model": model,
            "family": family,
            "series": series,
            "split": split_name,
            **metrics(y_true, y_pred, mase_train, m=m),
            "fit_time_s": fit_time_s,
        }
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    fit: Fit,
    *,
    model: str,
    family: str,
    series_name: str,
    series: pd.Series,
    table: ResultsTable,
) -> None:
    """Score one model on `series` and log val + test rows into `table`.

    Two fits, per the phase 02 protocol (single split, not N-fold
    backtesting): ``fit`` runs once on ``train`` for the validation forecast
    (model selection), and once more on ``train_and_val(series)`` for the
    test forecast (final table). MASE always scales by `train` (see `mase`).
    """
    train, val, test = split(series)

    t0 = time.perf_counter()
    val_predictor = fit(train)
    val_fit_time = time.perf_counter() - t0
    val_pred = val_predictor(len(val))
    table.append_result(
        model=model,
        family=family,
        series=series_name,
        split_name="val",
        y_true=val.to_numpy(),
        y_pred=val_pred,
        mase_train=train,
        fit_time_s=val_fit_time,
    )

    fit_data = train_and_val(series)
    t0 = time.perf_counter()
    test_predictor = fit(fit_data)
    test_fit_time = time.perf_counter() - t0
    test_pred = test_predictor(len(test))
    table.append_result(
        model=model,
        family=family,
        series=series_name,
        split_name="test",
        y_true=test.to_numpy(),
        y_pred=test_pred,
        mase_train=train,
        fit_time_s=test_fit_time,
    )
