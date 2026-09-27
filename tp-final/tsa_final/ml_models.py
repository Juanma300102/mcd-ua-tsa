"""ML forecasters for the TP Final phase 03 sweep.

Feature engineering is delegated to skforecast's own machinery rather than a
hand-built feature matrix: lags 1-24/48/168/336 and rolling mean/std at
windows 24/168 are configured on each ``ForecasterRecursive``/``ForecasterDirect``
via ``lags=``/``window_features=``, and calendar features (hour, dayofweek,
is_weekend, is_holiday, is_intervention) from ``tsa_final.data.calendar_features``
are passed as ``exog``. Every ``fit(train) -> predictor(horizon)`` closure
derives the future exog from the forecast index internally (calendar features
are a deterministic function of the timestamp), so the phase 02
``Predictor(horizon)`` contract in ``tsa_final.evaluation`` never changes. No
ALB exogenous is used for store_service (see ``status/03-ml-models.md``).

``ForecasterDirect`` trains one sub-model per horizon step, so it must be
constructed with the exact horizon it will be asked to predict; unlike the
recursive models, its fit cost cannot be measured inside a `fit(train)`
closure that does not yet know the horizon. ``evaluate_direct`` mirrors
``tsa_final.evaluation.evaluate``'s two-fit structure directly (val fit on
``train``, test fit on ``train_and_val``) so fit time is still recorded
correctly.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor

from . import data as _data
from .evaluation import Fit, Predictor, ResultsTable, split, train_and_val

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Shared feature configuration (task 2.1, 2.2)
# ---------------------------------------------------------------------------

LAGS: list[int] = list(range(1, 25)) + [48, 168, 336]
_ROLLING_STATS = ["mean", "std", "mean", "std"]
_ROLLING_WINDOWS = [24, 24, 168, 168]

_EXOG_DTYPES = {
    "hour": "int32",
    "dayofweek": "int32",
    "is_weekend": "int32",
    "is_holiday": "int32",
    "is_intervention": "int32",
}


def _window_features() -> RollingFeatures:
    return RollingFeatures(stats=list(_ROLLING_STATS), window_sizes=list(_ROLLING_WINDOWS))


def _exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar exog (task 2.3), cast to numeric dtypes skforecast can consume."""
    return _data.calendar_features(index).astype(_EXOG_DTYPES)


def _future_exog(train: pd.Series, horizon: int) -> pd.DataFrame:
    """Calendar exog for the `horizon` hours right after `train` (task 2.4).

    No ALB or other non-deterministic exogenous is ever included here (task 2.5)
    — only calendar features, which are computable for any future timestamp.
    """
    future_index = pd.date_range(train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return _exog(future_index)


# ---------------------------------------------------------------------------
# Recursive ML forecasters, default hyperparameters (task 3.1)
# ---------------------------------------------------------------------------


def _make_forecaster_fit(build_estimator: Callable[[], object]) -> Fit:
    """Build a `fit(train) -> predictor(horizon)` closure around a recursive forecaster."""

    def fit(train: pd.Series) -> Predictor:
        forecaster = ForecasterRecursive(
            estimator=build_estimator(), lags=list(LAGS), window_features=_window_features()
        )
        forecaster.fit(y=train, exog=_exog(train.index))

        def predict(horizon: int) -> np.ndarray:
            forecast = forecaster.predict(steps=horizon, exog=_future_exog(train, horizon))
            return forecast.to_numpy()

        return predict

    return fit


# name -> (estimator class, fixed (non-tuned) kwargs, Optuna search space)
#
# `search_space` returns ONLY the keys produced by `trial.suggest_*` —
# skforecast's `bayesian_search_forecaster` requires the returned dict's keys
# to match the suggested names exactly. Fixed kwargs (random_state, verbosity)
# live on the estimator instance itself (`_default_builder` below), which
# Optuna's `set_params(**search_space_dict)` only partially overrides.
MODEL_SPECS: dict[str, tuple[type, dict, Callable[["optuna.Trial"], dict]]] = {
    "lightgbm": (
        LGBMRegressor,
        {"random_state": 42, "verbose": -1},
        lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        },
    ),
    "xgboost": (
        XGBRegressor,
        {"random_state": 42, "verbosity": 0},
        lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        },
    ),
    "catboost": (
        CatBoostRegressor,
        {"random_state": 42, "verbose": 0},
        lambda trial: {
            "iterations": trial.suggest_int("iterations", 50, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        },
    ),
    "random_forest": (
        RandomForestRegressor,
        {"random_state": 42, "n_jobs": -1},
        lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        },
    ),
    "ridge": (
        Ridge,
        {},
        lambda trial: {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)},
    ),
}


def _default_builder(estimator_cls: type, fixed_kwargs: dict) -> Callable[[], object]:
    return lambda: estimator_cls(**fixed_kwargs)


RECURSIVE_MODELS: dict[str, Fit] = {
    name: _make_forecaster_fit(_default_builder(estimator_cls, fixed_kwargs))
    for name, (estimator_cls, fixed_kwargs, _) in MODEL_SPECS.items()
}


# ---------------------------------------------------------------------------
# Direct multi-step strategy, time-boxed (task 4)
# ---------------------------------------------------------------------------


def evaluate_direct(
    build_estimator: Callable[[], object],
    *,
    model: str,
    series_name: str,
    series: pd.Series,
    table: ResultsTable,
    n_jobs: int | str = -1,
) -> None:
    """Score a `ForecasterDirect` version of `build_estimator` under the shared protocol.

    Mirrors `tsa_final.evaluation.evaluate`'s two-fit structure (val fit on
    `train`, test fit on `train_and_val`), but `ForecasterDirect` needs its
    exact horizon (`steps`) at construction time, so — unlike the recursive
    closures — the forecaster is built fresh, with the correct `steps`, inside
    this function rather than inside a horizon-agnostic `Predictor` closure.
    This keeps `fit_time_s` accurate (one `ForecasterDirect` trains one
    sub-model per step, so its cost must be timed around the real `.fit()`
    call, not a lazily-deferred one).
    """
    train, val, test = split(series)

    t0 = time.perf_counter()
    val_forecaster = ForecasterDirect(
        estimator=build_estimator(), steps=len(val), lags=list(LAGS), window_features=_window_features(), n_jobs=n_jobs
    )
    val_forecaster.fit(y=train, exog=_exog(train.index))
    val_fit_time = time.perf_counter() - t0
    val_pred = val_forecaster.predict(steps=len(val), exog=_future_exog(train, len(val))).to_numpy()
    table.append_result(
        model=model,
        family="ML (direct)",
        series=series_name,
        split_name="val",
        y_true=val.to_numpy(),
        y_pred=val_pred,
        mase_train=train,
        fit_time_s=val_fit_time,
    )

    fit_data = train_and_val(series)
    t0 = time.perf_counter()
    test_forecaster = ForecasterDirect(
        estimator=build_estimator(), steps=len(test), lags=list(LAGS), window_features=_window_features(), n_jobs=n_jobs
    )
    test_forecaster.fit(y=fit_data, exog=_exog(fit_data.index))
    test_fit_time = time.perf_counter() - t0
    test_pred = test_forecaster.predict(steps=len(test), exog=_future_exog(fit_data, len(test))).to_numpy()
    table.append_result(
        model=model,
        family="ML (direct)",
        series=series_name,
        split_name="test",
        y_true=test.to_numpy(),
        y_pred=test_pred,
        mase_train=train,
        fit_time_s=test_fit_time,
    )


# ---------------------------------------------------------------------------
# Hyperparameter tuning, gated and time-boxed (task 5)
# ---------------------------------------------------------------------------


def tune_hyperparameters(
    name: str,
    series: pd.Series,
    *,
    horizon: int = 48,
    n_trials: int = 20,
    timeout_s: float = 180.0,
    metric: str = "mean_absolute_error",
    random_state: int = 42,
) -> dict:
    """Bayesian-search `name`'s hyperparameters on a single train/val fold.

    Uses a `TimeSeriesFold` configured with `initial_train_size` at the
    train/val boundary and `refit=False`: with `series` truncated to
    train+val and no data left afterwards, this produces exactly one fold
    (fit on train, score on val) — the phase 02 single-split protocol, not
    N-fold backtesting. Capped at `n_trials` trials or `timeout_s` seconds,
    whichever is reached first (task 5.2/5.3's time-box).
    """
    estimator_cls, fixed_kwargs, search_space = MODEL_SPECS[name]
    train, val, _test = split(series)
    train_and_val_series = pd.concat([train, val])

    forecaster = ForecasterRecursive(
        estimator=estimator_cls(**fixed_kwargs), lags=list(LAGS), window_features=_window_features()
    )
    cv = TimeSeriesFold(steps=horizon, initial_train_size=len(train), refit=False)

    _results, study = bayesian_search_forecaster(
        forecaster=forecaster,
        y=train_and_val_series,
        cv=cv,
        search_space=search_space,
        metric=metric,
        exog=_exog(train_and_val_series.index),
        n_trials=n_trials,
        random_state=random_state,
        return_best=False,
        n_jobs=-1,
        show_progress=False,
        suppress_warnings=True,
        kwargs_study_optimize={"timeout": timeout_s},
    )
    return dict(study.best_params)


def make_tuned_fit(name: str, best_params: dict) -> Fit:
    """`fit(train) -> predictor(horizon)` for `name` refit with `best_params` (task 5.4)."""
    estimator_cls, fixed_kwargs, _ = MODEL_SPECS[name]
    return _make_forecaster_fit(lambda: estimator_cls(**fixed_kwargs, **best_params))


def default_builder(name: str) -> Callable[[], object]:
    """Public accessor for `name`'s default (untuned) estimator builder, e.g. for `evaluate_direct`."""
    estimator_cls, fixed_kwargs, _ = MODEL_SPECS[name]
    return _default_builder(estimator_cls, fixed_kwargs)


# ---------------------------------------------------------------------------
# Stacking ensemble as a recursive forecaster (task 6)
# ---------------------------------------------------------------------------


def make_stacking_fit(estimators: list[tuple[str, object]], *, cv: int = 3) -> Fit:
    """Stack `estimators` (name, unfitted-template) with sklearn `StackingRegressor`.

    `cv` is set explicitly (sklearn's `StackingRegressor` default is 5, which
    would silently refit every base learner ~5-6x to build meta-features —
    task 6.2's time-box). The stack is wrapped as a `ForecasterRecursive`
    estimator (task 6.3), not fit flat, so it produces a genuine multi-step
    recursive forecast: each step's lags can depend on the stack's own
    earlier predictions, not on ground truth.
    """

    def build_stack() -> StackingRegressor:
        cloned = [(name, clone(estimator)) for name, estimator in estimators]
        return StackingRegressor(estimators=cloned, cv=cv, n_jobs=-1)

    return _make_forecaster_fit(build_stack)


# ---------------------------------------------------------------------------
# Feature importance for the single best model (task 7)
# ---------------------------------------------------------------------------


def training_matrix(name: str, estimator: object, train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild the (X, y) matrix a fitted forecaster trained on, for SHAP (task 7.1-7.2)."""
    forecaster = ForecasterRecursive(estimator=estimator, lags=list(LAGS), window_features=_window_features())
    X_train, y_train = forecaster.create_train_X_y(y=train, exog=_exog(train.index))
    return X_train, y_train
