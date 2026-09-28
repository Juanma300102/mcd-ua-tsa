"""Prophet family, hybrid, ensemble, AutoML and foundation-model forecasters for phase 05.

Every model plugs into the same `fit(train) -> predictor(horizon)` contract used by
`tsa_final.evaluation`, `tsa_final.baselines`, `tsa_final.ml_models` and
`tsa_final.dl_models`, and is scored by the unmodified `evaluation.evaluate` (two fits:
`train` for the validation forecast, `train_and_val(series)` for the test forecast). Where
a library cannot expose that contract directly (AutoGluon TimeSeries needs its horizon at
construction time, like `skforecast.ForecasterDirect` in phase 03), a thin two-fit mirror is
used instead (`evaluate_autogluon`, mirroring `ml_models.evaluate_direct`).

Leakage rule (from phase 04, applies to every model here): nothing may tune
hyperparameters, early-stop, or fit ensemble weights on the 48h selection window
`[evaluation.VAL_START, evaluation.VAL_END]`. Any tuning instead uses the 168h immediately
before that window, carved from the end of whatever `train` is passed into `fit` -- the same
`ES_WINDOW_HOURS` pattern `dl_models.DlModelFit` uses. Only Prophet (05.1) needs this: its
`ProphetTunedFit` is a stateful closure that tunes once (on the first `fit` call, over the
carved tuning window) and reuses the chosen priors, unmodified, on the second (refit) call --
mirroring `DlModelFit`'s `best_epoch` pattern. Every other model here either has no
hyperparameters to tune (NeuralProphet, the MSTL+LightGBM hybrid, the equal-weight ensemble,
Chronos zero-shot) or does its own internal, time-boxed search that never sees the selection
window (AutoGluon TimeSeries, AutoTS).

Family labels used when logging into a `ResultsTable`: "Prophet", "AutoML", "Hybrid",
"Foundation", "Ensemble" (see `status/05-hybrid-automl-foundation.md`).
"""

from __future__ import annotations

import tempfile
import time
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from autots import AutoTS
from lightgbm import LGBMRegressor
from neuralprophet import NeuralProphet, set_random_seed
from prophet import Prophet
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from statsmodels.tsa.seasonal import MSTL

from . import data as _data
from .evaluation import Fit, Predictor, ResultsTable, SEASONAL_PERIOD, split, train_and_val

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", module="chronos")
warnings.filterwarnings("ignore", module="neuralprophet")
warnings.filterwarnings("ignore", module="prophet")

# ---------------------------------------------------------------------------
# Shared helpers (calendar exog, reused from the same convention as ml_models.py)
# ---------------------------------------------------------------------------

_LAGS: list[int] = list(range(1, 25)) + [48, 168, 336]
_EXOG_DTYPES = {
    "hour": "int32",
    "dayofweek": "int32",
    "is_weekend": "int32",
    "is_holiday": "int32",
    "is_intervention": "int32",
}


def _window_features() -> RollingFeatures:
    return RollingFeatures(stats=["mean", "std", "mean", "std"], window_sizes=[24, 24, 168, 168])


def _exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    return _data.calendar_features(index).astype(_EXOG_DTYPES)


def _future_exog(train: pd.Series, horizon: int) -> pd.DataFrame:
    future_index = pd.date_range(train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return _exog(future_index)


def _naive_frame(series: pd.Series, value_col: str) -> pd.DataFrame:
    """A (ds, <value_col>) frame with tz-naive timestamps, the format Prophet/NeuralProphet expect."""
    return pd.DataFrame({"ds": series.index.tz_convert(None), value_col: series.to_numpy()})


# ---------------------------------------------------------------------------
# 05.1 -- Prophet: daily seasonality, Argentine holidays, tuned + untuned
# ---------------------------------------------------------------------------

PROPHET_TUNING_WINDOW_HOURS = 168  # leakage-safe tuning window, same convention as dl_models.ES_WINDOW_HOURS


def _build_prophet(changepoint_prior_scale: float = 0.05, seasonality_prior_scale: float = 10.0) -> Prophet:
    """Prophet with daily seasonality and Argentine holidays; no weekly/yearly component.

    Weekly seasonality is dropped (phase 01: negligible after go-live, weekday means differ
    <3%). Yearly seasonality is dropped: the clean series only covers ~13 weeks, far short of
    a year, so Prophet cannot estimate a yearly Fourier term from this range.
    """
    model = Prophet(
        growth="linear",
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    model.add_country_holidays(country_name="AR")
    return model


def _prophet_predict(model: Prophet, horizon: int) -> np.ndarray:
    future = model.make_future_dataframe(periods=horizon, freq="h", include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy()


def prophet_default_fit(train: pd.Series) -> Predictor:
    """Untuned Prophet (default `changepoint_prior_scale`/`seasonality_prior_scale`), for reference."""
    model = _build_prophet()
    model.fit(_naive_frame(train, "y"))

    def predict(horizon: int) -> np.ndarray:
        return _prophet_predict(model, horizon)

    return predict


class ProphetTunedFit:
    """Stateful `fit(train) -> predictor(horizon)`: tunes Prophet's priors once, reuses them after.

    First call: carves the `PROPHET_TUNING_WINDOW_HOURS` immediately before the end of `train`
    (which, on `evaluation.evaluate`'s first call, is exactly the 168h before `VAL_START` --
    the leakage-safe window, never the 48h selection window itself), fits an Optuna study that
    scores `changepoint_prior_scale`/`seasonality_prior_scale` on that carved-out slice, and
    stores `best_params`. Every later call (the test refit on `train_and_val`) reuses
    `best_params` unchanged and never re-tunes -- the same `best_epoch` pattern
    `dl_models.DlModelFit` uses for early stopping.
    """

    def __init__(self, n_trials: int = 18, tuning_window_hours: int = PROPHET_TUNING_WINDOW_HOURS, seed: int = 42):
        self.n_trials = n_trials
        self.tuning_window_hours = tuning_window_hours
        self.seed = seed
        self.best_params: dict | None = None

    def __call__(self, train: pd.Series) -> Predictor:
        if self.best_params is None:
            self.best_params = self._tune(train)
        model = _build_prophet(**self.best_params)
        model.fit(_naive_frame(train, "y"))

        def predict(horizon: int) -> np.ndarray:
            return _prophet_predict(model, horizon)

        return predict

    def _tune(self, train: pd.Series) -> dict:
        tune_fit_part = train.iloc[: -self.tuning_window_hours]
        tune_eval_part = train.iloc[-self.tuning_window_hours :]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
                "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
            }
            model = _build_prophet(**params)
            model.fit(_naive_frame(tune_fit_part, "y"))
            y_pred = _prophet_predict(model, len(tune_eval_part))
            return float(np.mean(np.abs(y_pred - tune_eval_part.to_numpy())))

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return dict(study.best_params)


def make_prophet_tuned_fit() -> Fit:
    """Public constructor for a fresh `ProphetTunedFit` closure (one per series/evaluation run)."""
    return ProphetTunedFit()


# ---------------------------------------------------------------------------
# 05.2 -- NeuralProphet: AR lags + daily seasonality
# ---------------------------------------------------------------------------

NEURALPROPHET_N_LAGS = 24
# n_forecasts is fixed to the largest horizon this phase ever needs (the 105h test window) so
# ONE model construction/training serves both `evaluate`'s calls: predicting `horizon` hours
# always means requesting the full 105-step direct multi-output forecast and slicing to
# `horizon` -- the model never needs its output length to match the requested horizon exactly.
NEURALPROPHET_MAX_HORIZON = 105
NEURALPROPHET_EPOCHS = 40
NEURALPROPHET_SEED = 42


def neuralprophet_fit(train: pd.Series) -> Predictor:
    """NeuralProphet with `n_lags` AR lags and daily seasonality (task 05.2).

    `n_forecasts=NEURALPROPHET_MAX_HORIZON` makes the model a direct multi-output AR model
    (like `skforecast.ForecasterDirect`, but one network with `NEURALPROPHET_MAX_HORIZON`
    output heads instead of one sub-model per step). Predicting 48h (validation) or 105h
    (test) both go through the exact same trained model: `make_future_dataframe` is always
    asked for `NEURALPROPHET_MAX_HORIZON` future rows, and the forecast dataframe returned by
    `predict` has one `yhat<k>` column per output step, with exactly one non-NaN value per
    future row -- the value at column `yhat<k>` for the k-th future row (there is only one
    valid origin, the last historical timestamp, once the future block has no historic rows
    mixed in). That single per-row value is extracted via the anti-diagonal of the `yhat*`
    matrix, then the first `horizon` values are kept.
    """
    set_random_seed(NEURALPROPHET_SEED)
    model = NeuralProphet(
        n_lags=NEURALPROPHET_N_LAGS,
        n_forecasts=NEURALPROPHET_MAX_HORIZON,
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        epochs=NEURALPROPHET_EPOCHS,
        # collect_metrics=False turns off NeuralProphet's internal metrics logger, which is
        # what makes the PyTorch Lightning trainer write a `lightning_logs/` directory with
        # tfevents/hparams files to the current working directory; no metrics from training
        # are used here (only the final forecast), so nothing is lost by disabling it.
        collect_metrics=False,
    )
    df = _naive_frame(train, "y")
    model.fit(df, freq="h", progress=None)

    def predict(horizon: int) -> np.ndarray:
        future = model.make_future_dataframe(df, periods=NEURALPROPHET_MAX_HORIZON)
        forecast = model.predict(future).tail(NEURALPROPHET_MAX_HORIZON).reset_index(drop=True)
        diag = np.array([forecast.loc[j, f"yhat{j + 1}"] for j in range(NEURALPROPHET_MAX_HORIZON)])
        return diag[:horizon]

    return predict


# ---------------------------------------------------------------------------
# 05.3 -- Hybrid: MSTL (period 24) decomposition + LightGBM on the deseasonalized series
# ---------------------------------------------------------------------------


def hybrid_mstl_lightgbm_fit(train: pd.Series) -> Predictor:
    """MSTL (period 24 only -- weekly seasonality negligible, phase 01) + LightGBM on the residual.

    `statsmodels.tsa.seasonal.MSTL` decomposes `train` into trend, a single seasonal-24
    component and a remainder, fit on `train` only (no lookahead). LightGBM (default
    hyperparameters -- no tuning, so no leakage risk) is then fit with the same lag/rolling
    feature set as `ml_models` on `train` minus the extracted seasonal component ("the
    deseasonalized series"), not on the raw MSTL residual: the deseasonalized series still
    carries the trend, which the recursive lag features can track, so LightGBM only has to
    learn what the fixed 24h seasonal shape does not explain. The future seasonal component is
    deterministic (period 24), so it is reconstructed by tiling the last 24 extracted seasonal
    values across the horizon; the final forecast adds that tile back to LightGBM's
    deseasonalized forecast.
    """
    values = train.to_numpy(dtype=float)
    decomposition = MSTL(values, periods=SEASONAL_PERIOD).fit()
    seasonal = pd.Series(np.asarray(decomposition.seasonal), index=train.index)
    deseasonalized = train - seasonal

    forecaster = ForecasterRecursive(
        estimator=LGBMRegressor(random_state=42, verbose=-1), lags=list(_LAGS), window_features=_window_features()
    )
    forecaster.fit(y=deseasonalized, exog=_exog(deseasonalized.index))

    season_tail = seasonal.iloc[-SEASONAL_PERIOD:].to_numpy()

    def predict(horizon: int) -> np.ndarray:
        deseason_fc = forecaster.predict(steps=horizon, exog=_future_exog(train, horizon)).to_numpy()
        reps = int(np.ceil(horizon / SEASONAL_PERIOD))
        season_fc = np.tile(season_tail, reps)[:horizon]
        return deseason_fc + season_fc

    return predict


# ---------------------------------------------------------------------------
# 05.4 -- Ensemble: equal-weight average of the best model per earlier family
# ---------------------------------------------------------------------------


def make_ensemble_fit(member_fits: list[Fit]) -> Fit:
    """Equal-weight average of `member_fits`' forecasts (task 05.4).

    Weights are fixed at `1/len(member_fits)` on purpose, never fitted: any weight search on
    the validation window would be exactly the ensemble-weight leakage the phase 05 protocol
    bans. Each member's own `fit` is called with whatever `train`/`train_and_val` `evaluate`
    passes in, so every member is refit through its *own* existing `fit(train) ->
    predictor(horizon)` closure from `baselines.py` / `ml_models.py` / `dl_models.py` (imported,
    not modified) -- the composed closure below adds no fitting logic of its own.
    """

    def fit(train: pd.Series) -> Predictor:
        predictors = [member(train) for member in member_fits]

        def predict(horizon: int) -> np.ndarray:
            forecasts = np.stack([predictor(horizon) for predictor in predictors], axis=0)
            return forecasts.mean(axis=0)

        return predict

    return fit


# ---------------------------------------------------------------------------
# 05.5a -- AutoML: AutoGluon TimeSeries (needs prediction_length at construction, like
# skforecast.ForecasterDirect in phase 03 -- mirrors ml_models.evaluate_direct's two-fit shape)
# ---------------------------------------------------------------------------

AUTOGLUON_TIME_LIMIT_S = 300.0  # per fit call; two calls (val, test) = 600s/series, the teammate's budget


def fit_autogluon(train: pd.Series, *, prediction_length: int, time_limit: float) -> dict:
    """Fit one AutoGluon `TimeSeriesPredictor` and return its prediction, fit time and chosen model.

    Public (not underscore-prefixed): `evaluate_autogluon` below calls it twice (once per
    horizon), and the notebook's own test-forecast plotting cell calls it a third time to get
    an actual prediction array for the figure -- the same "refit once more for the plot"
    pattern `04_dl_models.ipynb` uses for its own winner.
    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    frame = pd.DataFrame(
        {"item_id": "series", "timestamp": train.index.tz_convert(None), "y": train.to_numpy()}
    )
    ts_df = TimeSeriesDataFrame.from_data_frame(frame, id_column="item_id", timestamp_column="timestamp")

    with tempfile.TemporaryDirectory(prefix="tsa_final_autogluon_") as tmp_path:
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length, target="y", freq="h", path=tmp_path, verbosity=0, eval_metric="MAE"
        )
        t0 = time.perf_counter()
        predictor.fit(train_data=ts_df, time_limit=time_limit)
        fit_time_s = time.perf_counter() - t0
        forecast = predictor.predict(ts_df)
        y_pred = forecast["mean"].to_numpy()
        model_best = predictor.model_best

    return {"y_pred": y_pred, "fit_time_s": fit_time_s, "model_best": model_best}


def evaluate_autogluon(
    series: pd.Series, *, series_name: str, table: ResultsTable, time_limit: float = AUTOGLUON_TIME_LIMIT_S
) -> dict[str, str]:
    """Score AutoGluon TimeSeries under the shared protocol; mirrors `ml_models.evaluate_direct`.

    `TimeSeriesPredictor` needs `prediction_length` (the horizon) fixed at construction, so --
    like `ForecasterDirect` in phase 03 -- it cannot sit behind a horizon-agnostic `Predictor`
    closure. Two separate predictors are trained: one for the 48h validation horizon (`time_limit`
    seconds), one for the 105h test horizon (`time_limit` seconds), keeping the phase 02
    two-fit pattern -- and giving AutoGluon a validation row comparable to every other model,
    which its own internal validation does not otherwise produce.
    """
    train, val, test = split(series)

    val_result = fit_autogluon(train, prediction_length=len(val), time_limit=time_limit)
    table.append_result(
        model="autogluon_timeseries",
        family="AutoML",
        series=series_name,
        split_name="val",
        y_true=val.to_numpy(),
        y_pred=val_result["y_pred"],
        mase_train=train,
        fit_time_s=val_result["fit_time_s"],
    )

    fit_data = train_and_val(series)
    test_result = fit_autogluon(fit_data, prediction_length=len(test), time_limit=time_limit)
    table.append_result(
        model="autogluon_timeseries",
        family="AutoML",
        series=series_name,
        split_name="test",
        y_true=test.to_numpy(),
        y_pred=test_result["y_pred"],
        mase_train=train,
        fit_time_s=test_result["fit_time_s"],
    )

    return {"val_model": val_result["model_best"], "test_model": test_result["model_best"]}


# ---------------------------------------------------------------------------
# 05.5b -- AutoML: AutoTS (bounded search, fits the plain Fit/Predictor contract directly)
# ---------------------------------------------------------------------------

AUTOTS_MODEL_LIST = "superfast"
AUTOTS_MAX_GENERATIONS = 3
AUTOTS_NUM_VALIDATIONS = 1
AUTOTS_SEED = 42


def autots_fit(train: pd.Series) -> Predictor:
    """Bounded AutoTS search (task 05.5, time budget in `status/05-hybrid-automl-foundation.md`).

    Unlike AutoGluon, `AutoTS.predict(forecast_length=...)` accepts any horizon after a single
    `fit` (its own internal validation slicing during `fit` uses `forecast_length` from
    construction, but does not restrict `predict`), so this plugs into the shared
    `fit(train) -> predictor(horizon)` contract directly -- no two-fit mirror needed, and no
    risk of AutoTS re-validating against the phase 02 selection window since its internal
    `num_validations` folds are carved from `train` alone, never from `val`/`test`.
    """
    df = pd.DataFrame({"ds": train.index.tz_convert(None), "y": train.to_numpy(), "series_id": "series"})
    model = AutoTS(
        forecast_length=SEASONAL_PERIOD * 2,
        frequency="h",
        model_list=AUTOTS_MODEL_LIST,
        max_generations=AUTOTS_MAX_GENERATIONS,
        num_validations=AUTOTS_NUM_VALIDATIONS,
        ensemble=None,
        verbose=-1,
        n_jobs=1,
        random_seed=AUTOTS_SEED,
    )
    model = model.fit(df, date_col="ds", value_col="y", id_col="series_id")

    def predict(horizon: int) -> np.ndarray:
        prediction = model.predict(forecast_length=horizon, verbose=-1)
        return prediction.forecast["series"].to_numpy()

    return predict


# ---------------------------------------------------------------------------
# 05.6 -- Foundation model: Chronos zero-shot (no fit/tuning, GPU inference)
# ---------------------------------------------------------------------------

CHRONOS_MODEL_ID = "amazon/chronos-bolt-base"

_chronos_pipeline = None  # lazily loaded once, reused across every series/split/call


def _get_chronos_pipeline():
    global _chronos_pipeline
    if _chronos_pipeline is None:
        from chronos import BaseChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL_ID, device_map=device, torch_dtype=dtype)
    return _chronos_pipeline


def chronos_zero_shot_fit(train: pd.Series) -> Predictor:
    """Zero-shot Chronos-Bolt: `fit` only stores the context series, no training happens.

    `predict` asks for the 0.5 quantile (the median, per task 05.6) via `predict_df`, which
    handles horizons beyond the model's optimized <=64-step range internally (with a quality
    caveat the library itself warns about, recorded in `status/05-hybrid-automl-foundation.md`
    rather than worked around, since no tuning is allowed for a zero-shot model anyway).
    """
    pipeline = _get_chronos_pipeline()
    context_df = pd.DataFrame(
        {"item_id": "series", "timestamp": train.index.tz_convert(None), "target": train.to_numpy()}
    )

    def predict(horizon: int) -> np.ndarray:
        forecast = pipeline.predict_df(context_df, prediction_length=horizon, quantile_levels=[0.5])
        return forecast["0.5"].to_numpy()

    return predict


# ---------------------------------------------------------------------------
# 05.7 -- TimeGPT: skipped
# ---------------------------------------------------------------------------
#
# TimeGPT (Nixtla) requires sending the series to an external API with a key. No explicit
# user approval was granted for this project (see `status/index.md` "Open decisions" and
# `status/05-hybrid-automl-foundation.md`), so it is skipped entirely -- no code, no API call.
