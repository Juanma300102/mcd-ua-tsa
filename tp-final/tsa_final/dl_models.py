"""Deep-learning forecasters for the TP Final phase 04 sweep.

One library (darts) for every architecture so covariate handling, scaling and
the `fit(train) -> predictor(horizon)` contract in `tsa_final.evaluation` stay
identical across models — the same reason phase 03 kept every ML forecaster
inside skforecast.

Leakage rule (see `status/04-dl-models.md` Decisions): phase 03's Optuna tuning
scored trials on the *same* 48h window later used to rank/select models, which
let tuned models overfit that window (win validation, lose badly on test). For
DL, early stopping is deliberately kept off the selection window:

- The "tuning" window is the `ES_WINDOW_HOURS` (168h) immediately before
  `evaluation.VAL_START`, carved from the *end* of whatever `train` is passed
  into `fit`. The model is trained on `train` minus that window and monitors
  `val_loss` on it, stopping with patience.
- The best epoch (lowest monitored `val_loss`) is recorded. `evaluation.evaluate`
  calls `fit` a second time on `train_and_val(series)` for the test refit; that
  second call trains for the fixed recorded epoch count with no early stopping
  and no validation split at all — there is nothing left to leak into, because
  the model never sees `VAL_START:VAL_END` labels during training in either call.

`DlModelFit` (below) is the small stateful closure the module docstring in
`tsa_final.evaluation` anticipates: it stores `best_epoch` from the first call
and reuses it, unmodified, on the second.

Scaling: `darts.dataprocessing.transformers.Scaler` fit on the exact data each
call trains on (never on val/test) — `train` minus the tuning window on the
first call, `train_and_val(series)` on the second. Series are first mapped
through `np.log1p` (traffic counts are strictly positive and heavy-tailed;
log1p compresses the growth/ramp and the intervention-adjacent spikes before
the linear scaler sees them) and predictions are mapped back with `np.expm1`
after `Scaler.inverse_transform`.

Covariates: calendar features from `tsa_final.data.calendar_features`, cast to
cyclical `hour_sin/cos` and `dayofweek_sin/cos` plus `is_holiday`. The
`is_intervention` dummy is dropped (task instruction: it is always 0 in every
future window, so it carries no information a future-covariate model can use).
Whether a covariate series is passed as `future_covariates` or
`past_covariates` is decided once per model from its own
`supports_future_covariates`/`supports_past_covariates` flags (`_covariate_kwargs`),
so every architecture is wired the same way instead of branching by name.
"""

from __future__ import annotations

import logging
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel, NBEATSModel, NHiTSModel, TCNModel, TFTModel, TiDEModel
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping

from . import data as _data
from .evaluation import Predictor

warnings.filterwarnings("ignore", module="pytorch_lightning")
warnings.filterwarnings("ignore", module="lightning")
for _logger_name in ("pytorch_lightning", "lightning", "lightning.pytorch", "darts"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Shared sizing (task 04.1) and time budget (status/04-dl-models.md)
# ---------------------------------------------------------------------------

INPUT_CHUNK_LENGTH = 168  # one week of hourly history
OUTPUT_CHUNK_LENGTH = 48  # matches the phase 02 validation window / TP1 horizon
ES_WINDOW_HOURS = 168  # early-stopping monitor window, carved from train's tail
MAX_EPOCHS = 100
PATIENCE = 12
# darts' default batch_size (32) was measured faster wall-clock than 128 for the slowest
# architecture (TFT): a larger batch needs more epochs to reach the same val_loss with the
# same learning rate, which more than offset the fewer batches/epoch on this ~1,500-sample
# training set. Kept explicit (not just relying on the darts default) so the decision is visible.
BATCH_SIZE = 32
SEEDS = [42, 43]  # fixed 2 seeds, no 3rd seed (status/04-dl-models.md time budget)

_WORK_DIR = tempfile.mkdtemp(prefix="tsa_final_darts_")

# ---------------------------------------------------------------------------
# Covariates (shared across every architecture)
# ---------------------------------------------------------------------------


def _covariate_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar covariates for `index`; drops `is_intervention` (always 0 in the future)."""
    features = _data.calendar_features(index)
    hour = features["hour"].to_numpy(dtype=float)
    dow = features["dayofweek"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_holiday": features["is_holiday"].to_numpy(dtype=float),
        },
        index=index,
    )


def _covariate_series(index: pd.DatetimeIndex) -> TimeSeries:
    return TimeSeries.from_dataframe(_covariate_frame(index), freq="h")


def _covariate_kwargs(model: object, index: pd.DatetimeIndex) -> dict[str, TimeSeries]:
    """Route the same covariate series to whichever kwarg `model` actually supports."""
    if model.supports_future_covariates:
        return {"future_covariates": _covariate_series(index)}
    if model.supports_past_covariates:
        return {"past_covariates": _covariate_series(index)}
    return {}


def _log_series(values: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(np.log1p(values), freq="h")


# ---------------------------------------------------------------------------
# Best-epoch tracking for the leakage-safe early-stopping rule
# ---------------------------------------------------------------------------


class _BestEpochTracker(Callback):
    """Records the epoch with the lowest monitored `val_loss` seen so far."""

    def __init__(self) -> None:
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001 - pytorch_lightning hook signature
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = trainer.current_epoch


# ---------------------------------------------------------------------------
# Architecture registry (task 04.2-04.4): (darts class, architecture kwargs)
# ---------------------------------------------------------------------------

ARCHITECTURES: dict[str, tuple[type, dict]] = {
    "lstm": (BlockRNNModel, {"model": "LSTM", "hidden_dim": 64, "n_rnn_layers": 2, "dropout": 0.1}),
    "nbeats": (
        NBEATSModel,
        {"num_stacks": 3, "num_blocks": 1, "num_layers": 2, "layer_widths": 128, "dropout": 0.1},
    ),
    "nhits": (
        NHiTSModel,
        {"num_stacks": 3, "num_blocks": 1, "num_layers": 2, "layer_widths": 128, "dropout": 0.1},
    ),
    "tcn": (TCNModel, {"kernel_size": 3, "num_filters": 32, "dilation_base": 2, "weight_norm": True, "dropout": 0.1}),
    "tide": (TiDEModel, {"num_encoder_layers": 2, "num_decoder_layers": 2, "hidden_size": 128, "dropout": 0.1}),
    "tft": (
        TFTModel,
        {"hidden_size": 64, "lstm_layers": 1, "num_attention_heads": 4, "dropout": 0.1, "add_relative_index": True},
    ),
}


# ---------------------------------------------------------------------------
# fit(train) -> predictor(horizon) closure with leakage-safe early stopping
# ---------------------------------------------------------------------------


class DlModelFit:
    """`Fit` closure for one (architecture, seed) pair, reused across the two `evaluate` calls.

    The first call (`evaluation.evaluate`'s validation fit, argument `train`)
    trains with early stopping on the tuning window and records `best_epoch`.
    Every later call (the test refit on `train_and_val(series)`) trains for
    exactly `best_epoch` epochs with no early stopping and no validation
    split, per the phase 04 leakage rule.
    """

    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self.seed = seed
        self.model_cls, self.extra_kwargs = ARCHITECTURES[name]
        self.best_epoch: int | None = None

    def __call__(self, train: pd.Series) -> Predictor:
        if self.best_epoch is None:
            return self._fit_with_early_stopping(train)
        return self._fit_fixed_epochs(train)

    # -- model construction -------------------------------------------------

    def _build(self, *, n_epochs: int, callbacks: list) -> object:
        trainer_kwargs = {
            "accelerator": "gpu",
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "enable_checkpointing": False,
            "logger": False,
            "callbacks": callbacks,
        }
        return self.model_cls(
            input_chunk_length=INPUT_CHUNK_LENGTH,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            n_epochs=n_epochs,
            batch_size=BATCH_SIZE,
            random_state=self.seed,
            pl_trainer_kwargs=trainer_kwargs,
            work_dir=_WORK_DIR,
            model_name=f"{self.name}_{self.seed}",
            save_checkpoints=False,
            force_reset=True,
            **self.extra_kwargs,
        )

    # -- first call: early-stopping fit on train minus the tuning window ----

    def _fit_with_early_stopping(self, train: pd.Series) -> Predictor:
        seed_everything(self.seed, workers=False, verbose=False)

        fit_part = train.iloc[:-ES_WINDOW_HOURS]
        val_part = train.iloc[-(INPUT_CHUNK_LENGTH + ES_WINDOW_HOURS) :]

        scaler = Scaler()
        fit_ts = scaler.fit_transform(_log_series(fit_part))
        val_ts = scaler.transform(_log_series(val_part))

        tracker = _BestEpochTracker()
        model = self._build(
            n_epochs=MAX_EPOCHS,
            callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"), tracker],
        )

        fit_cov = _covariate_kwargs(model, fit_part.index)
        val_cov = {f"val_{k}": v for k, v in _covariate_kwargs(model, val_part.index).items()}
        model.fit(series=fit_ts, val_series=val_ts, **fit_cov, **val_cov, verbose=False)

        self.best_epoch = tracker.best_epoch + 1  # epochs are 0-indexed; run that many on the refit
        context = scaler.transform(_log_series(train))
        return self._make_predictor(model, scaler, context, train.index[0], train.index[-1])

    # -- second+ call: fixed epochs, no early stopping, no val split --------

    def _fit_fixed_epochs(self, train: pd.Series) -> Predictor:
        seed_everything(self.seed, workers=False, verbose=False)

        scaler = Scaler()
        context = scaler.fit_transform(_log_series(train))
        model = self._build(n_epochs=self.best_epoch, callbacks=[])
        cov = _covariate_kwargs(model, train.index)
        model.fit(series=context, **cov, verbose=False)

        return self._make_predictor(model, scaler, context, train.index[0], train.index[-1])

    # -- prediction -----------------------------------------------------------

    @staticmethod
    def _make_predictor(
        model: object, scaler: Scaler, context: TimeSeries, context_start: pd.Timestamp, context_end: pd.Timestamp
    ) -> Predictor:
        def predict(horizon: int) -> np.ndarray:
            # `context_start`/`context_end` come from the original tz-aware UTC pandas
            # index (darts strips tz internally), so covariates stay computable via
            # `data.calendar_features`, which requires a tz-aware UTC index.
            cov_index = pd.date_range(context_start, context_end + pd.Timedelta(hours=horizon), freq="h")
            cov = _covariate_kwargs(model, cov_index)
            pred_ts = model.predict(n=horizon, series=context, **cov, verbose=False, show_warnings=False)
            pred_log = scaler.inverse_transform(pred_ts)
            return np.expm1(pred_log.values().reshape(-1))

        return predict


def make_fit(name: str, seed: int) -> DlModelFit:
    """Public constructor for one (architecture, seed) `Fit` closure."""
    return DlModelFit(name, seed)
