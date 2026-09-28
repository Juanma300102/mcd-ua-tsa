"""Phase 06 -- model selection, validation/test consistency, and final forecast.

Consolidates the results tables produced by phases 02-05 (32 models per series:
naive/classical/SARIMA baselines, ML, DL, and the Prophet/hybrid/AutoML/foundation
family), flags the models whose validation score is optimistic because it was used
to tune hyperparameters or choose ensemble members (contaminated models, see
`CONTAMINATED_MODELS`), and implements the selection rule agreed with the user:

    selected model per series = lowest validation MAE among the *eligible*
    (non-contaminated) models. Test metrics are reported for every model but
    never used to select.

Also implements the final 48h forecast: each selected/worst/TP1-foundational/
foundation-model role is refit on ALL available clean data (`data.load_clean`)
and forecast 48h beyond the last observed hour, with prediction intervals (native
where the model has them, empirical from validation-window residuals otherwise).

Every `fit(train) -> predictor(horizon)` closure reused here comes unmodified
from `tsa_final.baselines`/`tsa_final.hybrid_models` -- this module adds no new
model code, only selection/consistency logic and thin interval-computation
wrappers around the same libraries those modules already use (statsforecast,
statsmodels, chronos).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA as _SFAutoARIMA

from . import baselines as _baselines
from . import data as _data
from . import hybrid_models as _hybrid
from .evaluation import Fit, SEASONAL_PERIOD, split, train_and_val

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PHASE_RESULT_FILES: dict[int, str] = {
    2: "02_baselines.csv",
    3: "03_ml.csv",
    4: "04_dl.csv",
    5: "05_hybrid.csv",
}

SERIES_NAMES: list[str] = ["alb", "store_service"]

# ---------------------------------------------------------------------------
# Pool loading and contamination flags
# ---------------------------------------------------------------------------


def load_pool(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Concatenate the phase 02-05 results tables into one 32-models-per-series pool."""
    frames = []
    for phase, fname in PHASE_RESULT_FILES.items():
        df = pd.read_csv(results_dir / fname)
        df["phase"] = phase
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# Models whose validation score was measured on the same window used to tune
# their hyperparameters or choose their members -- decided with the user
# (see status/06-selection-forecast.md "Decisions"). Verified against
# tsa_final.ml_models.tune_hyperparameters (scores on a TimeSeriesFold at the
# train/val boundary, i.e. the selection window itself), notebooks/03_ml_models.ipynb
# (stacking_top3 built from the top-3 *_tuned models, selected and parameterized
# using that same validation MAE) and status/05-hybrid-automl-foundation.md
# ("Ensemble members selected... best-by-validation-MAE"). Prophet's `prophet_tuned`
# is NOT included: it is tuned on a separate 168h leakage-safe window, never on
# the selection window (status/05-hybrid-automl-foundation.md task 05.1).
_TUNED_REASON = (
    "Hyperparameters tuned with Optuna (tsa_final.ml_models.tune_hyperparameters), "
    "scored on a TimeSeriesFold at the train/val boundary -- the same 48h window "
    "later used to rank and select models. Wins validation, then frequently loses "
    "on test (status/03-ml-models.md)."
)
_STACKING_REASON = (
    "Built from the top-3 *_tuned models by validation MAE (notebooks/03_ml_models.ipynb): "
    "inherits their validation-window leakage and additionally selects its own members "
    "by the same validation MAE metric being scored."
)
_ENSEMBLE_REASON = (
    "Equal-weight average of the best-by-validation-MAE model from phases 02/03/04 "
    "(status/05-hybrid-automl-foundation.md): its members are chosen using the same "
    "validation MAE metric this table ranks models by, so its own validation score is optimistic."
)

CONTAMINATED_MODELS: dict[str, dict[str, str]] = {
    "alb": {
        "catboost_tuned": _TUNED_REASON,
        "random_forest_tuned": _TUNED_REASON,
        "lightgbm_tuned": _TUNED_REASON,
        "stacking_top3": _STACKING_REASON,
        "ensemble_equal_weight": _ENSEMBLE_REASON,
    },
    "store_service": {
        "ridge_tuned": _TUNED_REASON,
        "lightgbm_tuned": _TUNED_REASON,
        "xgboost_tuned": _TUNED_REASON,
        "stacking_top3": _STACKING_REASON,
        "ensemble_equal_weight": _ENSEMBLE_REASON,
    },
}


def add_contamination_flags(pool: pd.DataFrame) -> pd.DataFrame:
    """Add `val_contaminated` (bool) and `contamination_reason` (str) columns.

    Applied to every row (val and test) of a flagged model, so the flag is
    visible everywhere that model appears, even though only the val split is
    actually excluded from the selection rule.
    """
    pool = pool.copy()

    def _reason(row: pd.Series) -> str:
        return CONTAMINATED_MODELS.get(row["series"], {}).get(row["model"], "")

    pool["contamination_reason"] = pool.apply(_reason, axis=1)
    pool["val_contaminated"] = pool["contamination_reason"] != ""
    return pool


# ---------------------------------------------------------------------------
# Ranking, selection, best-per-family
# ---------------------------------------------------------------------------


def add_rank_columns(pool: pd.DataFrame) -> pd.DataFrame:
    """Add `val_rank`/`test_rank` (1 = best, i.e. lowest MAE) within each series."""
    pool = pool.copy()
    pool["val_rank"] = np.nan
    pool["test_rank"] = np.nan
    for series_name in pool["series"].unique():
        for split_name, col in (("val", "val_rank"), ("test", "test_rank")):
            mask = (pool["series"] == series_name) & (pool["split"] == split_name)
            pool.loc[mask, col] = pool.loc[mask, "MAE"].rank(method="min")
    pool["val_rank"] = pool["val_rank"].astype("Int64")
    pool["test_rank"] = pool["test_rank"].astype("Int64")
    return pool


def eligible_val(pool: pd.DataFrame, series: str) -> pd.DataFrame:
    """Non-contaminated validation rows for `series`, sorted by MAE ascending."""
    sub = pool[(pool["series"] == series) & (pool["split"] == "val") & (~pool["val_contaminated"])]
    return sub.sort_values("MAE").reset_index(drop=True)


@dataclass
class SelectionResult:
    series: str
    selected: pd.Series
    runner_up: pd.Series
    worst: pd.Series


def select_model(pool: pd.DataFrame, series: str) -> SelectionResult:
    """Selected model = lowest validation MAE among eligible models (the adopted rule).

    Also returns the runner-up (2nd lowest) and the worst performer (highest
    validation MAE), both restricted to eligible models -- a contaminated model
    is never reported as "the worst" either, since its score is optimistic, not
    pessimistic, and reporting it as worst would misrepresent it.
    """
    elig = eligible_val(pool, series)
    if len(elig) < 2:
        raise ValueError(f"fewer than 2 eligible models for series '{series}'")
    return SelectionResult(series=series, selected=elig.iloc[0], runner_up=elig.iloc[1], worst=elig.iloc[-1])


def best_per_family(pool: pd.DataFrame, series: str, split_name: str = "val") -> pd.DataFrame:
    """Best (lowest-MAE) model of each family, `split_name` rows only, all models (contaminated included, flagged)."""
    sub = pool[(pool["series"] == series) & (pool["split"] == split_name)]
    idx = sub.groupby("family")["MAE"].idxmin()
    return sub.loc[idx].sort_values("MAE").reset_index(drop=True)


def tp1_and_foundation_reference(pool: pd.DataFrame, series: str, split_name: str = "val") -> pd.DataFrame:
    """The TP1 SARIMA refit ("foundational") and Chronos-Bolt ("foundation model") rows for `series`."""
    sub = pool[(pool["series"] == series) & (pool["split"] == split_name)]
    return sub[sub["family"].isin(["SARIMA (TP1 refit)", "Foundation"])].reset_index(drop=True)


def pct_improvement(baseline_mae: float, model_mae: float) -> float:
    """% reduction in MAE of `model_mae` over `baseline_mae` (negative = model is worse)."""
    return (baseline_mae - model_mae) / baseline_mae * 100.0


# ---------------------------------------------------------------------------
# Validation vs. test consistency
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyResult:
    rho: float
    p_value: float
    table: pd.DataFrame  # model, val_rank, test_rank, rank_diff, sorted by |rank_diff| desc


def _consistency_table(pool: pd.DataFrame, series: str, models: list[str] | None = None) -> pd.DataFrame:
    val = pool[(pool["series"] == series) & (pool["split"] == "val")][["model", "MAE"]]
    test = pool[(pool["series"] == series) & (pool["split"] == "test")][["model", "MAE"]]
    merged = val.merge(test, on="model", suffixes=("_val", "_test"))
    if models is not None:
        merged = merged[merged["model"].isin(models)]
    merged["val_rank"] = merged["MAE_val"].rank(method="min")
    merged["test_rank"] = merged["MAE_test"].rank(method="min")
    merged["rank_diff"] = (merged["test_rank"] - merged["val_rank"]).astype(int)
    return merged.sort_values("rank_diff", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def val_test_consistency(pool: pd.DataFrame, series: str, eligible_only: bool = False) -> ConsistencyResult:
    """Spearman rank correlation between validation and test MAE ranking for `series`.

    `eligible_only=False` (default) uses all 32 models; `True` restricts to the
    non-contaminated pool used by the selection rule. Either way this is
    reported for context only -- per the phase 02/03/04/05 evidence, the single
    48h validation window is noisy, which is a limitation, not a reason to
    select by test.
    """
    models = eligible_val(pool, series)["model"].tolist() if eligible_only else None
    table = _consistency_table(pool, series, models=models)
    rho, p = spearmanr(table["val_rank"], table["test_rank"])
    return ConsistencyResult(rho=float(rho), p_value=float(p), table=table)


# ---------------------------------------------------------------------------
# Sensitivity table (informational -- NOT the selection rule)
# ---------------------------------------------------------------------------


def sensitivity_table(pool: pd.DataFrame, series: str) -> pd.DataFrame:
    """Who would win under three criteria; only (a) is the adopted selection rule.

    (a) adopted rule: lowest validation MAE among eligible models.
    (b) including contaminated scores: lowest validation MAE over all 32 models.
    (c) test MAE oracle: lowest test MAE over all 32 models -- for discussion
        only, never usable at forecast time (test is unseen when a model must
        be chosen) and reported here purely to show how much the ranking would
        change if it were.
    """
    val_all = pool[(pool["series"] == series) & (pool["split"] == "val")].sort_values("MAE")
    test_all = pool[(pool["series"] == series) & (pool["split"] == "test")].sort_values("MAE")
    adopted = eligible_val(pool, series).iloc[0]
    incl_contaminated = val_all.iloc[0]
    oracle = test_all.iloc[0]
    return pd.DataFrame(
        [
            {"criterion": "adopted rule (eligible, val MAE)", "winner": adopted["model"], "MAE": adopted["MAE"]},
            {
                "criterion": "including contaminated scores (val MAE)",
                "winner": incl_contaminated["model"],
                "MAE": incl_contaminated["MAE"],
            },
            {"criterion": "test MAE oracle (discussion only)", "winner": oracle["model"], "MAE": oracle["MAE"]},
        ]
    )


# ---------------------------------------------------------------------------
# Final forecast: horizon and role registry
# ---------------------------------------------------------------------------

FORECAST_HOURS = 48
INTERVAL_LEVELS = (80, 95)


def forecast_index(series: pd.Series, horizon: int = FORECAST_HOURS) -> pd.DatetimeIndex:
    """The `horizon` hourly timestamps immediately after the last observed hour of `series`."""
    start = series.index[-1] + pd.Timedelta(hours=1)
    return pd.date_range(start, periods=horizon, freq="h")


@dataclass
class RoleSpec:
    role: str  # "selected" | "worst" | "tp1_foundational" | "foundation_model"
    model: str
    kind: str  # "auto_arima" | "sarima" | "chronos" | "empirical"
    fit_fn: Fit | None = None  # for kind == "empirical"
    sarima_spec: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None  # for kind == "sarima"


# Selected/worst per the phase 06 rule (see status/06-selection-forecast.md); TP1
# foundational = the refit TP1 SARIMA spec (phase 02); foundation model = Chronos-Bolt (phase 05).
FINAL_ROLES: dict[str, list[RoleSpec]] = {
    "alb": [
        RoleSpec("selected", "auto_arima", "auto_arima"),
        RoleSpec("worst", "average", "empirical", fit_fn=_baselines.average_fit),
        RoleSpec(
            "tp1_foundational",
            "SARIMA(1,1,1)x(1,1,1,24)",
            "sarima",
            sarima_spec=_baselines.SARIMA_SPECS["alb"],
        ),
        RoleSpec("foundation_model", "chronos_bolt_base", "chronos"),
    ],
    "store_service": [
        RoleSpec("selected", "seasonal_naive_m24", "empirical", fit_fn=_baselines.seasonal_naive_fit),
        RoleSpec("worst", "drift", "empirical", fit_fn=_baselines.drift_fit),
        RoleSpec(
            "tp1_foundational",
            "SARIMA(1,1,1)x(1,0,1,24)",
            "sarima",
            sarima_spec=_baselines.SARIMA_SPECS["store_service"],
        ),
        RoleSpec("foundation_model", "chronos_bolt_base", "chronos"),
    ],
}


def _fit_fn_for_role(role: RoleSpec) -> Fit:
    """The unmodified phase 02/05 `Fit` closure matching `role`, for reuse on the test overlay."""
    if role.kind == "sarima":
        order, seasonal_order = role.sarima_spec
        return _baselines.make_sarima_fit(order, seasonal_order)
    if role.kind == "chronos":
        return _hybrid.chronos_zero_shot_fit
    if role.kind == "auto_arima":
        return _baselines.auto_arima_fit
    return role.fit_fn


# ---------------------------------------------------------------------------
# Prediction intervals -- native where available, empirical otherwise
# ---------------------------------------------------------------------------


def auto_arima_forecast_with_intervals(
    train: pd.Series, horizon: int, levels: tuple[int, ...] = INTERVAL_LEVELS, m: int = SEASONAL_PERIOD
) -> dict[str, np.ndarray]:
    """Native statsforecast AutoARIMA intervals (`level=[...]`), same model as `baselines.auto_arima_fit`."""
    df = pd.DataFrame({"unique_id": "series", "ds": train.index.tz_convert(None), "y": train.to_numpy()})
    model = _SFAutoARIMA(season_length=m)
    sf = StatsForecast(models=[model], freq="h")
    sf.fit(df)
    fc = sf.predict(h=horizon, level=list(levels))
    out = {"yhat": fc["AutoARIMA"].to_numpy()}
    for lvl in levels:
        out[f"lower_{lvl}"] = fc[f"AutoARIMA-lo-{lvl}"].to_numpy()
        out[f"upper_{lvl}"] = fc[f"AutoARIMA-hi-{lvl}"].to_numpy()
    return out


def sarima_forecast_with_intervals(
    train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    horizon: int,
    levels: tuple[int, ...] = INTERVAL_LEVELS,
) -> dict[str, np.ndarray]:
    """Native statsmodels SARIMAX confidence intervals, same spec as `baselines.make_sarima_fit`."""
    model = sm.tsa.SARIMAX(
        train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False
    )
    result = model.fit(disp=False)
    forecast_obj = result.get_forecast(steps=horizon)
    out = {"yhat": forecast_obj.predicted_mean.to_numpy()}
    for lvl in levels:
        alpha = 1 - lvl / 100
        ci = forecast_obj.conf_int(alpha=alpha)
        # Column names depend on the series' own name ("lower <name>"/"upper <name>");
        # index positionally instead of by name to stay robust to that.
        out[f"lower_{lvl}"] = ci.iloc[:, 0].to_numpy()
        out[f"upper_{lvl}"] = ci.iloc[:, 1].to_numpy()
    return out


# Chronos-Bolt was trained on quantile levels [0.1, 0.9] only (library warning,
# verified with a smoke test): asking for 0.025/0.975 silently clips to that
# range, which would misrepresent a 95% interval as identical to the 80% one.
# Only the native 80% interval is reported for Chronos; 95% is left as NaN
# rather than fabricated -- "keep it simple and honest" per the phase brief.
def chronos_forecast_with_intervals(train: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    """Native Chronos-Bolt quantile intervals; only an 80% interval is meaningful (see note above)."""
    pipeline = _hybrid._get_chronos_pipeline()
    context_df = pd.DataFrame(
        {"item_id": "series", "timestamp": train.index.tz_convert(None), "target": train.to_numpy()}
    )
    fc = pipeline.predict_df(context_df, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    n = len(fc)
    return {
        "yhat": fc["0.5"].to_numpy(),
        "lower_80": fc["0.1"].to_numpy(),
        "upper_80": fc["0.9"].to_numpy(),
        "lower_95": np.full(n, np.nan),
        "upper_95": np.full(n, np.nan),
    }


def empirical_intervals(
    fit_fn: Fit, series: pd.Series, horizon: int, levels: tuple[int, ...] = INTERVAL_LEVELS
) -> dict[str, np.ndarray]:
    """Empirical intervals for a model with no native ones (naive/seasonal-naive/drift/average).

    Method (documented, deliberately simple): fit `fit_fn` on the phase 02
    `train` split, forecast the 48h validation window, and take the *pooled*
    residuals (val actual - val forecast, n=48) as the model's error
    distribution. The `(100-level)/2` / `100-(100-level)/2` empirical
    percentiles of those residuals are added as *constant* offsets (not
    time-varying) to the point forecast at every horizon step. This is a rough
    uncertainty proxy, not a calibrated interval: it comes from a single 48h
    window (n=48, the same noisy-window limitation phases 02-05 already
    documented), and it does not model error growth with the horizon step.
    """
    train, val, _test = split(series)
    val_pred = fit_fn(train)(len(val))
    residuals = val.to_numpy() - val_pred

    final_pred = fit_fn(series)(horizon)
    out = {"yhat": final_pred}
    for lvl in levels:
        lo_q, hi_q = (100 - lvl) / 2, 100 - (100 - lvl) / 2
        lo_off, hi_off = np.percentile(residuals, lo_q), np.percentile(residuals, hi_q)
        out[f"lower_{lvl}"] = final_pred + lo_off
        out[f"upper_{lvl}"] = final_pred + hi_off
    return out


def _forecast_role(role: RoleSpec, series: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    if role.kind == "auto_arima":
        return auto_arima_forecast_with_intervals(series, horizon)
    if role.kind == "sarima":
        order, seasonal_order = role.sarima_spec
        return sarima_forecast_with_intervals(series, order, seasonal_order, horizon)
    if role.kind == "chronos":
        return chronos_forecast_with_intervals(series, horizon)
    if role.kind == "empirical":
        return empirical_intervals(role.fit_fn, series, horizon)
    raise ValueError(f"unknown role kind '{role.kind}'")


def build_final_forecast_table(horizon: int = FORECAST_HOURS) -> pd.DataFrame:
    """Refit every role in `FINAL_ROLES` on all available clean data and forecast `horizon` hours ahead."""
    rows = []
    for series_name, roles in FINAL_ROLES.items():
        series = _data.load_clean(series_name)
        idx = forecast_index(series, horizon)
        for role in roles:
            result = _forecast_role(role, series, horizon)
            n = len(result["yhat"])
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": idx[:n],
                        "series": series_name,
                        "model": role.model,
                        "role": role.role,
                        "yhat": result["yhat"],
                        "lower_80": result.get("lower_80", np.full(n, np.nan)),
                        "upper_80": result.get("upper_80", np.full(n, np.nan)),
                        "lower_95": result.get("lower_95", np.full(n, np.nan)),
                        "upper_95": result.get("upper_95", np.full(n, np.nan)),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_overlay_forecasts(series_name: str) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Refit each `FINAL_ROLES[series_name]` model on train+val and forecast the held-out test window.

    Reuses the exact same unmodified `Fit` closures as `FINAL_ROLES` (just fit on
    `train_and_val(series)` instead of the full clean series), for the comparison
    figure only -- point forecasts, no intervals needed there.
    """
    series = _data.load_clean(series_name)
    _train, _val, test = split(series)
    fit_data = train_and_val(series)
    horizon = len(test)

    forecasts: dict[str, np.ndarray] = {}
    for role in FINAL_ROLES[series_name]:
        fit_fn = _fit_fn_for_role(role)
        forecasts[role.role] = fit_fn(fit_data)(horizon)
    return forecasts, test
