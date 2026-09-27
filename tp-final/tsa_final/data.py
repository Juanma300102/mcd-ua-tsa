"""Loading, cleaning and calendar-feature helpers for the TP Final series.

All three project series (two implemented here, room for a teammate's third
one) are AWS ALB/target-group hourly metrics exported as CSV with columns
``load_balancer, target_group, metric_name, period_start_utc, period_end_utc,
value``. This module centralizes:

- a small registry mapping a short series name to its CSV file and metric,
- the dated constants decided during the phase 01 EDA (go-live, steady-state
  start, the known intervention window, and the phase 02 test window),
- ``load_raw`` / ``load_clean`` to get a model-ready ``pd.Series``,
- ``impute_intervention`` to repair the known client-behaviour intervention,
- ``calendar_features`` to derive local-time calendar/holiday regressors.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import pandas as pd

# ---------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# name -> {"file": csv filename under DATA_DIR, "metric": metric_name to keep}
# Add a new entry here to register a third series; load_raw/load_clean work
# for any registry entry without further changes.
SERIES_REGISTRY: dict[str, dict[str, str]] = {
    "alb": {
        "file": "alb-request-count-hourly-since-2026-05-01.csv",
        "metric": "RequestCount",
    },
    "store_service": {
        "file": "store-service-request-count-per-target-hourly-since-2026-05-01.csv",
        "metric": "RequestCountPerTarget",
    },
}

# ---------------------------------------------------------------------------
# Dated constants agreed upon in the phase 01 EDA (see status/01-data-eda.md)
# ---------------------------------------------------------------------------

# Local timezone used for calendar features (business hours, holidays).
LOCAL_TZ = "America/Argentina/Buenos_Aires"

# The load balancer went live around 2026-06-26 12:00 UTC; traffic ramps up
# until steady state is reached. Series before this timestamp are dropped.
STEADY_STATE_START = pd.Timestamp("2026-07-03 00:00", tz="UTC")

# Known deployment that changed client behaviour and was later rolled back.
# Window is inclusive on both ends (116 hourly observations).
INTERVENTION_START = pd.Timestamp("2026-09-17 18:00", tz="UTC")
INTERVENTION_END = pd.Timestamp("2026-09-22 13:00", tz="UTC")

# Phase 02 held-out test window: observed, post-rollback data only.
TEST_START = pd.Timestamp("2026-09-22 14:00", tz="UTC")
TEST_END = pd.Timestamp("2026-09-26 22:00", tz="UTC")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(name: str) -> pd.Series:
    """Load the full raw hourly series for a registered series name.

    Reads the CSV, keeps only rows matching the registered metric, parses
    ``period_start_utc`` as a UTC ``DatetimeIndex`` with hourly frequency,
    and returns a float ``pd.Series`` named ``name``.
    """
    if name not in SERIES_REGISTRY:
        raise ValueError(f"Unknown series '{name}'. Known series: {sorted(SERIES_REGISTRY)}")

    entry = SERIES_REGISTRY[name]
    df = pd.read_csv(DATA_DIR / entry["file"])
    df = df[df["metric_name"] == entry["metric"]].copy()
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"], utc=True)
    df = df.sort_values("period_start_utc")

    series = pd.Series(
        df["value"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(df["period_start_utc"], name="timestamp"),
        name=name,
    )
    series = series.asfreq("h")
    return series


def intervention_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask, True for hours inside the known intervention window."""
    mask = (index >= INTERVENTION_START) & (index <= INTERVENTION_END)
    return pd.Series(mask, index=index, name="is_intervention")


def impute_intervention(series: pd.Series) -> pd.Series:
    """Replace intervention-window values with a level-corrected same-hour-last-week value.

    Method: for each hour ``t`` inside [INTERVENTION_START, INTERVENTION_END],
    the imputed value is ``series[t - 168h] * ratio``, where ``ratio`` is a
    single scalar level-correction factor shared across the whole window:

        ratio = mean(series[INTERVENTION_START - 168h : INTERVENTION_START - 1h])
              / mean(series[INTERVENTION_START - 336h : INTERVENTION_START - 169h])

    i.e. the ratio between the mean of the 168 hours immediately preceding the
    intervention and the mean of the same 168 hours one week earlier. This
    captures the organic week-over-week growth/level trend so the imputed
    values are consistent with the series' level right before the
    intervention, while still reusing the previous week's hourly shape
    (weekday/weekend, time of day) for the replaced hours.
    """
    recent_window = series.loc[
        INTERVENTION_START - pd.Timedelta(hours=168) : INTERVENTION_START - pd.Timedelta(hours=1)
    ]
    prior_window = series.loc[
        INTERVENTION_START - pd.Timedelta(hours=336) : INTERVENTION_START - pd.Timedelta(hours=169)
    ]
    ratio = recent_window.mean() / prior_window.mean()

    imputed = series.copy()
    mask = intervention_mask(series.index)
    for ts in series.index[mask]:
        imputed.loc[ts] = series.loc[ts - pd.Timedelta(hours=168)] * ratio
    return imputed


def load_clean(name: str, impute: bool = True) -> pd.Series:
    """Load, slice to steady state, optionally impute the intervention, and validate.

    Steps: ``load_raw`` -> slice from ``STEADY_STATE_START`` onward -> if
    ``impute`` is True, apply ``impute_intervention``. Raises ``ValueError``
    if the result does not have hourly frequency, contains NaNs or zeros, or
    is not monotonically increasing.
    """
    series = load_raw(name)
    series = series.loc[STEADY_STATE_START:]

    if impute:
        series = impute_intervention(series)

    if series.index.freq != pd.tseries.frequencies.to_offset("h"):
        raise ValueError(f"'{name}' clean series does not have hourly frequency.")
    if series.isna().any():
        raise ValueError(f"'{name}' clean series contains NaN values.")
    if (series == 0).any():
        raise ValueError(f"'{name}' clean series contains zero values.")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"'{name}' clean series index is not monotonically increasing.")

    return series


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build local-time calendar and intervention features for a UTC index.

    Columns: ``hour`` (0-23, local time), ``dayofweek`` (0=Monday), ``is_weekend``,
    ``is_holiday`` (Argentina, via the ``holidays`` package) and ``is_intervention``
    (the known intervention dummy, computed on the original UTC timestamps).
    """
    local_index = index.tz_convert(LOCAL_TZ)
    ar_holidays = holidays.Argentina(years=range(local_index.year.min(), local_index.year.max() + 1))

    features = pd.DataFrame(
        {
            "hour": local_index.hour,
            "dayofweek": local_index.dayofweek,
            "is_weekend": local_index.dayofweek >= 5,
            "is_holiday": [d.date() in ar_holidays for d in local_index],
            "is_intervention": intervention_mask(index).to_numpy(),
        },
        index=index,
    )
    return features
