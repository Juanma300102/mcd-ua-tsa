"""tsa_final: shared data-loading and feature helpers for the TP Final notebook.

Kept intentionally small: a registry of raw series, a handful of dated
constants agreed upon during the phase 01 EDA, and the loading/imputation
functions the notebook (and later phases) build on.
"""

from .data import (
    INTERVENTION_END,
    INTERVENTION_START,
    LOCAL_TZ,
    SERIES_REGISTRY,
    STEADY_STATE_START,
    TEST_END,
    TEST_START,
    calendar_features,
    impute_intervention,
    intervention_mask,
    load_clean,
    load_raw,
)

__all__ = [
    "INTERVENTION_END",
    "INTERVENTION_START",
    "LOCAL_TZ",
    "SERIES_REGISTRY",
    "STEADY_STATE_START",
    "TEST_END",
    "TEST_START",
    "calendar_features",
    "impute_intervention",
    "intervention_mask",
    "load_clean",
    "load_raw",
]
