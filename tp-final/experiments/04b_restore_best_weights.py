"""Recompute phase 04 validation rows after restoring best-epoch weights.

The first phase 04 run forecast the validation window with the weights of the
last early-stopping epoch (PATIENCE epochs past the best one), because darts'
EarlyStopping does not restore the best weights. `dl_models` now restores
them. The test refit already trained exactly `best_epoch` epochs, so it is
unaffected — unless the best epoch itself changes (the fix also stops counting
Lightning's pre-training sanity-check validation as epoch 0). This script:

1. re-runs only the early-stopping (validation) fit per series x architecture x seed,
2. re-runs the test refit only where the best epoch differs from the first run,
3. rewrites the affected rows of `results/04_dl_seeds.csv` and rebuilds
   `results/04_dl.csv` (seed-averaged metrics, summed fit time, as in the notebook),
4. writes `results/04_dl_best_epochs.csv` with old vs new best epochs.

Run from the repo root: uv run python tp-final/experiments/04b_restore_best_weights.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

TP_FINAL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TP_FINAL))

from tsa_final import data, dl_models as dl  # noqa: E402
from tsa_final import evaluation as ev  # noqa: E402

RESULTS_DIR = TP_FINAL / "results"
METRIC_COLUMNS = ["MAE", "RMSE", "MAPE_%", "MASE"]

# Best epochs of the first run (notebook 04, cell "mejor época"), as passed to the test refit.
FIRST_RUN_BEST_EPOCHS = {
    ("alb", "lstm"): {42: 1, 43: 1},
    ("alb", "nbeats"): {42: 83, 43: 97},
    ("alb", "nhits"): {42: 21, 43: 42},
    ("alb", "tcn"): {42: 10, 43: 23},
    ("alb", "tft"): {42: 21, 43: 16},
    ("alb", "tide"): {42: 21, 43: 27},
    ("store_service", "lstm"): {42: 18, 43: 17},
    ("store_service", "nbeats"): {42: 34, 43: 45},
    ("store_service", "nhits"): {42: 6, 43: 16},
    ("store_service", "tcn"): {42: 51, 43: 9},
    ("store_service", "tft"): {42: 23, 43: 19},
    ("store_service", "tide"): {42: 11, 43: 16},
}


def scored_row(model: str, series_name: str, split_name: str, y_true, y_pred, train, fit_time_s, arch, seed) -> dict:
    return {
        "model": model, "family": "DL", "series": series_name, "split": split_name,
        **ev.metrics(y_true, y_pred, train), "fit_time_s": fit_time_s, "architecture": arch, "seed": seed,
    }


def main() -> None:
    seeds_df = pd.read_csv(RESULTS_DIR / "04_dl_seeds.csv")
    new_rows, epochs = [], []

    for series_name in ("alb", "store_service"):
        series = data.load_clean(series_name)
        train, val, test = ev.split(series)
        for arch in dl.ARCHITECTURES:
            for seed in dl.SEEDS:
                model_name = f"{arch}_seed{seed}"
                fit = dl.make_fit(arch, seed)

                t0 = time.perf_counter()
                predictor = fit(train)
                fit_s = time.perf_counter() - t0
                new_rows.append(scored_row(model_name, series_name, "val", val.to_numpy(), predictor(len(val)),
                                           train, fit_s, arch, seed))

                old_epoch = FIRST_RUN_BEST_EPOCHS[(series_name, arch)][seed]
                changed = fit.best_epoch != old_epoch
                if changed:
                    history = ev.train_and_val(series)
                    t0 = time.perf_counter()
                    test_predictor = fit(history)
                    test_fit_s = time.perf_counter() - t0
                    new_rows.append(scored_row(model_name, series_name, "test", test.to_numpy(),
                                               test_predictor(len(test)), train, test_fit_s, arch, seed))

                epochs.append({"series": series_name, "architecture": arch, "seed": seed,
                               "best_epoch_first_run": old_epoch, "best_epoch": fit.best_epoch,
                               "test_refit_rerun": changed})
                print(f"{series_name:14} {model_name:14} best_epoch {old_epoch:3d} -> {fit.best_epoch:3d} "
                      f"val_MASE={new_rows[-1 - changed]['MASE']:.3f} fit={fit_s:6.1f}s"
                      + ("  (test refit re-run)" if changed else ""), flush=True)

    new_df = pd.DataFrame(new_rows)
    key = ["model", "series", "split"]
    kept = seeds_df.merge(new_df[key], on=key, how="left", indicator=True)
    kept = kept[kept["_merge"] == "left_only"].drop(columns="_merge")
    seeds_out = pd.concat([kept, new_df], ignore_index=True).sort_values(["series", "architecture", "seed", "split"],
                                                                        ascending=[True, True, True, False])
    seeds_out.to_csv(RESULTS_DIR / "04_dl_seeds.csv", index=False)

    main_out = (
        seeds_out.groupby(["architecture", "series", "split"], sort=False)
        .agg({**{c: "mean" for c in METRIC_COLUMNS}, "fit_time_s": "sum"})
        .reset_index()
        .rename(columns={"architecture": "model"})
        .assign(family="DL")
    )[["model", "family", "series", "split", *METRIC_COLUMNS, "fit_time_s"]]
    main_out.to_csv(RESULTS_DIR / "04_dl.csv", index=False)

    pd.DataFrame(epochs).to_csv(RESULTS_DIR / "04_dl_best_epochs.csv", index=False)
    print("\nValidation MASE per architecture (seed mean):")
    print(main_out[main_out["split"] == "val"].pivot(index="model", columns="series", values="MASE").round(3))


if __name__ == "__main__":
    main()
