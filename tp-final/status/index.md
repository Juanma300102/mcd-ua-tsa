# TP Final — Plan Index

Goal: model the two BodegaAI traffic series (ALB total and store-service target group) with ML, DL, hybrid and AutoML techniques, compare them against the TP1 classical baselines, forecast with the selected model per series, and deliver one notebook plus a LaTeX report (PDF, max 30 carillas excluding cover and appendices).

Sources: `tp-final/consignas.md`, `tp-final/CONTEXT.md`, `tp-1/notes.md`, `tp-1/notas_informe.md`, example report `Informe_Series_Temporales_TP2 (1).pdf` (repo root), class notebooks in `tp-final/example/`.

## Phases

| # | Phase | File | Status |
|---|---|---|---|
| 00 | Environment and tooling | [00-environment.md](00-environment.md) | done |
| 01 | Data preparation and EDA refresh | [01-data-eda.md](01-data-eda.md) | done |
| 02 | Evaluation protocol and baselines | [02-evaluation-baselines.md](02-evaluation-baselines.md) | done |
| 03 | Machine Learning models | [03-ml-models.md](03-ml-models.md) | done |
| 04 | Deep Learning models | [04-dl-models.md](04-dl-models.md) | done |
| 05 | Prophet family, AutoML, hybrids and foundation models | [05-hybrid-automl-foundation.md](05-hybrid-automl-foundation.md) | done |
| 06 | Comparison, selection and final forecast | [06-selection-forecast.md](06-selection-forecast.md) | done |
| 07 | Deliverable notebook consolidation | [07-notebook.md](07-notebook.md) | pending |
| 08 | LaTeX report | [08-report.md](08-report.md) | pending |

Status values: `pending`, `in-progress`, `done`, `blocked`.

## Data facts (profiled 2026-09-27, confirmed in phase 01)

| Series | File | Raw range | Effective clean range (steady state) | Effective hours | Intervention hours imputed |
|---|---|---|---|---:|---:|
| ALB total `RequestCount` | `data/alb-request-count-hourly-since-2026-05-01.csv` | 2026-05-01 00:00 → 2026-09-26 22:00 UTC | 2026-07-03 00:00 → 2026-09-26 22:00 UTC | 2,063 | 116 |
| store-service `RequestCountPerTarget` | `data/store-service-request-count-per-target-hourly-since-2026-05-01.csv` | same | 2026-07-03 00:00 → 2026-09-26 22:00 UTC | 2,063 | 116 |

- No timestamp gaps. The load balancer went live around 2026-06-26 12:00 UTC; before that both series are all zeros. Between go-live and 2026-07-03 there is a traffic ramp (ALB ~7M/day → ~53M/day); both the zero block and the ramp are dropped, so the clean/modeling range starts at `STEADY_STATE_START` = 2026-07-03 00:00 UTC, not at the first non-zero hour.
- After `STEADY_STATE_START` there are no zeros. The only material anomaly in that range is a known deployment/rollback intervention (2026-09-17 18:00 → 2026-09-22 13:00 UTC, 116 h inclusive) that changed client behaviour (ALB drops to ~72% of the prior week, store-service to ~80-90%); it is imputed in `load_clean` via `impute_intervention`, not treated as missing data or dropped.
- After the rollback, store-service stays ~7% below the prior week's level (likely a target-count/autoscaling level shift); treated as real, not corrected.
- store-service is `RequestCountPerTarget` (normalized by the number of healthy targets, fractional values). Autoscaling can shift its level without a traffic change; documented in `tsa_final/data.py` and `status/01-data-eda.md`.
- Weekly seasonality (period 168) is negligible after go-live (weekday means differ <3%, MSTL seasonal-168 strength well below seasonal-24 strength); daily seasonality (period 24) is the dominant seasonal component. See `status/01-data-eda.md` for numeric evidence.

## Open decisions

- TimeGPT (Nixtla) requires sending the series to an external API with a key; only with explicit approval.

## Cross-cutting decisions

- Time budget (deadline pressure, added 2026-09-27): every phase from 03 onward must complete its full run (all fits, any hyperparameter search, notebook execution) in a few hours, not days. Any sub-task with an open-ended cost (hyperparameter search trial count, per-step model multiplication, AutoML default budgets in phases 05-06, DL epoch/seed sweeps in phase 04) needs an explicit, stated time-box before it runs; if a time-box is hit, cut scope and record the gap as a reported limitation instead of letting it run unbounded. See `03-ml-models.md`'s "Time budget" section for the first concrete application (Optuna trial caps, `ForecasterDirect`'s per-step model multiplication, `StackingRegressor`'s default internal CV).

- Third series: owned by the teammate, who works in this same repo. It goes into the same notebook and the same LaTeX report. The teammate follows the shared protocol (phase 02) and helpers so results are comparable across the three series.

- TP1 series 2 was the POS API target group; TP Final uses store-service instead. There is no TP1 number for store-service, so the TP1 model specs are refit on the new data as baselines for both series (phase 02). This makes the "compare against TP1" objective fair for both series.
- Weekly seasonality (period 168) was a documented TP1 limitation; ~13 weeks of history now allow modeling it.
