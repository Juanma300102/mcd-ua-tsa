# TP Final — Plan Index

Goal: model the two BodegaAI traffic series (ALB total and store-service target group) with ML, DL, hybrid and AutoML techniques, compare them against the TP1 classical baselines, forecast with the selected model per series, and deliver one notebook plus a LaTeX report (PDF, max 30 carillas excluding cover and appendices).

Sources: `tp-final/consignas.md`, `tp-final/CONTEXT.md`, `tp-1/notes.md`, `tp-1/notas_informe.md`, example report `Informe_Series_Temporales_TP2 (1).pdf` (repo root), class notebooks in `tp-final/example/`.

## Phases

| # | Phase | File | Status |
|---|---|---|---|
| 00 | Environment and tooling | [00-environment.md](00-environment.md) | done |
| 01 | Data preparation and EDA refresh | [01-data-eda.md](01-data-eda.md) | pending |
| 02 | Evaluation protocol and baselines | [02-evaluation-baselines.md](02-evaluation-baselines.md) | pending |
| 03 | Machine Learning models | [03-ml-models.md](03-ml-models.md) | pending |
| 04 | Deep Learning models | [04-dl-models.md](04-dl-models.md) | pending |
| 05 | Prophet family, AutoML, hybrids and foundation models | [05-hybrid-automl-foundation.md](05-hybrid-automl-foundation.md) | pending |
| 06 | Comparison, selection and final forecast | [06-selection-forecast.md](06-selection-forecast.md) | pending |
| 07 | Deliverable notebook consolidation | [07-notebook.md](07-notebook.md) | pending |
| 08 | LaTeX report | [08-report.md](08-report.md) | pending |

Status values: `pending`, `in-progress`, `done`, `blocked`.

## Data facts (profiled 2026-09-27)

| Series | File | Raw range | Effective start (first non-zero) | Effective hours | Interior zeros |
|---|---|---|---|---:|---:|
| ALB total `RequestCount` | `data/alb-request-count-hourly-since-2026-05-01.csv` | 2026-05-01 00:00 → 2026-09-26 22:00 UTC | 2026-06-22 14:00 | ~2,313 (~96 days, ~13.7 weeks) | 14 |
| store-service `RequestCountPerTarget` | `data/store-service-request-count-per-target-hourly-since-2026-05-01.csv` | same | 2026-06-22 16:00 | ~2,311 | 61 |

- No timestamp gaps; the 2026-05-01 → 2026-06-22 block is all zeros (the load balancer did not exist yet) and must be dropped, not modeled.
- Interior zeros are likely outages or metric gaps; treatment decided in phase 01.
- store-service is `RequestCountPerTarget` (normalized by the number of healthy targets, fractional values). Autoscaling can shift its level without a traffic change; document it.

## Open decisions

- TimeGPT (Nixtla) requires sending the series to an external API with a key; only with explicit approval.

## Cross-cutting decisions

- Third series: owned by the teammate, who works in this same repo. It goes into the same notebook and the same LaTeX report. The teammate follows the shared protocol (phase 02) and helpers so results are comparable across the three series.

- TP1 series 2 was the POS API target group; TP Final uses store-service instead. There is no TP1 number for store-service, so the TP1 model specs are refit on the new data as baselines for both series (phase 02). This makes the "compare against TP1" objective fair for both series.
- Weekly seasonality (period 168) was a documented TP1 limitation; ~13 weeks of history now allow modeling it.
