# Phase 01 — Data preparation and EDA refresh

Status: pending

## Objective

Clean, model-ready hourly series for ALB and store-service, plus a short EDA that justifies the modeling choices (not a repeat of TP1).

## Tasks

- [ ] 01.1 Load both CSVs, parse `period_start_utc` as a UTC hourly index, keep `value`.
- [ ] 01.2 Trim the leading all-zero block (effective start 2026-06-22 14:00 / 16:00).
- [ ] 01.3 Inspect the 14 (ALB) and 61 (store-service) interior zeros: locate them, check if they cluster (outages, deploys, metric gaps). Decide: treat as missing and impute (seasonal interpolation using the same hour of the previous week/day) vs keep as real events. Record the decision.
- [ ] 01.4 Check the last partial day (series ends 2026-09-26 22:00) and decide whether to cut to the last full day.
- [ ] 01.5 Decide timezone for calendar features (UTC vs America/Argentina/Buenos_Aires). Daily/weekly patterns follow local business hours.
- [ ] 01.6 EDA: level plot, daily profile, day-of-week × hour heatmap, STL/MSTL with periods 24 and 168, ACF/PACF up to lag 336. Confirm weekly seasonality (TP1 pending observation).
- [ ] 01.7 Check for level shifts / trend over the ~13 weeks (growth of the product, autoscaling effect on per-target values).
- [ ] 01.8 Decide target transformation (`log1p` as in TP1 vs raw); models that need it get it, metrics always in original scale (TP1 correction 2).
- [ ] 01.9 Exogenous/calendar features: hour, day of week, weekend flag, Argentine holidays in range (e.g. 2026-07-09, 2026-08-17). Evaluate using ALB as exogenous for store-service (TP1 found Granger causality POS API → ALB, not the reverse; re-check direction here).

## Outputs

- Clean series saved or reproducible from a single loader function.
- 4–6 figures candidates for the report (Introduction / data description).

## Evidence

_(fill in: zero-treatment decision, effective ranges, seasonality findings)_
