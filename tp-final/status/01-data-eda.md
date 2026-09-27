# Phase 01 — Data preparation and EDA refresh

Status: done

## Objective

Clean, model-ready hourly series for ALB and store-service, plus a short EDA that justifies the modeling choices (not a repeat of TP1).

## Tasks

- [x] 01.1 Load both CSVs, parse `period_start_utc` as a UTC hourly index, keep `value`. (`tsa_final.data.load_raw`)
- [x] 01.2 Trim the leading all-zero block. Effective steady-state start is 2026-07-03 00:00 UTC (`STEADY_STATE_START`), not the first non-zero hour: the load balancer went live around 2026-06-26 12:00 UTC and traffic ramps up (ALB ~7M/day -> ~53M/day) until 2026-07-03; the ramp itself is dropped along with the zero block.
- [x] 01.3 Go-live zeros and the September intervention (re-scoped from "scattered interior outages"): all zeros are confined to before go-live (2026-06-26 12:00 UTC); after `STEADY_STATE_START` there are no zeros. The only material anomaly in the steady-state range is a known deployment (2026-09-17 18:00 -> 2026-09-22 13:00 UTC, 116 h inclusive) that changed client behaviour and was rolled back. Decision: impute that window (`impute_intervention`), keep everything else as observed.
- [x] 01.4 Last partial day (2026-09-26, 23 hours) is kept as-is; no cut needed for hourly modelling (confirmed via the sanity-check cell: last index value is 2026-09-26 22:00 UTC).
- [x] 01.5 Calendar features use `America/Argentina/Buenos_Aires` (`LOCAL_TZ`); UTC is kept for storage/indexing. Local time is needed because the daily minimum aligns with 04:00 local, not a UTC boundary.
- [x] 01.6 EDA notebook (`notebooks/01_data_eda.ipynb`): level plots, hourly profile, day-of-week x hour heatmap, MSTL (periods 24, 168) with seasonality-strength quantification, ACF/PACF to lag 336. Weekly seasonality confirmed negligible (see Evidence).
- [x] 01.7 Weekly mean-level plot over the ~12-week steady-state window (trend/growth check); no additional level shift found beyond the documented intervention and the post-rollback ~7% store-service drop.
- [x] 01.8 Transformation decision: `log1p` + first difference for stationarity analysis and future modelling; metrics reported in original scale (consistent with TP1 correction 2).
- [x] 01.9 Calendar/exogenous features implemented (`calendar_features`: hour, dayofweek, is_weekend, is_holiday via `holidays.Argentina`, is_intervention). Cross-correlation ALB vs store_service (log1p + diff) computed for lags -24..24; direction discussed in Evidence.

## Outputs

- `tp-final/tsa_final/data.py`: registry, constants, `load_raw`, `intervention_mask`, `impute_intervention`, `load_clean`, `calendar_features`.
- `tp-final/notebooks/01_data_eda.ipynb`: executed end to end, outputs kept.
- 10 figures in `tp-final/informe/figuras/` (prefix `01_`), listed in Evidence.

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| Steady-state start = 2026-07-03 00:00 UTC, dropping both the zero block and the go-live ramp (2026-06-26 12:00 -> 2026-07-03) | The ramp is a one-time onboarding transient (ALB grows ~7M/day to ~53M/day), not representative of the process to be forecast; keeping it would bias trend/seasonality estimates | Cutting only at the first non-zero hour (2026-06-22 14:00/16:00, per the earlier index.md profiling) — rejected because it still includes the ramp |
| Impute the intervention window (2026-09-17 18:00 -> 2026-09-22 13:00 UTC) with `t-168h` scaled by a pre-window level ratio | The window is a known, dated, external cause (deployment + rollback), not organic signal; leaving it as observed would teach models a one-off client-behaviour anomaly | Dropping the window (would break hourly continuity for models needing full history) or leaving it unmodified (would leak the anomaly into training) |
| `impute_intervention` uses a single scalar ratio (mean of the 168 h before intervention / mean of the same 168 h one week earlier), not per-hour ratios | Keeps the method simple and avoids overfitting the correction to noise in a single day; the trend correction matters more than hour-level shape, which is already taken from `t-168h` | Per-hour or per-day ratios — more moving parts for a 116 h window with no evidence of needing that granularity |
| Post-rollback ~7% store-service level drop treated as real, not corrected | No evidence it is caused by the same intervention; consistent with a target-count/autoscaling change (store-service is `RequestCountPerTarget`) | Extending the imputation past the intervention end — rejected, would erase real information |
| Local timezone (`America/Argentina/Buenos_Aires`) for calendar features; UTC kept for indexing/storage | Daily minimum aligns with 04:00 local time, not a UTC hour boundary; holidays are defined in local dates | Keeping everything in UTC — rejected, would misalign the daily profile and holiday calendar |
| Weekly seasonality (period 168) treated as secondary/negligible for future modelling | Weekday means differ <2-3%, MSTL seasonal-168 strength is far below seasonal-24 strength (see Evidence) | Modelling m=168 seasonal naive/SARIMA as the primary seasonal baseline — rejected given the weak empirical signal |
| `log1p` + first difference for stationarity/ACF analysis, metrics in original scale | Matches TP1's corrected convention and stabilizes variance/mean for the stationarity tests | Raw-level-only analysis — rejected, ADF/KPSS on raw levels alone would not distinguish trend from variance non-stationarity as cleanly |

## Evidence

- Effective clean range: 2026-07-03 00:00 UTC -> 2026-09-26 22:00 UTC, 2063 hourly observations per series, no NaN, no zeros, monotonic, hourly freq (verified by the notebook's sanity-check cell and the standalone verification command).
- Intervention hours imputed: 116 (matches the documented window exactly).
- Weekday-mean spread: ALB 1.80% of the mean, store_service 2.88% of the mean -> weekly seasonality negligible after go-live.
- MSTL seasonality strength (Wang/Smith/Hyndman definition, variance-based):
  - ALB: seasonal-24 = 0.771, seasonal-168 = 0.206, trend = 0.880.
  - store_service: seasonal-24 = 0.846, seasonal-168 = 0.311, trend = 0.876.
  - Daily cycle is a real, moderate-to-strong component; the weekly component is much weaker in both series, confirming task 01.6/01.9's expectation.
- Stationarity (ADF / KPSS, on `log1p` level and first difference):
  - ALB level: ADF p = 0.810 (fail to reject unit root), KPSS p = 0.010 (reject stationarity) -> non-stationary in level, as expected.
  - ALB first difference: ADF p ≈ 0.000, KPSS p = 0.100 (do not reject stationarity) -> stationary after one difference.
  - store_service level: ADF p = 0.174, KPSS p = 0.010 -> non-stationary in level.
  - store_service first difference: ADF p ≈ 0.000, KPSS p = 0.100 -> stationary after one difference.
  - Both tests agree: `log1p` + first difference is enough to reach stationarity for both series.
- Holidays in range (Argentina): 2026-07-09, 2026-07-10, 2026-08-17. Effect vs. same weekday one week earlier is mixed, not a uniform reduction: ALB ratios 102.5%/106.0%/104.3% (slightly above the prior week), store_service 91.8%/98.0%/112.5%. No consistent holiday dampening was found with only 3 holiday days in range; treated as a feature (`is_holiday`) for later models rather than a hard-coded correction.
- Cross-correlation ALB vs store_service (both `log1p` + first difference, lags -24..24 h): the correlation peaks at lag 0 with r = 0.852, and it is essentially the global maximum (no materially higher peak at nonzero lag). This indicates the two series move together contemporaneously rather than one leading the other by whole hours; ALB is a plausible contemporaneous exogenous signal for store_service, but not a lead indicator at this resolution — different from the TP1 POS-API -> ALB Granger-causality direction (which used a different pair of series).
- Figures saved to `tp-final/informe/figuras/`: `01_raw_ramp.png`, `01_intervention_zoom.png`, `01_weekly_level.png`, `01_hourly_profile_heatmap_alb.png`, `01_hourly_profile_heatmap_store_service.png`, `01_mstl_alb.png`, `01_mstl_store_service.png`, `01_acf_pacf_alb.png`, `01_acf_pacf_store_service.png`, `01_ccf_alb_store_service.png`.
