# Phase 04 — Deep Learning models

Status: pending

## Objective

Neural forecasters on the same protocol, sized for ~2,300 hourly observations and CPU training.

## Tasks

- [ ] 04.1 Scaling pipeline (fit scaler on train only) and input window (e.g. 168 or 336 h).
- [ ] 04.2 LSTM (darts `RNNModel` / `BlockRNNModel`, or keras as in class 6).
- [ ] 04.3 N-BEATS and N-HiTS (neuralforecast or darts).
- [ ] 04.4 TCN, TiDE, TFT (darts), with calendar covariates.
- [ ] 04.5 Optional: TSMixer, PatchTST if time allows.
- [ ] 04.6 Early stopping on a validation window; fix seeds; average over 2–3 seeds if variance is high.
- [ ] 04.7 Log results and training time into the shared table.

## Time budget

Decided 2026-09-27, together with the phase 02 single-split protocol: DL is the largest cost driver in the whole TP (6 architectures x 2-3 seeds x 2 fits under the phase 02 protocol = up to ~36 fits per series), so it needs explicit caps, not just "early stopping on a validation window":

- Max epochs 100, early-stopping patience ~10-15 epochs on the validation-window loss.
- 2 seeds, not 3, by default; only run a 3rd seed for a given architecture if the 2-seed spread on validation MAE is large enough to matter for ranking.
- The 6 required architectures (04.2-04.4) run first; TSMixer/PatchTST (04.5) only if the required 6 finished within budget.
- If a single architecture x seed fit exceeds ~15 min wall-clock, treat that as a signal to cut epochs/patience for that model rather than letting it run — do not silently accept multi-hour fits.

## Risk

With ~96 days, large DL models may overfit or underperform the boosters. That is a valid finding for the report, not a failure.

## Class references

`PRACTICA_AST_Clase_6_1_Intro_LSTM`, `_6_2A_LSTM`, `_6_2B_LSTM`, `_6_7A_NBEATS`, `_7_5_DARTS`, `AST_Clase_8_ejercicio_3` (TiDE, TCN, TFT).

## Evidence

_(fill in)_
