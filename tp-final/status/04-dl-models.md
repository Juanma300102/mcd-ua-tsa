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

## Risk

With ~96 days, large DL models may overfit or underperform the boosters. That is a valid finding for the report, not a failure.

## Class references

`PRACTICA_AST_Clase_6_1_Intro_LSTM`, `_6_2A_LSTM`, `_6_2B_LSTM`, `_6_7A_NBEATS`, `_7_5_DARTS`, `AST_Clase_8_ejercicio_3` (TiDE, TCN, TFT).

## Evidence

_(fill in)_
