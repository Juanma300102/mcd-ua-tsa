# Phase 08 — LaTeX report

Status: in-progress (everything is complete except the third-series content, which is explicitly out of scope for this phase and marked with visible "PENDIENTE" placeholders — see Decisions)

## Objective

Academic report in Spanish, APA 7 style, max 30 carillas (15 hojas) excluding cover and appendices, reusing the TP1 LaTeX setup (`tp-1/informe/`: `apa7` class, XeLaTeX, one file per section).

## Required structure (consigna)

i. Carátula · ii. Resumen Ejecutivo · iii. Índice · iv. Introducción · v. Marco Teórico (models) · vi. Análisis de Resultados · vii. Conclusiones · viii. Referencias · ix. Apéndices.

## Tasks

- [x] 08.1 Copy the TP1 skeleton to `tp-final/informe/` and adapt the cover (TP N° 2 title, members, date).
- [x] 08.2 Introducción: problem, motivation for the series, context (TP1 → full history), description of each series (third series left as a visible placeholder).
- [x] 08.3 Marco Teórico: one short subsection per model family actually used (naive/classical, ML boosters + stacking, DL (LSTM/N-BEATS/N-HiTS/TCN/TiDE/TFT), Prophet/NeuralProphet + MSTL-LightGBM hybrid, AutoML (AutoGluon/AutoTS) + Chronos) plus metrics and validation scheme. Every model family cites its primary paper.
- [x] 08.4 Análisis de Resultados: data prep (steady state, intervention imputation, leakage-safe tuning window), protocol (validation/test windows), per-family compact comparison tables for both series, validation-vs-test Spearman consistency, sensitivity table, cost table, comparison with TP1 baseline, final 48h forecast (third series left as a visible placeholder).
- [x] 08.5 Conclusiones per series + overall (did complex models beat TP1?), limitations (single validation window, history length, contamination, possible store-service tail anomaly, weekly seasonality), future work.
- [x] 08.6 Appendices: full 32-model tables per series, 12 supplementary EDA/phase figures, 2 full code listings (intervention imputation, contamination flags).
- [x] 08.7 Style pass: academic register, no gerunds, no first person (impersonal "se"), APA citations throughout.
- [x] 08.8 Compile, check page budget, cross-references, bibliography with no orphans.

## Reference

Example structure: `Informe_Series_Temporales_TP2 (1).pdf` (repo root).

## Decisions

| Decision | Reason | Discarded alternative |
|---|---|---|
| Cover uses only "Rodrigo Del Rosso" as docente | `tp-final/consignas.md` lists a single docente for TP2, unlike TP1's three | Reusing TP1's three docentes (Del Rosso, Calcagno, Drago) — not evidenced for TP2 |
| Cover lists only Carlos Aular and Juan Martín Pedrozo as integrantes | Only these two names are evidenced in the repo; the third group member (owner of series 3) is never named in any status doc. Name corrected mid-task from "Juan Manuel Pedrozo" (as it appeared in TP1's cover) to "Juan Martín Pedrozo" per explicit user correction | Inventing a third name, or omitting the "Integrantes" block |
| Delivery date set to "septiembre de 2026" | No delivery date is recorded anywhere in the repo; instructions explicitly authorize this fallback | Leaving the field blank (breaks the cover layout) |
| Third series handled as visible amber "PENDIENTE" boxes (`\fcolorbox`) in Introducción, Análisis and Conclusiones | Explicit requirement; content must never be invented | `tcolorbox` (extra package dependency, not needed) |
| "Preparación de datos y protocolo de evaluación" placed as a subsection inside "Análisis de Resultados" | Consigna's mandatory structure has no separate top-level section for it; the example PDF's "Preparación de datos" (its section 3) maps naturally onto "Análisis de Resultados" | A separate 6th top-level section not in the consigna's mandatory list |
| Section-level `\ref{sec:...}` cross-references removed, replaced by prose ("más adelante, en la sección de...") | `apa7`'s `\section`/`\subsection` do not use numbered LaTeX counters (APA style has no heading numbers), so `\ref` on a section label silently prints an empty string with no warning — verified by inspecting the compiled PDF | Forcing a numbered-heading style into `apa7` (would diverge from the verified TP1 template) |
| Reported that the TP1 SARIMA refit is actually the single best test-set model for ALB (test MASE 1.526, rank 1/32), and NeuralProphet the best for store-service (test MASE 1.071, rank 1/32) — neither is the selected model | Verified directly against `results/06_all_models.csv`'s `test_rank` column and an independent Spearman recomputation (`scipy.stats.spearmanr` on `val_rank`/`test_rank`), which corrected an imprecise claim from initial research (that N-BEATS was "the best test model of the project on both series" — false: N-BEATS ranks 4th on both) | Repeating the imprecise "N-BEATS wins both series on test" narrative |
| No DL figure regenerated (`04_*.png` excluded per instructions) | `06_best_per_family.png` and the DL correction table already visualize/tabulate the corrected numbers without duplicating a redundant chart (TP1's own editorial rule: a figure earns its place) | Writing `tp-final/informe/scripts/regen_04_figures.py` to produce a `04c_*.png` bar chart — judged redundant |
| Store-service tail anomaly reported only for store-service, not "both series" as originally hypothesized | Direct inspection of the raw CSV tail: store-service's last 3 hours (20:00–22:00 UTC, 2026-09-26) deviate from the same hours on the 4 preceding days; ALB's last 3 hours do not show a comparable deviation | Asserting the anomaly for both series without checking the raw data |
| AI-usage declaration cites Claude Code (Anthropic, Sonnet) only | This is the tool actually used for phase 08 (this session); no verifiable evidence of which tool(s) were used in phases 00–07 | Copying TP1's declaration verbatim (which names OpenAI Codex and Claude Opus, unverified for this project) |

## Evidence

- Compile sequence: `xelatex → biber → xelatex → xelatex`, all four steps exit 0.
- `rg -c "^! " informe.log` → no match (0 LaTeX errors).
- `rg -i "undefined" informe.log` → no match (no undefined references or citations).
- `rg -i "overfull \\hbox" informe.log` → no match (no overfull boxes after the cost-table fix; one fixed during iteration).
- `Output written on informe.pdf (37 pages).`
- Body page count (Resumen Ejecutivo + Introducción…Conclusiones, per `informe.toc`: Resumen page 1, Introducción page 6 → Declaración de uso de IA page 26): **21 pages**, within the 30-page limit (índice/lof/lot pages 2–5 excluded, as are Declaración/Referencias/Apéndices).
- Figures used: 10 in the body (`01_raw_ramp`, `01_intervention_zoom`, `01_weekly_level`, `02_split_train_val_test`, `03_shap_alb`, `05_val_mae_ranking`, `06_best_per_family`, `06_val_vs_test_scatter`, `06_test_overlay_forecast`, `06_final_forecast`) + 12 in Apéndice A (MSTL ×2, hourly heatmaps ×2, ACF/PACF ×2, CCF, SHAP store-service, `03_val_mae_ranking`, `02_val_mape_ranking`, `02_test_forecast_winners`, `05_test_forecast_best_model`) = 22 of 24 available PNGs. `04_*.png` (5 files) intentionally excluded (predate the phase 04 correction).
- Tables: 2 full 32-model appendix tables (one per series, from `results/06_all_models.csv`, cross-checked cell by cell against the CSV) + 8 body tables (protocol windows, DL correction, best-epoch, 2× compact per-family comparison, sensitivity, cost, best-epoch).
- Bibliography: 31 entries in `bibliografia.bib`, 31 unique citekeys found by biber, all 31 cited in the text (verified with `rg`) — every entry cited, every citation resolves.
- `git status --short` (informe-related): all files untracked/new (`tp-final/informe/*`), no earlier-phase file touched.

## Known limitation of this phase

The third series (owned by another group member) is not in the repository. Its slots — Introducción (series description), Análisis (results subsection) and Conclusiones (paragraph) — are visible amber "PENDIENTE" boxes. No numbers were invented for it.
