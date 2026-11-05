# Preregistration

Complete and commit this file **before** running `python pilot.py evaluate` on real data, so the git history shows the tests were fixed first. Edit anything below that you disagree with, but do it now, not after seeing results.

- **Author:** Alex Rop
- **Date fixed:** 2026-11-05
- **Data seen so far:** field labels only (crop counts, locations). No feature values or model scores.

## Question
Under spatially held-out validation, do frozen AlphaEarth annual embeddings classify crop type on western Kenyan smallholder fields better than conventional seasonal Sentinel-2 features?

## Hypotheses
- **H1 (primary).** Logistic regression on AlphaEarth has a higher macro-F1 than Random Forest on Sentinel-2. Supported only if the 95% block-bootstrap interval in `comparison.csv` is entirely above 0.
- **H2.** For every non-baseline model, mean macro-F1 is higher under random folds than under spatial blocks. Any gap is reported as the estimated leakage.
- **H3.** The embedding advantage (best AlphaEarth minus best Sentinel-2, spatial blocks) is larger with 10% of training labels than with 100%.

## Fixed settings
- All 2019 fields whose main crop (`Crop1`) has at least 10 fields: Maize, Cassava, Fallowland, Millet, Groundnut.
- Field-mean features; 5-fold validation; 5 km blocks; 10 repeats; model settings as written in `pilot.py`, with no tuning.
- Primary metric: macro-F1. Also reported: balanced accuracy and F1 per crop.
- Fields with no AlphaEarth value are dropped and counted. Missing Sentinel-2 values are filled with the training-fold median.

## Sensitivity checks (reported as exploratory, whatever they show)
- `--block-km 2` and `--block-km 10`
- `--pure-only` (fields with no second recorded crop)

## Reporting
All hypotheses are reported, including those not supported. Any change made after seeing results is listed here with its date and reason.

## Changes after results
None yet.
