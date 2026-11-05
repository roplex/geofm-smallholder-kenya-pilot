# Results (run 2026-11-05, after the preregistration commit 88ee008)

Features extracted once on Earth Engine project `knimeproject-447409` (personal, non-commercial): 305 fields, no missing values in either feature set. Each evaluation was run once; nothing was tuned after seeing scores. Full tables in `results/` (not in git).

## Main run (5 km blocks, 13 blocks, 10 repeats; macro-F1, mean)
| Model | Random folds | Spatial blocks |
|---|---|---|
| LogReg on Sentinel-2 | 0.379 | 0.345 |
| RF on Sentinel-2 (primary baseline) | 0.287 | 0.298 |
| LogReg on AlphaEarth | 0.294 | 0.250 |
| RF on AlphaEarth | 0.177 | 0.163 |
| Majority class | 0.157 | 0.157 |

## Hypotheses
- **H1 not supported.** LogReg on AlphaEarth minus RF on Sentinel-2, spatial blocks: **-0.048** (95% block-bootstrap -0.123 to +0.029). The interval includes 0 and the estimate favours Sentinel-2. RF on AlphaEarth: -0.135 (-0.192 to -0.088).
- **H2 partly supported (3 of 4 models).** Random minus spatial: LogReg AlphaEarth +0.044, LogReg Sentinel-2 +0.034, RF AlphaEarth +0.014, RF Sentinel-2 -0.011. Random folds overstated performance a little for three models, not for the RF on Sentinel-2.
- **H3: the embeddings never lead.** Best AlphaEarth minus best Sentinel-2 (spatial blocks): -0.071 at 10% of labels, -0.074 at 25%, -0.081 at 50%, -0.103 at 100%. The gap narrows with fewer labels (the preregistered direction), but there is no embedding advantage at any label volume.

## Sensitivity checks (exploratory)
H1 difference: 2 km blocks -0.043 (-0.108 to +0.030); 10 km blocks -0.047 (-0.139 to +0.015); fields with no second crop -0.037 (-0.098 to +0.025). Same conclusion in each.

## Per crop (spatial blocks, F1)
LogReg on Sentinel-2 is best for Cassava (0.59) and Fallowland (0.28); RF models mostly predict Maize. Groundnut (10 fields) and Millet (16) are near zero for every model.

## Reading
On one small area, one year and annual embeddings in a two-season system, frozen AlphaEarth embeddings did not beat seasonal Sentinel-2 features for smallholder crop type. All models are weak (best macro-F1 0.35 against a 0.16 majority baseline): small fields (median 0.14 ha), mixed cropping and thin rare classes limit every method. This is a question for larger, multi-year, sub-seasonal tests, not an answer.

## Changes after results
None to the preregistered settings. (The sensitivity runs were first launched with the flags passed as one string by the shell; they were rerun correctly. No scores were seen or used from the failed launch.)
