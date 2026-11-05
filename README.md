# GeoFM pilot: AlphaEarth embeddings on Kenyan smallholder fields

A small test of the question behind the Manchester PhD project: do geospatial foundation model embeddings beat conventional Sentinel-2 features for mapping smallholder crops, once validation stops nearby fields from leaking between training and test sets?

It compares logistic regression and Random Forest on two feature sets, under random and spatially blocked cross-validation, and with fewer and fewer training labels.

## Status
- **Run on 5 November 2026**, after the preregistration was committed (see the git history). Results: [`RESULTS.md`](RESULTS.md); tables and figures in `results/`.
- Headline: under spatially blocked validation, frozen AlphaEarth embeddings did **not** outperform seasonal Sentinel-2 features (macro-F1 difference −0.048, 95% block-bootstrap −0.123 to +0.029).
- Anything from `demo` is invented data.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_pilot.py                          # about a minute; no account needed
python pilot.py demo                          # optional: see the outputs on invented data

earthengine authenticate                      # needs a Cloud project registered for Earth Engine
python pilot.py labels                        # writes data/fields.geojson
python pilot.py extract --project YOUR_ID     # writes data/features.csv

# Complete and commit PREREGISTRATION.md now, before looking at any scores
python pilot.py evaluate                      # writes results/
python pilot.py evaluate --block-km 2  --out results/block2
python pilot.py evaluate --block-km 10 --out results/block10
python pilot.py evaluate --pure-only   --out results/pure
```
Earth Engine needs a Google Cloud project registered for it; this pilot was run on a personal, non-commercial research project. Check which tier applies to you before running it.

## Outputs
| File | Contents |
|---|---|
| `comparison.csv` | AlphaEarth minus Sentinel-2 macro-F1 with a 95% block-bootstrap interval (the H1 test) |
| `cv_summary.csv`, `cv_by_repeat.csv` | Macro-F1 and balanced accuracy by model and validation design |
| `per_crop_f1.csv` | F1 for each crop |
| `learning_curve.csv` | Macro-F1 at 10, 25, 50 and 100% of training labels |
| `cv_comparison.png`, `learning_curve.png` | Figures |
| `run_info.json` | Field counts, blocks, missing data and software versions |

## Data
PlantVillage Kenya Ground Reference Crop Type Dataset, downloaded as one GeoJSON file from Source Cooperative.

- 319 rainfed field polygons in about 27 × 27 km of western Kenya, surveyed 14 May to 3 June 2019.
- 305 fields stay in the experiment (Maize 197, Cassava 59, Fallowland 23, Millet 16, Groundnut 10). Eight crops with 3 fields or fewer are dropped.
- The median field is 0.14 ha, about 14 Sentinel-2 pixels; 46 fields cover under 0.05 ha, about 5 pixels. 27% of the retained fields record a second crop.

## What this pilot can and cannot show
- **It is small.** There are few fields, one area and one year, and at 5 km the blocks number only about 13. Expect wide intervals. An inconclusive result is a legitimate outcome and should be reported as one.
- **It does not test transfer** across regions or years, which is the PhD's central concern. Blocks 5 km apart are still neighbours.
- **It does not test annual versus sub-seasonal embeddings.** AlphaEarth is annual, so the Sentinel-2 features have seasonal detail that the embeddings cannot show directly.
- **It is not deep learning.** The embeddings are frozen and the classifiers are classical. It shows you can work with foundation model outputs and evaluate them properly; it does not close the fine-tuning gap you name in your Statement.
- **Labels are imperfect.** Many fields are mixed, and crop is by farmer report or the collector's observation.
- **Maize dominates,** so accuracy would flatter any model. That is why the main metric is macro-F1, with a majority-class baseline for scale.
- **Rare crops are thin.** Groundnut has only 10 fields, so its F1 will be noisy.

## Possible next steps
- Add TESSERA through the `geotessera` library, keeping the same folds.
- Use the larger `african-crops-kenya-02` set, whose labels are rasters and so need different loading code.
- Hold out one lead farmer's fields at a time; the first number of `Field ID` identifies the lead farmer.

## Credits and licences
- **Code:** MIT licence (see `LICENSE`).
- **Labels** (not included in this repository; `pilot.py labels` downloads them): PlantVillage (2019). *PlantVillage Kenya Ground Reference Crop Type Dataset*. Radiant MLHub. https://doi.org/10.34911/rdnt.u41j87. CC BY-SA 4.0: credit it, and share any redistributed adaptation of the labels under the same licence.
- **Embeddings:** The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind. CC-BY 4.0. Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., et al. (2025). arXiv:2507.22291. Earth Engine ID: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- **Imagery:** Copernicus Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`) with Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`).
