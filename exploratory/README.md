# Exploratory: a small neural network under the same spatial blocks

**Not preregistered.** This was added after the preregistered run, so it cannot support or overturn the pilot's hypotheses. It uses the same 305 fields, features, 5 km spatial blocks and 10 repeats, so its scores compare directly with the pilot's.

A multilayer perceptron (two hidden layers of 64, dropout 0.3, weight decay, class weights, early stopping on validation blocks taken from each training fold) is trained in PyTorch on AlphaEarth embeddings and on seasonal Sentinel-2 features.

## Results (macro-F1, 5 km spatial blocks, mean over 10 repeats)
| Model | Mean | SD |
|---|---|---|
| Logistic regression on Sentinel-2 (pilot) | 0.345 | 0.018 |
| Random Forest on Sentinel-2 (pilot) | 0.298 | 0.010 |
| MLP on Sentinel-2 | 0.279 | 0.043 |
| Logistic regression on AlphaEarth (pilot) | 0.250 | 0.018 |
| MLP on AlphaEarth | 0.201 | 0.026 |
| Majority class | 0.157 | 0.000 |

The network does not beat logistic regression and varies more between repeats. `results/overfitting.png` shows why: trained on one fold without early stopping, validation loss is lowest at about epoch 80–90 and then climbs while training loss falls towards zero. With about 240 training fields, a small network memorises them.

## Run
```bash
pip install -r exploratory/requirements.txt
python exploratory/mlp_spatial_blocks.py overfit
python exploratory/mlp_spatial_blocks.py cv
python exploratory/mlp_spatial_blocks.py report
```
