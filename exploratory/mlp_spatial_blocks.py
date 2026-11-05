#!/usr/bin/env python3
"""Exploratory: a small PyTorch neural network (MLP) on the pilot's Kenyan fields, under the same spatial blocks.

    python exploratory/mlp_spatial_blocks.py overfit     one spatial fold, trained far too long: watch overfitting happen
    python exploratory/mlp_spatial_blocks.py cv          the same network under the pilot's spatial blocks, with early stopping
    python exploratory/mlp_spatial_blocks.py report      compare with the pilot's Random Forest and logistic regression

Run from the repository root after `pilot.py extract` (it reads data/features.csv). Needs torch
(see exploratory/requirements.txt). Add --quick for a smoke test (not a result). Results go to exploratory/results/.

This is EXPLORATORY: it was not in the pilot's preregistration, so it cannot support or overturn the
pilot's hypotheses. Same 305 fields, same features, same 5 km spatial blocks and repeats as the pilot, so
the numbers are comparable. A multilayer perceptron (MLP) on 305 fields is a small network on very little
data: expect it to overfit, which is the point of the first step.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
PILOT = ROOT.parent
sys.path.insert(0, str(PILOT))
import pilot  # noqa: E402  (the pilot's own folds, features and metric, so nothing differs)

OUT = ROOT / "results"
DEVICE = torch.device("cpu")      # 305 rows: the CPU is faster than moving tiny batches to the GPU


class MLP(nn.Module):
    """Two hidden layers of 64 neurons. Each neuron: weights times inputs plus a bias, then ReLU.
    Dropout switches off 30% of neurons at random in training, so no single neuron is relied on."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


def load():
    frame = pd.read_csv(PILOT / "data" / "features.csv")
    classes = sorted(frame["crop"].unique())
    y = frame["crop"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return frame, y, classes


def standardise(X, train):
    """Scale with the TRAINING rows' mean and spread only; using all rows would leak test information."""
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-8
    return (X - mu) / sd


def fit(X, y, train, val, n_classes, epochs, patience, seed):
    """Full-batch training (all training fields at once, since there are only ~240). Class weights make a
    wrong Groundnut cost as much in total as a wrong Maize, as the pilot's balanced models do."""
    torch.manual_seed(seed)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    model = MLP(X.shape[1], n_classes)
    counts = np.bincount(y[train], minlength=n_classes).clip(min=1)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (n_classes * counts), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    history, best, best_state, waited = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt[train]), yt[train])    # forward pass and loss
        loss.backward()                                 # backpropagation
        opt.step()                                      # gradient descent step
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xt[val]), yt[val]).item()
        history.append((epoch, loss.item(), val_loss))
        if val_loss < best:
            best, waited, best_state = val_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def inner_split(frame, train, seed):
    """Early stopping needs its own validation fields, taken from the TRAINING fold by spatial block, so
    the outer test fold stays untouched and neighbours do not leak into validation."""
    blocks = pilot.spatial_blocks(frame.iloc[train], 5.0, seed)
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    val_blocks = set(rng.choice(ub, max(1, len(ub) // 5), replace=False))
    is_val = np.array([b in val_blocks for b in blocks])
    return train[~is_val], train[is_val]


def step_overfit(quick=False):
    """No early stopping, 1,500 epochs on one fold: training loss keeps falling, validation loss turns up."""
    import matplotlib.pyplot as plt
    frame, y, classes = load()
    folds = pilot.folds("spatial", y, frame, 5.0, 0)
    train, test = folds[0]
    tr, va = inner_split(frame, np.asarray(train), 0)
    epochs = 50 if quick else 1500
    where = OUT / "smoke_test" if quick else OUT
    where.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (name, cols) in zip(axes, [("AlphaEarth", pilot.EMBEDDING_BANDS), ("Sentinel-2", pilot.S2_COLUMNS)]):
        X = standardise(frame[cols].to_numpy(float), tr)
        _, h = fit(X, y, tr, va, len(classes), epochs, patience=10 ** 9, seed=0)
        e, tl, vl = zip(*h)
        best = int(np.argmin(vl))
        ax.plot(e, tl, label="train"); ax.plot(e, vl, label="validation")
        ax.axvline(e[best], ls=":", c="grey"); ax.set_title(f"MLP on {name}: best epoch {e[best]}", fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
        print(f"{name}: validation loss lowest at epoch {e[best]} ({vl[best]:.3f}); at the end {vl[-1]:.3f}, "
              f"while training loss fell from {tl[0]:.3f} to {tl[-1]:.3f}")
    fig.tight_layout(); fig.savefig(where / "overfitting.png", dpi=130)
    print("figure:", where / "overfitting.png")


def step_cv(quick=False):
    """Out-of-fold predictions for every field, as the pilot does: each prediction comes from a model that
    never saw that field or its 5 km block."""
    frame, y, classes = load()
    repeats = 1 if quick else pilot.N_REPEATS
    rows = []
    for name, cols in [("MLP on AlphaEarth", pilot.EMBEDDING_BANDS), ("MLP on Sentinel-2", pilot.S2_COLUMNS)]:
        Xraw = frame[cols].to_numpy(float)
        for r in range(repeats):
            pred = np.empty(len(y), dtype=int)
            for train, test in pilot.folds("spatial", y, frame, 5.0, r):
                tr, va = inner_split(frame, np.asarray(train), r)
                X = standardise(Xraw, np.asarray(train))
                model, _ = fit(X, y, tr, va, len(classes), 30 if quick else 2000, 50, seed=r)
                with torch.no_grad():
                    pred[test] = model(torch.tensor(X[test], dtype=torch.float32)).argmax(1).numpy()
            f1 = pilot.macro_f1(np.array(classes)[y], np.array(classes)[pred])
            rows.append({"model": name, "repeat": r, "macro_f1": f1})
            print(f"{name} repeat {r}: macro-F1 {f1:.3f}")
    where = OUT / "smoke_test" if quick else OUT
    where.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(where / "cv_by_repeat.csv", index=False)


def step_report():
    mine = pd.read_csv(OUT / "cv_by_repeat.csv").groupby("model").macro_f1.agg(["mean", "std"])
    theirs = pd.read_csv(PILOT / "results" / "cv_summary.csv", header=[0, 1], index_col=[0, 1])
    spatial = theirs.loc["spatial", "macro_f1"][["mean", "std"]]
    table = pd.concat([spatial, mine]).sort_values("mean", ascending=False).round(3)
    print("Macro-F1 under 5 km spatial blocks (pilot models: preregistered; MLPs: exploratory)")
    print(table.to_string())
    table.to_csv(OUT / "comparison_with_pilot.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["overfit", "cv", "report"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    {"overfit": lambda: step_overfit(a.quick), "cv": lambda: step_cv(a.quick), "report": step_report}[a.step]()


if __name__ == "__main__":
    main()
