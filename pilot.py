#!/usr/bin/env python3
"""Do AlphaEarth embeddings beat conventional Sentinel-2 features on Kenyan smallholder fields?

    python pilot.py labels                      download the PlantVillage field labels
    python pilot.py extract --project MY_GEE_ID sample both feature sets in Earth Engine
    python pilot.py evaluate                    run the comparison, write results/
    python pilot.py demo                        check the install on synthetic data

See README.md. Fix the hypotheses in PREREGISTRATION.md before running `evaluate` on real data.
"""
import argparse
import json
import math
import platform
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_URL = ("https://data.source.coop/radiantearth/african-crops-kenya-01/"
             "ref_african_crops_kenya_01_tile_001.geojson")
LABEL_FILE = Path("data/fields.geojson")
FEATURE_FILE = Path("data/features.csv")
RESULT_DIR = Path("results")

YEAR = 2019  # every field was surveyed between 14 May and 3 June 2019
MIN_FIELDS_PER_CROP = 10
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# Western Kenya's calendar: dry start of year, long rains, long-rains harvest, short rains.
S2_PERIODS = {
    "jan_feb": (f"{YEAR}-01-01", f"{YEAR}-03-01"),
    "mar_jun": (f"{YEAR}-03-01", f"{YEAR}-07-01"),
    "jul_sep": (f"{YEAR}-07-01", f"{YEAR}-10-01"),
    "oct_dec": (f"{YEAR}-10-01", f"{YEAR + 1}-01-01"),
}
CLOUD_SCORE_MIN = 0.6
S2_COLUMNS = [f"s2_{p}_{b}" for p in S2_PERIODS for b in S2_BANDS + ["NDVI"]]

N_FOLDS = 5
N_REPEATS = 10
N_REPEATS_CURVE = 5
N_BOOTSTRAP = 1000
LABEL_FRACTIONS = (0.1, 0.25, 0.5, 1.0)
PRIMARY_BASELINE = "RF on Sentinel-2"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Colour follows the model; marker shape separates AlphaEarth (circle) from Sentinel-2 (square).
MODEL_STYLES = {"LogReg on AlphaEarth": ("#2a78d6", "o"), "RF on Sentinel-2": ("#eb6834", "s"),
                "RF on AlphaEarth": ("#1baf7a", "o"), "LogReg on Sentinel-2": ("#eda100", "s")}


# ---------------------------------------------------------------- labels

def download_labels(path=LABEL_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LABEL_URL, path)
    print(f"wrote {path}")


def ring_stats(ring):
    """Centroid (lon, lat) and area (ha) of a lon/lat ring, on a local flat projection."""
    lat0 = sum(p[1] for p in ring) / len(ring)
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540
    xs, ys = [p[0] * kx for p in ring], [p[1] * ky for p in ring]
    x0, y0 = xs[0], ys[0]  # shift to the first vertex to keep precision
    xs, ys = [x - x0 for x in xs], [y - y0 for y in ys]
    cross = [xs[i] * ys[i + 1] - xs[i + 1] * ys[i] for i in range(len(xs) - 1)]
    area = sum(cross) / 2
    if area == 0:
        return sum(p[0] for p in ring) / len(ring), lat0, 0.0
    cx = sum((xs[i] + xs[i + 1]) * cross[i] for i in range(len(cross))) / (6 * area)
    cy = sum((ys[i] + ys[i + 1]) * cross[i] for i in range(len(cross))) / (6 * area)
    return (cx + x0) / kx, (cy + y0) / ky, abs(area) / 10000


def load_fields(path=LABEL_FILE, min_fields=MIN_FIELDS_PER_CROP):
    """Returns (table, geometries). Crops with too few fields to split into folds are dropped."""
    features = json.loads(Path(path).read_text())["features"]
    rows, geometries = [], {}
    for f in features:
        p = f["properties"]
        lon, lat, area = ring_stats(f["geometry"]["coordinates"][0])
        rows.append({"fid": f["id"], "crop": p["Crop1"], "mixed": bool(p.get("Crop2")),
                     "lon": lon, "lat": lat, "area_ha": area})
        geometries[f["id"]] = f["geometry"]
    table = pd.DataFrame(rows)
    counts = table["crop"].value_counts()
    keep = counts[counts >= min_fields].index
    print(f"{len(table)} fields; keeping {counts[keep].to_dict()}; "
          f"dropping {counts.drop(keep).to_dict()}")
    table = table[table["crop"].isin(keep)].reset_index(drop=True)
    return table, {fid: geometries[fid] for fid in table["fid"]}


# ---------------------------------------------------------------- Earth Engine

def s2_composite(ee, region, name, start, end):
    """Cloud-masked median for one period, with NDVI, renamed to its feature columns."""
    scores = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    median = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region).filterDate(start, end)
              .linkCollection(scores, ["cs_cdf"])
              .map(lambda image: image.updateMask(image.select("cs_cdf").gte(CLOUD_SCORE_MIN)))
              .select(S2_BANDS).median())
    bands = S2_BANDS + ["NDVI"]
    return (median.addBands(median.normalizedDifference(["B8", "B4"]).rename("NDVI"))
            .select(bands).rename([f"s2_{name}_{b}" for b in bands]))


def extract(project, labels=LABEL_FILE, out=FEATURE_FILE, batch=50, ee=None):
    """Field-mean AlphaEarth and Sentinel-2 values for every labelled field."""
    if ee is None:
        import ee
    ee.Initialize(project=project)
    table, geometries = load_fields(labels)
    lons, lats = table["lon"], table["lat"]
    region = ee.Geometry.Rectangle([lons.min() - 0.02, lats.min() - 0.02, lons.max() + 0.02, lats.max() + 0.02])
    embeddings = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                  .filterBounds(region).filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01"))
    image = embeddings.mosaic().select(EMBEDDING_BANDS)
    for name, (start, end) in S2_PERIODS.items():
        image = image.addBands(s2_composite(ee, region, name, start, end))

    sampled = []
    fids = list(table["fid"])
    for i in range(0, len(fids), batch):
        chunk = ee.FeatureCollection([ee.Feature(ee.Geometry(geometries[fid]), {"fid": fid})
                                      for fid in fids[i:i + batch]])
        result = image.reduceRegions(collection=chunk, reducer=ee.Reducer.mean(), scale=10,
                                     crs="EPSG:32636", tileScale=2).getInfo()
        sampled += [f["properties"] for f in result["features"]]
        print(f"sampled {len(sampled)}/{len(fids)}")

    values = pd.DataFrame(sampled)
    frame = table.merge(values.reindex(columns=["fid"] + EMBEDDING_BANDS + S2_COLUMNS), on="fid", validate="1:1")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}")
    print("share of fields with no value:", frame[EMBEDDING_BANDS[0]].isna().mean().round(3), "(AlphaEarth);",
          frame[S2_COLUMNS].isna().mean().round(3).groupby(lambda c: c.split("_B")[0].split("_NDVI")[0]).max().to_dict())
    return frame


# ---------------------------------------------------------------- evaluation

def build_models(seed, embedding_cols, s2_cols):
    def rf():
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))

    def logreg():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))

    return {"RF on Sentinel-2": (rf(), s2_cols), "LogReg on Sentinel-2": (logreg(), s2_cols),
            "RF on AlphaEarth": (rf(), embedding_cols), "LogReg on AlphaEarth": (logreg(), embedding_cols),
            "Majority class": (DummyClassifier(strategy="most_frequent"), s2_cols)}


def spatial_blocks(frame, block_km, seed=None):
    """Grid-square id of each field, so nearby fields stay together in a fold. A seed shifts
    the grid: StratifiedGroupKFold returns almost the same folds for a fixed set of groups,
    so without the shift, repeats would add nothing."""
    shift_x, shift_y = (0.0, 0.0) if seed is None else np.random.default_rng(seed).uniform(0, block_km, 2)
    km_x = frame["lon"] * 111.32 * np.cos(np.radians(frame["lat"].mean())) + shift_x
    km_y = frame["lat"] * 110.54 + shift_y
    return (np.floor(km_x / block_km).astype(int).astype(str) + "_"
            + np.floor(km_y / block_km).astype(int).astype(str)).to_numpy()


def folds(mode, y, frame, block_km, seed):
    if mode == "random":
        return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(y, y))
    blocks = spatial_blocks(frame, block_km, seed)
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(y, y, blocks))


def subsample(train, y, fraction, rng):
    """Stratified share of the training fold, keeping at least two fields per crop."""
    if fraction >= 1:
        return train
    keep = []
    for crop in np.unique(y[train]):
        members = train[y[train] == crop]
        n = max(min(2, len(members)), round(len(members) * fraction))
        keep.append(rng.choice(members, n, replace=False))
    return np.concatenate(keep)


def out_of_fold(model, X, y, splits, fraction=1.0, rng=None):
    """Predictions for every field, each from a model that never saw that field."""
    predicted = np.empty(len(y), dtype=object)
    for train, test in splits:
        train = subsample(train, y, fraction, rng)
        predicted[test] = model.fit(X[train], y[train]).predict(X[test])
    return predicted


def macro_f1(y, predicted):
    return f1_score(y, predicted, average="macro", labels=np.unique(y), zero_division=0)


def block_bootstrap(y, preds_a, preds_b, blocks, seed):
    """95% interval for macro-F1 (a - b), resampling spatial blocks rather than fields."""
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in unique}
    differences = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([members[b] for b in rng.choice(unique, len(unique))])
        r = rng.integers(len(preds_a))
        differences.append(macro_f1(y[idx], preds_a[r][idx]) - macro_f1(y[idx], preds_b[r][idx]))
    return np.percentile(differences, [2.5, 97.5])


def evaluate(features=FEATURE_FILE, out=RESULT_DIR, block_km=5.0, pure_only=False, note="",
             repeats=N_REPEATS, repeats_curve=N_REPEATS_CURVE):
    frame = pd.read_csv(features)
    if pure_only:
        frame = frame[~frame["mixed"]].reset_index(drop=True)
    embedding_cols = [c for c in EMBEDDING_BANDS if frame[c].notna().any()]
    s2_cols = [c for c in S2_COLUMNS if frame[c].notna().any()]
    frame = frame[frame[embedding_cols].notna().all(axis=1)].reset_index(drop=True)
    y = frame["crop"].to_numpy()
    blocks = spatial_blocks(frame, block_km)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_info.json").write_text(json.dumps({
        "note": note, "fields": len(frame), "crops": frame["crop"].value_counts().to_dict(),
        "spatial_blocks": len(set(blocks)), "block_km": block_km, "pure_only": pure_only,
        "median_field_ha": round(float(frame["area_ha"].median()), 3),
        "sentinel2_missing_share": round(float(frame[s2_cols].isna().mean().mean()), 3),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__}, indent=2))
    print(f"{len(frame)} fields in {len(set(blocks))} blocks of {block_km} km")

    # 1. Random versus spatially blocked cross-validation
    rows, per_class, preds = [], [], {}
    for mode in ("random", "spatial"):
        for repeat in range(repeats):
            splits = folds(mode, y, frame, block_km, repeat)
            for name, (model, cols) in build_models(repeat, embedding_cols, s2_cols).items():
                predicted = out_of_fold(model, frame[cols].to_numpy(dtype=float), y, splits)
                preds.setdefault((mode, name), []).append(predicted)
                rows.append({"validation": mode, "model": name, "repeat": repeat, "macro_f1": macro_f1(y, predicted),
                             "balanced_accuracy": balanced_accuracy_score(y, predicted)})
                scores = f1_score(y, predicted, average=None, labels=np.unique(y), zero_division=0)
                per_class += [{"validation": mode, "model": name, "repeat": repeat, "crop": c, "f1": s}
                              for c, s in zip(np.unique(y), scores)]
        print(f"finished {mode} validation")
    results = pd.DataFrame(rows)
    results.to_csv(out / "cv_by_repeat.csv", index=False)
    summary = (results.groupby(["validation", "model"])[["macro_f1", "balanced_accuracy"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(out / "cv_summary.csv")
    (pd.DataFrame(per_class).groupby(["validation", "model", "crop"])["f1"].mean().round(3)
     .unstack("crop").to_csv(out / "per_crop_f1.csv"))

    # 2. Each model against the conventional baseline, under spatial validation
    comparison = []
    spatial_means = results[results["validation"] == "spatial"].groupby("model")["macro_f1"].mean()
    for name in ("LogReg on AlphaEarth", "RF on AlphaEarth"):
        low, high = block_bootstrap(y, preds[("spatial", name)], preds[("spatial", PRIMARY_BASELINE)], blocks, 0)
        comparison.append({"model": name, "baseline": PRIMARY_BASELINE,
                           "macro_f1_difference": round(spatial_means[name] - spatial_means[PRIMARY_BASELINE], 3),
                           "ci95_low": round(low, 3), "ci95_high": round(high, 3)})
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)

    # 3. Learning curves: does the gain grow when labels are scarce?
    curve = []
    for fraction in LABEL_FRACTIONS:
        for repeat in range(repeats_curve):
            splits = folds("spatial", y, frame, block_km, repeat)
            for name, (model, cols) in build_models(repeat, embedding_cols, s2_cols).items():
                if name == "Majority class":
                    continue
                rng = np.random.default_rng([repeat, int(fraction * 100)])
                predicted = out_of_fold(model, frame[cols].to_numpy(dtype=float), y, splits, fraction, rng)
                curve.append({"fraction": fraction, "model": name, "repeat": repeat, "macro_f1": macro_f1(y, predicted)})
        print(f"finished learning curve at {fraction:.0%} of labels")
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "learning_curve.csv", index=False)

    plot_cv(results, out / "cv_comparison.png", note)
    plot_curve(curve, out / "learning_curve.png", note)
    print(summary.xs("macro_f1", axis=1, level=0).to_string())
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"results in {out}/")
    return results, pd.DataFrame(comparison), curve


# ---------------------------------------------------------------- figures

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)


def plot_cv(results, path, note):
    import matplotlib.pyplot as plt
    means = results.groupby(["validation", "model"])["macro_f1"].mean().unstack("validation")
    means = means.sort_values("spatial")
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    positions = range(len(means))
    ax.hlines(positions, means["spatial"], means["random"], color=AXIS, linewidth=2, zorder=2)
    ax.scatter(means["random"], positions, s=70, facecolor=SURFACE, edgecolor=INK_2, linewidth=1.8, zorder=3,
               label="Random folds")
    ax.scatter(means["spatial"], positions, s=70, color=INK_2, edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="Spatial blocks")
    ax.set_yticks(list(positions), means.index, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1, mean over repeats", color=INK_2)
    title = "Field-level crop classification by validation design"
    ax.set_title(f"{note}: {title}" if note else title, loc="left", color=INK, fontsize=12, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_curve(curve, path, note):
    import matplotlib.pyplot as plt
    means = curve.groupby(["fraction", "model"])["macro_f1"].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for name, (colour, marker) in MODEL_STYLES.items():
        ax.plot(means.index, means[name], color=colour, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
    ax.set_xscale("log")
    ax.set_xticks(list(LABEL_FRACTIONS), [f"{f:.0%}" for f in LABEL_FRACTIONS])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of training labels used (log scale)", color=INK_2)
    ax.set_ylabel("Macro-F1, spatial blocks", color=INK_2)
    title = "How performance changes as labels become scarce"
    ax.set_title(f"{note}: {title}" if note else title, loc="left", color=INK, fontsize=12, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9, labelcolor=INK_2,
              handlelength=1.5, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- synthetic check

def synthetic_features(n=320, seed=0, clusters=15):
    """Fake data with the real file's columns. Fields cluster around villages that favour
    certain crops and have their own local signature, which is what lets random folds leak.
    Not a result."""
    rng = np.random.default_rng(seed)
    names = np.array(["Maize", "Cassava", "Fallowland", "Millet", "Groundnut"])
    columns = EMBEDDING_BANDS + S2_COLUMNS
    centres = np.column_stack([rng.uniform(34.12, 34.37, clusters), rng.uniform(0.47, 0.71, clusters)])
    crop_shares = rng.dirichlet(np.array([.62, .19, .08, .06, .05]), clusters)
    local = rng.normal(size=(clusters, len(columns)))
    crop_signature = rng.normal(size=(len(names), len(columns)))
    weight = np.array([0.3 if c in EMBEDDING_BANDS else 0.15 for c in columns])

    cluster = rng.integers(clusters, size=n)
    crop = np.array([rng.choice(len(names), p=crop_shares[c]) for c in cluster])
    values = weight * crop_signature[crop] + local[cluster] + rng.normal(size=(n, len(columns)))
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "fid", [f"synthetic_{i}" for i in range(n)])
    frame.insert(1, "crop", names[crop])
    frame.insert(2, "mixed", rng.random(n) < 0.25)
    frame.insert(3, "lon", centres[cluster, 0] + rng.normal(0, 0.006, n))
    frame.insert(4, "lat", centres[cluster, 1] + rng.normal(0, 0.006, n))
    frame.insert(5, "area_ha", rng.gamma(2, 0.2, n))
    return frame


def demo(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "synthetic_features.csv"
    synthetic_features().to_csv(path, index=False)
    (out / "SYNTHETIC_NOT_RESULTS.txt").write_text("These files come from invented data. They only show the code runs.\n")
    evaluate(path, out, note="SYNTHETIC", repeats=3, repeats_curve=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("labels")
    e = commands.add_parser("extract")
    e.add_argument("--project", required=True, help="your Google Cloud project registered for Earth Engine")
    v = commands.add_parser("evaluate")
    v.add_argument("--block-km", type=float, default=5.0)
    v.add_argument("--pure-only", action="store_true", help="drop fields with a second recorded crop")
    v.add_argument("--out", default=str(RESULT_DIR))
    d = commands.add_parser("demo")
    d.add_argument("--out", default="demo_output")
    args = parser.parse_args()
    if args.command == "labels":
        download_labels()
    elif args.command == "extract":
        extract(args.project)
    elif args.command == "evaluate":
        evaluate(block_km=args.block_km, pure_only=args.pure_only, out=args.out)
    else:
        demo(args.out)


if __name__ == "__main__":
    main()
