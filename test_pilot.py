"""python test_pilot.py [path/to/fields.geojson]. No Earth Engine account or network needed."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

import pilot


def test_ring_stats():
    side = 0.001  # about 111 m
    lon, lat, area = pilot.ring_stats([[34, 0], [34 + side, 0], [34 + side, side], [34, side], [34, 0]])
    assert abs(lon - 34.0005) < 1e-6 and abs(lat - 0.0005) < 1e-6
    assert 1.1 < area < 1.3, area


def test_spatial_folds_keep_blocks_apart():
    frame = pilot.synthetic_features()
    y = frame["crop"].to_numpy()
    blocks = pilot.spatial_blocks(frame, 5, seed=3)
    for train, test in pilot.folds("spatial", y, frame, 5, 3):
        assert not set(blocks[train]) & set(blocks[test])
    first = pilot.folds("spatial", y, frame, 5, 0)[0][1]
    assert any(set(pilot.folds("spatial", y, frame, 5, s)[0][1]) != set(first) for s in (1, 2, 3)), "repeats are identical"


def test_subsample_keeps_every_crop():
    y = np.array(["a"] * 90 + ["b"] * 10)
    picked = pilot.subsample(np.arange(100), y, 0.1, np.random.default_rng(0))
    assert (y[picked] == "b").sum() == 2 and (y[picked] == "a").sum() == 9


def test_validation_design_and_null_signal():
    """Shuffled labels must score near the baseline; spatial clustering must inflate random folds."""
    with tempfile.TemporaryDirectory() as tmp:
        frame = pilot.synthetic_features()
        shuffled = frame.copy()
        shuffled["crop"] = np.random.default_rng(1).permutation(frame["crop"].to_numpy())
        path = Path(tmp) / "shuffled.csv"
        shuffled.to_csv(path, index=False)
        results, _, _ = pilot.evaluate(path, Path(tmp) / "a", repeats=2, repeats_curve=1)
        best = results[results["model"] != "Majority class"]["macro_f1"].max()
        assert best < 0.35, f"shuffled labels scored {best:.2f}: something is leaking"

        path = Path(tmp) / "real.csv"
        frame.to_csv(path, index=False)
        results, comparison, curve = pilot.evaluate(path, Path(tmp) / "b", repeats=2, repeats_curve=1)
        means = results.groupby(["validation", "model"])["macro_f1"].mean()
        for model in ("RF on AlphaEarth", "RF on Sentinel-2"):
            gap = means["random", model] - means["spatial", model]
            assert gap > 0.05, f"random folds should beat spatial blocks for {model}, gap {gap:.3f}"
        assert {"cv_comparison.png", "learning_curve.png", "comparison.csv"} <= {p.name for p in (Path(tmp) / "b").iterdir()}


def test_extract_control_flow(labels):
    """Runs extract against a stand-in `ee`. This checks our logic, not Earth Engine itself."""
    ee = MagicMock()
    def fake_feature(_geometry, props):
        return props
    ee.Feature.side_effect = fake_feature
    ee.FeatureCollection.side_effect = lambda chunk: chunk
    image = MagicMock()
    ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value.mosaic.return_value.select.return_value = image
    image.addBands.return_value = image
    def reduce_regions(collection, **_):
        values = {c: 0.1 for c in pilot.EMBEDDING_BANDS + pilot.S2_COLUMNS}
        return MagicMock(getInfo=lambda: {"features": [{"properties": {**f, **values}} for f in collection]})
    image.reduceRegions.side_effect = reduce_regions
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "features.csv"
        frame = pilot.extract("test", labels, out, ee=ee)
        assert len(frame) == 305 and frame["crop"].nunique() == 5, (len(frame), frame["crop"].nunique())
        assert frame["area_ha"].gt(0).all()
        assert set(pilot.EMBEDDING_BANDS) <= set(frame.columns)


if __name__ == "__main__":
    test_ring_stats()
    test_spatial_folds_keep_blocks_apart()
    test_subsample_keeps_every_crop()
    test_validation_design_and_null_signal()
    if len(sys.argv) > 1:
        test_extract_control_flow(sys.argv[1])
    print("ALL PASSED")
