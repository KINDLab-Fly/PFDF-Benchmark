"""Verify the actual offline data/split/feature contract used by every rerun."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.reproduce_checkpointed_baselines import _load_complete_case, _metrics
from experiments.eval_literature_baselines import _build_feature_matrix
from features.load import load_data
from features.split import zone_stratified_split_3way
from training.hp_search import SPLIT_RATIOS, HP_GRID, HP_SEEDS, EVAL_SEEDS


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.paired, cls.weather = _load_complete_case()

    def test_dataset_and_all_canonical_partitions(self):
        self.assertEqual(len(load_data()), 1243)
        self.assertEqual(len(self.df), 1131)
        self.assertEqual(int(self.df.pfdf.sum()), 292)
        for train_ratio, val_ratio, _ in SPLIT_RATIOS:
            for seed in range(42, 47):
                a = zone_stratified_split_3way(self.paired, train_ratio, val_ratio, seed)
                b = zone_stratified_split_3way(self.paired, train_ratio, val_ratio, seed)
                self.assertEqual(a, b)
                sets = [set(x) for x in a]
                self.assertEqual(len(set.union(*sets)), 1131)
                self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_feature_shapes_and_train_only_scaling(self):
        tr, va, te = zone_stratified_split_3way(self.paired, .70, .15, 42)
        for name, dim in [("n", 14), ("selected_aug", 23)]:
            X = _build_feature_matrix(name, self.df, tr, self.weather)
            self.assertEqual(X.shape, (1131, dim))
            self.assertTrue(np.isfinite(X).all())
            changed = self.df.copy()
            changed.loc[te, "storm_accum"] += 1e6
            other = _build_feature_matrix(name, changed, tr, self.weather)
            np.testing.assert_allclose(X[tr], other[tr], rtol=0, atol=0)

    def test_metric_and_grid_contract(self):
        m = _metrics([1, 1, 0, 0], [.5, .1, .7, .2])
        self.assertEqual((m["TP"], m["FP"], m["FN"], m["TN"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["TS"], 1/3)
        self.assertTrue(np.isnan(_metrics([0, 0], [.1, .2])["TS"]))
        self.assertEqual(len(HP_GRID), 200)
        self.assertEqual(list(HP_SEEDS), list(range(42, 47)))
        self.assertEqual(list(EVAL_SEEDS), list(range(42, 52)))


if __name__ == "__main__":
    unittest.main()
