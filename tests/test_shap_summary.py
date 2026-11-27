import unittest

import numpy as np
import pandas as pd

from scripts.shap_summary import IDENT, aggregate_folds, normalize_importance


class ShapSummaryTest(unittest.TestCase):
    def test_normalize_before_selecting_features(self):
        values = np.array([[1, -3, 6], [-3, 1, 6]])
        raw, relative = normalize_importance(values)
        np.testing.assert_allclose(raw, [2, 2, 6])
        np.testing.assert_allclose(relative, [0.2, 0.2, 0.6])
        np.testing.assert_allclose(normalize_importance(values * 20)[1], relative)

    def test_fold_weighting_and_zero_contributions(self):
        a = normalize_importance(np.array([[9, 1]]))[1]
        b = normalize_importance(np.repeat([[1, 3]], 10, axis=0))[1]
        np.testing.assert_allclose(np.mean([a, b], axis=0), [0.575, 0.425])
        self.assertTrue(np.isnan(normalize_importance(np.zeros((2, 3)))[1]).all())

    def test_absent_feature_is_zero_in_other_fold(self):
        d = pd.DataFrame(
            dict(
                outer_fold=[0, 0, 1],
                feature=['a', 'b', 'a'],
                relative_importance=[0.4, 0.6, 1.0],
                mean_abs_shap=[4.0, 6.0, 2.0],
                output='positive',
                n_explained=[10, 10, 5],
                same_feature_set=True,
            )
        )
        for key in IDENT:
            if key not in d:
                d[key] = 'test'
        result = aggregate_folds(d).set_index('feature')
        np.testing.assert_allclose(result.loc[['a', 'b'], 'relative_mean'], [0.7, 0.3])
        np.testing.assert_allclose(result.relative_mean.sum(), 1.0)
        self.assertEqual(result.loc['b', 'n_folds'], 2)
        self.assertEqual(result.loc['b', 'n_explained'], 15)


if __name__ == '__main__':
    unittest.main()
