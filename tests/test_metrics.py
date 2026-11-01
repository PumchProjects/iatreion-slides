import unittest

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from scripts.metrics import BinaryMetrics, paired_difference


class BinaryMetricTest(unittest.TestCase):
    def test_ties_and_weighted_bootstrap_match_sklearn(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        p = np.array([0.1, 0.2, 0.2, 0.7, 0.9, 0.5, 0.7, 0.7])
        pred = p >= 0.5
        for weights in [np.ones(8, dtype=int), np.array([1, 2, 0, 3, 0, 2, 1, 1])]:
            index = np.repeat(np.arange(8), weights)
            got = BinaryMetrics(y, p, pred).calculate(weights)
            precision, recall, _ = precision_recall_curve(
                y[index], p[index], drop_intermediate=False
            )
            expected = dict(
                AUROC=roc_auc_score(y[index], p[index]),
                AP=average_precision_score(y[index], p[index]),
                PR_AUC=auc(recall, precision),
                Brier=brier_score_loss(y[index], p[index]),
                LogLoss=log_loss(y[index], p[index]),
                F1_macro=f1_score(y[index], pred[index], average='macro'),
            )
            for name, value in expected.items():
                self.assertAlmostEqual(got[name][0], value, places=12, msg=name)

    def test_constant_probability_and_no_positive_prediction(self):
        got = BinaryMetrics([0, 0, 0, 1], [0.2] * 4, [0] * 4).calculate()
        self.assertAlmostEqual(got['AUROC'][0], 0.5)
        self.assertAlmostEqual(got['AP'][0], 0.25)
        self.assertAlmostEqual(got['PR_AUC'][0], 0.625)
        self.assertTrue(np.isnan(got['PPV'][0]))
        self.assertAlmostEqual(got['NPV'][0], 0.75)

    def test_predictive_values_known_counts(self):
        got = BinaryMetrics(
            [1, 1, 0, 0, 1], [0.9, 0.8, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0]
        ).calculate()
        self.assertAlmostEqual(got['PPV'][0], 2 / 3)
        self.assertAlmostEqual(got['NPV'][0], 1 / 2)
        self.assertAlmostEqual(got['Brier'][0], (0.01 + 0.04 + 0.49 + 0.04 + 0.81) / 5)

    def test_identical_models_have_zero_paired_difference(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.8, 0.4, 0.6])
        pred = p >= 0.5
        d = paired_difference(y, p, pred, p, pred, 100, 42)
        np.testing.assert_allclose(d[['difference', 'ci_low', 'ci_high']], 0)


if __name__ == '__main__':
    unittest.main()
