import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.collect import compare_implementations, evaluate, predictions


class ImplementationIdentityTest(unittest.TestCase):
    def test_engines_and_single_fusion_survive_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for engine in ['rrl_neural', 'rrl_parser']:
                for kind in ['raw', 'calibrated_fusion']:
                    folder = root / 'predictions' / engine / kind
                    folder.mkdir(parents=True)
                    np.savez_compressed(
                        folder / 'predictions.npz',
                        sample_id=['a', 'b', 'c', 'd'],
                        y_true=[0, 1, 0, 1],
                        y_score=[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]],
                        y_pred=[0, 1, 0, 1],
                        y_mask=[False] * 4,
                        outer_fold=[0, 0, 1, 1],
                        inner_fold=[5] * 4,
                        kind=['outer_test'] * 4,
                        threshold=[0.6] * 4,
                    )
                    meta = dict(
                        run_id='r',
                        setting='logs_not_imputed',
                        model='rrl',
                        stage='internal',
                        cohort='4500',
                        subset='h-mmse',
                        train_cohort='4500',
                        feature_scheme='clinical',
                        task='MMSE',
                        inference_engine=engine,
                        prediction_kind=kind,
                        positive_index=1,
                        label_version='v',
                        threshold_source='development_inner_oof',
                        fusion_fit_engine='rrl_neural',
                    )
                    (folder / 'predictions.json').write_text(json.dumps(meta))
            data = predictions(root, ['r'], None)
            self.assertEqual(data.configuration.nunique(), 4)
            self.assertEqual(len(data), 16)
            compare_implementations(data, root / 'D5', 20, 42)
            import pandas as pd

            agreement = pd.read_csv(root / 'D5/implementation_agreement.csv')
            self.assertEqual(len(agreement), 2)
            np.testing.assert_allclose(agreement.classification_agreement, 1)
            np.testing.assert_allclose(agreement.max_absolute_probability_difference, 0)

    def test_current_labels_are_evaluated_once(self):
        import pandas as pd

        data = pd.DataFrame(
            dict(
                configuration='c',
                run_id='r',
                setting='logs_not_imputed',
                model='rrl',
                inference_engine='rrl_parser',
                prediction_kind='calibrated_fusion',
                stage='internal',
                cohort='4500',
                subset='h-mmse',
                train_cohort='4500',
                feature_scheme='clinical',
                task='MMSE',
                model_label_version='baseline_ols_le3_v1',
                y_true=[0, 1, 0, 1],
                y_true_baseline=[0, 1, 0, 1],
                probability=[0.1, 0.8, 0.3, 0.7],
                y_pred=[0, 1, 0, 1],
                unavailable=False,
                subject_id=['a', 'b', 'c', 'd'],
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evaluate(data, root, 20, 42)
            result = pd.read_csv(root / 'metrics.csv')
            self.assertFalse(
                result.duplicated(
                    [
                        'configuration',
                        'evaluation_label_version',
                        'threshold_mode',
                        'metric',
                    ]
                ).any()
            )


if __name__ == '__main__':
    unittest.main()
