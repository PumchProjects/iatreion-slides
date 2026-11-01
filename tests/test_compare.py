import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.compare import main


class CompareCommandTest(TestCase):
    def test_comparison_with_and_without_updated_labels(self):
        for updated_labels in [False, True]:
            with (
                self.subTest(updated_labels=updated_labels),
                TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                data = pd.DataFrame(
                    {
                        'subject_id': ['001', '002', '003', '004'],
                        'run_id': 'example',
                        'cohort': 'AIBL',
                        'stage': 'external',
                        'unavailable': False,
                        'subset': 'h-nonimaging_h-mri',
                        'setting': 'imputed',
                        'model': 'xgboost',
                        'y_true': [0, 1, 0, 1],
                        'probability': [0.1, 0.8, 0.4, 0.6],
                        'y_pred': [0, 1, 0, 1],
                    }
                )
                if updated_labels:
                    data['y_true_baseline'] = [0, 1, 1, 1]
                source = root / 'predictions.csv'
                output = root / 'comparison.csv'
                data.to_csv(source, index=False)
                args = [
                    'compare',
                    '--left',
                    str(source),
                    '--right',
                    str(source),
                    '--left-run',
                    'example',
                    '--right-run',
                    'example',
                    '--cohort',
                    'AIBL',
                    '--output',
                    str(output),
                    '--bootstrap',
                    '100',
                ]
                with patch.object(sys, 'argv', args):
                    main()
                result = pd.read_csv(output)
                expected = (
                    {'y_true', 'y_true_baseline'} if updated_labels else {'y_true'}
                )
                self.assertEqual(set(result.label_column), expected)
                self.assertTrue(result.n_common.eq(4).all())
                self.assertEqual(len(result), 7 * len(expected))
                np.testing.assert_allclose(
                    result[['difference', 'ci_low', 'ci_high']], 0
                )
