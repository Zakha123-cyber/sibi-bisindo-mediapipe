import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sibi_bisindo.training import evaluate_predictions


class TrainingMetricTests(unittest.TestCase):
    def test_invalid_feature_rows_count_as_system_rejections(self):
        result=evaluate_predictions(
            np.array(['A','A','B','B']),
            np.array(['A','A','B','']),
            np.array([True,True,True,False]),
            ['A','B'])
        self.assertEqual(result['feature_valid'],3)
        self.assertAlmostEqual(result['feature_coverage'],0.75)
        self.assertEqual(result['classifier_valid_only']['accuracy'],1.0)
        self.assertEqual(result['system_all_images']['accuracy'],0.75)
        self.assertEqual(result['system_all_images']['rejected_due_to_invalid_features'],1)
        self.assertEqual(result['system_all_images']['confusion_matrix_with_reject_column'][1][2],1)

    def test_mismatched_metric_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions(['A'],['A','B'],[True],['A','B'])


if __name__=='__main__':unittest.main()
