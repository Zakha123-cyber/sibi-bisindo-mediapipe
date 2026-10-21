from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from sibi_bisindo.npy_inference import PredictionWindow, load_selected_model, predict_hands
from sibi_bisindo.npy_landmarks import FEATURE_VERSION, normalized_features, sha256_file
from webcam_npy_landmarks import finish_trial, run_preview


class RecordingClassifier:
    classes_ = ['A', 'B']

    def predict(self, x):
        self.seen = x.copy()
        return ['A']

    def decision_function(self, x):
        return [[2.0, 1.0]]


class NpyInferenceTests(unittest.TestCase):
    def test_webcam_raw_landmarks_use_the_training_transform(self):
        raw = np.random.default_rng(9).normal(size=(21, 3))
        model = RecordingClassifier()
        result = predict_hands({'model': model}, raw[None])
        np.testing.assert_array_equal(model.seen[0], normalized_features(raw.reshape(63)))
        self.assertEqual(result['letter'], 'A')
        self.assertEqual(result['score_type'], 'svm_decision_uncalibrated')

    def test_no_hand_extra_hands_and_invalid_geometry_are_not_classified(self):
        # The bundle has no classifier: any attempt to classify a rejected frame fails.
        self.assertEqual(predict_hands({}, np.empty((0, 21, 3)))['status'], 'no_hand')
        self.assertEqual(predict_hands({}, np.ones((2, 21, 3)))['status'], 'unexpected_extra_hand')
        self.assertEqual(predict_hands({}, np.zeros((1, 21, 3)))['status'], 'invalid_geometry')

    def test_mirror_comparison_changes_only_relative_x_and_does_not_mutate_input(self):
        raw = np.random.default_rng(12).normal(size=(1, 21, 3))
        before = raw.copy()
        model = RecordingClassifier()
        predict_hands({'model': model}, raw)
        original = model.seen.reshape(21, 3).copy()
        predict_hands({'model': model}, raw, orientation='mirror-x')
        mirrored = model.seen.reshape(21, 3)
        np.testing.assert_allclose(mirrored[:, 0], -original[:, 0], atol=1e-7)
        np.testing.assert_allclose(mirrored[:, 1:], original[:, 1:], atol=1e-7)
        np.testing.assert_array_equal(raw, before)
        with self.assertRaises(ValueError):
            predict_hands({'model': model}, raw, orientation='automatic')

    def test_stable_prediction_resets_on_lost_hand_and_pose_change(self):
        window = PredictionWindow(5)
        for _ in range(3):
            self.assertIsNone(window.update('A'))
        self.assertEqual(window.update('A'), 'A')
        self.assertIsNone(window.update('B'))
        self.assertIsNone(window.update(None))
        self.assertEqual(len(window.history), 0)
        self.assertIsNone(window.update('B'))

    def test_trial_counts_rejections_and_ties_as_no_prediction(self):
        trial = {'expected': 'V', 'started': 1.0, 'frames': 8,
                 'raw_counts': Counter({'R': 8}),
                 'stable_counts': Counter({'__reject__': 5, 'R': 3}),
                 'statuses': Counter({'ok': 8})}
        result = finish_trial(trial, 4.1, completed=True)
        self.assertIsNone(result['trial_prediction'])
        self.assertFalse(result['correct'])
        trial['stable_counts'] = Counter({'R': 4, 'V': 4})
        self.assertIsNone(finish_trial(trial, 4.1, True)['trial_prediction'])

    def test_loader_verifies_artifact_and_model_contract(self):
        rng = np.random.default_rng(10)
        x = np.stack([normalized_features(rng.normal(size=(21, 3))) for _ in range(8)])
        model = make_pipeline(StandardScaler(), SVC(kernel='linear')).fit(x, ['A']*4 + ['B']*4)
        metadata = {
            'source_language': 'asl', 'target_language': 'sibi', 'pose_mapping': 'unverified',
            'classes': ['A', 'B'], 'dataset_version': 'fixture', 'feature_version': FEATURE_VERSION,
            'feature_dimension': 63,
            'split_lock': {'feature_implementation_sha256': sha256_file(ROOT / 'src/sibi_bisindo/npy_landmarks.py')},
            'scikit_learn_version': importlib.metadata.version('scikit-learn'),
            'joblib_version': importlib.metadata.version('joblib'),
            'numpy_version': np.__version__,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            artifact = path / 'selected.joblib'
            joblib.dump(dict(metadata, model=model, candidate='svm_fixture'), artifact)
            selection = dict(metadata, artifact=artifact.name, artifact_sha256=sha256_file(artifact),
                             selected_candidate='svm_fixture')
            (path / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
            loaded, _ = load_selected_model(path)
            np.testing.assert_array_equal(loaded['model'].predict(x), model.predict(x))
            artifact.write_bytes(artifact.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                load_selected_model(path)

    def test_comparison_trial_reports_both_branches_without_selecting_from_ground_truth(self):
        trial = {'expected': 'A', 'started': 1.0, 'frames': 10,
                 'raw_counts': Counter({'O': 10}), 'stable_counts': Counter({'O': 10}),
                 'statuses': Counter({'ok': 10}),
                 'comparison': {'orientation': 'mirror-x', 'raw_counts': Counter({'A': 10}),
                                'stable_counts': Counter({'A': 10})}}
        result = finish_trial(trial, 4.1, completed=True)
        self.assertEqual(result['trial_prediction'], 'O')
        self.assertFalse(result['correct'])
        self.assertEqual(result['comparison']['trial_prediction'], 'A')
        self.assertTrue(result['comparison']['correct'])

    def test_preview_comparison_records_both_branches_and_rgb_without_real_camera(self):
        import cv2

        class Classifier:
            classes_ = ['A', 'O']

            def predict(self, x):
                return np.where(x[:, 3] < 0, 'A', 'O')

            def decision_function(self, x):
                return np.array([[1, 0] if x[0, 3] < 0 else [0, 1]])

        class Camera:
            def isOpened(self):
                return True

            def set(self, prop, value):
                pass

            def get(self, prop):
                return 64

            def read(self):
                return True, np.full((64, 64, 3), [1, 2, 3], dtype=np.uint8)

            def release(self):
                self.released = True

        raw = np.full((1, 21, 3), 0.3, dtype=np.float64)
        raw[0, 1, 0] = 0.5
        expected_rgb = np.full((64, 64, 3), [3, 2, 1], dtype=np.uint8)

        class Detector:
            def detect(inner, rgb):
                np.testing.assert_array_equal(rgb, expected_rgb)
                return raw.copy()

        tick = [0.0]

        def clock():
            tick[0] += 0.1
            return tick[0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text('{}', encoding='utf-8')
            args = SimpleNamespace(device=0, width=64, height=64, history=1, config=config,
                                   trial_seconds=0.01, output_dir=root / 'reports',
                                   landmark_orientation='original', compare_mirror=True)
            selection = {'classes': ['A', 'O'], 'selected_candidate': 'fixture',
                         'artifact_sha256': 'fixture', 'feature_version': FEATURE_VERSION}
            mp = SimpleNamespace(solutions=SimpleNamespace(hands=SimpleNamespace(HAND_CONNECTIONS=[])))
            camera = Camera()
            with patch.object(cv2, 'VideoCapture', return_value=camera), \
                    patch.object(cv2, 'imshow'), patch.object(cv2, 'destroyAllWindows'), \
                    patch.object(cv2, 'waitKey', side_effect=[ord('a'), -1, 27]), \
                    patch('webcam_npy_landmarks.time.perf_counter', side_effect=clock):
                run_preview(args, root, {'model': Classifier()}, selection, Detector(),
                            cv2, mp, np, PredictionWindow, predict_hands, sha256_file)
            summary = json.loads(next(args.output_dir.glob('*.json')).read_text())
            self.assertEqual(summary['completed_trials'], 1)
            self.assertEqual(summary['correct_completed_trials'], 0)
            self.assertEqual(summary['correct_comparison_trials'], 1)
            self.assertEqual(summary['trials'][0]['comparison']['trial_prediction'], 'A')
            self.assertTrue(camera.released)


if __name__ == '__main__':
    unittest.main()
