"""Inference for the ASL-source 63D candidate, separate from the old SIBI model."""
from collections import Counter, deque
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np

from .npy_landmarks import FEATURE_VERSION, normalized_features, sha256_file


def load_selected_model(run_dir):
    run_dir = Path(run_dir)
    selection = json.loads((run_dir / 'selection.json').read_text(encoding='utf-8'))
    artifact = Path(selection['artifact'])
    if artifact.name != selection['artifact'] or artifact.is_absolute():
        raise ValueError('Model artifact must be a filename inside its run directory')
    path = run_dir / artifact
    if sha256_file(path) != selection['artifact_sha256']:
        raise ValueError('Selected model artifact hash mismatch')
    if (selection['feature_version'] != FEATURE_VERSION or selection['feature_dimension'] != 63
            or selection['source_language'] != 'asl' or selection['target_language'] != 'sibi'
            or selection['pose_mapping'] != 'unverified'):
        raise ValueError('Unexpected feature contract or source/target label provenance')
    if sha256_file(Path(__file__).with_name('npy_landmarks.py')) != selection['split_lock']['feature_implementation_sha256']:
        raise ValueError('Preprocessing implementation differs from the selected training run')
    for package, key in [('scikit-learn', 'scikit_learn_version'), ('joblib', 'joblib_version'),
                         ('numpy', 'numpy_version')]:
        if importlib.metadata.version(package) != selection[key]:
            raise RuntimeError(f'{package} differs from the saved model environment')
    bundle = joblib.load(path)
    for key in ('source_language', 'target_language', 'pose_mapping', 'feature_version',
                'feature_dimension', 'classes', 'split_lock', 'dataset_version'):
        if bundle[key] != selection[key]:
            raise ValueError(f'Classifier bundle metadata mismatch: {key}')
    model = bundle['model']
    if model.n_features_in_ != 63 or list(model.classes_) != selection['classes']:
        raise ValueError('Classifier feature count or label ordering mismatch')
    if bundle['candidate'] != selection['selected_candidate']:
        raise ValueError('Classifier does not match the selected candidate')
    return bundle, selection


def predict_hands(bundle, hands, orientation='original'):
    if orientation not in {'original', 'mirror-x'}:
        raise ValueError('Unsupported landmark orientation')
    hands = np.asarray(hands)
    if hands.size == 0:
        return {'status': 'no_hand', 'letter': None, 'score': None, 'score_type': None}
    if hands.ndim != 3 or hands.shape[1:] != (21, 3):
        raise ValueError('Expected hands with shape (n, 21, 3)')
    if len(hands) > 1:
        return {'status': 'unexpected_extra_hand', 'letter': None, 'score': None, 'score_type': None}
    points = hands[0]
    if orientation == 'mirror-x':
        # Reflect only the raw image x coordinate. Wrist subtraction in the
        # shared training transform then negates relative x, preserving y/z.
        # Copy to avoid changing the detected points used by the other branch.
        points = points.copy()
        points[:, 0] = 1.0 - points[:, 0]
    try:
        features = normalized_features(points).reshape(1, 63)
    except ValueError:
        return {'status': 'invalid_geometry', 'letter': None, 'score': None, 'score_type': None}
    model = bundle['model']
    letter = str(model.predict(features)[0])
    class_index = list(model.classes_).index(letter)
    if hasattr(model, 'decision_function'):
        scores = np.asarray(model.decision_function(features)).reshape(-1)
        score = float(scores[class_index]) if len(scores) > 1 else float(scores[0])
        score_type = 'svm_decision_uncalibrated'
    else:
        score = float(model.predict_proba(features)[0, class_index])
        score_type = 'probability_uncalibrated'
    return {'status': 'ok', 'letter': letter, 'score': score, 'score_type': score_type}


class PredictionWindow:
    def __init__(self, frames=5):
        if frames < 1:
            raise ValueError('History must be positive')
        self.history = deque(maxlen=frames)
        self.minimum_votes = (4 * frames + 4) // 5

    def reset(self):
        self.history.clear()

    def update(self, letter):
        if letter is None:
            self.reset()
            return None
        self.history.append(letter)
        winner, count = Counter(self.history).most_common(1)[0]
        # Do not continue showing an old majority once the current prediction changes.
        return winner if winner == letter and count >= self.minimum_votes else None


class RawHandDetector:
    """Use only raw MediaPipe coordinates; never invoke the old 132D transform."""
    def __init__(self, config):
        if (config['feature_version'] != FEATURE_VERSION or config['input_color'] != 'RGB'
                or config['input_mirrored'] is not False):
            raise ValueError('Unsupported webcam input contract')
        for package, expected in config['dependencies'].items():
            installed = importlib.metadata.version(package)
            if installed != expected:
                raise RuntimeError(f'{package}: expected {expected}, found {installed}; use requirements-features.txt')
        package_dir = Path(importlib.util.find_spec('mediapipe').origin).parent
        for relative, digest in config['bundled_model_sha256'].items():
            if sha256_file(package_dir / relative) != digest:
                raise RuntimeError(f'MediaPipe bundled model changed: {relative}')
        import mediapipe as mp
        self.graph = mp.solutions.hands.Hands(**config['detector'])

    def detect(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Expected a nonempty uint8 RGB image')
        result = self.graph.process(np.ascontiguousarray(rgb))
        return np.asarray([[[p.x, p.y, p.z] for p in hand.landmark]
                           for hand in (result.multi_hand_landmarks or [])],
                          dtype=np.float64).reshape(-1, 21, 3)

    def close(self):
        self.graph.close()
