"""Load a reset experiment selection and predict from valid raw landmarks."""
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch

from .coordinate_model import CoordinateNet
from .coordinates import VERSION, coordinate_tensor
from .npy_landmarks import sha256_file, canonical


def verify_detector_config(config_path, extraction_metadata):
    """Match the canonical JSON hash used by extract_landmarks, not file bytes."""
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(config)).hexdigest() != extraction_metadata['config_sha256']:
        raise ValueError('Detector config changed since extraction')
    return config


class CoordinatePredictor:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.selection = json.loads((self.run_dir/'selection.json').read_text())
        if self.selection['feature_version'] != VERSION:
            raise ValueError('Unsupported coordinate model')
        local = Path(__file__).parent
        for file, key in [('coordinates.py', 'coordinate_implementation_sha256'),
                          ('coordinate_model.py', 'model_implementation_sha256')]:
            if sha256_file(local/file) != self.selection[key]:
                raise ValueError(f'Implementation changed since training: {file}')
        for name, digest in self.selection['artifact_sha256'].items():
            if Path(name).name != name or sha256_file(self.run_dir/name) != digest:
                raise ValueError('Selected artifact path/hash mismatch')
        self.classes = self.selection['classes']
        selected = self.selection['selected']
        checkpoint = torch.load(self.run_dir/selected['checkpoint'], map_location='cpu', weights_only=True)
        if (checkpoint['classes'] != self.classes or checkpoint['language'] != self.selection['language']
                or checkpoint['kind'] != selected['kind'] or checkpoint['feature_version'] != VERSION):
            raise ValueError('Checkpoint metadata mismatch')
        self.model = CoordinateNet(len(self.classes), selected['kind'])
        self.model.load_state_dict(checkpoint['state_dict']); self.model.eval()
        self.svm = joblib.load(self.run_dir/selected['svm']) if selected['svm'] else None

    def predict(self, hands, aspect_ratio):
        if self.selection['language'] == 'sibi' and len(hands) != 1:
            raise ValueError('SIBI reset requires exactly one detected hand')
        tensor = coordinate_tensor(hands, aspect_ratio)
        x = torch.from_numpy(tensor[None])
        with torch.no_grad():
            if self.svm is None:
                scores = torch.softmax(self.model(x), dim=1).numpy()[0]
                index = int(scores.argmax())
                return {'label': self.classes[index], 'score': float(scores[index]), 'score_type': 'softmax_uncalibrated'}
            embedding = self.model.embed(x).numpy()
        label = str(self.svm.predict(embedding)[0])
        scores = np.sort(np.asarray(self.svm.decision_function(embedding)).reshape(-1))
        return {'label': label, 'score': float(scores[-1]-scores[-2]), 'score_type': 'svm_margin_not_probability'}
