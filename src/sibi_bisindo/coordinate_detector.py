"""Construct the extraction policy recorded by a coordinate training run."""
import json
from pathlib import Path

from .coordinate_inference import verify_detector_config
from .landmarks import HandExtractor
from .npy_landmarks import sha256_file


def build_detector(dataset, selection, root, run_dir):
    language = selection['language']
    policy = dataset.get('runtime_preprocessing')
    expected = selection.get('runtime_preprocessing_sha256')
    if bool(policy) != bool(expected) or (policy and policy.get('sha256') != expected):
        raise ValueError('Model and dataset recovery policies differ')
    if selection.get('dataset_sha256') and sha256_file(Path(run_dir)/'dataset.json') != selection['dataset_sha256']:
        raise ValueError('Run dataset metadata changed after training')
    lock = dataset['extraction_metadata']['train']
    if policy:
        from .recovery_runtime import RuntimeRecoveryExtractor, verify_policy
        from .npy_landmarks import canonical
        import hashlib
        verify_policy(policy)
        if language != 'bisindo' or any(dataset['extraction_metadata'][s].get('runtime_preprocessing') != policy
                                         for s in ('train', 'val')):
            raise ValueError('Training/validation recovery policy mismatch')
        config = policy['base_config']
        if hashlib.sha256(canonical(config)).hexdigest() != lock['config_sha256']:
            raise ValueError('Recovery base config differs from extraction')
    else:
        name = 'landmarks_v1.json' if language == 'sibi' else 'landmarks_bisindo_kaggle_v2_conf03.json'
        config = verify_detector_config(Path(root)/'configs'/name, lock)
    if lock['feature_implementation_sha256'] != sha256_file(Path(__file__).parent/'landmarks.py'):
        raise ValueError('Detector implementation changed since extraction')
    detector = HandExtractor(config)
    if policy:
        try:
            return RuntimeRecoveryExtractor(detector, policy)
        except Exception:
            detector.close()
            raise
    return detector
