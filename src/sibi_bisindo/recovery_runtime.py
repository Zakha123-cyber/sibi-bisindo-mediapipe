"""Versioned, label-free recovery shared by dataset cleanup and webcam."""
import hashlib
import time
from pathlib import Path

import numpy as np

from .landmark_recovery import RecoveryExtractor, palm_scale
from .recovery_quality import duplicate_distance
from .npy_landmarks import canonical, sha256_file

IMPLEMENTATIONS = ('landmarks.py', 'landmark_recovery.py', 'recovery_quality.py', 'recovery_runtime.py')


def point_set_distance(hands, aspect):
    """Symmetric nearest-point distance, independent of finger numbering."""
    points = np.asarray(hands)
    if len(points) != 2:
        return float('inf')
    scale = palm_scale(points, aspect).mean()
    if scale <= 1e-6 or not np.isfinite(points).all():
        return 0.
    xy = points[:, :, :2]*[1, aspect]
    distances = np.linalg.norm(xy[0, :, None]-xy[1, None, :], axis=-1)/scale
    return float((distances.min(0).mean()+distances.min(1).mean())/2)


def suspicious_pair(hands, aspect, quality):
    return (duplicate_distance(hands, aspect) < quality['minimum_corresponding_distance_palms']
            or point_set_distance(hands, aspect) < quality['minimum_point_set_distance_palms'])


def make_policy(base_config, recovery_config, quality):
    local = Path(__file__).parent
    policy = dict(version='bisindo-runtime-recovery-v2', language='bisindo',
                  base_config=base_config, recovery_config=recovery_config, quality=quality,
                  implementations={name: sha256_file(local/name) for name in IMPLEMENTATIONS},
                  manual_image_overrides=False, baseline_two_hands='preserve')
    policy['sha256'] = hashlib.sha256(canonical(policy)).hexdigest()
    return policy


def verify_policy(policy):
    unsigned = {k: v for k, v in policy.items() if k != 'sha256'}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != policy.get('sha256'):
        raise ValueError('Recovery policy identity changed')
    if (policy.get('version') != 'bisindo-runtime-recovery-v2' or policy.get('language') != 'bisindo'
            or policy.get('manual_image_overrides') is not False
            or policy.get('baseline_two_hands') != 'preserve'):
        raise ValueError('Unsupported runtime recovery policy')
    expected = {name: sha256_file(Path(__file__).parent/name) for name in IMPLEMENTATIONS}
    if policy.get('implementations') != expected:
        raise ValueError('Recovery implementation changed since extraction')
    return policy


class RuntimeRecoveryExtractor:
    def __init__(self, detector, policy):
        self.policy = verify_policy(policy)
        self.detector = detector
        quality = policy['quality']

        class CheckedViews:
            def extract(self, rgb, *args, **kwargs):
                result = detector.extract(rgb, *args, **kwargs)
                if suspicious_pair(result.raw, rgb.shape[0]/rgb.shape[1], quality):
                    result.valid = False
                    result.features = None
                return result

        self.recovery = RecoveryExtractor(CheckedViews(), policy['recovery_config'])
        self.last_details = None

    def extract(self, rgb, language='bisindo', input_mirrored=False, baseline=None):
        if language != 'bisindo' or input_mirrored:
            raise ValueError('BISINDO recovery requires unmirrored RGB')
        started = time.perf_counter()
        if baseline is None:
            baseline = self.detector.extract(rgb, language, input_mirrored=False)
        result, details = self.recovery.extract(rgb, baseline)
        # Baseline timing also exists in cached results; measure this call, not old timing.
        from dataclasses import replace
        result = replace(result, process_ms=(time.perf_counter()-started)*1000)
        self.last_details = details
        return result

    def close(self):
        self.detector.close()
