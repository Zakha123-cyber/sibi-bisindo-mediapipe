"""Reject near-duplicate landmark sets on one physical hand during fallback."""
import numpy as np

from .landmark_recovery import RecoveryExtractor, palm_scale


def duplicate_distance(hands, aspect):
    points = np.asarray(hands)
    if len(points) != 2:
        return float('inf')
    scales = palm_scale(points, aspect)
    if min(scales) <= 1e-6:
        return 0.
    delta = (points[0,:,:2]-points[1,:,:2])*[1, aspect]
    return float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))/scales.mean())


class QualityRecoveryExtractor:
    def __init__(self, detector, recovery_config, minimum_separation=.85):
        self.detector = detector
        owner = self
        class ViewDetector:
            def extract(self, rgb, *args, **kwargs):
                result = detector.extract(rgb, *args, **kwargs)
                if duplicate_distance(result.raw, rgb.shape[0]/rgb.shape[1]) < minimum_separation:
                    result.valid = False; result.features = None
                return result
        self.recovery = RecoveryExtractor(ViewDetector(), recovery_config)

    def extract(self, rgb, baseline=None):
        if baseline is None:
            baseline = self.detector.extract(rgb, 'bisindo', input_mirrored=False)
        return self.recovery.extract(rgb, baseline)
