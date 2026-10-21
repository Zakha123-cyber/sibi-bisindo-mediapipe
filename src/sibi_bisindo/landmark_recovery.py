"""Label-free image views with inverse coordinate mapping; baseline stays intact."""
import time

import cv2
import numpy as np

from .landmarks import Extraction, geometric_features, policy_status


def make_view(rgb, spec):
    height, width = rgb.shape[:2]
    roi = spec.get('roi', [0, 0, 1, 1])
    x0, y0, x1, y1 = [int(round(v*s)) for v, s in zip(roi, [width, height, width, height])]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Empty ROI')
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    angle, scale = spec.get('angle', 0), spec.get('scale', 1.)
    if scale <= 0:
        raise ValueError('View scale must be positive')
    matrix = cv2.getRotationMatrix2D(((cw-1)/2, (ch-1)/2), angle, scale)
    vw = int(np.ceil(abs(matrix[0, 0])*cw+abs(matrix[0, 1])*ch-1e-9))
    vh = int(np.ceil(abs(matrix[1, 0])*cw+abs(matrix[1, 1])*ch-1e-9))
    matrix[:, 2] += [(vw-cw)/2, (vh-ch)/2]
    view = cv2.warpAffine(crop, matrix, (vw, vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inverse = cv2.invertAffineTransform(matrix)
    inverse[:, 2] += [x0, y0]
    if spec.get('clahe'):
        lab = cv2.cvtColor(view, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[:, :, 0])
        view = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.ascontiguousarray(view), inverse


def map_landmarks(raw, view_shape, inverse, original_shape):
    points = np.asarray(raw, dtype=np.float64).copy()
    vh, vw = view_shape[:2]; height, width = original_shape[:2]
    xy = points[:, :, :2]*[vw, vh]
    xy = xy@inverse[:, :2].T+inverse[:, 2]
    points[:, :, :2] = xy/[width, height]
    # Relative z has the scale of normalized x; remove view zoom and width change.
    inverse_scale = np.sqrt(abs(np.linalg.det(inverse[:, :2])))
    points[:, :, 2] *= vw*inverse_scale/width
    return points.astype(np.float32)


def palm_scale(points, aspect):
    delta = points[:, [5, 9, 13, 17], :2]-points[:, :1, :2]
    delta[:, :, 1] *= aspect
    return np.linalg.norm(delta, axis=2).mean(axis=1)


def acceptable(points, original_shape, baseline, config):
    if not np.isfinite(points).all() or not 1 <= len(points) <= 2:
        return False
    margin = config['maximum_landmark_outside_margin']
    if ((points[:, :, :2] < -margin) | (points[:, :, :2] > 1+margin)).any():
        return False
    aspect = original_shape[0]/original_shape[1]
    scales = palm_scale(points, aspect)
    if (scales <= 1e-6).any():
        return False
    if len(points) == 2:
        delta = (points[0, 0, :2]-points[1, 0, :2])*[1, aspect]
        if np.linalg.norm(delta) < config['minimum_wrist_separation_palms']*scales.mean():
            return False
    if len(baseline) == 1:
        # A two-hand recovery must retain the previously detected hand spatially.
        delta = (points[:, :, :2]-baseline[0, :, :2])*[1, aspect]
        distances = np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=1))/scales
        if distances.min() > config['baseline_anchor_rms_palms']:
            return False
    try:
        geometric_features(points, original_shape[1], original_shape[0])
    except ValueError:
        return False
    return True


class RecoveryExtractor:
    def __init__(self, detector, config):
        self.detector, self.config = detector, config

    def extract(self, rgb, baseline=None):
        started = time.perf_counter()
        if baseline is None:
            baseline = self.detector.extract(rgb, 'bisindo', input_mirrored=False)
        original_count = len(baseline.raw)
        details = dict(method='baseline', attempts=0, baseline_hands=original_count, recovered=False)
        if original_count >= 2 or (original_count == 1 and not self.config['recover_one_hand']):
            return baseline, details
        for spec in self.config['views']:
            view, inverse = make_view(rgb, spec)
            result = self.detector.extract(view, 'bisindo', input_mirrored=False)
            details['attempts'] += 1
            if not result.valid or len(result.raw) <= original_count:
                continue
            mapped = map_landmarks(result.raw, view.shape, inverse, rgb.shape)
            if not acceptable(mapped, rgb.shape, baseline.raw, self.config):
                continue
            feature, order = geometric_features(mapped, rgb.shape[1], rgb.shape[0])
            mapped = mapped[order]
            status, valid = policy_status('bisindo', len(mapped), True)
            extraction = Extraction(feature, mapped, [result.handedness[i] for i in order],
                                    [result.handedness_scores[i] for i in order], status, valid,
                                    (time.perf_counter()-started)*1000,
                                    ['recovered_coordinates_require_visual_review'], rgb.shape[1], rgb.shape[0])
            details.update(method=spec['name'], recovered=True)
            return extraction, details
        return baseline, details
