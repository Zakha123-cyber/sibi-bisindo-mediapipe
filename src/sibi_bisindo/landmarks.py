"""Versioned preprocessing shared by offline extraction and future inference.

Only geometry checks are implemented: valid features do not certify a correct
sign, complete two-hand pose, or that all detected hands belong to one actor.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import time

import numpy as np

FEATURE_DIMENSION = 132
MCP_INDICES = [5, 9, 13, 17]


def feature_names():
    return (['slot0_present', 'slot1_present'] +
            [f'slot{s}_landmark{i}_{axis}' for s in range(2) for i in range(21) for axis in 'xyz'] +
            ['wrist_delta_x', 'wrist_delta_y', 'slot0_palm_scale_ratio', 'slot1_palm_scale_ratio'])


def preprocess_rgb(rgb, config, input_mirrored=False):
    """Canonical unmirrored RGB; webcam preview mirroring is a separate UI choice."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Expected a nonempty HxWx3 uint8 RGB image')
    if input_mirrored:
        image = image[:, ::-1, :]
    image = np.ascontiguousarray(image)
    contrast = config['preprocessing']['contrast']
    if contrast == 'clahe':
        import cv2
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif contrast != 'none':
        raise ValueError('Unknown contrast preprocessing')
    return image


def geometric_features(hands, width, height, minimum_scale=1e-6):
    """Return a 132-vector and the raw-hand permutation, or None for no hands.

    Slots denote image position, never anatomical Left/Right. z is local to each
    hand; inter-hand depth is deliberately not fabricated from separate origins.
    """
    raw = np.asarray(hands, dtype=np.float64)
    if raw.size == 0:
        return None, []
    if raw.ndim != 3 or raw.shape[1:] != (21, 3) or not 1 <= len(raw) <= 2:
        raise ValueError('Expected one or two hands with 21 xyz landmarks each')
    if width <= 0 or height <= 0 or not np.isfinite(raw).all():
        raise ValueError('Invalid image dimensions or nonfinite landmarks')
    order = sorted(range(len(raw)), key=lambda i:(raw[i,0,0],raw[i,0,1],raw[i,9,0],raw[i,9,1]))
    points = raw[order].copy()
    points[:, :, 1] *= height / width
    relative = points - points[:, :1, :]
    scales = np.linalg.norm(relative[:, MCP_INDICES, :2], axis=2).mean(axis=1)
    if (scales <= minimum_scale).any():
        raise ValueError('Degenerate palm geometry')
    local = np.zeros((2,21,3), dtype=np.float64)
    local[:len(points)] = relative / scales[:,None,None]
    presence = np.array([1, int(len(points)==2)], dtype=np.float64)
    pair_delta = np.zeros(2)
    ratios = np.array([1.,0.])
    if len(points)==2:
        common_scale = scales.mean()
        pair_delta = (points[1,0,:2]-points[0,0,:2]) / common_scale
        ratios = scales/common_scale
    features = np.concatenate([presence,local.reshape(-1),pair_delta,ratios]).astype(np.float32)
    if features.shape != (FEATURE_DIMENSION,) or not np.isfinite(features).all():
        raise ValueError('Invalid derived feature vector')
    return features, order


def policy_status(language, count, geometry_valid):
    if count == 0:
        return 'no_hand', False
    if not geometry_valid:
        return 'invalid_geometry', False
    if language == 'sibi':
        return ('ok',True) if count == 1 else ('unexpected_extra_hand',False)
    if language == 'bisindo':
        # The extractor cannot infer pose completeness from the ground-truth
        # letter, because that label does not exist at webcam inference time.
        return ('one_hand_completeness_unverified',True) if count == 1 else ('two_hands_actor_unverified',True)
    raise ValueError('Unknown language')


@dataclass
class Extraction:
    features: object
    raw: np.ndarray
    handedness: list
    handedness_scores: list
    status: str
    valid: bool
    process_ms: float
    warnings: list
    width: int
    height: int


class HandExtractor:
    """One independent static-image graph per extractor; close it after use."""
    def __init__(self, config):
        self.config = config
        if config['api'] != 'mediapipe.solutions.hands.Hands':
            raise ValueError('Unsupported detector API')
        if config['detector']['static_image_mode'] is not True:
            raise ValueError('v1 uses static mode for both offline images and initial service frames')
        if config['detector']['max_num_hands'] != 2 or config['feature_dimension'] != FEATURE_DIMENSION:
            raise ValueError('Unsupported v1 geometry schema')
        if config['preprocessing']['color'] != 'RGB' or config['preprocessing']['resize'] != 'none':
            raise ValueError('Unsupported preprocessing schema')
        if config['feature_policy']['minimum_palm_scale'] != 1e-6:
            raise ValueError('Changing geometry scale requires a new feature implementation/version')
        if config['feature_policy']['slot_order'] != 'ascending_image_wrist_x_then_y':
            raise ValueError('Unsupported slot ordering')
        policy = config['feature_policy']
        fixed_policy = {
            'coordinates': 'x_y_corrected_for_aspect_ratio_and_relative_z',
            'normalization': 'subtract_each_wrist_divide_by_mean_2d_wrist_to_MCP_length',
            'retain_orientation': True,
            'pair_features': 'relative_wrist_xy_and_palm_scale_ratios',
            'handedness_is_feature': False,
            'no_hand_features': 'NaN_with_valid_mask_false',
            'sibi_required_hands': 1,
            'bisindo_required_hands': None,
        }
        if any(policy.get(key) != value for key, value in fixed_policy.items()):
            raise ValueError('Feature policy differs from implemented hand-geometry-v1 schema')
        prep = config['preprocessing']
        if (prep.get('exif_transpose') is not True or prep.get('offline_input_mirrored') is not False
                or prep.get('contrast') not in {'none', 'clahe'}):
            raise ValueError('Unsupported orientation or contrast preprocessing')
        self.versions = {}
        for package,expected in config['dependencies'].items():
            installed = importlib.metadata.version(package)
            if installed != expected:
                raise RuntimeError(f'{package}: expected {expected}, found {installed}. Use requirements-features.txt.')
            self.versions[package] = installed
        package_dir = Path(importlib.util.find_spec('mediapipe').origin).parent
        self.model_hashes = {}
        for relative, expected in config['bundled_model_sha256'].items():
            actual = hashlib.sha256((package_dir/relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Bundled model differs from pinned snapshot: {relative}')
            self.model_hashes[relative] = actual
        import mediapipe as mp
        self.detector = mp.solutions.hands.Hands(**config['detector'])

    def close(self):
        self.detector.close()

    def extract(self, rgb, language, input_mirrored=False):
        start = time.perf_counter()
        image = preprocess_rgb(rgb,self.config,input_mirrored=input_mirrored)
        height,width = image.shape[:2]
        result = self.detector.process(image)
        raw = np.array([[[p.x,p.y,p.z] for p in hand.landmark]
                        for hand in (result.multi_hand_landmarks or [])],dtype=np.float32).reshape(-1,21,3)
        handedness = [h.classification[0].label for h in result.multi_handedness or []]
        scores = [h.classification[0].score for h in result.multi_handedness or []]
        warnings = []
        try:
            features,order = geometric_features(raw,width,height)
        except ValueError as exc:
            features,order = None,list(range(len(raw)))
            warnings.append(str(exc))
        if raw.size and ((raw[:,:,:2]<0).any() or (raw[:,:,:2]>1).any()):
            warnings.append('landmarks_outside_image_review_required')
        status,valid = policy_status(language,len(raw),features is not None)
        # Preserve raw detections even if policy rejects the frame. Invalid
        # feature rows are explicitly NaN in storage, never labeled zero vectors.
        return Extraction(features if valid else None,raw[order],
            [handedness[i] if i<len(handedness) else '' for i in order],
            [scores[i] if i<len(scores) else float('nan') for i in order],
            status,valid,(time.perf_counter()-start)*1000,warnings,width,height)
