"""63D features shared by ASL-source dataset and future webcam inference.

Source label compatibility with SIBI is unverified. These are not the 132D
features used by the existing SIBI model.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

FEATURE_VERSION = 'wrist-maxdist63-v1'


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized_features(landmarks):
    """Center on wrist; divide all xyz by maximum 3D wrist distance.

    Coordinates retain input orientation. No aspect or handedness correction is
    inferred because source capture dimensions and handedness are unavailable.
    """
    values = np.asarray(landmarks)
    if values.shape not in {(63,), (21, 3)} or not np.issubdtype(values.dtype, np.floating):
        raise ValueError('Expected floating-point (63,) or (21, 3) landmarks')
    points = values.astype(np.float64).reshape(21, 3)
    if not np.isfinite(points).all():
        raise ValueError('Landmarks contain nonfinite coordinates')
    relative = points - points[:1]
    scale = np.linalg.norm(relative, axis=1).max()
    if scale <= 1e-6:
        raise ValueError('Degenerate hand geometry')
    return (relative / scale).reshape(63).astype(np.float32)


def frame_number(relative_path):
    path = PurePosixPath(relative_path)
    if path.is_absolute() or '..' in path.parts or ':' in relative_path or '\\' in relative_path:
        raise ValueError('Unsafe sample path')
    match = re.fullmatch(r'frame_(\d+)\.npy', path.name)
    if not match:
        raise ValueError(f'Unsupported source filename: {path.name}')
    return int(match.group(1))


def split_assignment(index, count, train_fraction=0.7, validation_fraction=0.15, gap=10):
    """Heuristic file-order holdout, not verified time/signer/session separation."""
    if (not 0 <= index < count or gap < 0 or not 0 < train_fraction < 1
            or not 0 < validation_fraction < 1 - train_fraction):
        raise ValueError('Invalid split parameters')
    first = int(count * train_fraction)
    second = int(count * (train_fraction + validation_fraction))
    if index < first - gap:
        return 'train'
    if index < first + gap:
        return 'gap'
    if index < second - gap:
        return 'val'
    if index < second + gap:
        return 'gap'
    return 'test'
