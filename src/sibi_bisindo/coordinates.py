"""Shared reset experiment: coordinate tensors for offline data and webcam."""
import numpy as np

VERSION = 'two-hand-coordinates-v1'
INPUT_SHAPE = (42, 7)
CHANNELS = ['local_x', 'local_y', 'local_z', 'wrist_x', 'wrist_y', 'palm_ratio', 'present']


def coordinate_tensor(hands, aspect_ratio=1.0):
    """Image-position slots; no inferred anatomical side or inter-hand z depth.

    Empty slots are zeros with present=0. Hands must already be valid detections.
    Aspect ratio is height/width; ASL capture dimensions are unknown (explicit 1).
    """
    points = np.asarray(hands, dtype=np.float64)
    if points.shape == (63,):
        points = points.reshape(1, 21, 3)
    elif points.shape == (21, 3):
        points = points[None]
    if points.ndim != 3 or points.shape[1:] != (21, 3) or not 1 <= len(points) <= 2:
        raise ValueError('Expected one/two hands, each with 21 xyz points')
    if not np.isfinite(points).all() or not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError('Nonfinite coordinates or invalid aspect ratio')
    order = sorted(range(len(points)), key=lambda i: tuple(points[i, 0, :2]))
    points = points[order].copy()
    points[:, :, 1] *= aspect_ratio
    relative = points-points[:, :1]
    scales = np.linalg.norm(relative[:, [5, 9, 13, 17], :2], axis=2).mean(axis=1)
    if (scales <= 1e-6).any():
        raise ValueError('Degenerate palm geometry')
    common = scales.mean()
    wrists = (points[:, 0, :2]-points[:, 0, :2].mean(axis=0))/common
    result = np.zeros((2, 21, 7), dtype=np.float32)
    for slot in range(len(points)):
        result[slot, :, :3] = relative[slot]/scales[slot]
        result[slot, :, 3:5] = wrists[slot]
        result[slot, :, 5] = scales[slot]/common
        result[slot, :, 6] = 1
    return result.reshape(INPUT_SHAPE)


def augment_coordinates(tensors, rng, rotation_degrees=12., noise_std=.008):
    """Small joint rotation/noise; never fabricates an absent hand or mirrors signs."""
    output = np.asarray(tensors, dtype=np.float32).copy()
    for sample in output:
        present = sample[:, 6] == 1
        angle = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for channels in ([0, 1], [3, 4]):
            sample[:, channels] = sample[:, channels]@rotation.T
        noise = rng.normal(0, noise_std, (int(present.sum()), 3))
        sample[present, :3] += noise
        for start in (0, 21):
            if sample[start, 6]:
                sample[start:start+21, :3] -= sample[start, :3].copy()
        sample[~present] = 0
        if sample[0, 6] and sample[21, 6] and tuple(sample[0, 3:5]) > tuple(sample[21, 3:5]):
            sample[:] = np.concatenate([sample[21:].copy(), sample[:21].copy()])
    return output
