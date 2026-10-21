"""Small sequence-template experiment for J/Z; scores are not probabilities."""
import numpy as np

VERSION = 'sibi-jz-template-v1'


def sequence_features(times, hands, aspect, steps=32):
    times, hands = np.asarray(times, float), np.asarray(hands, float)
    if hands.shape != (len(times), 21, 3) or len(times) < 6:
        raise ValueError('Need at least six single-hand observations')
    if not np.isfinite(hands).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid sequence coordinates or timestamps')
    if times[-1]-times[0] < .4 or np.max(np.diff(times)) > .3:
        raise ValueError('Sequence too short or landmark gap exceeds 0.3 seconds')
    points = hands.copy()
    points[:, :, 1] *= aspect
    relative = points-points[:, :1]
    scales = np.linalg.norm(relative[:, [5, 9, 13, 17], :2], axis=2).mean(axis=1)
    scale = float(np.median(scales))
    if scale < 1e-6 or (scales < 1e-6).any():
        raise ValueError('Degenerate hand geometry')
    # Keep wrist movement; centering each frame alone would erase the trajectory.
    local = (relative/scales[:, None, None]).reshape(len(times), -1)
    trajectory = (points[:, [0, 8, 20], :2]-points[0, 0, :2])/scale
    trajectory -= trajectory[:1]
    raw = np.concatenate([local*.25, trajectory.reshape(len(times), -1)], axis=1)
    grid = np.linspace(times[0], times[-1], steps)
    features = np.stack([np.interp(grid, times, raw[:, i]) for i in range(raw.shape[1])], axis=1)
    motion = float(np.max(np.linalg.norm(trajectory-trajectory[:1], axis=2)))
    return features.astype(np.float32), motion


def match(features, templates, labels):
    distances = np.sqrt(np.mean((np.asarray(templates)-features)**2, axis=(1, 2)))
    scores = {label: float(distances[np.asarray(labels) == label].min()) for label in sorted(set(labels))}
    ordered = sorted(scores, key=scores.get)
    return ordered[0], scores[ordered[0]], scores[ordered[1]]-scores[ordered[0]]


class MotionRecognizer:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as bundle:
            if str(bundle['version']) != VERSION:
                raise ValueError('Unsupported temporal artifact')
            self.templates = bundle['templates'].copy()
            self.labels = bundle['labels'].copy()
            self.threshold = float(bundle['distance_threshold'])
        self.samples = []
        self.last_check = -float('inf')
        self.last_event = -float('inf')

    def update(self, timestamp, hands, aspect):
        if len(hands) != 1:
            self.samples.clear()
            return None
        if self.samples and timestamp-self.samples[-1][0] > .3:
            self.samples.clear()
        self.samples.append((timestamp, np.asarray(hands[0]).copy()))
        self.samples = [(t, h) for t, h in self.samples if timestamp-t <= 2.5]
        if timestamp-self.last_check < .2 or timestamp-self.last_event < 1.:
            return None
        self.last_check = timestamp
        candidates = []
        for duration in (.8, 1.2, 1.6, 2., 2.4):
            selected = [(t, h) for t, h in self.samples if timestamp-t <= duration]
            if not selected or selected[-1][0]-selected[0][0] < duration*.8:
                continue
            try:
                features, motion = sequence_features(*zip(*selected), aspect)
            except ValueError:
                continue
            label, distance, margin = match(features, self.templates, self.labels)
            # Initial heuristic gates, not calibrated without non-J/Z sequences.
            if motion >= .5 and distance <= self.threshold and margin >= .03:
                candidates.append((distance, label))
        if candidates:
            self.last_event = timestamp
            distance, label = min(candidates)
            self.samples.clear()
            return {'label': label, 'distance': distance}
        return None
