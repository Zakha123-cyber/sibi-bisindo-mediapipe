"""Compare unused ASL training coordinates with the 42 examples already used."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.coordinates import coordinate_tensor
from sibi_bisindo.npy_landmarks import sha256_file


def audit(features_dir, output_dir, root=ROOT):
    dataset = json.loads((features_dir/'dataset.json').read_text())
    with (features_dir/'manifest.csv').open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    allowed = [r for r in records if r['source'] == 'asl_static_pose' and r['split'] in ('train', 'unused_train')]
    values = {}
    for row in allowed:
        path = root/row['id']
        if sha256_file(path) != row['sha256']:
            raise ValueError('Source NPY changed since baseline preparation')
        values[row['id']] = coordinate_tensor(np.load(path, allow_pickle=False), dataset['config']['asl_aspect_ratio'])[:21, :3]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'classes': {}, 'loaded_splits': ['train', 'unused_train'], 'validation_or_test_coordinates_loaded': False}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    finger_chains = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
    colors = ['#d97706', '#2563eb', '#9333ea', '#059669', '#dc2626']
    for column, label in enumerate(['J', 'Z']):
        selected = [r for r in allowed if r['label'] == label]
        used = np.stack([values[r['id']] for r in selected if r['split'] == 'train'])
        unused = np.stack([values[r['id']] for r in selected if r['split'] == 'unused_train'])
        full = np.concatenate([used, unused])
        distances = np.sqrt(np.mean((unused[:, None]-used[None])**2, axis=(2, 3))).min(axis=1)
        # Median scatter around each group's median quantifies within-group spread,
        # not semantic pose diversity or a calibrated landmark noise tolerance.
        spread = lambda x: float(np.median(np.sqrt(np.mean((x-np.median(x, axis=0))**2, axis=(1, 2)))))
        summary['classes'][label] = dict(used=len(used), unused=len(unused), total=len(full),
                                        used_median_spread_rms=spread(used), full_median_spread_rms=spread(full),
                                        unused_nearest_used_rms_quantiles={str(q): float(np.percentile(distances, q)) for q in [0, 50, 90, 100]})
        for row, points in enumerate([used, full]):
            axis = axes[row, column]
            for sample in points:
                for chain, color in zip(finger_chains, colors):
                    axis.plot(sample[list(chain), 0], sample[list(chain), 1], color=color, alpha=.07, linewidth=.7)
            # One actual example, not an invented average hand.
            center = points[np.argmin(np.mean((points-np.median(points, axis=0))**2, axis=(1, 2)))]
            for chain, color in zip(finger_chains, colors):
                axis.plot(center[list(chain), 0], center[list(chain), 1], color=color, linewidth=2)
            axis.set_xlim(-1.5, 2); axis.set_ylim(1, -2.5); axis.set_aspect('equal'); axis.grid(alpha=.2)
            axis.set_title(f'{label}: '+('baseline' if row == 0 else 'seluruh pool training')+f' ({len(points)} sampel)')
    fig.suptitle('Apakah sisa data training menambah variasi J/Z?\nGaris tipis = sampel nyata, tebal = contoh dekat median; tidak membaca validation/test', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(output_dir/'training_pool_comparison.png', dpi=160); plt.close(fig)
    (output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    audit(ROOT/'data/features/coordinate-reset-v1/sibi', ROOT/'data/reports/sibi-jz-training-pool')
