"""Audit the ASL-source NumPy landmarks without training or changing source data."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'dataset/ASL Dataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/audits/asl-landmarks-v1')
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    paths = sorted((root / 'landmarks').rglob('*.npy'))
    if not paths:
        raise ValueError(f'No NumPy samples under {root / "landmarks"}')
    shapes, dtypes, counts = Counter(), Counter(), Counter()
    invalid, records, arrays = [], [], []
    duplicates = defaultdict(list)
    raw_hashes = hashlib.sha256()
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        raw_hashes.update((relative + '\0' + digest + '\n').encode('utf-8'))
        record = {'path': relative, 'label': path.parent.name, 'sha256': digest,
                  'shape': '', 'dtype': '', 'valid': 0, 'error': ''}
        counts[path.parent.name] += 1
        try:
            a = np.load(io.BytesIO(content), allow_pickle=False)
            shapes[str(a.shape)] += 1
            dtypes[str(a.dtype)] += 1
            record.update(shape=str(a.shape), dtype=str(a.dtype))
            if a.shape not in {(63,), (21, 3)}:
                raise ValueError(f'Unsupported landmark shape: {a.shape}')
            if not np.issubdtype(a.dtype, np.floating) or not np.isfinite(a).all():
                raise ValueError('Expected finite floating-point coordinates')
            a = a.astype(np.float64).reshape(21, 3)
            if np.linalg.norm(a-a[:1], axis=1).max() <= 1e-6:
                raise ValueError('Degenerate hand geometry')
            duplicate_key = hashlib.sha256(a.tobytes()).hexdigest()
            duplicates[duplicate_key].append((path.parent.name, relative))
            arrays.append(a)
            record['valid'] = 1
        except (ValueError, TypeError, OSError, EOFError) as error:
            record['error'] = str(error)
            invalid.append({'path': relative, 'error': str(error)})
        records.append(record)
        if index % 1000 == 0:
            print(f'Audited {index}/{len(paths)} arrays', flush=True)
    x = np.stack(arrays) if arrays else np.empty((0, 21, 3))
    if len(x):
        wrists = np.linalg.norm(x[:, 0], axis=1)
        scales = np.linalg.norm(x-x[:, :1], axis=2).max(axis=1)
        numeric = {
            'coordinate_min_xyz': x.min(axis=(0, 1)).tolist(),
            'coordinate_max_xyz': x.max(axis=(0, 1)).tolist(),
            'wrist_at_zero_count': int((wrists < 1e-7).sum()),
            'wrist_norm_min_median_max': np.percentile(wrists, [0, 50, 100]).tolist(),
            'maximum_wrist_distance_min_median_max': np.percentile(scales, [0, 50, 100]).tolist(),
            'unit_maximum_wrist_distance_count': int(np.isclose(scales, 1.0, atol=1e-5).sum()),
        }
    else:
        numeric = {}
    report = {
        'dataset': 'ASL Alphabet Hand Landmarks - MediaPipe 3D',
        'source_language': 'asl', 'target_sibi_pose_mapping': 'unverified',
        'file_count': len(paths), 'valid_samples': len(arrays),
        'class_counts': dict(sorted(counts.items())),
        'shapes': dict(shapes), 'dtypes': dict(dtypes),
        'invalid_samples': invalid, 'numeric': numeric,
        'exact_numeric_duplicate_groups': sum(len(v) > 1 for v in duplicates.values()),
        'exact_numeric_duplicate_excess_samples': sum(len(v)-1 for v in duplicates.values()),
        'cross_label_exact_duplicate_groups': sum(len(set(t[0] for t in v)) > 1 for v in duplicates.values()),
        'snapshot_sha256': raw_hashes.hexdigest(),
        'source_documentation_claims': {
            'sample_shape': [21, 3], 'centered_at_wrist': True,
            'normalized_by_maximum_wrist_distance': True,
            'single_signer': True, 'static_samples_only': True,
        },
        'limitations': [
            'ASL labels are not verified SIBI labels.',
            'Participant/session metadata and original frame dimensions are unavailable locally.',
            'The frame_<number> naming does not establish independent capture trials.',
            'Landmark arrays alone cannot evaluate hand detector failures on images.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (output / 'inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Reports written to {output}', flush=True)


if __name__ == '__main__':
    main()
