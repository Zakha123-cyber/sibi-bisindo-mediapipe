"""Lock an audited landmark snapshot and prepare only train/validation features."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sibi_bisindo.npy_landmarks import (FEATURE_VERSION, canonical, frame_number,
                                      normalized_features, sha256_file, split_assignment)


def prepare(config, root=ROOT):
    if config['feature_version'] != FEATURE_VERSION or config['feature_dimension'] != 63:
        raise ValueError('Unsupported feature contract')
    if config['split_method'] != 'per_class_numeric_filename_order_holdout_with_gap':
        raise ValueError('Unsupported split method')
    if (config['source_language'], config['target_language'], config['pose_mapping']) != ('asl', 'sibi', 'unverified'):
        raise ValueError('Dataset provenance must remain explicit')
    classes = config['classes']
    if len(classes) != len(set(classes)) or not classes or set(classes) & set(config['excluded_classes']):
        raise ValueError('Invalid or overlapping classes')
    audit_dir = root / config['audit_dir']
    audit = json.loads((audit_dir / 'summary.json').read_text(encoding='utf-8'))
    if audit['snapshot_sha256'] != config['audit_snapshot_sha256'] or audit['invalid_samples']:
        raise ValueError('Audit snapshot differs from config or contains invalid samples')
    with (audit_dir / 'inventory.csv').open(encoding='utf-8', newline='') as stream:
        inventory = list(csv.DictReader(stream))
    snapshot = hashlib.sha256()
    by_class = defaultdict(list)
    for row in inventory:
        number = frame_number(row['path'])
        parts = PurePosixPath(row['path']).parts
        if (len(parts) != 3 or parts[0] != 'landmarks' or parts[1] != row['label']
                or row['valid'] != '1' or row['label'] not in set(classes) | set(config['excluded_classes'])):
            raise ValueError(f'Unexpected audit row: {row["path"]}')
        snapshot.update((row['path'] + '\0' + row['sha256'] + '\n').encode('utf-8'))
        by_class[row['label']].append((number, row))
    if snapshot.hexdigest() != config['audit_snapshot_sha256'] or len(inventory) != audit['file_count']:
        raise ValueError('Audit inventory does not match its snapshot hash')
    rows = []
    for label, numbered in sorted(by_class.items()):
        numbered.sort(key=lambda item: item[0])
        if len({n for n, _ in numbered}) != len(numbered):
            raise ValueError(f'Duplicate frame index in class {label}')
        for index, (number, row) in enumerate(numbered):
            split = ('excluded' if label not in classes else split_assignment(
                index, len(numbered), config['train_fraction'], config['validation_fraction'],
                config['gap_samples_each_side']))
            rows.append({'path': row['path'], 'label': label, 'frame_number': number,
                         'split': split, 'sha256': row['sha256']})
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(classes):
            raise ValueError(f'{split} does not contain every selected class; review gap/split configuration')
    output = root / config['output_dir']
    implementation_path = ROOT / 'src/sibi_bisindo/npy_landmarks.py'
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    manifest_bytes = buffer.getvalue().encode('utf-8')
    identity = {
        'version': config['version'], 'config_sha256': hashlib.sha256(canonical(config)).hexdigest(),
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'feature_implementation_sha256': sha256_file(implementation_path),
        'audit_snapshot_sha256': config['audit_snapshot_sha256'],
    }
    lock_path = output / 'split_lock.json'
    if output.exists() and any(output.iterdir()):
        if not lock_path.exists():
            raise ValueError(f'Incomplete output exists: {output}; use a new directory')
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        if any(lock.get(k) != v for k, v in identity.items()):
            raise ValueError('Locked snapshot changed; create a new config version and output directory')
        saved_config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if hashlib.sha256(canonical(saved_config)).hexdigest() != identity['config_sha256']:
            raise ValueError('Saved cache config changed after lock')
        for filename, digest in lock['files_sha256'].items():
            if sha256_file(output / filename) != digest:
                raise ValueError(f'Locked cache changed: {filename}')
        print(f'Existing locked features verified: {output}', flush=True)
        return lock

    output.mkdir(parents=True, exist_ok=True)
    counts = dict(Counter(row['split'] for row in rows))
    for split in ('train', 'val'):
        selected = [row for row in rows if row['split'] == split]
        features = []
        for index, row in enumerate(selected, 1):
            path = root / config['dataset_root'] / row['path']
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError(f'Source sample differs from audit: {path}')
            features.append(normalized_features(np.load(io.BytesIO(content), allow_pickle=False)))
            if index % 2000 == 0:
                print(f'Prepared {split}: {index}/{len(selected)}', flush=True)
        metadata = dict(identity, split=split, feature_version=FEATURE_VERSION,
                        source_language='asl', target_language='sibi', pose_mapping='unverified')
        np.savez_compressed(output / f'{split}.npz', X=np.stack(features),
                            labels=np.asarray([r['label'] for r in selected]),
                            paths=np.asarray([r['path'] for r in selected]),
                            metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    (output / 'manifest.csv').write_bytes(manifest_bytes)
    lock = dict(identity, classes=classes, source_language='asl', target_language='sibi',
                pose_mapping='unverified', split_counts=counts, test_arrays_loaded=False,
                files_sha256={name: sha256_file(output / name)
                              for name in ('manifest.csv', 'train.npz', 'val.npz')})
    (output / 'config.json').write_bytes(canonical(config))
    lock_path.write_bytes(canonical(lock))
    print(json.dumps(lock, indent=2), flush=True)
    print('Test rows are reserved in manifest.csv; test arrays were not loaded or cached.', flush=True)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/npy_landmarks_v1.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    prepare(config)


if __name__ == '__main__':
    main()
