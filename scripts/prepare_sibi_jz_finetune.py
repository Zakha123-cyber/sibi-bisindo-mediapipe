"""Expand ASL training pool while copying baseline validation byte for byte."""
import argparse
from collections import Counter
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.coordinates import coordinate_tensor
from sibi_bisindo.npy_landmarks import sha256_file


def prepare(base, output, root=ROOT):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; baseline is never overwritten')
    old = json.loads((base/'dataset.json').read_text())
    if old['language'] != 'sibi':
        raise ValueError('SIBI only')
    for split in ('train', 'val'):
        if sha256_file(base/f'{split}.npz') != old['package_sha256'][f'{split}.npz']:
            raise ValueError('Baseline package hash mismatch')
    with (base/'manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(base/'train.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ('X', 'labels', 'valid_mask', 'sources', 'paths')}
    additions = []
    for row in rows:
        if row['split'] != 'unused_train' or row['source'] != 'asl_static_pose':
            continue
        if row['label'] not in ('J', 'Z'):
            raise ValueError('Unexpected ASL class')
        path = root/row['id']
        if sha256_file(path) != row['sha256']:
            raise ValueError('ASL training source changed')
        tensor = coordinate_tensor(np.load(path, allow_pickle=False), old['config']['asl_aspect_ratio'])
        additions.append((tensor, row.copy()))
        row.update(split='train', valid='True', reason='expanded_training_pool')
    if not additions:
        raise ValueError('No unused training pool')
    for key, values in [('X', [t for t, _ in additions]), ('labels', [r['label'] for _, r in additions]),
                         ('valid_mask', [True]*len(additions)), ('sources', [r['source'] for _, r in additions]),
                         ('paths', [r['id'] for _, r in additions])]:
        data[key] = np.concatenate([data[key], np.asarray(values)], axis=0)
    if len(set(data['paths'])) != len(data['paths']):
        raise ValueError('Duplicate training paths')
    identity = copy.deepcopy(old)
    identity.update(version='coordinate-jz-ft-v1', parent_dataset_sha256=sha256_file(base/'dataset.json'))
    identity['config'].update(version='coordinate-jz-ft-v1', learning_rate=.0001, epochs=60, patience=12,
                              balanced_sampling=True, asl_train_limit_per_class=272)
    identity['counts']['train'] = {'total': len(data['labels']), 'valid': int(data['valid_mask'].sum()),
                                  'classes_valid': dict(sorted(Counter(data['labels'][data['valid_mask']].tolist()).items()))}
    output.mkdir(parents=True)
    np.savez_compressed(output/'train.npz', **data, metadata_json=json.dumps(identity, sort_keys=True))
    shutil.copyfile(base/'val.npz', output/'val.npz')
    with (output/'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    identity['manifest_sha256'] = sha256_file(output/'manifest.csv')
    identity['package_sha256'] = {f'{s}.npz': sha256_file(output/f'{s}.npz') for s in ('train', 'val')}
    assert identity['package_sha256']['val.npz'] == old['package_sha256']['val.npz']
    (output/'dataset.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps({'version': identity['version'], 'counts': identity['counts'],
                      'validation_unchanged': True, 'test_loaded': False}, indent=2))
    return identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'data/features/coordinate-reset-v1/sibi')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/features/coordinate-jz-ft-v1/sibi')
    args = parser.parse_args()
    prepare(args.base, args.output_dir)
