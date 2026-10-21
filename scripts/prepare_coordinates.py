"""Prepare reset coordinate datasets from existing caches; never load test arrays."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.coordinates import VERSION, coordinate_tensor
from sibi_bisindo.npy_landmarks import frame_number, split_assignment, sha256_file


def prepare(config, language, root=ROOT):
    if config['feature_version'] != VERSION or language not in ('sibi', 'bisindo'):
        raise ValueError('Unknown feature version/language')
    output = root/config['output_dir']/language
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output already exists; use a new version/output_dir: {output}')
    cache = root/config[f'{language}_cache']
    classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    records, data, source_hashes, cache_metadata = [], defaultdict(list), {}, {}
    runtime_policy = None
    if config.get('runtime_policy'):
        if language != 'bisindo':
            raise ValueError('Recovery policy only supports BISINDO')
        from sibi_bisindo.recovery_runtime import verify_policy
        runtime_policy = verify_policy(json.loads((root/config['runtime_policy']).read_text()))
    seen_hashes, seen_groups = {}, {}
    for split in ('train', 'val'):
        path = cache/f'{language}_{split}.npz'
        source_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ('processed_count', 'labels', 'metadata_json', 'paths',
                                                   'sha256', 'source_groups', 'status', 'valid_mask',
                                                   'raw_landmarks', 'height', 'width')}
            if int(arrays['processed_count']) != len(arrays['labels']):
                raise ValueError('Incomplete extraction cache')
            meta = json.loads(str(arrays['metadata_json']))
            if meta['language'] != language or meta['split'] != split:
                raise ValueError('Cache metadata mismatch')
            cache_metadata[split] = meta
            if meta.get('runtime_preprocessing') != runtime_policy:
                raise ValueError('Cache runtime policy differs from preparation config')
            if split == 'val' and any(meta.get(key) != cache_metadata['train'].get(key) for key in
                                       ('config_sha256', 'feature_implementation_sha256', 'feature_version')):
                raise ValueError('Train/validation detector provenance differs')
            for i, label in enumerate(arrays['labels'].astype(str)):
                source = f'{language}_mediapipe'
                identifier = str(arrays['paths'][i])
                digest, group = str(arrays['sha256'][i]), str(arrays['source_groups'][i])
                for seen, key in ((seen_hashes, digest), (seen_groups, group)):
                    if key in seen and seen[key] != split:
                        raise ValueError('Train/validation hash or source-group overlap')
                    seen[key] = split
                row = dict(id=identifier, label=label, split=split, source=source,
                           sha256=digest, group=group, valid=False, reason=str(arrays['status'][i]))
                tensor = np.zeros((42, 7), dtype=np.float32)
                if arrays['valid_mask'][i]:
                    hands = arrays['raw_landmarks'][i]
                    hands = hands[np.isfinite(hands).all(axis=(1, 2))]
                    try:
                        tensor = coordinate_tensor(hands, float(arrays['height'][i])/int(arrays['width'][i]))
                        row.update(valid=True, reason='ok')
                    except ValueError as error:
                        row['reason'] = str(error)
                records.append(row)
                data[split].append((tensor, row))
    if language == 'sibi':
        asl_records = []
        if config['asl_labels'] != ['J', 'Z']:
            raise ValueError('Only ASL J/Z are approved additions to SIBI reset')
        for label in config['asl_labels']:
            paths = sorted((root/config['asl_root']/label).glob('*.npy'), key=lambda p: frame_number(p.name))
            if not paths:
                raise ValueError(f'Missing ASL {label}')
            for index, path in enumerate(paths):
                split = split_assignment(index, len(paths), config['asl_train_fraction'],
                                         config['asl_val_fraction'], config['asl_gap'])
                asl_records.append(dict(id=path.relative_to(root).as_posix(), label=label, split=split,
                                        source='asl_static_pose', sha256=sha256_file(path),
                                        group=f'asl:{label}:{path.stem}', valid=False,
                                        reason='not_loaded' if split != 'gap' else 'split_boundary_gap'))
        grouped = defaultdict(list)
        for row in asl_records:
            grouped[row['sha256']].append(row)
        for items in grouped.values():
            if len({r['label'] for r in items}) > 1:
                raise ValueError('Conflicting ASL duplicate labels')
            if len({r['split'] for r in items}) > 1:
                for row in items:
                    row.update(split='gap', reason='exact_duplicate_cross_split')
            else:
                for row in items[1:]:
                    row.update(split='gap', reason='exact_duplicate')
        for label in config['asl_labels']:
            eligible = [r for r in asl_records if r['label'] == label and r['split'] == 'train']
            limit = min(len(eligible), config['asl_train_limit_per_class'])
            keep = set(np.linspace(0, len(eligible)-1, limit, dtype=int).tolist())
            for index, row in enumerate(eligible):
                if index not in keep:
                    row.update(split='unused_train', reason='training_class_balance_cap')
        for row in asl_records:
            if row['split'] in ('train', 'val'):
                # Held-out test/gap coordinates are never loaded or packaged for training.
                tensor = coordinate_tensor(np.load(root/row['id'], allow_pickle=False), config['asl_aspect_ratio'])
                row.update(valid=True, reason='ok')
                data[row['split']].append((tensor, row))
            records.append(row)
    for split in ('train', 'val'):
        available = {r['label'] for _, r in data[split] if r['valid']}
        if available != set(classes):
            raise ValueError(f'Missing valid classes in {split}: {set(classes)-available}')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(records)
    manifest_bytes = buffer.getvalue().encode()
    identity = dict(version=config['version'], language=language, feature_version=VERSION, classes=classes,
                    input_shape=[42, 7], sources=source_hashes, config=config, extraction_metadata=cache_metadata,
                    coordinate_implementation_sha256=sha256_file(ROOT/'src/sibi_bisindo/coordinates.py'),
                    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    limitations=['Old SIBI test already seen historically; not a fresh independent test.',
                                 'ASL J/Z are accepted static-pose candidates, not movement recognition; source confounded with labels.',
                                 'ASL capture aspect ratio unknown; uses explicit config assumption. Signer independence unverified.'])
    if language == 'bisindo':
        identity['limitations'] = ['Signer independence is unverified.',
                                   'Valid coordinates do not certify accurate landmarks or a complete two-hand pose.',
                                   'Test arrays were not loaded; no unknown/background class.']
    if runtime_policy is not None:
        identity['runtime_preprocessing'] = runtime_policy
        identity['limitations'].append('Recovery uses multiple image views; webcam latency must be measured.')
    output.mkdir(parents=True)
    (output/'manifest.csv').write_bytes(manifest_bytes)
    for split in ('train', 'val'):
        items = data[split]
        np.savez_compressed(output/f'{split}.npz', X=np.stack([x for x, _ in items]),
                            labels=[r['label'] for _, r in items], valid_mask=[r['valid'] for _, r in items],
                            sources=[r['source'] for _, r in items], paths=[r['id'] for _, r in items],
                            metadata_json=json.dumps(identity, sort_keys=True))
    identity['counts'] = {s: {'total': len(items), 'valid': sum(r['valid'] for _, r in items),
                              'classes_valid': dict(sorted(Counter(r['label'] for _, r in items if r['valid']).items()))}
                          for s, items in data.items()}
    identity['package_sha256'] = {f'{s}.npz': sha256_file(output/f'{s}.npz') for s in data}
    (output/'dataset.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(json.dumps({'language': language, 'counts': identity['counts'], 'output': str(output)}, indent=2))
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/coordinate_reset_v1.json')
    parser.add_argument('--languages', nargs='+', choices=['sibi', 'bisindo'], default=['sibi'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for language in args.languages:
        prepare(config, language)


if __name__ == '__main__':
    main()
