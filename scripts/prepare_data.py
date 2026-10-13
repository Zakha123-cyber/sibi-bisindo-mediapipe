"""Build reproducible stage-1 manifests without modifying raw data.

python scripts/prepare_data.py --verify-images
Outputs are locked: changed inputs require a new version/output directory.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    'path', 'language', 'variant', 'label', 'label_source', 'source_group',
    'source_group_basis', 'filename_series', 'signer_id', 'session_id',
    'original_split', 'split', 'include', 'decision', 'exclusion_reason',
    'annotation_class_ids', 'sha256', 'evaluation_status',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(data):
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def normalized_path(value):
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in value:
        raise ValueError(f'Unsafe dataset-relative path: {value}')
    return path.as_posix()


def csv_bytes(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def input_fingerprints(inventory, decisions, audit, label_map):
    """Hash parsed content so CRLF, CSV row order and path separators are harmless."""
    def canonical_rows(rows):
        result = [dict(row, path=normalized_path(row['path'])) for row in rows]
        return sorted(result, key=lambda row:row['path'])
    audit_copy = dict(audit, orphan_labels=sorted(normalized_path(p) for p in audit['orphan_labels']))
    return {name:digest(canonical_json(value)) for name,value in {
        'inventory':canonical_rows(inventory), 'curation_decisions':canonical_rows(decisions),
        'audit_summary':audit_copy, 'label_map':label_map,
    }.items()}


def identify_source(row):
    if row['language'] == 'SIBI':
        core = re.sub(r'^bg_\d+__', '', row['origin'])
        if not re.fullmatch(r'[A-Z]_\d+', core):
            raise ValueError(f'Unknown SIBI origin format: {row["origin"]}')
        return 'sibi:' + core, 'filename_capture_shared_across_variants', ''
    if row['language'] == 'BISINDO':
        match = re.fullmatch(r'([A-Z])_(\d+)(?:_(AR|2|3))?', row['origin'])
        if not match or match[1] != row['letter']:
            raise ValueError(f'Unknown BISINDO origin format: {row["origin"]}')
        series = match[3] or 'plain'
        return 'bisindo:series:' + series, 'unverified_filename_series_not_person_id', series
    raise ValueError(f'Unknown language: {row["language"]}')


def assign_sibi_splits(groups_by_label, config):
    """Hash ordering is deterministic and independent of input file ordering."""
    result = {}
    for label, groups in sorted(groups_by_label.items()):
        ordered = sorted(groups, key=lambda g: (digest(f'{config["seed"]}:{g}'.encode()), g))
        n_val = max(1, int(len(ordered) * config['sibi']['validation_fraction'] + 0.5))
        n_test = max(1, int(len(ordered) * config['sibi']['test_fraction'] + 0.5))
        if len(ordered) - n_val - n_test < 1:
            raise ValueError(f'Insufficient independent source groups for class {label}')
        for i, group in enumerate(ordered):
            result[group] = 'test' if i < n_test else 'val' if i < n_test+n_val else 'train'
    return result


def build_manifest(inventory, decisions, config, label_map, orphan_labels):
    normalized = []
    for original in inventory:
        row = dict(original, path=normalized_path(original['path']))
        group, basis, series = identify_source(row)
        row.update(group=group, basis=basis, series=series)
        normalized.append(row)
    lookup = {r['path']: r for r in normalized}
    if len(lookup) != len(normalized):
        raise ValueError('Duplicate inventory paths')
    orphans = {normalized_path(p) for p in orphan_labels}
    decision_map = {}
    for d in decisions:
        path = normalized_path(d['path'])
        if path in decision_map:
            raise ValueError(f'Duplicate curation decision: {path}')
        if d['action'] != 'exclude':
            raise ValueError('v1 only supports reviewed exclusions; label overrides need an explicit implementation/review')
        if not d['reason']:
            raise ValueError('Every decision requires a reason')
        if d['record_type'] == 'image':
            if path not in lookup or lookup[path]['sha256'] != d['image_sha256']:
                raise ValueError(f'Curation decision does not match image snapshot: {path}')
        elif d['record_type'] == 'orphan_label':
            if path not in orphans:
                raise ValueError(f'Unknown orphan annotation: {path}')
        else:
            raise ValueError('Unknown decision record type')
        decision_map[path] = d
    if not orphans.issubset(decision_map):
        raise ValueError('Unreviewed orphan annotations')
    quarantined = {lookup[p]['group'] for p,d in decision_map.items() if d['record_type'] == 'image'}
    if not config['sibi']['quarantine_affected_source_groups']:
        raise ValueError('v1 requires source-level quarantine')
    sibi_expected = {letter: cls for cls,letter in label_map['sibi']['annotation_id_to_letter'].items()}
    sibi_groups = defaultdict(set)
    source_labels = defaultdict(set)
    mono_sources = Counter()
    multi_sources = defaultdict(set)
    for row in normalized:
        problem = bool(row['error'] or row['annotation_error'])
        if row['language'] == 'SIBI':
            if row['letter'] not in sibi_expected:
                raise ValueError('Unknown SIBI class')
            problem |= row['class_ids'] != sibi_expected[row['letter']]
            source_labels[row['group']].add(row['letter'])
            if row['variant'] == config['sibi']['active_variant']:
                mono_sources[row['group']] += 1
                if row['group'] not in quarantined:
                    sibi_groups[row['letter']].add(row['group'])
            if row['variant'] == 'sibi_alphabet_multi_bg_no_augmentation':
                multi_sources[row['group']].add(row['origin'].split('__')[0])
        elif row['letter'] not in label_map['bisindo']['candidate_letters']:
            raise ValueError('Unknown BISINDO candidate class')
        if problem and row['path'] not in decision_map:
            raise ValueError(f'Unreviewed data problem: {row["path"]}')
    if any(len(labels) != 1 for labels in source_labels.values()):
        raise ValueError('A SIBI source group has conflicting labels')
    if not mono_sources or set(mono_sources) != set(multi_sources):
        raise ValueError('SIBI mono/multi source-key coverage changed; review grouping')
    if any(n != 1 for n in mono_sources.values()) or any(len(bgs) != 5 for bgs in multi_sources.values()):
        raise ValueError('Expected one mono original and five backgrounds per SIBI source')
    assignment = assign_sibi_splits(sibi_groups, config)
    manifest = []
    for row in sorted(normalized, key=lambda r:r['path']):
        reason = ''
        decision = 'keep'
        if row['group'] in quarantined:
            split, reason, decision = 'excluded', 'quarantined_review_source', 'exclude'
        elif row['language'] == 'SIBI':
            split = assignment[row['group']]
            if row['variant'] != config['sibi']['active_variant']:
                reason, decision = 'inactive_variant_reference_only', 'exclude'
        else:
            split = config['bisindo']['filename_series_splits'][row['series']]
        if row['path'] in decision_map:
            reason += '; reviewed_record: ' + decision_map[row['path']]['reason']
        manifest.append(dict(
            path=row['path'], language=row['language'].lower(), variant=row['variant'],
            label=row['letter'], label_source='filename_annotation_crosscheck' if row['language']=='SIBI' else 'class_folder',
            source_group=row['group'], source_group_basis=row['basis'], filename_series=row['series'],
            signer_id='', session_id='', original_split=row['split'], split=split,
            include='0' if reason else '1', decision=decision, exclusion_reason=reason,
            annotation_class_ids=row['class_ids'], sha256=row['sha256'],
            evaluation_status=config[row['language'].lower()]['evaluation_status'],
        ))
    return manifest, sorted(quarantined), len(mono_sources)


def validate_manifest(rows, label_map):
    groups, hashes, hash_labels = defaultdict(set), defaultdict(set), defaultdict(set)
    included_classes = defaultdict(set)
    for row in rows:
        groups[row['source_group']].add(row['split'])
        hashes[row['sha256']].add(row['split'])
        hash_labels[row['sha256']].add((row['language'], row['label']))
        if row['signer_id'] or row['session_id']:
            raise ValueError('v1 must not invent signer/session identities')
        if row['include'] == '1':
            if row['split'] not in {'train','val','test'} or row['exclusion_reason']:
                raise ValueError('Invalid included row')
            if 'with_augmentation' in row['variant']:
                raise ValueError('Pre-generated augmentation is not part of the v1 baseline')
            included_classes[(row['language'],row['split'])].add(row['label'])
    if any(len(s)>1 for s in groups.values()):
        raise ValueError('Source group crosses splits')
    if any(len(s)>1 for s in hashes.values()):
        raise ValueError('Identical image bytes cross splits')
    if any(len(s)>1 for s in hash_labels.values()):
        raise ValueError('Identical image bytes have conflicting labels')
    expected = {'sibi':set(label_map['sibi']['annotation_id_to_letter'].values()),
                'bisindo':set(label_map['bisindo']['candidate_letters'])}
    for language, letters in expected.items():
        for split in ['train','val','test']:
            if included_classes[(language,split)] != letters:
                raise ValueError(f'Missing class coverage for {language}/{split}')
    return dict(source_group_overlap=0, identical_image_overlap=0, conflicting_exact_duplicate_labels=0,
                all_candidate_classes_in_each_split=True, signer_independence_verified=False)


def verify_images(rows):
    def check(row):
        path = ROOT / 'dataset' / row['path']
        actual = digest(path.read_bytes())
        if actual != row['sha256']:
            raise ValueError(f'Image changed since audit: {row["path"]}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(check,rows),1):
            if i % 5000 == 0:
                print(f'Verified image hashes: {i}/{len(rows)}',flush=True)


def enforce_lock(output_dir, lock, latest_report=False):
    path = output_dir / 'split_lock.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        # The tracked report is a latest-version summary. A new version may
        # replace it only after the separate local output-directory lock passes.
        if latest_report and previous['version'] != lock['version']:
            return
        if previous != lock:
            raise ValueError('Locked snapshot changed. Create a new config version AND output directory; do not silently replace test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/data_v1.json')
    parser.add_argument('--verify-images', action='store_true', help='Rehash every raw image against audit inventory')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    label_map = json.loads((ROOT/config['label_map']).read_text(encoding='utf-8'))
    inventory = read_csv(ROOT/config['inventory'])
    decisions = read_csv(ROOT/config['curation_decisions'])
    audit = json.loads((ROOT/config['audit_summary']).read_text(encoding='utf-8'))
    rows, quarantined, source_count = build_manifest(inventory,decisions,config,label_map,audit['orphan_labels'])
    checks = validate_manifest(rows,label_map)
    if args.verify_images:
        verify_images(rows)
        print('All raw-image hashes match the audit inventory.',flush=True)
    else:
        for row in rows:
            if not (ROOT/'dataset'/row['path']).is_file():
                raise ValueError(f'Missing raw image: {row["path"]}')
    files = {'all_samples.csv':csv_bytes(rows)}
    for language in ['sibi','bisindo']:
        for split in ['train','val','test']:
            selected = [r for r in rows if r['include']=='1' and r['language']==language and r['split']==split]
            files[f'{language}_{split}.csv'] = csv_bytes(selected)
    files['excluded.csv'] = csv_bytes([r for r in rows if r['include']=='0'])
    assignment = sorted({(r['source_group'],r['split']) for r in rows})
    lock = dict(version=config['version'], seed=config['seed'],
        input_hash_format='parsed-canonical-json-v1',
        input_sha256=input_fingerprints(inventory,decisions,audit,label_map),
        config_sha256=digest(canonical_json(config)),
        assignment_sha256=digest(canonical_json(assignment)),
        manifests_sha256={name:digest(data) for name,data in sorted(files.items())},
        test_policy='Do not tune against test; no silent split changes; no signer-independent claim.')
    out = ROOT/config['output_dir']
    report = ROOT/config['report_dir']
    enforce_lock(out,lock)
    # The tracked lock also protects a fresh checkout with no local manifests yet.
    enforce_lock(report,lock,latest_report=True)
    counts = defaultdict(Counter)
    per_class = defaultdict(lambda:defaultdict(Counter))
    for row in rows:
        if row['include']=='1':
            counts[row['language']][row['split']] += 1
            per_class[row['language']][row['label']][row['split']] += 1
    summary = dict(version=config['version'], total_images=len(rows),
        included_images=sum(r['include']=='1' for r in rows),
        excluded_images=sum(r['include']=='0' for r in rows),
        quarantine_images=sum(r['split']=='excluded' for r in rows),
        quarantined_source_groups=quarantined, sibi_source_groups_before_quarantine=source_count,
        reviewed_records=len(decisions), orphan_annotations_excluded=len(audit['orphan_labels']),
        counts={k:dict(v) for k,v in counts.items()},
        per_class={k:{c:dict(s) for c,s in v.items()} for k,v in per_class.items()}, checks=checks,
        limitations=['BISINDO suffix groups are unverified filename series, not participant/session IDs.',
                    'SIBI grouping uses common capture keys across variants, supported by structural and sample visual checks.',
                    'Near duplicates and common participants may remain across groups.',
                    'BISINDO classes are candidates pending static-pose validation in stage 2.',
                    'Official SIBI annotation class configuration is not available; mapping is empirically cross-checked.'])
    out.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        (out/name).write_bytes(data)
    (out/'split_lock.json').write_bytes(canonical_json(lock))
    (out/'summary.json').write_bytes(canonical_json(summary))
    report.mkdir(parents=True,exist_ok=True)
    (report/'summary.json').write_bytes(canonical_json(summary))
    (report/'split_lock.json').write_bytes(canonical_json(lock))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_class'},indent=2))


if __name__ == '__main__':
    main()
