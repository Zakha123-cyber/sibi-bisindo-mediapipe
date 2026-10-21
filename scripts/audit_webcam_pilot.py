"""Audit pilot trial integrity and coverage; render development-only review sheets."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sibi_bisindo.pilot import sha, validate_config


def render_sheet(items, destination, title, temporal=False):
    if not items:
        return
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    columns = 3 if temporal else 5
    tiles = []
    for item in items:
        selected = [0, len(item['frames'])//2, len(item['frames'])-1] if temporal else [len(item['frames'])//2]
        for index in selected:
            tiles.append((item, index))
    rows = math.ceil(len(tiles)/columns)
    sheet = Image.new('RGB', (columns*320, 75+rows*270), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=font, fill='black')
    draw.text((12, 36), 'Label pengguna, belum disahkan. Gambar lokal; test tidak ditampilkan.', font=font, fill='black')
    for offset, (item, frame_index) in enumerate(tiles):
        x, y = (offset % columns)*320+8, (offset//columns)*270+75
        with Image.open(item['frames'][frame_index]) as source:
            tile = ImageOps.contain(source.convert('RGB'), (304, 228))
        sheet.paste(tile, (x+(304-tile.width)//2, y))
        draw.text((x, y+230), f"{item['metadata']['label']} | {item['ordinal']} | frame {frame_index}", font=font, fill='black')
        draw.text((x, y+249), item['metadata']['trial_id'][:10], font=font, fill='black')
    sheet.save(destination, quality=92)


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol_bytes = (root/'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    validate_config(protocol)
    protocol_sha = sha(protocol_bytes)
    records, issues, views = [], [], defaultdict(list)
    coverage = Counter()
    frame_total, decoded, test_decoded = 0, 0, 0
    directories = sorted(root.glob('*/*/*/trials/*/trial.json'))
    seen_ids = set()
    for path in directories:
        t = json.loads(path.read_text(encoding='utf-8'))
        trial_dir, errors = path.parent, []
        mode, participant, session, _, trial_id, _ = path.relative_to(root).parts
        for key, expected in [('mode', mode), ('participant_id', participant), ('session_id', session),
                              ('trial_id', trial_id), ('protocol_sha256', protocol_sha)]:
            if t.get(key) != expected:
                errors.append(f'metadata_mismatch:{key}')
        if trial_id in seen_ids:
            errors.append('duplicate_trial_id')
        seen_ids.add(trial_id)
        expected_split = protocol['participants'].get(participant)
        if expected_split is None or t.get('split') != expected_split:
            errors.append('participant_split_mismatch')
        if session not in protocol['sessions'] or mode not in protocol['languages']:
            errors.append('unknown_session_or_mode')
        valid_labels = (protocol['languages'].get(mode, {}).get('classes', []) if t['sample_type'] == 'positive'
                        else list(protocol['negative_keys'].values()))
        if t['label'] not in valid_labels:
            errors.append('unsupported_label')
        if not t.get('context', {}).get('consent_recorded') or t.get('saved_images_mirrored') is not False:
            errors.append('consent_or_orientation_mismatch')
        manifest_path = trial_dir/'frames.csv'
        manifest_bytes = manifest_path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode('utf-8'))))
        if t['status'] != 'recording':
            if t.get('frames_manifest_sha256') != sha(manifest_bytes):
                errors.append('frame_manifest_hash_mismatch')
            if t.get('frame_count') != len(rows):
                errors.append('frame_count_mismatch')
        if t['status'] == 'completed' and not rows:
            errors.append('completed_without_frames')
        seen_paths, frames, last_elapsed = set(), [], -1.0
        for row in rows:
            relative = PurePosixPath(row['path'])
            if relative.is_absolute() or '..' in relative.parts or ':' in row['path'] or '\\' in row['path']:
                errors.append('unsafe_frame_path')
                continue
            image_path = trial_dir/row['path']
            if row['path'] in seen_paths:
                errors.append('duplicate_frame_path')
            seen_paths.add(row['path'])
            try:
                elapsed = float(row['elapsed_ms'])
                if not math.isfinite(elapsed) or elapsed < last_elapsed or elapsed < 0:
                    errors.append('invalid_frame_time')
                last_elapsed = elapsed
                if elapsed >= protocol['capture_seconds']*1000:
                    errors.append('frame_after_trial_interval')
                content = image_path.read_bytes()
                if sha(content) != row['sha256']:
                    errors.append('image_hash_mismatch')
                # Integrity hashes may be checked for all splits; visual review and
                # image decoding are restricted to development participants.
                if expected_split != 'test':
                    with Image.open(io.BytesIO(content)) as image:
                        image.load()
                        if list(image.size) != t['context']['resolution']:
                            errors.append('resolution_mismatch')
                    decoded += 1
                    frames.append(image_path)
            except (OSError, ValueError) as error:
                errors.append(f'frame_error:{type(error).__name__}')
        actual_files = {p.relative_to(trial_dir).as_posix() for p in (trial_dir/'frames').glob('*.jpg')}
        if actual_files != seen_paths:
            errors.append('orphan_or_missing_images')
        frame_total += len(rows)
        if t['status'] == 'completed':
            coverage[(mode, participant, session, t['label'])] += 1
        record = {key: t.get(key) for key in ('trial_id', 'mode', 'participant_id', 'session_id', 'split',
                                              'label', 'sample_type', 'status', 'trial_review_status')}
        record.update(frame_count=len(rows), pose_review_status=(t.get('pose_review') or {}).get('status', 'not_applicable'),
                      lighting=t['context'].get('lighting'), background=t['context'].get('background'),
                      signing_hand=t['context'].get('signing_hand'), errors=';'.join(sorted(set(errors))))
        records.append(record)
        if errors:
            issues.append({'trial_id': trial_id, 'issues': sorted(set(errors))})
        if not errors and frames and t['status'] == 'completed' and expected_split != 'test':
            views[(mode, participant, session, t['sample_type'])].append({'metadata': t, 'frames': frames})
    targets = []
    for mode in protocol['languages']:
        for participant, split in protocol['participants'].items():
            for session in protocol['sessions']:
                for label in protocol['languages'][mode]['classes']+list(protocol['negative_keys'].values()):
                    target = (protocol['positive_trials_per_class_per_session'] if len(label) == 1
                              else protocol['negative_trials_per_type_per_session'])
                    observed = coverage[(mode, participant, session, label)]
                    targets.append({'mode': mode, 'participant_id': participant, 'session_id': session,
                                    'split': split, 'label': label, 'completed': observed,
                                    'target': target, 'missing': max(0, target-observed)})
    summary = {
        'version': protocol['version'], 'trial_count': len(records), 'frame_count': frame_total,
        'status_counts': dict(Counter(r['status'] for r in records)),
        'trials_by_mode': dict(Counter(r['mode'] for r in records)),
        'frames_by_mode': {mode: sum(r['frame_count'] for r in records if r['mode'] == mode) for mode in protocol['languages']},
        'observed_sessions': sorted({'/'.join((r['mode'], r['participant_id'], r['session_id'])) for r in records}),
        'integrity_issue_count': len(issues), 'integrity_issues': issues,
        'decoded_development_frames': decoded, 'decoded_test_frames': test_decoded,
        'pose_review_counts': dict(Counter(r['pose_review_status'] for r in records)),
        'trial_review_counts': dict(Counter(r['trial_review_status'] for r in records)),
        'metadata_unspecified_counts': {key: sum(r[key] in ('unspecified', None, '') for r in records)
                                       for key in ('lighting', 'background', 'signing_hand')},
        'target_trials': sum(row['target'] for row in targets),
        'missing_target_trials': sum(row['missing'] for row in targets),
        'note': 'Technical integrity and coverage only; does not certify sign labels. Test images are not decoded or rendered.',
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('trials.csv', records), ('coverage.csv', targets)]:
        if rows:
            with (output/name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
    for (mode, participant, session, kind), items in views.items():
        items.sort(key=lambda item: (item['metadata']['label'], item['metadata']['started_at_utc']))
        for index, item in enumerate(items, 1):
            item['ordinal'] = index
        name = f'{mode}_{participant}_{session}_{kind}.jpg'
        render_sheet(items, output/name, f'{mode.upper()} {participant}/{session} | {kind}', temporal=kind=='negative')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'data/webcam_pilot_v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/pilot_audit/v1')
    args = parser.parse_args()
    audit(args.root, args.output_dir)


if __name__ == '__main__':
    main()
