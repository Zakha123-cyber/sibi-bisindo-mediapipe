"""Participant-separated webcam pilot data, with complete trial-level provenance."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)+'\n').encode('utf-8')


def sha(content):
    return hashlib.sha256(content).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(config):
    if set(config['languages']) != {'sibi', 'bisindo'}:
        raise ValueError('Pilot requires separate sibi and bisindo modes')
    if config['split_policy'] != 'fixed_participant_role_across_languages_and_sessions':
        raise ValueError('Participant-separated split policy is required')
    for name in list(config['participants']) + config['sessions']:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', name):
            raise ValueError('Unsafe participant/session ID')
    if set(config['participants'].values()) != {'train', 'val', 'test'}:
        raise ValueError('Participant registry must cover train, val and test')
    for settings in config['languages'].values():
        labels = settings['classes']
        if len(labels) != 5 or len(set(labels)) != 5 or any(not re.fullmatch('[A-Z]', x) for x in labels):
            raise ValueError('Each pilot mode must define five unique letters')
    if (min(config['capture_seconds'], config['frame_interval_seconds']) <= 0
            or config['countdown_seconds'] < 0 or config['frame_interval_seconds'] > config['capture_seconds']
            or not 1 <= config['jpeg_quality'] <= 100 or config['saved_images_mirrored'] is not False):
        raise ValueError('Invalid capture settings or saved-image orientation')


def read_pose_review(path, mode, label):
    content = Path(path).read_bytes()
    rows = list(csv.DictReader(content.decode('utf-8-sig').splitlines()))
    matches = [row for row in rows if row['mode'] == mode and row['label'] == label]
    if len(matches) != 1:
        raise ValueError(f'Pose review must contain exactly one entry for {mode}/{label}')
    row = matches[0]
    if row['status'] not in {'pending', 'verified', 'rejected'}:
        raise ValueError('Unsupported pose review status')
    if row['status'] == 'verified' and not all(row[key].strip() for key in ('reference_url', 'reviewer', 'reviewed_on')):
        raise ValueError('Verified poses require a reference, reviewer and review date')
    return dict(row), sha(content)


def lock_collection(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    content = canonical(config)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Collection protocol changed; use a new version/output directory')
    else:
        with path.open('xb') as stream:
            stream.write(content)
    return sha(content)


class TrialWriter:
    def __init__(self, output, config, mode, participant, session, label, sample_type,
                 context, pose_review=None, pose_review_sha256=None):
        validate_config(config)
        if mode not in config['languages'] or participant not in config['participants'] or session not in config['sessions']:
            raise ValueError('Unknown mode, participant or session')
        if sample_type == 'positive':
            if label not in config['languages'][mode]['classes'] or not pose_review or pose_review['status'] == 'rejected':
                raise ValueError('Unsupported/rejected pose or missing pose review')
        elif sample_type == 'negative':
            if label not in config['negative_keys'].values():
                raise ValueError('Unsupported negative type')
        else:
            raise ValueError('Unsupported sample type')
        if context.get('consent_recorded') is not True:
            raise ValueError('Record participant consent before saving camera samples')
        protocol_sha = lock_collection(output, config)
        self.trial_id = uuid.uuid4().hex
        self.directory = Path(output) / mode / participant / session / 'trials' / self.trial_id
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'frames').mkdir()
        self.frames = 0
        self.metadata = {
            'version': config['version'], 'trial_id': self.trial_id, 'mode': mode,
            'participant_id': participant, 'session_id': session,
            'split': config['participants'][participant], 'label': label, 'sample_type': sample_type,
            'status': 'recording', 'frame_count': 0, 'started_at_utc': utc_now(),
            'protocol_sha256': protocol_sha, 'pose_review': pose_review,
            'pose_review_sha256': pose_review_sha256, 'saved_images_mirrored': False,
            'trial_review_status': 'pending', 'context': context,
        }
        self.closed = False
        self._write_metadata()
        self.stream = (self.directory / 'frames.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.stream,
                                     fieldnames=['path', 'elapsed_ms', 'captured_at_utc', 'sha256'], lineterminator='\n')
        self.writer.writeheader()
        self.stream.flush()

    def _write_metadata(self):
        temporary = self.directory / 'trial.json.tmp'
        temporary.write_bytes(canonical(self.metadata))
        temporary.replace(self.directory / 'trial.json')

    def add_frame(self, jpg, elapsed_seconds):
        if self.closed:
            raise ValueError('Trial is already finalized')
        if elapsed_seconds < 0:
            raise ValueError('Frame time must be nonnegative')
        name = f'frames/frame_{self.frames:04d}.jpg'
        (self.directory / name).write_bytes(jpg)
        self.writer.writerow({'path': name, 'elapsed_ms': round(elapsed_seconds*1000, 3),
                              'captured_at_utc': utc_now(), 'sha256': sha(jpg)})
        self.stream.flush()
        self.frames += 1

    def finish(self, completed, reason=None):
        if self.closed:
            return self.metadata
        self.closed = True
        self.stream.close()
        self.metadata.update(status='completed' if completed and self.frames else 'aborted',
                             frame_count=self.frames, ended_at_utc=utc_now(), end_reason=reason,
                             frames_manifest_sha256=sha((self.directory / 'frames.csv').read_bytes()))
        self._write_metadata()
        return self.metadata


def completed_counts(session_dir):
    counts = Counter()
    for path in Path(session_dir).glob('trials/*/trial.json'):
        trial = json.loads(path.read_text(encoding='utf-8'))
        if trial['status'] == 'completed':
            counts[trial['label']] += 1
    return counts
