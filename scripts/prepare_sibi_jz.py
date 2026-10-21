"""Extract timestamped landmarks from unique J/Z videos, preserving failed frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.landmarks import HandExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/features/sibi-jz-v1')
    args = parser.parse_args()
    config_path = ROOT/'configs/landmarks_v1.json'
    config = json.loads(config_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen, records = {}, []
    try:
        for label in ('J', 'Z'):
            for path in sorted((ROOT/'dataset'/label).glob('*')):
                if path.suffix.lower() not in ('.mp4', '.avi', '.mov'):
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                record = {'source': path.relative_to(ROOT).as_posix(), 'label': label, 'sha256': digest}
                if digest in seen:
                    record.update(status='duplicate', duplicate_of=seen[digest])
                    records.append(record)
                    continue
                seen[digest] = record['source']
                cap = cv2.VideoCapture(str(path))
                extractor = None
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not cap.isOpened() or fps <= 0:
                        raise ValueError(f'Cannot read video {path}')
                    extractor = HandExtractor(config)
                    timestamps, hands, statuses = [], [], []
                    index = 0
                    width = height = None
                    while True:
                        ok, bgr = cap.read()
                        if not ok:
                            break
                        height, width = bgr.shape[:2]
                        # Sample using physical time, consistently across 25/50 FPS sources.
                        if index == 0 or index/fps-timestamps[-1] >= 1/15-1e-6:
                            result = extractor.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 'sibi')
                            timestamps.append(index/fps)
                            hands.append(result.raw[0] if result.valid else np.full((21, 3), np.nan))
                            statuses.append(result.status)
                        index += 1
                    if not timestamps:
                        raise ValueError(f'Empty video {path}')
                    target = f'{path.stem}_{digest[:12]}.npz'
                    np.savez_compressed(args.output_dir/target, times=timestamps, hands=hands,
                                        statuses=statuses, aspect=height/width)
                    valid = int(np.isfinite(np.asarray(hands)).all(axis=(1, 2)).sum())
                    record.update(status='extracted', file=target, sampled_frames=len(timestamps),
                                  valid_frames=valid, fps=fps, duration_seconds=index/fps)
                    print(f'{path.stem}: {valid}/{len(timestamps)} frames valid', flush=True)
                finally:
                    cap.release()
                    if extractor is not None:
                        extractor.close()
                records.append(record)
    finally:
        manifest = {'version': 'sibi-jz-extraction-v1', 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    'implementation_sha256': hashlib.sha256((ROOT/'src/sibi_bisindo/landmarks.py').read_bytes()).hexdigest(),
                    'labels': 'Dataset folder labels; source pose reference not independently verified', 'records': records}
        (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
