"""Fit lightweight temporal templates. No GPU or epoch loop is needed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.temporal import VERSION, match, sequence_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, default=ROOT/'data/features/sibi-jz-v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/sibi-jz-v1')
    args = parser.parse_args()
    manifest_path = args.features_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    templates, labels, sources, excluded = [], [], [], []
    for row in manifest['records']:
        if row['status'] != 'extracted':
            continue
        with np.load(args.features_dir/row['file'], allow_pickle=False) as sample:
            times, hands = sample['times'], sample['hands']
            valid = np.isfinite(hands).all(axis=(1, 2))
            try:
                if valid.mean() < .6:
                    raise ValueError('Less than 60% of sampled frames have valid landmarks')
                feature, motion = sequence_features(times[valid], hands[valid], float(sample['aspect']))
                if motion < .5:
                    raise ValueError('Insufficient visible motion')
            except ValueError as error:
                excluded.append({'source': row['source'], 'reason': str(error)})
                continue
        templates.append(feature)
        labels.append(row['label'])
        sources.append(row['source'])
    templates, labels = np.asarray(templates), np.asarray(labels)
    if any(np.sum(labels == label) < 2 for label in ('J', 'Z')):
        raise ValueError(f'Need at least two usable clips per class; exclusions: {excluded}')
    predictions, same_class = [], []
    for index, (feature, label) in enumerate(zip(templates, labels)):
        mask = np.arange(len(labels)) != index
        predicted, distance, margin = match(feature, templates[mask], labels[mask])
        same_class.append(float(np.sqrt(np.mean((templates[mask & (labels == label)]-feature)**2, axis=(1, 2))).min()))
        predictions.append({'source': sources[index], 'label': str(label), 'prediction': predicted,
                            'distance': distance, 'margin': margin})
    threshold = float(np.quantile(same_class, .9))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir/'templates.npz', version=VERSION, templates=templates, labels=labels,
                        distance_threshold=threshold)
    report = {'version': VERSION, 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'clips': len(labels), 'class_counts': {c: int(np.sum(labels == c)) for c in ('J', 'Z')},
              'leave_one_video_out_accuracy': float(np.mean([p['label'] == p['prediction'] for p in predictions])),
              'distance_threshold': threshold, 'excluded': excluded, 'predictions': predictions,
              'limitation': 'Exploratory clip-level evaluation, not signer-independent. No non-J/Z rejection validation. Threshold is heuristic; no final test score.'}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
