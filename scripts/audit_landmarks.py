"""Exploratory MediaPipe coverage audit; does not train a classifier.

Uses the legacy Hands API already installed locally (mediapipe==0.10.21).
This is not a benchmark of the newer Tasks API or webcam throughput.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
OUT = ROOT / 'reports' / 'dataset_audit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.per_class <= 0:
        parser.error('--per-class must be positive')
    with (OUT / 'inventory.csv').open(encoding='utf-8', newline='') as f:
        inventory = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in inventory:
        if 'with_augmentation' not in row['variant'] and not row['error']:
            groups[(row['language'], row['variant'], row['letter'])].append(row)
    rng = random.Random(args.seed)
    selected = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda r: r['path'])
        selected.extend(rng.sample(candidates, min(len(candidates), args.per_class)))
    results = []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                  model_complexity=1, min_detection_confidence=0.5) as detector:
        for i, row in enumerate(selected, 1):
            with Image.open(DATA / row['path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            start = time.perf_counter()
            result = detector.process(rgb)
            elapsed = (time.perf_counter() - start) * 1000
            n = len(result.multi_hand_landmarks or [])
            handedness = [c.classification[0] for c in result.multi_handedness or []]
            results.append(dict(path=row['path'], language=row['language'], variant=row['variant'],
                letter=row['letter'], split=row['split'], hands_detected=n,
                handedness=','.join(c.label for c in handedness),
                handedness_scores=','.join(f'{c.score:.5f}' for c in handedness),
                process_ms=round(elapsed, 3)))
            if i % 100 == 0:
                print(f'Landmarks: {i}/{len(selected)}', flush=True)
    with (OUT / 'landmark_sample.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summaries = {}
    for language, variant, letter in sorted(groups):
        rows = [r for r in results if r['variant'] == variant and r['letter'] == letter]
        key = language + '/' + variant
        summaries.setdefault(key, {})[letter] = dict(
            sampled=len(rows), hands_detected=dict(Counter(r['hands_detected'] for r in rows)))
    totals = {}
    for key in summaries:
        rows = [r for r in results if r['language'] + '/' + r['variant'] == key]
        times = [r['process_ms'] for r in rows]
        totals[key] = dict(sampled=len(rows), hands_detected=dict(Counter(r['hands_detected'] for r in rows)),
            at_least_one_rate=sum(r['hands_detected'] > 0 for r in rows) / len(rows),
            median_process_ms=float(np.median(times)), p95_process_ms=float(np.percentile(times, 95)))
    summary = dict(mediapipe_version=mp.__version__, api='solutions.hands.Hands (legacy)',
        config=dict(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5),
        seed=args.seed, per_class=args.per_class, totals=totals, per_class_results=summaries,
        limitations=['Sampling only; not full dataset coverage.',
            'A detected hand is not a verified correct landmark or correct letter.',
            'One detected hand can be incomplete for a two-hand sign; no expected hand-count ground truth.',
            'Static image latency excludes IO, network, classifier and temporal tracking; not webcam FPS.',
            'Legacy API results must be rechecked if production uses the Tasks API.'])
    (OUT / 'landmark_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Up to one failed example per class, for visual review rather than deletion.
    for key in summaries:
        failures = {}
        for row in results:
            if row['language'] + '/' + row['variant'] == key and row['hands_detected'] == 0:
                failures.setdefault(row['letter'], row)
        if not failures:
            continue
        sheet = Image.new('RGB', (6 * 180, ((len(failures)+5)//6)*170), 'white')
        draw = ImageDraw.Draw(sheet)
        manifest = []
        for i, (letter, row) in enumerate(sorted(failures.items())):
            with Image.open(DATA / row['path']) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (176, 140))
            x,y = (i%6)*180,(i//6)*170
            sheet.paste(thumb, (x,y+24))
            draw.text((x+5,y+5), f'{letter}: 0 hands', fill='black')
            manifest.append(row['path'])
        name = key.replace('/', '_') + '_no_hand_samples'
        sheet.save(OUT / (name + '.jpg'))
        (OUT / (name + '.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
