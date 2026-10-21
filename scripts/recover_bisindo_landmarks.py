"""Recover cached BISINDO train/val detections with label-free ROI/rotation views."""
import argparse
from collections import Counter
import csv
import copy
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.landmarks import HandExtractor, Extraction
from sibi_bisindo.landmark_recovery import RecoveryExtractor
from sibi_bisindo.npy_landmarks import canonical, sha256_file
from extract_landmarks import atomic_npz, load_manifest, validate_arrays, summarize, write_log


def init_worker(base_config, config, baseline_path, image_root):
    global WORKER
    import atexit
    detector = HandExtractor(base_config)
    atexit.register(detector.close)
    with np.load(baseline_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    WORKER = (RecoveryExtractor(detector, config), arrays, Path(image_root))


def recover_worker(task):
    import hashlib
    i, relative, digest = task
    recovery, baseline, image_root = WORKER
    content = (image_root/relative).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Image changed since manifest')
    with Image.open(io.BytesIO(content)) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    started = time.perf_counter()
    result, details = recovery.extract(rgb, baseline_result(baseline, i))
    return i, result, details, (time.perf_counter()-started)*1000


def apply_result(arrays, i, result, details, milliseconds):
    arrays['recovery_ms'][i] = milliseconds
    arrays['recovery_attempts'][i] = details['attempts']
    if details['recovered']:
        count = len(result.raw)
        arrays['raw_landmarks'][i] = np.nan; arrays['raw_landmarks'][i,:count] = result.raw
        arrays['handedness'][i] = ''; arrays['handedness'][i,:count] = result.handedness
        arrays['handedness_score'][i] = np.nan; arrays['handedness_score'][i,:count] = result.handedness_scores
        arrays['hand_count'][i] = count; arrays['valid_mask'][i] = result.valid
        arrays['X'][i] = result.features; arrays['status'][i] = result.status
        arrays['warnings'][i] = '; '.join(result.warnings)
        arrays['recovery_method'][i] = details['method']; arrays['review_required'][i] = True
    arrays['recovery_processed'][i] = True


def baseline_result(arrays, i):
    count = int(arrays['hand_count'][i]); valid = bool(arrays['valid_mask'][i])
    return Extraction(arrays['X'][i] if valid else None, arrays['raw_landmarks'][i, :count].copy(),
                      arrays['handedness'][i, :count].tolist(), arrays['handedness_score'][i, :count].tolist(),
                      str(arrays['status'][i]), valid, float(arrays['process_ms'][i]), [],
                      int(arrays['width'][i]), int(arrays['height'][i]))


def write_review_grids(directory, arrays, baseline, rows, image_root, split):
    import cv2
    from PIL import ImageDraw
    chains = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
    recovered = np.flatnonzero(arrays['review_required'])
    for start, end in [(0,1), (0,2), (1,2)]:
        pool = [int(i) for i in recovered if baseline['hand_count'][i] == start and arrays['hand_count'][i] == end]
        # Spread the examples by class; do not show just the first class in a split.
        selected, labels = [], set()
        for i in pool:
            if str(arrays['labels'][i]) not in labels:
                selected.append(i); labels.add(str(arrays['labels'][i]))
            if len(selected) == 6: break
        if not selected: continue
        sheet = Image.new('RGB', (960, len(selected)*360+45), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'{split}: {start}->{end} | left=baseline, right=recovery; geometry valid != pose certified', fill='black')
        for row_number, index in enumerate(selected):
            with Image.open(image_root/rows[index]['path']) as im:
                image = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
            for column, cache in enumerate([baseline, arrays]):
                frame = image.copy(); h, w = frame.shape[:2]
                for hand in cache['raw_landmarks'][index, :int(cache['hand_count'][index])]:
                    pixels = np.round(hand[:, :2]*[w,h]).astype(int)
                    for chain in chains:
                        for a,b in zip(chain[:-1], chain[1:]):
                            cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), (0,220,0), 2)
                    for point in pixels:
                        cv2.circle(frame, tuple(point), 3, (255,60,20), -1)
                tile = Image.fromarray(frame); tile.thumbnail((470, 320))
                x,y = column*480, row_number*360+45
                sheet.paste(tile, (x+(470-tile.width)//2, y))
                text = f"{arrays['labels'][index]} | #{index} | hands={cache['hand_count'][index]}"
                if column: text += f" | {arrays['recovery_method'][index]}"
                draw.text((x+5,y+325), text, fill='black')
        sheet.save(directory/f'{split}_recovered_{start}_to_{end}.jpg', quality=90)


def main():
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bisindo_recovery_v1.json')
    parser.add_argument('--splits', nargs='+', choices=['train','val'], default=['train','val'])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:parser.error('--workers must be positive')
    config = json.loads(args.config.read_text())
    base_config_path = ROOT/config['base_config']; base_config = json.loads(base_config_path.read_text())
    output = ROOT/config['output_dir']; output.mkdir(parents=True, exist_ok=True)
    identity = {'version': config['version'], 'config_sha256': hashlib.sha256(canonical(config)).hexdigest(),
                'base_config_sha256': hashlib.sha256(canonical(base_config)).hexdigest(),
                'recovery_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/landmark_recovery.py'),
                'baseline_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/landmarks.py'),
                'note': 'Automatic proposals requiring review. No annotations/class labels consumed by recovery. Train/val only.'}
    lock_path = output/'recovery_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != identity:
        raise ValueError('Recovery implementation/config changed; use a new output version')
    lock_path.write_text(json.dumps(identity, indent=2)+'\n')
    detector = HandExtractor(base_config); recovery = RecoveryExtractor(detector, config)
    summaries = {}
    report_path = output/'summary.json'
    if report_path.exists(): summaries = json.loads(report_path.read_text()).get('splits', {})
    try:
        for split in args.splits:
            rows, manifest_hash = load_manifest(base_config, 'bisindo', split)
            base_path = ROOT/config['base_cache']/f'bisindo_{split}.npz'
            with np.load(base_path, allow_pickle=False) as archive:
                baseline = {key: archive[key].copy() for key in archive.files}
            validate_arrays(baseline)
            meta = json.loads(str(baseline['metadata_json']))
            if (meta['config_sha256'] != identity['base_config_sha256'] or
                    meta['manifest_sha256'] != manifest_hash or
                    meta['feature_implementation_sha256'] != identity['baseline_implementation_sha256'] or
                    list(baseline['paths']) != [r['path'] for r in rows]):
                raise ValueError('Baseline cache/config/manifest mismatch')
            metadata = dict(meta, recovery_identity=identity, baseline_cache_sha256=sha256_file(base_path))
            target = output/f'bisindo_{split}.npz'; partial = output/f'bisindo_{split}.partial.npz'
            cache_path = target if target.exists() else partial
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                if json.loads(str(arrays['metadata_json'])) != metadata:
                    raise ValueError('Recovery cache provenance changed')
            else:
                arrays = copy.deepcopy(baseline)
                arrays['metadata_json'] = np.array(json.dumps(metadata, sort_keys=True))
                arrays['recovery_processed'] = np.zeros(len(rows), dtype=bool)
                arrays['recovery_method'] = np.full(len(rows), 'baseline', dtype='<U32')
                arrays['recovery_attempts'] = np.zeros(len(rows), dtype=np.int16)
                arrays['recovery_ms'] = np.zeros(len(rows), dtype=np.float32)
                arrays['review_required'] = np.zeros(len(rows), dtype=bool)
            image_root = ROOT/'dataset'/base_config['dataset_root']
            try:
                if args.workers > 1:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    tasks = []
                    for i,row in enumerate(rows):
                        if arrays['recovery_processed'][i]:continue
                        if baseline['hand_count'][i] >= (2 if config['recover_one_hand'] else 1):
                            arrays['recovery_processed'][i] = True
                        else:tasks.append((i,row['path'],row['sha256']))
                    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                             initargs=(base_config,config,str(base_path),str(image_root))) as pool:
                        futures = [pool.submit(recover_worker, task) for task in tasks]
                        for done,future in enumerate(as_completed(futures),1):
                            apply_result(arrays,*future.result())
                            if done % config['checkpoint_every'] == 0:
                                atomic_npz(partial, arrays)
                                print(f"{split}: processed={int(arrays['recovery_processed'].sum())}/{len(rows)}; "
                                      f"recovered={int(arrays['review_required'].sum())}; valid={int(arrays['valid_mask'].sum())}",flush=True)
                for i, row in enumerate(rows):
                    if arrays['recovery_processed'][i]: continue
                    original_count = int(baseline['hand_count'][i])
                    if original_count < (2 if config['recover_one_hand'] else 1):
                        contents = (image_root/row['path']).read_bytes()
                        if hashlib.sha256(contents).hexdigest() != row['sha256']:
                            raise ValueError('Image changed since manifest')
                        with Image.open(io.BytesIO(contents)) as im:
                            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
                        started = time.perf_counter()
                        result, details = recovery.extract(rgb, baseline_result(baseline, i))
                        arrays['recovery_ms'][i] = (time.perf_counter()-started)*1000
                        arrays['recovery_attempts'][i] = details['attempts']
                        if details['recovered']:
                            count = len(result.raw)
                            arrays['raw_landmarks'][i] = np.nan; arrays['raw_landmarks'][i,:count] = result.raw
                            arrays['handedness'][i] = ''; arrays['handedness'][i,:count] = result.handedness
                            arrays['handedness_score'][i] = np.nan; arrays['handedness_score'][i,:count] = result.handedness_scores
                            arrays['hand_count'][i] = count; arrays['valid_mask'][i] = result.valid
                            arrays['X'][i] = result.features; arrays['status'][i] = result.status
                            arrays['warnings'][i] = '; '.join(result.warnings)
                            arrays['recovery_method'][i] = details['method']; arrays['review_required'][i] = True
                    arrays['recovery_processed'][i] = True
                    if (i+1) % config['checkpoint_every'] == 0:
                        atomic_npz(partial, arrays)
                        print(f"{split}: {i+1}/{len(rows)}; recovered={int(arrays['review_required'].sum())}; "
                              f"valid={int(arrays['valid_mask'].sum())}; two_hands={int((arrays['hand_count']==2).sum())}", flush=True)
            finally:
                if not target.exists(): atomic_npz(partial, arrays)
            validate_arrays(arrays)
            assert arrays['recovery_processed'].all()
            assert (arrays['hand_count'] >= baseline['hand_count']).all()
            atomic_npz(target, arrays)
            write_log(output/f'bisindo_{split}_log.csv', arrays, 'bisindo', split)
            stats = summarize(arrays)
            stats['baseline'] = summarize(baseline)
            stats['transitions'] = dict(Counter(f'{a}->{b}' for a,b in zip(baseline['hand_count'], arrays['hand_count'])))
            stats['recovered_by_method'] = dict(Counter(arrays['recovery_method'][arrays['review_required']].tolist()))
            stats['review_required'] = int(arrays['review_required'].sum())
            stats['mean_extra_ms'] = float(arrays['recovery_ms'].mean())
            stats['p95_extra_ms'] = float(np.percentile(arrays['recovery_ms'],95))
            stats['median_total_estimated_ms'] = float(np.median(baseline['process_ms']+arrays['recovery_ms']))
            stats['p95_total_estimated_ms'] = float(np.percentile(baseline['process_ms']+arrays['recovery_ms'],95))
            with (output/f'bisindo_{split}_recovery.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['path','label','baseline_hands','recovered_hands','method',
                                                            'attempts','extra_ms','review_required'])
                writer.writeheader()
                for i,row in enumerate(rows):
                    writer.writerow(dict(path=row['path'],label=row['label'], baseline_hands=int(baseline['hand_count'][i]),
                                         recovered_hands=int(arrays['hand_count'][i]),method=str(arrays['recovery_method'][i]),
                                         attempts=int(arrays['recovery_attempts'][i]),extra_ms=float(arrays['recovery_ms'][i]),
                                         review_required=bool(arrays['review_required'][i])))
            # A train run may finish after a separate validation process; preserve both summaries.
            if report_path.exists():
                summaries.update(json.loads(report_path.read_text()).get('splits', {}))
            summaries[f'bisindo/{split}'] = stats
            write_review_grids(output, arrays, baseline, rows, image_root, split)
            report = {'identity': identity, 'splits': summaries,
                      'limitations': ['Automatic geometric recovery is not landmark accuracy or pose certification.',
                                      'Recovered rows require visual review before adoption; baseline cache remains unchanged.',
                                      'Extra timings exclude cached baseline detection and image I/O; not webcam FPS.',
                                      'process_ms is the original cached baseline time; recovery_ms is the new added work.',
                                      'Ground-truth boxes and alphabet labels are not used to generate views.']}
            report_path.write_text(json.dumps(report, indent=2)+'\n')
            print(f'{split} complete: {stats["feature_valid"]}/{len(rows)} valid; {stats["transitions"]}', flush=True)
    finally:
        detector.close()


if __name__ == '__main__':
    main()
