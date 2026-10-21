"""Build automatic-only v2 recovery caches; reuse unchanged verified proposals."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.recovery_runtime import make_policy, RuntimeRecoveryExtractor, suspicious_pair
from sibi_bisindo.landmarks import HandExtractor
from sibi_bisindo.npy_landmarks import canonical, sha256_file
from recover_bisindo_landmarks import baseline_result, apply_result, write_review_grids
from extract_landmarks import atomic_npz, validate_arrays, summarize, load_manifest, write_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/bisindo_recovery_v2.json').read_text())
    recovery = json.loads((ROOT/config['recovery_config']).read_text())
    base_config = json.loads((ROOT/recovery['base_config']).read_text())
    policy = make_policy(base_config, recovery, config['quality'])
    output = ROOT/config['output_dir']; output.mkdir(parents=True, exist_ok=True)
    policy_path = output/'runtime_policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError('Policy changed; choose a new output version')
    policy_path.write_text(json.dumps(policy, indent=2)+'\n')
    reports = json.loads((output/'summary.json').read_text())['splits'] if (output/'summary.json').exists() else {}
    detector = None
    try:
        for split in args.splits:
            proposal_path = ROOT/config['proposal_cache']/f'bisindo_{split}.npz'
            base_path = ROOT/recovery['base_cache']/f'bisindo_{split}.npz'
            with np.load(proposal_path, allow_pickle=False) as z: proposal = {k:z[k] for k in z.files}
            with np.load(base_path, allow_pickle=False) as z: baseline = {k:z[k] for k in z.files}
            meta = json.loads(str(proposal['metadata_json']))
            expected = meta['recovery_identity']
            if (meta['baseline_cache_sha256'] != sha256_file(base_path)
                    or expected['config_sha256'] != hashlib.sha256(canonical(recovery)).hexdigest()
                    or expected['recovery_implementation_sha256'] != sha256_file(ROOT/'src/sibi_bisindo/landmark_recovery.py')
                    or meta['feature_implementation_sha256'] != policy['implementations']['landmarks.py']
                    or meta['config_sha256'] != hashlib.sha256(canonical(base_config)).hexdigest()):
                raise ValueError('Proposal preprocessing provenance mismatch')
            for key in ['paths', 'sha256', 'labels', 'width', 'height', 'source_groups']:
                if not np.array_equal(proposal[key], baseline[key]):
                    raise ValueError('Proposal and baseline identity mismatch')
            if not proposal['recovery_processed'].all(): raise ValueError('Incomplete proposals')
            rows, _ = load_manifest(base_config, 'bisindo', split)
            if [r['path'] for r in rows] != proposal['paths'].tolist(): raise ValueError('Manifest row mismatch')
            meta['runtime_preprocessing'] = policy
            meta['proposal_cache_sha256'] = sha256_file(proposal_path)
            target = output/f'bisindo_{split}.npz'
            if target.exists():
                with np.load(target, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
                if json.loads(str(arrays['metadata_json'])) != meta: raise ValueError('Cached v2 provenance differs')
            else:
                arrays = copy.deepcopy(proposal)
                arrays['quality_retried'] = np.zeros(len(rows), bool)
                for i in np.flatnonzero(proposal['review_required'] & (proposal['hand_count'] == 2)):
                    if not suspicious_pair(proposal['raw_landmarks'][i], proposal['height'][i]/proposal['width'][i], config['quality']):
                        continue
                    if detector is None: detector = RuntimeRecoveryExtractor(HandExtractor(base_config), policy)
                    path = ROOT/'dataset'/base_config['dataset_root']/rows[i]['path']
                    if sha256_file(path) != rows[i]['sha256']: raise ValueError('Source image changed')
                    with Image.open(path) as im: rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
                    start = time.perf_counter()
                    result = detector.extract(rgb, baseline=baseline_result(baseline, i))
                    details = detector.last_details
                    for key in ['X','raw_landmarks','valid_mask','hand_count','handedness','handedness_score','status','warnings']:
                        arrays[key][i] = baseline[key][i]
                    arrays['review_required'][i] = False
                    arrays['recovery_method'][i] = 'quality_reverted'
                    previous_ms = float(arrays['recovery_ms'][i]); previous_attempts = int(arrays['recovery_attempts'][i])
                    apply_result(arrays, i, result, details, (time.perf_counter()-start)*1000)
                    arrays['recovery_ms'][i] += previous_ms
                    arrays['recovery_attempts'][i] += previous_attempts
                    arrays['quality_retried'][i] = True
                    print(f'{split} #{i}: retry -> {len(result.raw)} hands ({details["method"]})', flush=True)
                arrays['metadata_json'] = np.array(json.dumps(meta, sort_keys=True))
                validate_arrays(arrays); atomic_npz(target, arrays)
            if (not np.all(arrays['valid_mask'] | ~baseline['valid_mask'])
                    or not np.all(arrays['hand_count'] >= baseline['hand_count'])):
                raise ValueError('Recovery decreased baseline availability')
            write_log(output/f'bisindo_{split}_log.csv', arrays, 'bisindo', split)
            write_review_grids(output, arrays, baseline, rows, ROOT/'dataset'/base_config['dataset_root'], split)
            stats = summarize(arrays); stats['baseline'] = summarize(baseline)
            stats['quality_retried'] = int(arrays['quality_retried'].sum())
            stats['transitions'] = dict(Counter(f'{a}->{b}' for a,b in zip(baseline['hand_count'],arrays['hand_count'])))
            stats['review_required'] = int(arrays['review_required'].sum())
            stats['manual_image_overrides'] = 0
            stats['recovered_by_method'] = dict(Counter(arrays['recovery_method'][arrays['review_required']]))
            reports['bisindo/'+split] = stats
            (output/'summary.json').write_text(json.dumps(dict(config=config, policy_sha256=policy['sha256'], splits=reports), indent=2)+'\n')
            print(f'{split}: {stats["feature_valid"]}/{stats["total"]} valid; quality retries {stats["quality_retried"]}', flush=True)
    finally:
        if detector is not None: detector.close()


if __name__ == '__main__': main()
