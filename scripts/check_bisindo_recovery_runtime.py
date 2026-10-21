"""Check cached versus runtime landmark/tensor parity on a few existing images."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.recovery_runtime import RuntimeRecoveryExtractor
from sibi_bisindo.landmarks import HandExtractor
from sibi_bisindo.coordinates import coordinate_tensor
from sibi_bisindo.npy_landmarks import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'data/features/bisindo-recovery-clean-v2')
    parser.add_argument('--train-index', type=int, action='append', default=[])
    parser.add_argument('--val-index', type=int, action='append', default=[])
    args = parser.parse_args()
    policy = json.loads((args.cache/'runtime_policy.json').read_text())
    extractor = RuntimeRecoveryExtractor(HandExtractor(policy['base_config']), policy)
    records = []
    try:
        for split in ('train', 'val'):
            with np.load(args.cache/f'bisindo_{split}.npz', allow_pickle=False) as z:
                a = {k:z[k] for k in ['paths','sha256','raw_landmarks','hand_count','valid_mask','height','width','recovery_method','quality_retried']}
            selected = set(getattr(args, split+'_index'))
            for mask in [a['hand_count']==0, a['hand_count']==1, a['hand_count']==2,
                         a['quality_retried'], a['recovery_method']=='rotate45', a['recovery_method']=='left72']:
                found = np.flatnonzero(mask)
                if len(found): selected.add(int(found[0]))
            for i in sorted(selected):
                path = ROOT/'dataset'/policy['base_config']['dataset_root']/str(a['paths'][i])
                if sha256_file(path) != a['sha256'][i]: raise ValueError('Image changed')
                with Image.open(path) as im: rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
                result = extractor.extract(rgb)
                if result.valid != bool(a['valid_mask'][i]) or len(result.raw) != int(a['hand_count'][i]):
                    raise ValueError(f'Cache/runtime availability differs: {split} #{i}')
                if result.valid:
                    expected = coordinate_tensor(a['raw_landmarks'][i,:int(a['hand_count'][i])],a['height'][i]/a['width'][i])
                    actual = coordinate_tensor(result.raw,result.height/result.width)
                    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
                record = dict(split=split,index=i,path=str(a['paths'][i]),hands=len(result.raw),
                              method=extractor.last_details['method'],runtime_ms=result.process_ms,parity=True)
                records.append(record)
                print(f'{split} #{i}: parity OK; hands={len(result.raw)}; {result.process_ms:.1f}ms',flush=True)
    finally:
        extractor.close()
    output = ROOT/'data/reports/bisindo-recovery-v2'; output.mkdir(parents=True,exist_ok=True)
    (output/'runtime_parity.json').write_text(json.dumps(dict(policy_sha256=policy['sha256'],samples=records,
        note='Checks preprocessing consistency only, not classifier accuracy or camera FPS.'),indent=2)+'\n')


if __name__ == '__main__': main()
