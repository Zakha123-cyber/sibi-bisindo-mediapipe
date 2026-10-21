"""Retry suspected duplicate-hand recovery proposals without modifying raw caches."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sibi_bisindo.recovery_quality import duplicate_distance, QualityRecoveryExtractor
from sibi_bisindo.landmarks import HandExtractor
from sibi_bisindo.npy_landmarks import canonical, sha256_file
from recover_bisindo_landmarks import baseline_result, write_review_grids
from extract_landmarks import atomic_npz, validate_arrays, summarize, write_log, load_manifest


def main():
    import hashlib,csv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits',nargs='+',choices=['train','val'],default=['train','val'])
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/bisindo_recovery_quality_v1.json').read_text())
    recovery_config=json.loads((ROOT/config['recovery_config']).read_text())
    base_config=json.loads((ROOT/recovery_config['base_config']).read_text())
    output=ROOT/config['output_dir'];output.mkdir(parents=True,exist_ok=True)
    reports={}
    review_path=ROOT/'docs/bisindo_recovery_review.csv'
    with review_path.open(newline='',encoding='utf-8') as stream:decisions=list(csv.DictReader(stream))
    if (output/'summary.json').exists():reports=json.loads((output/'summary.json').read_text())['splits']
    detector=None
    try:
        for split in args.splits:
            proposal_path=ROOT/config['proposal_cache']/f'bisindo_{split}.npz'
            baseline_path=ROOT/recovery_config['base_cache']/f'bisindo_{split}.npz'
            with np.load(proposal_path,allow_pickle=False) as a:proposals={k:a[k].copy() for k in a.files}
            with np.load(baseline_path,allow_pickle=False) as a:baseline={k:a[k].copy() for k in a.files}
            validate_arrays(proposals);validate_arrays(baseline)
            if not proposals['recovery_processed'].all() or not np.array_equal(proposals['paths'],baseline['paths']):
                raise ValueError('Incomplete proposals or baseline row mismatch')
            metadata=json.loads(str(proposals['metadata_json']))
            metadata['quality_policy']={'config':config,'recovery_config':recovery_config,
                'implementation_sha256':sha256_file(ROOT/'src/sibi_bisindo/recovery_quality.py'),
                'proposal_cache_sha256':sha256_file(proposal_path),
                'technical_review_sha256':sha256_file(review_path)}
            target=output/f'bisindo_{split}.npz'
            if target.exists():
                with np.load(target,allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
                stored=json.loads(str(arrays['metadata_json']))
                compared=copy.deepcopy(metadata)
                if 'technical_review_sha256' not in stored['quality_policy']:
                    compared['quality_policy'].pop('technical_review_sha256')
                if stored!=compared:raise ValueError('Quality policy changed; new output required')
            else:
                arrays=copy.deepcopy(proposals)
                arrays['metadata_json']=np.array(json.dumps(metadata,sort_keys=True))
                arrays['duplicate_proposal']=np.zeros(len(arrays['paths']),bool)
                rows,_=load_manifest(base_config,'bisindo',split)
                for i in np.flatnonzero(proposals['review_required'] & (proposals['hand_count']==2)):
                    distance=duplicate_distance(proposals['raw_landmarks'][i,:2],proposals['height'][i]/proposals['width'][i])
                    if distance >= config['minimum_mean_point_separation_palms']:continue
                    arrays['duplicate_proposal'][i]=True
                    if detector is None:detector=HandExtractor(base_config)
                    extractor=QualityRecoveryExtractor(detector,recovery_config,config['minimum_mean_point_separation_palms'])
                    path=ROOT/'dataset'/base_config['dataset_root']/rows[i]['path']
                    if sha256_file(path)!=rows[i]['sha256']:raise ValueError('Source image changed')
                    with Image.open(path) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
                    started=time.perf_counter();result,details=extractor.extract(rgb,baseline_result(baseline,i))
                    arrays['recovery_ms'][i]+=(time.perf_counter()-started)*1000
                    arrays['recovery_attempts'][i]+=details['attempts']
                    for key in ['X','raw_landmarks','valid_mask','hand_count','handedness','handedness_score','status','warnings']:
                        arrays[key][i]=baseline[key][i]
                    count=len(result.raw)
                    if details['recovered']:
                        arrays['raw_landmarks'][i]=np.nan;arrays['raw_landmarks'][i,:count]=result.raw
                        arrays['X'][i]=result.features;arrays['valid_mask'][i]=result.valid;arrays['hand_count'][i]=count
                        arrays['handedness'][i]='';arrays['handedness'][i,:count]=result.handedness
                        arrays['handedness_score'][i]=np.nan;arrays['handedness_score'][i,:count]=result.handedness_scores
                        arrays['status'][i]=result.status;arrays['warnings'][i]='; '.join(result.warnings)
                    arrays['review_required'][i]=details['recovered']
                    arrays['recovery_method'][i]=details['method'] if details['recovered'] else 'duplicate_reverted'
                    print(f'{split} #{i}: duplicate proposal; {proposals["hand_count"][i]} -> {count} hands',flush=True)
                validate_arrays(arrays);atomic_npz(target,arrays)
            if 'technical_review_reverted' not in arrays:
                arrays['technical_review_reverted']=np.zeros(len(arrays['paths']),bool)
            for decision in decisions:
                if decision['split']!=split:continue
                found=np.flatnonzero(arrays['paths']==decision['path'])
                if len(found)!=1 or str(arrays['sha256'][found[0]])!=decision['sha256']:
                    raise ValueError('Technical review image identity mismatch')
                i=int(found[0])
                if decision['decision']!='revert_to_baseline':raise ValueError('Unsupported review decision')
                if not arrays['technical_review_reverted'][i]:
                    import shutil
                    backup=target.with_suffix('.automatic.npz')
                    if not backup.exists():shutil.copyfile(target,backup)
                    for key in ['X','raw_landmarks','valid_mask','hand_count','handedness','handedness_score','status','warnings']:
                        arrays[key][i]=baseline[key][i]
                    arrays['review_required'][i]=False
                    arrays['technical_review_reverted'][i]=True
                    arrays['recovery_method'][i]='manual_revert_duplicate'
            arrays['metadata_json']=np.array(json.dumps(metadata,sort_keys=True))
            validate_arrays(arrays);atomic_npz(target,arrays)
            rows,_=load_manifest(base_config,'bisindo',split)
            write_log(output/f'bisindo_{split}_log.csv',arrays,'bisindo',split)
            write_review_grids(output,arrays,baseline,rows,ROOT/'dataset'/base_config['dataset_root'],split)
            stats=summarize(arrays);stats['baseline']=summarize(baseline)
            stats['transitions']=dict(Counter(f'{a}->{b}' for a,b in zip(baseline['hand_count'],arrays['hand_count'])))
            stats['duplicate_proposals_retried']=int(arrays['duplicate_proposal'].sum())
            stats['technical_review_reverted']=int(arrays['technical_review_reverted'].sum())
            stats['recovered_by_method']=dict(Counter(arrays['recovery_method'][arrays['review_required']].tolist()))
            stats['review_required']=int(arrays['review_required'].sum())
            stats['mean_extra_ms']=float(arrays['recovery_ms'].mean())
            stats['p95_total_estimated_ms']=float(np.percentile(baseline['process_ms']+arrays['recovery_ms'],95))
            with (output/f'bisindo_{split}_recovery.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['path','label','baseline_hands','hands','method','duplicate_proposal','review_required'])
                for i,row in enumerate(rows):writer.writerow([row['path'],row['label'],int(baseline['hand_count'][i]),int(arrays['hand_count'][i]),str(arrays['recovery_method'][i]),bool(arrays['duplicate_proposal'][i]),bool(arrays['review_required'][i])])
            reports[f'bisindo/{split}']=stats
            (output/'summary.json').write_text(json.dumps({'config':config,'splits':reports,
                'limitations':['Duplicate filter is geometric, not a proof of detection correctness.',
                              'No image dropped, labels unchanged. Train/val only; no classifier training.',
                              'Recovered rows still require review; no classifier accuracy claim.']},indent=2)+'\n')
            print(f'{split}: {stats["feature_valid"]}/{len(rows)} valid, duplicate retries={stats["duplicate_proposals_retried"]}',flush=True)
    finally:
        if detector:detector.close()


if __name__=='__main__':main()
