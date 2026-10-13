"""Extract pinned features with split-level caching and resumable checkpoints.

Pilot: python scripts/extract_landmarks.py --splits val
Full:  python scripts/extract_landmarks.py --splits train val test
The test split requires the selected config to have been frozen first.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageOps
from sibi_bisindo.landmarks import HandExtractor,FEATURE_DIMENSION,feature_names


def canonical(data):
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_npz(path, arrays):
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**arrays)
    temporary.replace(path)


def load_manifest(config, language, split):
    directory=ROOT/config['manifest_dir']
    lock=json.loads((directory/'split_lock.json').read_text(encoding='utf-8'))
    tracked=json.loads((ROOT/'reports/data_preparation/split_lock.json').read_text(encoding='utf-8'))
    if lock!=tracked or lock['version']!=config['data_version']:
        raise ValueError('Manifest lock does not match the tracked data snapshot')
    name=f'{language}_{split}.csv'
    contents=(directory/name).read_bytes()
    if sha(contents)!=lock['manifests_sha256'][name]:
        raise ValueError(f'Manifest changed after lock: {name}')
    rows=list(csv.DictReader(io.StringIO(contents.decode('utf-8'))))
    for row in rows:
        if row['include']!='1' or row['language']!=language or row['split']!=split:
            raise ValueError('Unexpected excluded/mismatched manifest record')
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in row['path']:
            raise ValueError('Unsafe relative image path')
    return rows,sha(contents)


def init_arrays(rows, metadata):
    n=len(rows)
    return dict(
        X=np.full((n,FEATURE_DIMENSION),np.nan,dtype=np.float32),
        raw_landmarks=np.full((n,2,21,3),np.nan,dtype=np.float32),
        valid_mask=np.zeros(n,dtype=np.bool_), hand_count=np.zeros(n,dtype=np.int8),
        handedness=np.full((n,2),'',dtype='<U8'), handedness_score=np.full((n,2),np.nan,dtype=np.float32),
        status=np.full(n,'pending',dtype='<U48'), warnings=np.full(n,'',dtype='<U240'),
        process_ms=np.full(n,np.nan,dtype=np.float32), width=np.zeros(n,dtype=np.int32),height=np.zeros(n,dtype=np.int32),
        paths=np.array([r['path'] for r in rows]),labels=np.array([r['label'] for r in rows]),
        source_groups=np.array([r['source_group'] for r in rows]),
        sha256=np.array([r['sha256'] for r in rows]),
        feature_names=np.array(feature_names()), metadata_json=np.array(json.dumps(metadata,sort_keys=True)),
        processed_count=np.array(0,dtype=np.int64),
    )


def load_cache(path, metadata, rows):
    with np.load(path,allow_pickle=False) as saved:
        arrays={name:saved[name] for name in saved.files}
    if json.loads(str(arrays['metadata_json']))!=metadata:
        raise ValueError(f'Cache identity changed: {path}. Use a new extraction version/output directory.')
    if list(arrays['paths'])!=[r['path'] for r in rows] or arrays['X'].shape!=(len(rows),FEATURE_DIMENSION):
        raise ValueError('Cached row ordering/schema does not match manifest')
    processed=int(arrays['processed_count'])
    if not 0 <= processed <= len(rows):
        raise ValueError('Cached checkpoint has an invalid processed_count')
    return arrays


def validate_arrays(arrays):
    n=len(arrays['paths'])
    if int(arrays['processed_count'])!=n or (arrays['status']=='pending').any():
        raise ValueError('Incomplete cache')
    mask=arrays['valid_mask']
    if not np.isfinite(arrays['X'][mask]).all() or not np.isnan(arrays['X'][~mask]).all():
        raise ValueError('Invalid feature-mask representation')
    if not np.isfinite(arrays['process_ms']).all():
        raise ValueError('Missing timings')


def write_log(path, arrays, language, split):
    columns=['path','language','split','label','hands','feature_valid','status','warnings','process_ms']
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for i,p in enumerate(arrays['paths']):
            writer.writerow(dict(path=p,language=language,split=split,label=arrays['labels'][i],
                hands=int(arrays['hand_count'][i]),feature_valid=int(arrays['valid_mask'][i]),
                status=arrays['status'][i],warnings=arrays['warnings'][i],process_ms=round(float(arrays['process_ms'][i]),3)))


def summarize(arrays):
    mask=arrays['valid_mask'];count=arrays['hand_count'];n=len(count)
    by_class={}
    for label in sorted(set(arrays['labels'])):
        subset=arrays['labels']==label
        by_class[str(label)]=dict(total=int(subset.sum()),feature_valid=int(mask[subset].sum()),
            hands=dict(Counter(map(int,count[subset]))),statuses=dict(Counter(arrays['status'][subset])))
    return dict(total=n,feature_valid=int(mask.sum()),feature_valid_rate=float(mask.mean()),
        detected_at_least_one=int((count>0).sum()),hand_counts=dict(Counter(map(int,count))),
        statuses=dict(Counter(arrays['status'])),warning_rows=int((arrays['warnings']!='').sum()),
        median_process_ms=float(np.median(arrays['process_ms'])),p95_process_ms=float(np.percentile(arrays['process_ms'],95)),
        per_class=by_class)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/landmarks_v1.json')
    parser.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    parser.add_argument('--languages',nargs='+',choices=['sibi','bisindo'],default=['sibi','bisindo'])
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    config_hash=sha(canonical(config))
    if 'test' in args.splits:
        selection=ROOT/'configs/landmark_selection_v1.json'
        if not selection.exists() or json.loads(selection.read_text(encoding='utf-8'))['selected_config_sha256']!=config_hash:
            raise ValueError('Freeze the detector configuration using validation before extracting test')
    directory=ROOT/config['output_dir'];directory.mkdir(parents=True,exist_ok=True)
    extractor=HandExtractor(config)
    identity=dict(config_sha256=config_hash,feature_version=config['feature_version'],
        feature_implementation_sha256=sha((ROOT/'src/sibi_bisindo/landmarks.py').read_bytes()),
        dependencies=extractor.versions,models=extractor.model_hashes)
    lock_path=directory/'extraction_lock.json'
    if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf-8'))!=identity:
        extractor.close();raise ValueError('Extraction identity changed; use a new output directory/version')
    lock_path.write_bytes(canonical(identity))
    summaries={}
    try:
        for language in args.languages:
            for split in args.splits:
                rows,manifest_hash=load_manifest(config,language,split)
                metadata=dict(identity,manifest_sha256=manifest_hash,language=language,split=split)
                target=directory/f'{language}_{split}.npz';partial=directory/f'{language}_{split}.partial.npz'
                if target.exists():
                    arrays=load_cache(target,metadata,rows);validate_arrays(arrays)
                    print(f'Reused {target.name}',flush=True)
                else:
                    arrays=load_cache(partial,metadata,rows) if partial.exists() else init_arrays(rows,metadata)
                    start=int(arrays['processed_count'])
                    for i in range(start,len(rows)):
                        row=rows[i];path=ROOT/'dataset'/row['path'];contents=path.read_bytes()
                        if sha(contents)!=row['sha256']:
                            raise ValueError(f'Raw image changed since manifest: {row["path"]}')
                        with Image.open(io.BytesIO(contents)) as im:
                            if config['preprocessing']['exif_transpose']:
                                im=ImageOps.exif_transpose(im)
                            rgb=np.asarray(im.convert('RGB'))
                        result=extractor.extract(rgb,language,input_mirrored=config['preprocessing']['offline_input_mirrored'])
                        count=len(result.raw);arrays['hand_count'][i]=count
                        arrays['raw_landmarks'][i,:count]=result.raw
                        arrays['handedness'][i,:count]=result.handedness
                        arrays['handedness_score'][i,:count]=result.handedness_scores
                        arrays['status'][i]=result.status;arrays['warnings'][i]='; '.join(result.warnings)
                        arrays['valid_mask'][i]=result.valid
                        if result.valid: arrays['X'][i]=result.features
                        arrays['process_ms'][i]=result.process_ms;arrays['width'][i]=result.width;arrays['height'][i]=result.height
                        arrays['processed_count'][...]=i+1
                        if (i+1)%250==0:
                            atomic_npz(partial,arrays)
                            print(f'{language}/{split}: {i+1}/{len(rows)}',flush=True)
                    validate_arrays(arrays);atomic_npz(target,arrays)
                write_log(directory/f'{language}_{split}_log.csv',arrays,language,split)
                summaries[f'{language}/{split}']=summarize(arrays)
                print(f'{language}/{split}: {summaries[f"{language}/{split}"]["feature_valid"]}/{len(rows)} feature-valid',flush=True)
    finally:
        extractor.close()
    # Merge previously extracted splits; frozen caches are verified by identity.
    for language in ['sibi','bisindo']:
        for split in ['train','val','test']:
            key=f'{language}/{split}';p=directory/f'{language}_{split}.npz'
            if p.exists() and key not in summaries:
                rows,mhash=load_manifest(config,language,split)
                arrays=load_cache(p,dict(identity,manifest_sha256=mhash,language=language,split=split),rows)
                validate_arrays(arrays);summaries[key]=summarize(arrays)
    report=dict(version=config['version'],config_sha256=config_hash,identity=identity,splits=summaries,
        limitations=['Feature-valid is geometry/policy validity, not classification or landmark accuracy.',
          'BISINDO expected hand count and actor assignment are unverified.',
          'At most two hands are detected; additional people/hands may remain unseen.',
          'Test features are frozen preprocessing outputs, not detector-selection data.',
          'Runtime timings include preprocessing and detection but exclude image IO/network/classifier.'])
    (directory/'summary.json').write_bytes(canonical(report))
    print(json.dumps({k:{name:v[name] for name in ['total','feature_valid','hand_counts']} for k,v in summaries.items()},indent=2),flush=True)


if __name__=='__main__':main()
