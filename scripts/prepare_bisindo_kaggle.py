"""Audit and group the added Kaggle BISINDO dataset without changing raw files.

Builds a new source-image grouped split. It does not modify data-v1 and does
not claim signer/session independence because the dataset has no such IDs.
"""
import argparse
from collections import Counter, defaultdict
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import random
from pathlib import Path, PurePosixPath
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from PIL import Image


FIELDS=['path','label_path','language','label','class_id','original_split','split',
        'include','exclusion_reason','annotation_status','object_count','bbox_status',
        'width','height','sha256','phash64','source_group','source_group_basis']
LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
POPCOUNT=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(axis=1)


def canonical(data):
    return (json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf-8')


def sha(data):return hashlib.sha256(data).hexdigest()


def normalized_path(path):
    value=PurePosixPath(str(path).replace('\\','/'))
    if value.is_absolute() or '..' in value.parts or ':' in str(path):
        raise ValueError(f'Unsafe relative path: {path}')
    return value.as_posix()


def read_annotation(path,class_id):
    if not path.exists():return 'missing_label_file',0,'not_available'
    raw=path.read_text(encoding='utf-8',errors='replace')
    if not raw.strip():return 'empty_label_file',0,'empty'
    count=0
    for line in raw.splitlines():
        cols=line.split()
        if len(cols)!=5:return 'malformed_label',count,'malformed'
        try:
            found=int(cols[0]);x,y,w,h=map(float,cols[1:])
        except (TypeError,ValueError):return 'malformed_label',count,'malformed'
        if found!=class_id:return 'class_id_mismatch',count,'class_id_mismatch'
        if not np.isfinite([x,y,w,h]).all() or min(x,y,w,h)<0 or max(x,y,w,h)>1 or w<=0 or h<=0:
            return 'invalid_normalized_bbox',count,'invalid'
        if x-w/2 < -1e-4 or y-h/2 < -1e-4 or x+w/2 > 1.0001 or y+h/2 > 1.0001:
            return 'bbox_outside_image',count,'out_of_bounds'
        count+=1
    return ('ok',count,'normalized_xywh') if count else ('empty_label_file',0,'empty')


def inspect_item(dataset_root,image_path):
    rel=normalized_path(image_path.relative_to(dataset_root))
    parts=PurePosixPath(rel).parts
    if len(parts)!=4 or parts[0]!='images' or parts[1] not in {'train','val'} or parts[2] not in LETTERS:
        raise ValueError(f'Unexpected image layout: {rel}')
    original_split,label=parts[1],parts[2]
    class_id=LETTERS.index(label)
    label_rel=f'labels/{original_split}/{label}/{image_path.stem}.txt'
    label_path=dataset_root.joinpath(*PurePosixPath(label_rel).parts)
    contents=image_path.read_bytes()
    try:
        with Image.open(io.BytesIO(contents)) as im:
            im.load()
            width,height=im.size
            rgb=np.asarray(im.convert('RGB'))
    except Exception as exc:
        width=height=0;rgb=None
        image_status=f'corrupt_image:{type(exc).__name__}'
    else:image_status='ok'
    annotation_status,object_count,bbox_status=read_annotation(label_path,class_id)
    if image_status!='ok':exclusion=image_status
    elif annotation_status!='ok':exclusion=annotation_status
    else:exclusion=''
    image_hash=sha(contents)
    if rgb is None:phash=''
    else:
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        phash=int.from_bytes(bytes(cv2.img_hash.pHash(bgr).reshape(-1)),'little')
    return {'path':rel,'label_path':normalized_path(label_rel),'language':'bisindo',
        'label':label,'class_id':class_id,'original_split':original_split,'split':'',
        'include':'0' if exclusion else '1','exclusion_reason':exclusion,
        'annotation_status':annotation_status,'object_count':object_count,
        'bbox_status':bbox_status,'width':width,'height':height,'sha256':image_hash,
        'phash64':f'{phash:016x}' if phash else '', 'source_group':'',
        'source_group_basis':'perceptual_hash_component_within_class_d2'}


def hamming_matrix(values):
    values=np.asarray(values,dtype=np.uint64)
    x=np.bitwise_xor(values[:,None],values[None,:])
    bytes_view=x.view(np.uint8).reshape(len(values),len(values),8)
    return POPCOUNT[bytes_view].sum(axis=2).astype(np.uint8)


def component_ids(rows,max_distance=2):
    """Connected components of pHash-near images, restricted to same label."""
    result={}
    per_class={}
    for label in LETTERS:
        selected=[r for r in rows if r['label']==label and r['phash64']]
        values=[int(r['phash64'],16) for r in selected]
        dist=hamming_matrix(values)
        parent=list(range(len(selected)))
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        def union(a,b):
            a=find(a);b=find(b)
            if a!=b:parent[b]=a
        for i in range(len(selected)):
            for j in np.flatnonzero(dist[i,i+1:]<=max_distance)+i+1:
                union(i,int(j))
        groups=defaultdict(list)
        for i,row in enumerate(selected):groups[find(i)].append(row)
        for members in groups.values():
            # Hash membership list makes the group ID stable across row order.
            identifier=sha('\n'.join(sorted(r['sha256'] for r in members)).encode())[:20]
            for row in members:result[row['path']]=f'{label}:{identifier}'
        per_class[label]={'source_groups':len(groups),'multi_image_groups':sum(len(v)>1 for v in groups.values()),
            'largest_group':max((len(v) for v in groups.values()),default=0),
            'cross_original_split_groups':sum(len({r['original_split'] for r in v})>1 for v in groups.values()),
            'cross_original_split_images':sum(len(v) for v in groups.values() if len({r['original_split'] for r in v})>1)}
    return result,per_class


def assign_group_splits(rows,groups_by_path,config):
    by_class=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['include']=='1':by_class[row['label']][groups_by_path[row['path']]].append(row)
    fractions=config['split_fractions'];assignments={}
    for label,groups in sorted(by_class.items()):
        total=sum(len(v) for v in groups.values())
        if config.get('split_method')=='balanced_group_count_v2':
            group_ids=sorted(groups)
            if len(group_ids)<3:raise ValueError(f'Class {label} has fewer than three source groups')
            n_train=max(1,round(len(group_ids)*fractions['train']))
            n_val=max(1,round(len(group_ids)*fractions['val']))
            if n_train+n_val>=len(group_ids):n_train=len(group_ids)-2;n_val=1
            target=[fractions['train']*total,fractions['val']*total,fractions['test']*total]
            best=None
            # Keep group counts close to the requested fractions while searching
            # deterministic group assignments for the closest image-count split.
            for trial in range(1024):
                seed=int(sha(f"{config['seed']}:{label}:{trial}".encode())[:16],16)
                order=group_ids.copy();random.Random(seed).shuffle(order)
                partitions=(order[:n_train],order[n_train:n_train+n_val],order[n_train+n_val:])
                counts=[sum(len(groups[g]) for g in part) for part in partitions]
                score=sum(((counts[i]-target[i])/max(target[i],1))**2 for i in range(3))
                key=(score,tuple(order))
                if best is None or key<best[0]:best=(key,partitions)
            for split,part in zip(('train','val','test'),best[1]):
                for group in part:assignments[(label,group)]=split
        else:
            ordered=sorted(groups.items(),key=lambda item:(sha(f"{config['seed']}:{label}:{item[0]}".encode()),item[0]))
            cumulative=0
            for group,members in ordered:
                midpoint=(cumulative+len(members)/2)/total
                if midpoint<fractions['train']:split='train'
                elif midpoint<fractions['train']+fractions['val']:split='val'
                else:split='test'
                assignments[(label,group)]=split;cumulative+=len(members)
    for row in rows:
        if row['include']=='1':row['split']=assignments[(row['label'],groups_by_path[row['path']])]
        else:row['split']='excluded'
        if row['include']=='1':row['source_group']=groups_by_path[row['path']]
    return rows


def build_summary(rows,group_stats,config_hash,config):
    splits={}
    for split in ['train','val','test']:
        included=[r for r in rows if r['include']=='1' and r['split']==split]
        splits[split]={'total':len(included),'per_class':dict(Counter(r['label'] for r in included)),
            'source_groups':len({r['source_group'] for r in included}),
            'original_split_images':dict(Counter(r['original_split'] for r in included))}
    return {'version':config['version'],'config_sha256':config_hash,
        'total_images':len(rows),'included_images':sum(r['include']=='1' for r in rows),
        'excluded_images':dict(Counter(r['exclusion_reason'] for r in rows if r['include']=='0')),
        'splits':splits,'source_grouping':{'method':config['perceptual_hash']['algorithm'],
            'max_hamming_distance':config['perceptual_hash']['max_hamming_distance'],
            'within_class_only':True,'groups_by_class':group_stats,
            'exact_hash_train_val_overlap_after_assignment':0},
        'feature_note':'YOLO box annotations are checked but not consumed by the current hand-landmark pipeline.',
        'limitations':['Perceptual-hash components are conservative image-source candidates, not verified original-capture IDs.',
            'No participant or session metadata was supplied; this is not signer-independent.',
            'Original Kaggle train/val split had cross-split exact and near-duplicate content; the new split regroups perceptually similar images.',
            'License/citation metadata was absent from the downloaded directory and requires verification on the dataset source page.',
            'No model training or test metrics are included.']}


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/bisindo_kaggle_v1.json')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    if config['perceptual_hash']['max_hamming_distance']!=2:
        raise ValueError('This implementation is locked to pHash threshold 2')
    dataset_root=ROOT/'dataset'/config['dataset_root']
    images=sorted(p for p in (dataset_root/'images').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not images:raise FileNotFoundError(f'No images found under {dataset_root}/images')
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(lambda p:inspect_item(dataset_root,p),images))
    print(f'Inventoried {len(rows)} images; valid={sum(r["include"]=="1" for r in rows)}; excluded={sum(r["include"]=="0" for r in rows)}',flush=True)
    hash_labels=defaultdict(set)
    for row in rows:hash_labels[row['sha256']].add(row['label'])
    conflicts=[h for h,labels in hash_labels.items() if len(labels)>1]
    if conflicts:raise ValueError(f'Exact duplicate images have conflicting class-folder labels: {len(conflicts)} hashes')
    groups_by_path,group_stats=component_ids(rows,config['perceptual_hash']['max_hamming_distance'])
    assign_group_splits(rows,groups_by_path,config)
    included={(r['label'],r['split']) for r in rows if r['include']=='1'}
    if any((label,split) not in included for label in LETTERS for split in ('train','val','test')):
        raise ValueError('One or more classes are absent from an assigned split')
    split_groups=defaultdict(set)
    split_hashes=defaultdict(set)
    for row in rows:
        if row['include']=='1':
            split_groups[row['source_group']].add(row['split']);split_hashes[row['sha256']].add(row['split'])
    if any(len(v)>1 for v in split_groups.values()) or any(len(v)>1 for v in split_hashes.values()):
        raise ValueError('A source group or exact image hash crosses new splits')
    config_hash=sha(canonical(config))
    output_dir=ROOT/config['output_dir'];report_dir=ROOT/config['report_dir']
    output_dir.mkdir(parents=True,exist_ok=True);report_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=output_dir/'manifest.csv'
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_stream=io.StringIO(newline='')
    writer=csv.DictWriter(manifest_stream,fieldnames=FIELDS,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    manifest_bytes=manifest_stream.getvalue().encode('utf-8')
    summary=build_summary(rows,group_stats,config_hash,config)
    summary_bytes=canonical(summary)
    identity={'version':config['version'],'config_sha256':config_hash,
        'manifest_sha256':sha(manifest_bytes),'summary_sha256':sha(summary_bytes),
        'phash_components':'opencv-contrib-pHash-hamming<=2-within-class'}
    lock_path=output_dir/'split_lock.json';lock_bytes=canonical(identity)
    if lock_path.exists() and lock_path.read_bytes()!=lock_bytes:
        raise ValueError('Manifest snapshot is locked; create a new config/output version for changes')
    manifest_path.write_bytes(manifest_bytes)
    summary_path=output_dir/'summary.json';summary_path.write_bytes(summary_bytes)
    lock_path.write_bytes(lock_bytes)
    (report_dir/'summary.json').write_bytes(canonical(summary))
    (report_dir/'split_lock.json').write_bytes(lock_bytes)
    report='''# Kaggle BISINDO dataset - audit and split proposal\n\n'''
    report+=f"Source directory: `dataset/{config['dataset_root']}/`. Snapshot: `{config['version']}`.\n\n"
    report+=f"Inventory: **{summary['total_images']:,} images**; **{summary['included_images']:,} included**, **{summary['total_images']-summary['included_images']:,} excluded**. Original dimensions are 640 x 640. Class folders A-Z map to YOLO class IDs 0-25.\n\n"
    report+='## New split after perceptual grouping\n\n| Split | Images | Source-image groups | From original train | From original val |\n|---|---:|---:|---:|---:|\n'
    for name in ('train','val','test'):
        s=summary['splits'][name];report+=f"| {name} | {s['total']:,} | {s['source_groups']:,} | {s['original_split_images'].get('train',0):,} | {s['original_split_images'].get('val',0):,} |\n"
    report+='\n## Findings and limits\n\n'
    report+='- The provided train/val split had exact duplicate image groups across splits. The new assignment groups files within each letter by connected components of 64-bit perceptual hashes with Hamming distance <= 2, then assigns each component to one split. Visual spot-checks of candidate pairs showed same-capture images with augmentation changes.\n'
    report+='- This perceptual grouping reduces augmented-image leakage but is a heuristic, not a recovered capture/session or signer ID. We cannot claim signer-independent evaluation.\n'
    if config.get('split_method')=='balanced_group_count_v2':
        report+='- V2 balances source-group counts per class close to 70/15/15, then searches assignments to reduce image-count drift. It is a new snapshot; earlier v1 validation results do not transfer.\n'
    report+='- Missing and empty YOLO annotation files are excluded from this classification candidate manifest; original files remain unchanged. YOLO boxes are audited but the current landmark classifier reads full images.\n'
    report+='- The downloaded folder has no README/license metadata. Verify Kaggle attribution and license before external release.\n'
    report+=f"- Full rows, hashes, assignments, and per-class summary are local under `{config['output_dir']}/`; generated data is ignored by Git.\n\n"
    report+='Per-class counts and exact configuration/manifest hashes are in `summary.json` and `split_lock.json`. No classifier training or test metrics are included in this split-generation report.\n'
    (report_dir/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps({'summary':summary,'manifest':str(manifest_path.relative_to(ROOT))},indent=2),flush=True)


if __name__=='__main__':main()
