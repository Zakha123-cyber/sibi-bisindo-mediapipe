"""Summarize extraction and render train/validation overlays for review.

python scripts/review_landmark_features.py --pilot
python scripts/review_landmark_features.py --full
Test images are never rendered by this review tool.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from sibi_bisindo.landmarks import preprocess_rgb

OUT=ROOT/'reports/landmarks'
CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
             (13,17),(0,17),(17,18),(18,19),(19,20)]


def read_npz(path):
    with np.load(path,allow_pickle=False) as saved:
        return {k:saved[k] for k in saved.files}


def save_sheet(name, arrays, indices, config):
    if not indices:return
    cols=6;cell_w,cell_h=240,210
    sheet=Image.new('RGB',(cols*cell_w,((len(indices)+cols-1)//cols)*cell_h),'white')
    title=ImageDraw.Draw(sheet);manifest=[]
    for j,i in enumerate(indices):
        with Image.open(ROOT/'dataset'/str(arrays['paths'][i])) as im:
            image=ImageOps.exif_transpose(im).convert('RGB')
            image=Image.fromarray(preprocess_rgb(np.asarray(image),config,input_mirrored=config['preprocessing']['offline_input_mirrored']))
        draw=ImageDraw.Draw(image);w,h=image.size
        for slot in range(int(arrays['hand_count'][i])):
            points=arrays['raw_landmarks'][i,slot]
            coords=[(int(p[0]*w),int(p[1]*h)) for p in points]
            color=(0,255,80) if slot==0 else (255,60,200)
            for a,b in CONNECTIONS:draw.line([coords[a],coords[b]],fill=color,width=3)
            for x,y in coords:draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
            draw.text(coords[0],f'slot{slot}',fill=(255,255,0))
        thumb=ImageOps.contain(image,(236,180));x,y=(j%cols)*cell_w,(j//cols)*cell_h
        sheet.paste(thumb,(x,y+28))
        title.text((x+3,y+3),f"{arrays['labels'][i]}: {arrays['hand_count'][i]} hands; valid={int(arrays['valid_mask'][i])}",fill='black')
        manifest.append(dict(path=str(arrays['paths'][i]),label=str(arrays['labels'][i]),
            hands=int(arrays['hand_count'][i]),status=str(arrays['status'][i])))
    sheet.save(OUT/(name+'.jpg'))
    (OUT/(name+'.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def per_class_examples(arrays, predicate):
    chosen=[]
    for label in sorted(set(arrays['labels'])):
        indices=[i for i in range(len(arrays['paths'])) if arrays['labels'][i]==label and predicate(i)]
        if indices:
            chosen.append(min(indices,key=lambda i:str(arrays['sha256'][i])))
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    if not args.pilot and not args.full:parser.error('Choose --pilot or --full')
    OUT.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'configs/landmarks_v1.json').read_text())
    contrast=json.loads((ROOT/'configs/landmarks_pilot_clahe_v1.json').read_text())
    comparisons={}
    for language in ['sibi','bisindo']:
        arrays=read_npz(ROOT/base['output_dir']/f'{language}_val.npz')
        for n in [0,1,2]:
            indices=per_class_examples(arrays,lambda i:arrays['hand_count'][i]==n)
            save_sheet(f'{language}_val_{n}_hands',arrays,indices,base)
        if args.pilot:
            other=read_npz(ROOT/contrast['output_dir']/f'{language}_val.npz')
            if list(arrays['paths'])!=list(other['paths']):raise ValueError('Pilot rows differ')
            a=arrays['valid_mask'];b=other['valid_mask']
            comparisons[language]=dict(total=len(a),baseline_feature_valid=int(a.sum()),clahe_feature_valid=int(b.sum()),
                baseline_hand_counts=dict(Counter(map(int,arrays['hand_count']))),
                clahe_hand_counts=dict(Counter(map(int,other['hand_count']))),
                gains=int((~a&b).sum()),losses=int((a&~b).sum()),
                baseline_median_ms=float(np.median(arrays['process_ms'])),clahe_median_ms=float(np.median(other['process_ms'])))
            gained=np.flatnonzero(~a&b)[:18].tolist()
            save_sheet(f'{language}_clahe_gained',other,gained,contrast)
            lost=np.flatnonzero(a&~b)[:18].tolist()
            save_sheet(f'{language}_clahe_lost_baseline',arrays,lost,base)
        if args.full:
            train=read_npz(ROOT/base['output_dir']/f'{language}_train.npz')
            for n in [0,1,2]:
                indices=per_class_examples(train,lambda i:train['hand_count'][i]==n)
                save_sheet(f'{language}_train_{n}_hands',train,indices,base)
    if args.pilot:
        result=dict(evaluation_split='validation_only',compared='RGB unchanged vs LAB-luminance CLAHE clip=2 grid=8',
            results=comparisons,limitations=['No negative ground-truth validation set available yet.',
                'Coverage gains do not establish landmark correctness, same-actor assignment or lower false acceptance.'])
        (OUT/'pilot_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2))
    if args.full:
        summary=json.loads((ROOT/base['output_dir']/'summary.json').read_text())
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        lock=json.loads((ROOT/base['output_dir']/'extraction_lock.json').read_text())
        (OUT/'extraction_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
        print('Full summary and train/validation overlays saved.')


if __name__=='__main__':main()
