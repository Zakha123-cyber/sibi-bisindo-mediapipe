"""Render validation examples with and without MediaPipe landmarks.

Reads only the BISINDO Kaggle v2 confidence-0.3 validation cache.
"""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20)]

sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw,ImageOps


def pick_examples(arrays,manifest_rows):
    manifest={r['path']:r for r in manifest_rows if r['include']=='1' and r['split']=='val'}
    selected={}
    for letter in LETTERS:
        candidates=[i for i,label in enumerate(arrays['labels']) if str(label)==letter]
        good=[i for i in candidates if bool(arrays['valid_mask'][i])]
        bad=[i for i in candidates if not bool(arrays['valid_mask'][i])]
        if not good:
            raise ValueError(f'Validation lacks feature-valid example for class {letter}')
        good_by_group={};bad_by_group={}
        for i in good:good_by_group.setdefault(manifest[str(arrays['paths'][i])]['source_group'],[]).append(i)
        for i in bad:bad_by_group.setdefault(manifest[str(arrays['paths'][i])]['source_group'],[]).append(i)
        shared=sorted(set(good_by_group)&set(bad_by_group))
        if shared:
            group=max(shared,key=lambda g:(len(good_by_group[g])+len(bad_by_group[g]),g))
            good_idx=min(good_by_group[group],key=lambda i:str(arrays['paths'][i]))
            bad_idx=min(bad_by_group[group],key=lambda i:str(arrays['paths'][i]))
            paired=True
        else:
            good_idx=min(good,key=lambda i:str(arrays['paths'][i]))
            bad_idx=min(bad,key=lambda i:str(arrays['paths'][i])) if bad else None
            paired=False
        selected[letter]=(good_idx,bad_idx,paired)
    return selected,manifest


def tile(image_path,landmarks,label,status,hand_count,same_group,filename):
    with Image.open(image_path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
    if landmarks is not None:
        draw=ImageDraw.Draw(image)
        width,height=image.size
        for slot,points in enumerate(landmarks):
            color=(0,220,70) if slot==0 else (245,40,190)
            coords=[(round(float(p[0])*width),round(float(p[1])*height)) for p in points]
            for a,b in CONNECTIONS:draw.line((coords[a],coords[b]),fill=color,width=max(2,width//250))
            radius=max(2,width//200)
            for x,y in coords:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=(255,30,30))
    thumb=ImageOps.contain(image,(310,205))
    canvas=Image.new('RGB',(320,260),'white')
    canvas.paste(thumb,((320-thumb.width)//2,5))
    draw=ImageDraw.Draw(canvas)
    if status=='valid':caption=f'{label}  LANDMARK OK  ({hand_count} hand)'
    else:caption=f'{label}  NO HAND'
    draw.text((6,215),caption,fill=(0,110,30) if status=='valid' else (170,25,25))
    draw.text((6,232),'same pHash group' if same_group else 'different source group',fill=(50,50,50))
    draw.text((6,247),filename[:46],fill=(70,70,70))
    return canvas


def empty_tile(label,message):
    canvas=Image.new('RGB',(320,260),(248,248,248))
    draw=ImageDraw.Draw(canvas)
    draw.text((10,12),f'{label}  {message}',fill=(70,70,70))
    return canvas


def main():
    cache=ROOT/'data/features/landmarks-bisindo-kaggle-v2-conf03/bisindo_val.npz'
    manifest_path=ROOT/'data/manifests/bisindo-kaggle-v2/manifest.csv'
    out=ROOT/'reports/bisindo_kaggle_v2/landmark-comparison-conf03'
    out.mkdir(parents=True,exist_ok=True)
    with np.load(cache,allow_pickle=False) as saved:
        arrays={name:saved[name] for name in saved.files}
    rows=list(csv.DictReader(manifest_path.open(encoding='utf-8',newline='')))
    selected,manifest=pick_examples(arrays,rows)
    metadata=[]
    for page,letters in [('A-M',LETTERS[:13]),('N-Z',LETTERS[13:])]:
        columns=4;cell_w,cell_h=320,260
        title_h=34
        canvas=Image.new('RGB',(columns*cell_w,title_h+((len(letters)*2+columns-1)//columns)*cell_h),(235,238,242))
        draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'BISINDO v2 validation, confidence 0.3 - {page}',fill='black')
        y_offset=title_h
        for li,letter in enumerate(letters):
            good_idx,bad_idx,paired=selected[letter]
            for col_idx,(idx,status) in enumerate(((good_idx,'valid'),(bad_idx,'no_hand'))):
                col=li%2*2+col_idx
                row=li//2
                # Alternate class pairs across four columns for compact side-by-side review.
                if idx is None:
                    cell=empty_tile(letter,'no no-hand example in validation')
                    group='';same=False;relative=''
                else:
                    relative=str(arrays['paths'][idx])
                    image_path=ROOT/'dataset/Indonesian Sign Language - BISINDO'/relative
                    group=manifest[relative]['source_group'];same=paired
                    points=arrays['raw_landmarks'][idx,:int(arrays['hand_count'][idx])] if status=='valid' else None
                    cell=tile(image_path,points,letter,status,int(arrays['hand_count'][idx]),same,Path(relative).name)
                canvas.paste(cell,(col*cell_w,y_offset+row*cell_h))
                metadata.append({'class':letter,'status':status,'path':relative or None,
                    'source_group':group,'paired_same_group':paired,'hand_count':int(arrays['hand_count'][idx]) if idx is not None else None})
        canvas.save(out/f'{page}.jpg',quality=92)
    (out/'selection.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(out)


if __name__=='__main__':main()
