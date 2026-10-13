"""Read-only dataset audit. Run: python scripts/audit_dataset.py"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import re

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
OUT = ROOT / 'reports' / 'dataset_audit'


def inspect(path):
    relative = path.relative_to(DATA)
    parts = relative.parts
    language = 'BISINDO' if parts[0] == 'BISINDO DATASET' else 'SIBI'
    variant = parts[1]
    split = parts[parts.index('images')+1] if language == 'SIBI' else 'unsplit'
    match = re.search(r'(?:^|__)([A-Z])_', path.name)
    letter = path.parent.name if language == 'BISINDO' else (match[1] if match else '')
    # Roboflow suffix and extension tokens do not identify independent captures.
    origin = re.split(r'\.rf\.', path.stem)[0]
    origin = re.sub(r'_(?:jpg|png|jpeg)$', '', origin)
    row = dict(path=str(relative), language=language, variant=variant, split=split,
               letter=letter, origin=origin, sha256='', width='', height='',
               mode='', class_ids='', boxes=0, error='', annotation_error='')
    try:
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode)
            im.load()
    except Exception as exc:
        row['error'] = str(exc)
    if language == 'SIBI':
        variant_root = DATA / parts[0] / variant
        annotation = variant_root / 'labels' / Path(*parts[3:]).with_suffix('.txt')
        try:
            lines = annotation.read_text().splitlines()
            ids = []
            for line in lines:
                tokens = line.split()
                if len(tokens) != 5:
                    raise ValueError('Expected YOLO class cx cy width height')
                cls = int(tokens[0])
                x, y, w, h = map(float, tokens[1:])
                if cls < 0 or not all(0 <= v <= 1 for v in (x,y,w,h)) or w <= 0 or h <= 0:
                    raise ValueError('Invalid class or normalized box')
                if x-w/2 < -0.001 or y-h/2 < -0.001 or x+w/2 > 1.001 or y+h/2 > 1.001:
                    raise ValueError('Box extends beyond image')
                ids.append(cls)
            row['class_ids'] = ','.join(map(str, sorted(set(ids))))
            row['boxes'] = len(lines)
            if not lines:
                row['annotation_error'] = 'Empty annotation'
        except Exception as exc:
            row['annotation_error'] = str(exc)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in DATA.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for i, row in enumerate(pool.map(inspect, paths), 1):
            rows.append(row)
            if i % 5000 == 0:
                print(f'Inspected {i}/{len(paths)}', flush=True)
    with (OUT/'inventory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['language']+'/'+row['variant']].append(row)
    summary = {}
    for key, items in groups.items():
        summary[key] = dict(images=len(items), classes=dict(sorted(Counter(r['letter'] for r in items).items())),
            splits=dict(Counter(r['split'] for r in items)),
            dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in items)),
            corrupt=sum(bool(r['error']) for r in items),
            annotation_errors=sum(bool(r['annotation_error']) for r in items),
            empty_annotations=sum(r['annotation_error']=='Empty annotation' for r in items))
    duplicates = []
    for field in ['sha256','origin']:
        buckets = defaultdict(list)
        for r in rows:
            if r[field]:
                # Origins are only comparable within a language.
                buckets[(r['language'],r[field])].append(r)
        for (language, value), items in buckets.items():
            if len(items) < 2:
                continue
            variants = {r['variant'] for r in items}
            split_sets = defaultdict(set)
            for r in items:
                split_sets[r['variant']].add(r['split'])
            duplicates.append(dict(method=field, language=language, value=value,
                count=len(items), cross_variant=len(variants)>1,
                cross_split_within_variant=any(len(s)>1 for s in split_sets.values()),
                conflicting_letters=len({r['letter'] for r in items})>1,
                paths=[r['path'] for r in items]))
    mapping = defaultdict(Counter)
    for r in rows:
        if r['language']=='SIBI':
            mapping[r['letter']][r['class_ids']] += 1
    image_stems = {str((DATA/r['path']).with_suffix('')) for r in rows}
    orphan_labels = []
    for p in DATA.rglob('*.txt'):
        if 'labels' in p.parts:
            image_path = Path(str(p).replace('\\labels\\','\\images\\').replace('/labels/','/images/'))
            if str(image_path.with_suffix('')) not in image_stems:
                orphan_labels.append(str(p.relative_to(DATA)))
    result = dict(variants=summary, sibi_filename_to_annotation_ids={k:dict(v) for k,v in sorted(mapping.items())},
                  orphan_labels=orphan_labels,
                  duplicate_summary={field:dict(groups=sum(d['method']==field for d in duplicates),
                    cross_split_groups=sum(d['method']==field and d['cross_split_within_variant'] for d in duplicates),
                    cross_variant_groups=sum(d['method']==field and d['cross_variant'] for d in duplicates),
                    conflicting_letter_groups=sum(d['method']==field and d['conflicting_letters'] for d in duplicates))
                    for field in ['sha256','origin']})
    (OUT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT/'duplicates.json').write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    for key, items in groups.items():
        if 'with_augmentation' in key:
            continue
        chosen = {}
        for r in items:
            if not r['error']:
                chosen.setdefault(r['letter'], r)
        sheet = Image.new('RGB', (6*180, 5*170), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (letter,r) in enumerate(sorted(chosen.items())):
            with Image.open(DATA/r['path']) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (176,140))
                x,y = (i%6)*180,(i//6)*170
                sheet.paste(thumb,(x,y+24))
                draw.text((x+5,y+5),letter,fill='black')
        sheet.save(OUT/(key.replace('/','_')+'_samples.jpg'))
    lines = ['# Audit dataset SIBI dan BISINDO', '',
             'Audit membaca seluruh gambar tanpa mengubah dataset. Detail tersedia di `inventory.csv`, `summary.json`, dan `duplicates.json`.', '',
             '| Varian | Gambar | Kelas | Gambar rusak | Masalah anotasi |',
             '|---|---:|---:|---:|---:|']
    for key,s in summary.items():
        lines.append(f"| {key} | {s['images']} | {len(s['classes'])} | {s['corrupt']} | {s['annotation_errors']} |")
    lines += ['', '## Duplikasi', '', '```json', json.dumps(result['duplicate_summary'],indent=2), '```', '',
        'SHA-256 mendeteksi file identik, bukan kemiripan visual. Origin adalah kandidat sumber berdasarkan nama file; perlu verifikasi sebelum digunakan sebagai ID grup.',
        'ID kelas SIBI disandingkan dengan huruf dari nama file, bukan metadata resmi. Periksa pemetaan dalam summary.json.',
        f'Label tanpa pasangan gambar: {len(orphan_labels)}.', '',
        'Pemeriksaan struktur ini belum mengukur keberhasilan MediaPipe, kesamaan visual, identitas pengguna, atau kebenaran linguistik pose. Jumlah kelas bukan jaminan dukungan pose statis.',
        'Lihat review_summary.json untuk ketidaksesuaian kelas dan kandidat kebocoran sumber lintas latar. Tidak adanya duplikasi hash antar split bukan bukti split bebas kebocoran.',
        'Hasil audit landmark terpisah tersedia di landmark_summary.json setelah menjalankan scripts/audit_landmarks.py. Interpretasi snapshot tersedia di FINDINGS.md.',
        'Gunakan varian tanpa augmentasi untuk baseline. Pisahkan sumber/pengguna/sesi sebelum augmentasi; jangan gabungkan seluruh varian secara langsung.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
    from review_dataset_audit import main as review
    review()
