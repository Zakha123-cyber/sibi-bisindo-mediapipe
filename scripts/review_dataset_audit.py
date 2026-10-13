"""Create actionable review lists from the completed inventory (no image rescan)."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'
OUT = ROOT / 'reports' / 'dataset_audit'


def main():
    with (OUT / 'inventory.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    mappings = defaultdict(Counter)
    for row in rows:
        if row['language'] == 'SIBI' and row['class_ids'] and not row['annotation_error']:
            mappings[row['letter']][row['class_ids']] += 1
    expected = {letter: counts.most_common(1)[0][0] for letter, counts in mappings.items()}
    issues = []
    for row in rows:
        reasons = []
        if row['error']:
            reasons.append('image_error: ' + row['error'])
        if row['annotation_error']:
            reasons.append('annotation_error: ' + row['annotation_error'])
        if row['language'] == 'SIBI' and row['class_ids'] and row['class_ids'] != expected.get(row['letter']):
            reasons.append('filename_annotation_disagreement')
        if not row['letter']:
            reasons.append('unparsed_letter')
        if reasons:
            issues.append(dict(path=row['path'], variant=row['variant'], split=row['split'],
                filename_letter=row['letter'], annotation_ids=row['class_ids'],
                majority_id_for_filename_letter=expected.get(row['letter'], ''), reason='; '.join(reasons)))
    columns = ['path','variant','split','filename_letter','annotation_ids','majority_id_for_filename_letter','reason']
    with (OUT / 'review_required.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(issues)
    # Background replacement can keep the same foreground capture but change all bytes.
    # These are filename-derived candidates, not verified source/session identifiers.
    candidates = defaultdict(list)
    for row in rows:
        if row['language'] == 'SIBI' and 'multi_bg' in row['variant']:
            base = re.sub(r'^bg_\d+__', '', row['origin'])
            candidates[(row['variant'], base)].append(row)
    crossings = []
    for (variant, base), items in sorted(candidates.items()):
        splits = sorted({r['split'] for r in items})
        backgrounds = sorted({r['origin'].split('__')[0] for r in items})
        if len(splits) > 1 and len(backgrounds) > 1:
            crossings.append(dict(variant=variant, candidate_capture=base, splits=splits,
                backgrounds=backgrounds, paths=[r['path'] for r in items]))
    (OUT / 'background_source_candidates.json').write_text(json.dumps(crossings, indent=2), encoding='utf-8')
    per_split = defaultdict(Counter)
    for row in rows:
        per_split[(row['language'], row['variant'], row['split'])][row['letter']] += 1
    coverage = []
    for (language, variant, split), counts in sorted(per_split.items()):
        all_letters = {r['letter'] for r in rows if r['variant'] == variant}
        coverage.append(dict(language=language, variant=variant, split=split,
            classes=len(counts), min_class_images=min(counts.values()), max_class_images=max(counts.values()),
            missing_classes=sorted(all_letters - counts.keys()), counts=dict(sorted(counts.items()))))
    result = dict(inferred_sibi_mapping=dict(sorted(expected.items())),
        annotation_disagreements=sum('filename_annotation_disagreement' in r['reason'] for r in issues),
        review_images=len(issues), candidate_cross_background_split_groups=len(crossings),
        candidate_groups_by_variant=dict(Counter(r['variant'] for r in crossings)), split_class_coverage=coverage)
    (OUT / 'review_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    chosen = [r for r in issues if 'with_augmentation' not in r['variant']]
    if chosen:
        sheet = Image.new('RGB', (3*300, ((len(chosen)+2)//3)*260), 'white')
        draw = ImageDraw.Draw(sheet)
        for i,row in enumerate(chosen):
            with Image.open(DATA / row['path']) as im:
                thumb = ImageOps.contain(im.convert('RGB'), (296,220))
            x,y = (i%3)*300,(i//3)*260
            sheet.paste(thumb,(x,y+35))
            draw.text((x+3,y+4), f"file: {row['filename_letter']} | annotation: {row['annotation_ids'] or 'empty'}", fill='black')
        sheet.save(OUT / 'annotation_review_samples.jpg')
    if crossings:
        sample = crossings[0]
        paths = sample['paths'][:6]
        sheet = Image.new('RGB', (len(paths)*200,220), 'white')
        draw = ImageDraw.Draw(sheet)
        for i,path in enumerate(paths):
            row = next(r for r in rows if r['path'] == path)
            with Image.open(DATA / path) as im:
                thumb = ImageOps.contain(im.convert('RGB'),(196,185))
            sheet.paste(thumb,(i*200,30))
            draw.text((i*200+3,5),row['split']+' '+row['origin'].split('__')[0],fill='black')
        sheet.save(OUT / 'background_source_candidate_sample.jpg')
        (OUT / 'background_source_candidate_sample.json').write_text(json.dumps(sample,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'split_class_coverage'}, indent=2))


if __name__ == '__main__':
    main()
