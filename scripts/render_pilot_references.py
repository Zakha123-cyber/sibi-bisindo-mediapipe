"""Create train-only candidate pose sheets; these are not approved sign references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/webcam_pilot_v1.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/pilot_references/v1')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    sources = {
        'sibi': (ROOT / 'data/manifests/data-v1/sibi_train.csv', ROOT / 'dataset'),
        'bisindo': (ROOT / 'data/manifests/bisindo-kaggle-v2/manifest.csv',
                    ROOT / 'dataset/Indonesian Sign Language - BISINDO'),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    except OSError:
        font = ImageFont.load_default()
    chosen = []
    for mode, (manifest_path, dataset_root) in sources.items():
        with manifest_path.open(encoding='utf-8', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['split'] == 'train' and r['include'] == '1']
        sheet = Image.new('RGB', (1500, 720), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((16, 12), f'{mode.upper()} | KANDIDAT POSE DATASET - BELUM DIVERIFIKASI', fill='black', font=font)
        draw.text((16, 40), 'Hanya contoh train; cocokkan dengan referensi/praktisi sebelum menetapkan label.', fill='black', font=font)
        for column, label in enumerate(config['languages'][mode]['classes']):
            candidates = [r for r in rows if r['label'] == label]
            candidates.sort(key=lambda r: (not Path(r['path']).name.startswith('image_'), r['sha256']))
            selected, groups = [], set()
            for row in candidates:
                if row['source_group'] not in groups:
                    selected.append(row)
                    groups.add(row['source_group'])
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                raise ValueError(f'Not enough distinct train groups for {mode}/{label}')
            for index, row in enumerate(selected):
                path = (dataset_root / row['path']).resolve()
                if not path.is_relative_to(dataset_root.resolve()):
                    raise ValueError('Source image path escapes its dataset root')
                if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                    raise ValueError(f'Source image differs from manifest: {path}')
                with Image.open(path) as source:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (280, 260))
                x, y = column*300+10, 95+index*310
                sheet.paste(thumb, (x+(280-thumb.width)//2, y))
                draw.text((x, y+265), f'{label} | contoh {index+1} | pending', fill='black', font=font)
                chosen.append({'mode': mode, 'label': label, 'source_path': str(path.relative_to(ROOT)),
                               'sha256': row['sha256'], 'source_group': row['source_group'],
                               'split': 'train', 'pose_review_status': 'pending'})
        path = output / f'{mode}_candidate_poses.jpg'
        sheet.save(path, quality=92)
        print(path)
    (output / 'sources.json').write_text(json.dumps(chosen, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
