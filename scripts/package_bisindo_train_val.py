"""Package only locked BISINDO v2 train/validation images for Colab transfer.

The test split is excluded from the archive and its image files are never read.
JPEGs are stored without recompression to keep packaging fast.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config', type=Path, default=ROOT / 'configs/bisindo_kaggle_v2.json')
    parser.add_argument('--manifest-dir', type=Path, default=Path('data/manifests/bisindo-kaggle-v2'))
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'dataset/Indonesian Sign Language - BISINDO')
    parser.add_argument('--output', type=Path, default=ROOT / 'bisindo_train_val_images.zip')
    args = parser.parse_args()

    config_path = args.dataset_config if args.dataset_config.is_absolute() else ROOT / args.dataset_config
    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else ROOT / args.manifest_dir
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest_path = manifest_dir / 'manifest.csv'
    lock = json.loads((manifest_dir / 'split_lock.json').read_text(encoding='utf-8'))
    config_bytes = (json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (lock.get('config_sha256') != config_hash or lock.get('manifest_sha256') != manifest_hash
            or lock.get('version') != config['version']):
        raise ValueError('Config/manifest does not match the locked BISINDO split')

    rows = []
    with manifest_path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['include'] == '1' and row['split'] in {'train', 'val'}:
                rows.append(row)
    if not rows or any(row['split'] == 'test' for row in rows):
        raise ValueError('Unexpected train/validation selection')

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, mode='w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for index, row in enumerate(rows, 1):
            path = dataset_root / row['path']
            content = path.read_bytes()
            if sha256_bytes(content) != row['sha256']:
                raise ValueError(f'Missing or changed train/validation image: {row["path"]}')
            archive.writestr(row['path'], content)
            if index % 2000 == 0:
                print(f'Packaged {index}/{len(rows)} train/validation images', flush=True)

    print(json.dumps({
        'output': str(output),
        'train_images': sum(row['split'] == 'train' for row in rows),
        'validation_images': sum(row['split'] == 'val' for row in rows),
        'test_images_in_archive': 0,
        'archive_bytes': output.stat().st_size,
        'manifest_sha256': manifest_hash,
    }, indent=2))


if __name__ == '__main__':
    main()
