"""One-time final evaluation of the frozen BISINDO image classifier on test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'artifacts/bisindo-image-run-01')
    parser.add_argument('--dataset-config', type=Path, default=ROOT / 'configs/bisindo_kaggle_v2.json')
    parser.add_argument('--manifest-dir', type=Path, default=ROOT / 'data/manifests/bisindo-kaggle-v2')
    parser.add_argument('--dataset-root', type=Path,
                        default=ROOT / 'dataset/Indonesian Sign Language - BISINDO')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--confirm-frozen-selection', action='store_true',
                        help='Confirm the model and all settings are frozen; test is evaluated once.')
    args = parser.parse_args()
    if not args.confirm_frozen_selection:
        parser.error('Review validation, freeze the selection, then pass --confirm-frozen-selection.')
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch-size must be positive and num-workers cannot be negative')

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    config_path = args.dataset_config if args.dataset_config.is_absolute() else ROOT / args.dataset_config
    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else ROOT / args.manifest_dir
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    checkpoint_path = run_dir / 'best.pt'
    metrics_path = run_dir / 'best_validation_metrics.json'
    state_path = run_dir / 'training_state.json'
    marker_path = run_dir / 'test_evaluation_started.json'
    report_path = run_dir / 'test_metrics.json'

    if marker_path.exists() or report_path.exists():
        raise FileExistsError(f'Test evaluation already started or completed for this run: {run_dir}')
    for path in (checkpoint_path, metrics_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('status') != 'complete' or state.get('test_accessed') is not False:
        raise ValueError('Training state is incomplete or does not confirm test_accessed=false')
    val_metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    checkpoint_sha = sha256_file(checkpoint_path)

    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest_path = manifest_dir / 'manifest.csv'
    lock_path = manifest_dir / 'split_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    canonical_config = (json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    config_sha = hashlib.sha256(canonical_config).hexdigest()
    manifest_sha = sha256_file(manifest_path)
    if (lock.get('version') != config.get('version')
            or lock.get('config_sha256') != config_sha
            or lock.get('manifest_sha256') != manifest_sha):
        raise ValueError('Config or manifest does not match its split lock')

    test_rows = []
    with manifest_path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['include'] == '1' and row['split'] == 'test':
                if row['language'] != 'bisindo' or row['label'] not in LETTERS:
                    raise ValueError(f'Unexpected test row: {row}')
                test_rows.append(row)
    if not test_rows or {row['label'] for row in test_rows} != set(LETTERS):
        raise ValueError('Test split is empty or does not contain all 26 classes')

    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('architecture') != 'mobilenet_v3_small'
            or checkpoint.get('dataset_version') != config['version']
            or checkpoint.get('manifest_sha256') != manifest_sha
            or checkpoint.get('class_names') != LETTERS
            or checkpoint.get('epoch') != val_metrics.get('epoch')
            or abs(checkpoint.get('validation_macro_f1', -1) - val_metrics.get('macro_f1', -2)) > 1e-12):
        raise ValueError('Checkpoint metadata does not match the selected validation checkpoint')

    # Commit the one-time marker before opening any test image. An interrupted run
    # must not silently become another attempt against the same test split.
    write_json(marker_path, {
        'status': 'started', 'test_manifest_sha256': manifest_sha,
        'checkpoint_sha256': checkpoint_sha, 'checkpoint_epoch': checkpoint['epoch'],
        'started_unix_seconds': time.time(),
    })

    for index, row in enumerate(test_rows, 1):
        path = dataset_root / row['path']
        if not path.is_file() or sha256_file(path) != row['sha256']:
            raise ValueError(f'Missing or changed test image: {row["path"]}')
        if index % 500 == 0:
            print(f'Verified {index}/{len(test_rows)} test images', flush=True)

    mean = checkpoint['normalization']['mean']
    std = checkpoint['normalization']['std']
    transform = transforms.Compose([
        transforms.Resize((checkpoint['image_size'], checkpoint['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    class TestDataset(Dataset):
        def __len__(self):
            return len(test_rows)

        def __getitem__(self, index):
            row = test_rows[index]
            with Image.open(dataset_root / row['path']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            return transform(image), LETTERS.index(row['label'])

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LETTERS))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    loader = DataLoader(TestDataset(), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                        persistent_workers=args.num_workers > 0)
    truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=device.type == 'cuda'))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.tolist())

    report = classification_report(truth, predictions, labels=list(range(len(LETTERS))),
                                   target_names=LETTERS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=list(range(len(LETTERS))))
    mistakes = [
        {'path': row['path'], 'true': LETTERS[actual], 'predicted': LETTERS[predicted]}
        for row, actual, predicted in zip(test_rows, truth, predictions) if actual != predicted
    ]
    result = {
        'version': 'bisindo-image-test-report-v1',
        'dataset_version': config['version'],
        'test_manifest_sha256': manifest_sha,
        'checkpoint': str(checkpoint_path.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_sha,
        'checkpoint_epoch': checkpoint['epoch'],
        'device': str(device),
        'sample_count': len(test_rows),
        'accuracy': float(accuracy_score(truth, predictions)),
        'macro_f1': float(f1_score(truth, predictions, labels=list(range(len(LETTERS))),
                                   average='macro', zero_division=0)),
        'confusion_matrix_labels': LETTERS,
        'confusion_matrix': matrix.tolist(),
        'per_class': {letter: report[letter] for letter in LETTERS},
        'mistakes': mistakes,
        'limitations': [
            'Labeled still images only; no webcam trials or no-hand/unknown examples.',
            'The split is grouped by perceptual similarity, not verified signer/session IDs.',
        ],
    }
    write_json(report_path, result)
    write_json(marker_path, {
        'status': 'complete', 'test_manifest_sha256': manifest_sha,
        'checkpoint_sha256': checkpoint_sha, 'checkpoint_epoch': checkpoint['epoch'],
        'report': report_path.name,
    })
    print(f"Test evaluated once: n={len(test_rows)} accuracy={result['accuracy']:.4f} "
          f"macro-F1={result['macro_f1']:.4f}; report={report_path}", flush=True)


if __name__ == '__main__':
    main()
