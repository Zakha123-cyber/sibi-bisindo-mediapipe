"""Train a direct-image BISINDO alphabet classifier (no hand landmarks).

Uses only the locked v2 train and validation splits. The test split is never
loaded. Checkpoints and per-epoch metrics are written to the local artifacts
directory.
"""
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
IMAGE_SIZE = 224
SEED = 42


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_manifest(dataset_config_path, manifest_dir):
    config = json.loads(dataset_config_path.read_text(encoding='utf-8'))
    manifest = ROOT / manifest_dir / 'manifest.csv'
    lock_path = ROOT / manifest_dir / 'split_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    manifest_hash = sha256_file(manifest)
    canonical_config = (json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    config_hash = hashlib.sha256(canonical_config).hexdigest()
    if (lock.get('manifest_sha256') != manifest_hash or lock.get('config_sha256') != config_hash
            or lock.get('version') != config['version']):
        raise ValueError('Manifest does not match the locked BISINDO v2 snapshot')

    selected = {'train': [], 'val': []}
    with manifest.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            split = row['split']
            if split not in selected or row['include'] != '1':
                continue
            if row['language'] != 'bisindo' or row['label'] not in LETTERS:
                raise ValueError(f'Unexpected label/language in manifest: {row}')
            selected[split].append(row)
    for split, rows in selected.items():
        if not rows or {row['label'] for row in rows} != set(LETTERS):
            raise ValueError(f'{split} is empty or does not contain all 26 letters')
    return config, lock, manifest_hash, selected


def verify_images(rows, image_root):
    for index, row in enumerate(rows, 1):
        path = image_root / row['path']
        if not path.is_file() or sha256_file(path) != row['sha256']:
            raise ValueError(f'Missing or changed training/validation image: {row["path"]}')
        if index % 2000 == 0:
            print(f'Verified {index}/{len(rows)} train/validation images', flush=True)
    print(f'Verified all {len(rows)} train/validation images; test files were not opened.', flush=True)


def make_dataset_class(torch, Image, ImageOps, image_root):
    class SignImageDataset(torch.utils.data.Dataset):
        def __init__(self, rows, image_transform):
            self.rows = rows
            self.transform = image_transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            path = image_root / row['path']
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            return self.transform(image), LETTERS.index(row['label'])

    return SignImageDataset


def score_epoch(model, loader, criterion, device, optimizer=None, torch=None, frozen_features=False):
    training = optimizer is not None
    model.train(training)
    if training and frozen_features:
        model.features.eval()
    total_loss = 0.0
    total = 0
    true_labels = []
    predictions = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == 'cuda')
            labels = labels.to(device, non_blocking=device.type == 'cuda')
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch = labels.size(0)
            total_loss += float(loss.detach()) * batch
            total += batch
            true_labels.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(total, 1), true_labels, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config', type=Path, default=ROOT / 'configs/bisindo_kaggle_v2.json')
    parser.add_argument('--manifest-dir', type=Path, default=Path('data/manifests/bisindo-kaggle-v2'))
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'dataset/Indonesian Sign Language - BISINDO')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Train only the new classifier head for the first N epochs')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--weights', choices=['imagenet', 'none'], default='imagenet')
    parser.add_argument('--skip-image-hash-check', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error('Require epochs >= 1 and 0 <= warmup-epochs < epochs')
    if args.batch_size < 1 or args.num_workers < 0 or args.patience < 1:
        parser.error('batch-size and patience must be positive; num-workers cannot be negative')

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset_config_path = args.dataset_config if args.dataset_config.is_absolute() else ROOT / args.dataset_config
    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else ROOT / args.manifest_dir
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    config, split_lock, manifest_hash, rows = load_manifest(dataset_config_path, manifest_dir)
    image_root = dataset_root
    train_rows, val_rows = rows['train'], rows['val']
    if not args.skip_image_hash_check:
        verify_images(train_rows + val_rows, image_root)

    run_dir = args.output_dir
    if run_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = ROOT / 'artifacts' / f'bisindo-image-v1_{stamp}'
    elif not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.weights == 'imagenet':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        mean, std = weights.transforms().mean, weights.transforms().std
    else:
        weights = None
        model = models.mobilenet_v3_small(weights=None)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(LETTERS))
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, fill=(114, 114, 114)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    Dataset = make_dataset_class(torch, Image, ImageOps, image_root)
    train_dataset = Dataset(train_rows, train_transform)
    val_dataset = Dataset(val_rows, val_transform)
    pin_memory = torch.cuda.is_available() and args.device != 'cpu'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory,
                            persistent_workers=args.num_workers > 0)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python environment')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    print(f'Classes: {len(LETTERS)}; train={len(train_dataset)}; val={len(val_dataset)}; '
          f'device={device}; image={IMAGE_SIZE}x{IMAGE_SIZE}; weights={args.weights}', flush=True)
    if args.weights == 'imagenet':
        print('If ImageNet weights are not cached, torchvision downloads them on first use.', flush=True)

    def make_optimizer(fine_tune):
        if fine_tune:
            groups = [
                {'params': model.features.parameters(), 'lr': 1e-4},
                {'params': model.classifier.parameters(), 'lr': 5e-4},
            ]
            return torch.optim.AdamW(groups, weight_decay=1e-4)
        return torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    optimizer = make_optimizer(fine_tune=args.warmup_epochs == 0)
    if args.warmup_epochs == 0:
        for parameter in model.features.parameters():
            parameter.requires_grad = True

    history = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    class_names = LETTERS
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1 and args.warmup_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = True
            optimizer = make_optimizer(fine_tune=True)
            print('Unfroze MobileNetV3 features for fine-tuning.', flush=True)
        epoch_start = time.perf_counter()
        is_frozen = epoch <= args.warmup_epochs
        train_loss, train_true, train_pred = score_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            torch=torch, frozen_features=is_frozen)
        val_loss, val_true, val_pred = score_epoch(model, val_loader, criterion, device, torch=torch)
        val_macro_f1 = float(f1_score(val_true, val_pred, labels=list(range(len(LETTERS))),
                                     average='macro', zero_division=0))
        val_accuracy = float(accuracy_score(val_true, val_pred))
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': float(accuracy_score(train_true, train_pred)),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_macro_f1': val_macro_f1,
            'epoch_seconds': time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {row['train_accuracy']:.4f} "
              f"| val loss {val_loss:.4f} acc {val_accuracy:.4f} macro-F1 {val_macro_f1:.4f} "
              f"| {row['epoch_seconds']:.1f}s", flush=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'architecture': 'mobilenet_v3_small',
            'class_names': class_names,
            'image_size': IMAGE_SIZE,
            'normalization': {'mean': list(mean), 'std': list(std)},
            'weights': 'MobileNet_V3_Small_Weights.DEFAULT' if weights is not None else 'none',
            'dataset_version': config['version'],
            'manifest_sha256': manifest_hash,
            'split_lock': split_lock,
            'epoch': epoch,
            'validation_macro_f1': val_macro_f1,
        }
        torch.save(checkpoint, run_dir / 'last.pt')
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, run_dir / 'best.pt')
            matrix = confusion_matrix(val_true, val_pred, labels=list(range(len(LETTERS))))
            report = classification_report(val_true, val_pred, labels=list(range(len(LETTERS))),
                                           target_names=LETTERS, output_dict=True, zero_division=0)
            write_json(run_dir / 'best_validation_metrics.json', {
                'epoch': epoch,
                'accuracy': val_accuracy,
                'macro_f1': val_macro_f1,
                'confusion_matrix_labels': LETTERS,
                'confusion_matrix': matrix.tolist(),
                'per_class': {letter: report[letter] for letter in LETTERS},
                'note': '26-class metrics on labeled sign images only; no no-hand/negative set is included.',
            })
        else:
            stale_epochs += 1

        with (run_dir / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(history)
        write_json(run_dir / 'training_state.json', {
            'status': 'running', 'current_epoch': epoch, 'best_epoch': best_epoch,
            'best_validation_macro_f1': best_f1, 'history': history,
        })
        if stale_epochs >= args.patience:
            print(f'Early stopping after {args.patience} epochs without validation macro-F1 improvement.', flush=True)
            break

    elapsed = time.perf_counter() - start
    write_json(run_dir / 'training_state.json', {
        'status': 'complete',
        'best_epoch': best_epoch,
        'best_validation_macro_f1': best_f1,
        'epochs_completed': len(history),
        'duration_seconds': round(elapsed, 2),
        'device': str(device),
        'train_images': len(train_dataset),
        'validation_images': len(val_dataset),
        'test_accessed': False,
        'history': history,
        'limitations': [
            'Validation contains labeled dataset images, not independent webcam trials.',
            'This 26-class baseline has no explicit no-hand/unknown class or negative examples.',
            'Dataset split is grouped by perceptual similarity, not verified participant/session IDs.',
        ],
    })
    print(f'Best validation macro-F1={best_f1:.4f} at epoch {best_epoch}; artifacts: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
