"""Compare SVM and an epoch-monitored MLP on the new 63D landmark cache.

Uses train and validation only. Labels originate from ASL; SIBI pose mapping is
unverified. Full training is intended to be launched by the user.
"""
import argparse
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sibi_bisindo.npy_landmarks import FEATURE_VERSION, canonical, sha256_file


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def read_features(directory):
    lock = json.loads((directory / 'split_lock.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(config)).hexdigest() != lock['config_sha256']:
        raise ValueError('Cache config changed after split lock')
    if (lock['source_language'], lock['target_language'], lock['pose_mapping']) != ('asl', 'sibi', 'unverified'):
        raise ValueError('Unexpected provenance in feature cache')
    if sha256_file(ROOT / 'src/sibi_bisindo/npy_landmarks.py') != lock['feature_implementation_sha256']:
        raise ValueError('Feature implementation changed; prepare a new feature version')
    classes = lock['classes']
    data = {}
    # This loop must never be extended to test during candidate selection.
    for name in ('manifest.csv', 'train.npz', 'val.npz'):
        if sha256_file(directory / name) != lock['files_sha256'][name]:
            raise ValueError(f'Locked input changed: {name}')
    for split in ('train', 'val'):
        with np.load(directory / f'{split}.npz', allow_pickle=False) as saved:
            x, y, paths = saved['X'], saved['labels'].astype(str), saved['paths'].astype(str)
            metadata = json.loads(str(saved['metadata_json']))
        for key in ('version', 'config_sha256', 'manifest_sha256', 'feature_implementation_sha256'):
            if metadata[key] != lock[key]:
                raise ValueError(f'{split} cache identity mismatch: {key}')
        if metadata['split'] != split or metadata['feature_version'] != FEATURE_VERSION:
            raise ValueError(f'Unexpected {split} feature schema')
        if (x.shape != (len(y), 63) or len(paths) != len(y) or x.dtype != np.float32
                or not np.isfinite(x).all() or set(y) != set(classes)):
            raise ValueError(f'Invalid {split} arrays')
        points = x.reshape(-1, 21, 3)
        if not np.allclose(points[:, 0], 0, atol=1e-6) or not np.allclose(
                np.linalg.norm(points, axis=2).max(axis=1), 1, atol=1e-5):
            raise ValueError(f'{split} feature normalization differs from contract')
        data[split] = (x, y, paths)
    if set(data['train'][2]) & set(data['val'][2]):
        raise ValueError('Train/validation paths overlap')
    return lock, config, data


def metrics(truth, prediction, classes):
    report = classification_report(truth, prediction, labels=classes, output_dict=True, zero_division=0)
    return {
        'sample_count': len(truth),
        'accuracy': float(accuracy_score(truth, prediction)),
        'macro_f1': float(f1_score(truth, prediction, labels=classes, average='macro', zero_division=0)),
        'confusion_matrix_labels': classes,
        'confusion_matrix': confusion_matrix(truth, prediction, labels=classes).tolist(),
        'per_class': {label: report[label] for label in classes},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, default=ROOT / 'data/features/asl-source-sibi-candidate-v1')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--families', nargs='+', choices=['svm', 'mlp'], default=['svm', 'mlp'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--check-only', action='store_true', help='Validate caches without training')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    directory = args.features_dir.resolve()
    lock, config, data = read_features(directory)
    x_train, y_train, _ = data['train']
    x_val, y_val, _ = data['val']
    classes = lock['classes']
    print(f'ASL-source / SIBI-candidate (unverified pose mapping): '
          f'train={len(y_train)}, val={len(y_val)}, classes={len(classes)}, features=63', flush=True)
    if args.check_only:
        print('Input checks passed. Test arrays were not loaded.', flush=True)
        return
    output = args.output_dir or ROOT / 'artifacts' / f"{lock['version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(output / 'training_state.json', {'status': 'running', 'test_accessed': False})
    identity = {
        'dataset_version': lock['version'], 'source_language': 'asl', 'target_language': 'sibi',
        'pose_mapping': 'unverified', 'classes': classes, 'feature_version': FEATURE_VERSION,
        'feature_dimension': 63, 'split_lock': lock,
        'scikit_learn_version': sklearn.__version__, 'joblib_version': joblib.__version__,
        'numpy_version': np.__version__, 'seed': config['seed'],
    }
    candidates = []

    def save_candidate(name, model, epoch=None):
        scored = metrics(y_val, model.predict(x_val), classes)
        artifact = output / f'{name}.joblib'
        joblib.dump(dict(identity, model=model, candidate=name, best_epoch=epoch), artifact)
        row = {'candidate': name, 'artifact': artifact.name,
               'artifact_sha256': sha256_file(artifact), 'validation_accuracy': scored['accuracy'],
               'validation_macro_f1': scored['macro_f1'], 'best_epoch': epoch}
        return row, scored

    if 'svm' in args.families:
        for kernel in ('linear', 'rbf'):
            for c in (0.1, 1.0, 10.0):
                candidate_started = time.perf_counter()
                name = f'svm_{kernel}_c{c:g}'
                model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c,
                                           class_weight='balanced', cache_size=512))
                model.fit(x_train, y_train)
                row, scored = save_candidate(name, model)
                row['fit_seconds'] = round(time.perf_counter()-candidate_started, 3)
                candidates.append(row)
                print(f"{name}: val acc={scored['accuracy']:.4f}, macro-F1={scored['macro_f1']:.4f} "
                      f"({row['fit_seconds']:.1f}s)", flush=True)

    if 'mlp' in args.families:
        scaler = StandardScaler().fit(x_train)
        train_scaled = scaler.transform(x_train).astype(np.float32)
        val_scaled = scaler.transform(x_val).astype(np.float32)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                            batch_size=min(128, len(y_train)), random_state=config['seed'],
                            max_iter=1, early_stopping=False)
        best_key, best_epoch, stale = (-1.0, -1.0), 0, 0
        fields = ['epoch', 'train_loss', 'train_accuracy', 'val_accuracy', 'val_macro_f1',
                  'epoch_seconds', 'best_epoch', 'stale_epochs']
        with (output / 'mlp_128x64_epochs.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                epoch_started = time.perf_counter()
                mlp.partial_fit(train_scaled, y_train, classes=np.asarray(classes) if epoch == 1 else None)
                prediction = mlp.predict(val_scaled)
                val_acc = float(accuracy_score(y_val, prediction))
                val_f1 = float(f1_score(y_val, prediction, labels=classes, average='macro', zero_division=0))
                if (val_f1, val_acc) > best_key:
                    best_key, best_epoch, stale = (val_f1, val_acc), epoch, 0
                    joblib.dump(dict(identity, model=make_pipeline(scaler, mlp),
                                     candidate='mlp_128x64', best_epoch=epoch), output / 'mlp_128x64.joblib')
                else:
                    stale += 1
                row = {'epoch': epoch, 'train_loss': float(mlp.loss_),
                       'train_accuracy': float(accuracy_score(y_train, mlp.predict(train_scaled))),
                       'val_accuracy': val_acc, 'val_macro_f1': val_f1,
                       'epoch_seconds': round(time.perf_counter()-epoch_started, 3),
                       'best_epoch': best_epoch, 'stale_epochs': stale}
                writer.writerow(row)
                stream.flush()
                print(f"MLP epoch {epoch:03d}/{args.epochs}: loss={mlp.loss_:.5f} "
                      f"train acc={row['train_accuracy']:.4f} | val acc={val_acc:.4f} "
                      f"macro-F1={val_f1:.4f} | best={best_epoch} | {row['epoch_seconds']:.1f}s", flush=True)
                if stale >= args.patience:
                    print(f'Early stopping; best epoch={best_epoch}', flush=True)
                    break
        # Reload the saved best epoch; the estimator in memory may be worse.
        model = joblib.load(output / 'mlp_128x64.joblib')['model']
        row, _ = save_candidate('mlp_128x64', model, epoch=best_epoch)
        candidates.append(row)

    winner = max(candidates, key=lambda item: (item['validation_macro_f1'], item['validation_accuracy']))
    selected = joblib.load(output / winner['artifact'])['model']
    write_json(output / 'best_validation_metrics.json', metrics(y_val, selected.predict(x_val), classes))
    selection = dict(identity, selected_candidate=winner['candidate'], artifact=winner['artifact'],
                     artifact_sha256=winner['artifact_sha256'], candidates=candidates,
                     training_script_sha256=sha256_file(Path(__file__)), test_accessed=False,
                     limitations=[
                         'ASL source labels are not verified SIBI poses.',
                         'Filename-order holdout is a heuristic; participant/session independence is unverified.',
                         'Only valid landmark samples are evaluated; no detector failures or unknown/no-hand examples.',
                         'Decision scores/probabilities are not calibrated on webcam inputs.',
                     ])
    write_json(output / 'selection.json', selection)
    write_json(output / 'training_state.json', {
        'status': 'complete', 'test_accessed': False, 'source_language': 'asl', 'target_language': 'sibi',
        'pose_mapping': 'unverified', 'selected_candidate': winner['candidate'],
        'validation_macro_f1': winner['validation_macro_f1'], 'train_samples': len(y_train),
        'validation_samples': len(y_val), 'duration_seconds': round(time.perf_counter()-started, 3),
    })
    print(f"Selected {winner['candidate']}: val macro-F1={winner['validation_macro_f1']:.4f}", flush=True)
    print(f'Artifacts: {output}; test arrays were not loaded.', flush=True)


if __name__ == '__main__':
    main()
