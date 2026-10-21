"""Monitored coordinate CNN/MLP training, then SVM on learned embeddings."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.coordinate_model import CoordinateNet
from sibi_bisindo.coordinates import VERSION, augment_coordinates
from sibi_bisindo.npy_landmarks import sha256_file
from sibi_bisindo.training import evaluate_predictions


def epoch_order(target, rng, balanced=False):
    if not balanced:
        return rng.permutation(len(target))
    counts = np.bincount(target)
    weights = 1./counts[target]
    return rng.choice(len(target), size=len(target), replace=True, p=weights/weights.sum())


def forward_batches(model, x, device, embedding=False):
    model.eval()
    with torch.no_grad():
        return np.concatenate([(model.embed if embedding else model)(torch.from_numpy(x[i:i+256]).to(device)).cpu().numpy()
                               for i in range(0, len(x), 256)])


def metrics(labels, prediction, mask, sources, classes):
    result = evaluate_predictions(labels, prediction, mask, classes)
    result['by_source'] = {}
    for source in sorted(set(sources)):
        chosen = sources == source
        source_classes = [c for c in classes if c in set(labels[chosen])]
        result['by_source'][str(source)] = evaluate_predictions(labels[chosen], prediction[chosen], mask[chosen], source_classes)
    return result


def train(args):
    torch.set_num_threads(args.threads)
    dataset_dir = args.features_dir or ROOT/'data/features/coordinate-reset-v1'/args.language
    dataset = json.loads((dataset_dir/'dataset.json').read_text())
    if dataset['language'] != args.language or dataset['feature_version'] != VERSION:
        raise ValueError('Dataset language/feature contract mismatch')
    if sha256_file(ROOT/'src/sibi_bisindo/coordinates.py') != dataset['coordinate_implementation_sha256']:
        raise ValueError('Coordinate implementation changed after dataset preparation')
    config, classes = dataset['config'], dataset['classes']
    cache = {}
    for split in ('train', 'val'):
        path = dataset_dir/f'{split}.npz'
        if sha256_file(path) != dataset['package_sha256'][path.name]:
            raise ValueError('Prepared dataset hash mismatch')
        with np.load(path, allow_pickle=False) as a:
            cache[split] = {key: a[key].copy() for key in ('X', 'labels', 'valid_mask', 'sources')}
    t, v = cache['train'], cache['val']
    if any(not np.isfinite(a['X']).all() or a['X'].shape[1:] != (42, 7) for a in cache.values()):
        raise ValueError('Invalid coordinate tensors')
    tx, ty = t['X'][t['valid_mask']], t['labels'][t['valid_mask']]
    vx, vy = v['X'][v['valid_mask']], v['labels'][v['valid_mask']]
    target = np.array([classes.index(c) for c in ty], dtype=np.int64)
    counts = np.bincount(target, minlength=len(classes))
    if not counts.all():
        raise ValueError('Missing train classes')
    init_run = getattr(args, 'init_run', None)
    parent = None
    if init_run:
        from sibi_bisindo.coordinate_inference import CoordinatePredictor
        parent = CoordinatePredictor(init_run)
        if (parent.classes != classes or parent.selection['language'] != args.language
                or args.models != [parent.selection['selected']['kind']]
                or parent.selection['selected']['head'] != 'softmax'):
            raise ValueError('Fine-tune one matching neural Softmax architecture')
        if sha256_file(dataset_dir/'val.npz') != parent.selection['val_package_sha256']:
            raise ValueError('Fine-tuning must preserve baseline validation package')
        if getattr(args, 'heads', None) not in (None, ['softmax']):
            raise ValueError('Fine-tuning comparison uses Softmax only to preserve the baseline guard')
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    if args.device == 'cuda' and device.type != 'cuda':
        raise ValueError('CUDA requested but unavailable')
    run = args.output_dir or ROOT/'artifacts'/config['version']/f'{args.language}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True, exist_ok=False)
    (run/'dataset.json').write_text(json.dumps(dataset, indent=2)+'\n')
    print(f'{args.language}: train valid={len(tx)} val valid={len(vx)} classes={len(classes)} device={device}', flush=True)
    candidates, all_logs = [], {}
    epochs = args.epochs if args.epochs is not None else config['epochs']
    for kind in args.models:
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rng = np.random.default_rng(config['seed'])
        model = CoordinateNet(len(classes), kind).to(device)
        if parent:
            model.load_state_dict(parent.model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        weights = torch.tensor(len(target)/(len(classes)*counts), dtype=torch.float32, device=device)
        balanced = config.get('balanced_sampling', False)
        criterion = torch.nn.CrossEntropyLoss(weight=None if balanced else weights)
        best, stale, best_epoch, logs = -1., 0, 0, []
        checkpoint = run/f'{kind}_best.pt'
        native_floor = None
        if parent:
            initial = np.full(len(v['labels']), '', dtype='<U1')
            initial[v['valid_mask']] = np.asarray(classes)[forward_batches(model, vx, device).argmax(1)]
            best = float(f1_score(v['labels'], initial, labels=classes, average='macro', zero_division=0))
            (run/'baseline_validation.json').write_text(json.dumps(
                metrics(v['labels'], initial, v['valid_mask'], v['sources'], classes), indent=2)+'\n')
            native = v['sources'] == 'sibi_mediapipe'
            native_floor = int(np.sum((initial == v['labels']) & native))
            torch.save({'state_dict': {k: value.detach().cpu() for k, value in model.state_dict().items()},
                        'kind': kind, 'classes': classes, 'feature_version': VERSION, 'language': args.language,
                        'epoch': 0}, checkpoint)
            print(f'Baseline epoch 0: macro-F1={best:.4f}; native correct floor={native_floor}', flush=True)
        with (run/f'{kind}_epochs.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'loss', 'train_accuracy', 'val_accuracy_valid',
                                                       'val_macro_f1_valid', 'val_macro_f1_system', 'seconds', 'best_epoch',
                                                       'native_correct', 'j_correct', 'z_correct'])
            writer.writeheader()
            for epoch in range(1, epochs+1):
                started = time.perf_counter()
                augmented = augment_coordinates(tx, rng, config['rotation_degrees'], config['coordinate_noise_std'])
                model.train(); total_loss = 0.
                order = epoch_order(target, rng, balanced)
                for start in range(0, len(order), config['batch_size']):
                    idx = order[start:start+config['batch_size']]
                    bx = torch.from_numpy(augmented[idx]).to(device)
                    by = torch.from_numpy(target[idx]).to(device)
                    optimizer.zero_grad(); loss = criterion(model(bx), by)
                    loss.backward(); optimizer.step()
                    total_loss += float(loss.detach())*len(idx)
                train_pred = np.asarray(classes)[forward_batches(model, tx, device).argmax(1)]
                val_pred = np.asarray(classes)[forward_batches(model, vx, device).argmax(1)]
                full_pred = np.full(len(v['labels']), '', dtype='<U1'); full_pred[v['valid_mask']] = val_pred
                score = float(f1_score(v['labels'], full_pred, labels=classes, average='macro', zero_division=0))
                preserves_native = native_floor is None or int(np.sum((full_pred == v['labels']) & native)) >= native_floor
                if score > best+1e-8 and preserves_native:
                    best, stale, best_epoch = score, 0, epoch
                    torch.save({'state_dict': {k: value.detach().cpu() for k, value in model.state_dict().items()},
                                'kind': kind, 'classes': classes, 'feature_version': VERSION, 'language': args.language,
                                'epoch': epoch}, checkpoint)
                else:
                    stale += 1
                row = dict(epoch=epoch, loss=total_loss/len(tx), train_accuracy=float(np.mean(train_pred == ty)),
                           val_accuracy_valid=float(np.mean(val_pred == vy)),
                           val_macro_f1_valid=float(f1_score(vy, val_pred, labels=classes, average='macro', zero_division=0)),
                           val_macro_f1_system=score, seconds=time.perf_counter()-started, best_epoch=best_epoch,
                           native_correct=int(np.sum((full_pred == v['labels']) & (v['sources'] == 'sibi_mediapipe'))),
                           j_correct=int(np.sum((full_pred == 'J') & (v['labels'] == 'J'))),
                           z_correct=int(np.sum((full_pred == 'Z') & (v['labels'] == 'Z'))))
                writer.writerow(row); stream.flush(); logs.append(row)
                print(f"{kind} epoch {epoch:03}/{epochs}: loss={row['loss']:.5f} train={row['train_accuracy']:.4f} "
                      f"val={row['val_accuracy_valid']:.4f} macro-F1={score:.4f} best={best_epoch} | {row['seconds']:.1f}s", flush=True)
                if parent:
                    print(f"  Native correct={row['native_correct']} (floor {native_floor}); "
                          f"J={row['j_correct']}; Z={row['z_correct']}", flush=True)
                if stale >= config['patience']:
                    print(f'Early stop; best epoch={best_epoch}', flush=True); break
        all_logs[kind] = logs
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['state_dict'])
        embeddings_train = forward_batches(model, tx, device, True)
        embeddings_val = forward_batches(model, vx, device, True)
        heads = getattr(args, 'heads', None) or (['softmax'] if parent else ['softmax', 'svm_linear', 'svm_rbf'])
        for head in heads:
            artifact = None
            if head == 'softmax':
                prediction = np.asarray(classes)[forward_batches(model, vx, device).argmax(1)]
            else:
                svm = make_pipeline(StandardScaler(), SVC(C=1., kernel='linear' if head == 'svm_linear' else 'rbf',
                                                          class_weight='balanced'))
                svm.fit(embeddings_train, ty); prediction = svm.predict(embeddings_val)
                artifact = f'{kind}_{head}.joblib'; joblib.dump(svm, run/artifact)
            full = np.full(len(v['labels']), '', dtype='<U1'); full[v['valid_mask']] = prediction
            measured = metrics(v['labels'], full, v['valid_mask'], v['sources'], classes)
            name = f'{kind}_{head}'
            timings = []
            for repeat in range(23):
                started = time.perf_counter()
                with torch.no_grad():
                    single = torch.from_numpy(vx[:1]).to(device)
                    if head == 'softmax':
                        torch.softmax(model(single), dim=1).cpu().numpy()
                    else:
                        svm.predict(model.embed(single).cpu().numpy())
                if repeat >= 3:
                    timings.append((time.perf_counter()-started)*1000)
            measured['classifier_batch1_median_ms'] = float(np.median(timings))
            measured['timing_note'] = f'Classifier only on {device}; excludes MediaPipe, camera and coordinate preprocessing.'
            (run/f'{name}_validation.json').write_text(json.dumps(measured, indent=2)+'\n')
            candidates.append(dict(name=name, kind=kind, head=head, checkpoint=checkpoint.name, svm=artifact,
                                   macro_f1=measured['system_all_images']['macro_f1'],
                                   accuracy=measured['system_all_images']['accuracy'], best_epoch=best_epoch,
                                   classifier_batch1_median_ms=measured['classifier_batch1_median_ms']))
            print(f"{name}: validation system macro-F1={candidates[-1]['macro_f1']:.4f}", flush=True)
    selected = max(candidates, key=lambda r: r['macro_f1'])
    identity = {'version': config['version'], 'language': args.language, 'feature_version': VERSION,
                'classes': classes, 'selected': selected, 'candidates': candidates, 'test_loaded': False,
                'train_package_sha256': dataset['package_sha256']['train.npz'],
                'val_package_sha256': dataset['package_sha256']['val.npz'],
                'coordinate_implementation_sha256': dataset['coordinate_implementation_sha256'],
                'model_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/coordinate_model.py'),
                'training_script_sha256': sha256_file(Path(__file__)),
                'initialization': {'parent_selection_sha256': sha256_file(Path(init_run)/'selection.json'),
                                   'parent_run': str(init_run), 'native_validation_correct_floor': native_floor}
                                  if parent else None,
                'balanced_sampling': config.get('balanced_sampling', False),
                'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
                'artifact_sha256': {name: sha256_file(run/name) for name in
                                    [selected['checkpoint']]+([selected['svm']] if selected['svm'] else [])},
                'limitations': dataset['limitations']+['Validation selected; scores are not calibrated confidence. No unknown/background class training.']}
    if dataset.get('runtime_preprocessing'):
        selection['runtime_preprocessing_sha256'] = dataset['runtime_preprocessing']['sha256']
        selection['dataset_sha256'] = sha256_file(run/'dataset.json')
    (run/'selection.json').write_text(json.dumps(identity, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for kind, logs in all_logs.items():
        axes[0].plot([r['epoch'] for r in logs], [r['loss'] for r in logs], label=kind)
        axes[1].plot([r['epoch'] for r in logs], [r['val_macro_f1_system'] for r in logs], label=kind)
    axes[0].set_title('Train loss'); axes[1].set_title('Validation system macro-F1')
    for axis in axes:
        axis.set_xlabel('Epoch'); axis.legend()
    fig.tight_layout(); fig.savefig(run/'learning_curves.png', dpi=150); plt.close(fig)
    print(f"Selected {selected['name']}; artifacts: {run}", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['sibi', 'bisindo'], default='sibi')
    parser.add_argument('--features-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--models', nargs='+', choices=['cnn', 'mlp'], default=['cnn', 'mlp'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--init-run', type=Path, help='Warm-start from a compatible selected Softmax run')
    parser.add_argument('--heads', nargs='+', choices=['softmax', 'svm_linear', 'svm_rbf'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1 or args.threads < 1:
        parser.error('epochs and threads must be positive')
    train(args)


if __name__ == '__main__':
    main()
