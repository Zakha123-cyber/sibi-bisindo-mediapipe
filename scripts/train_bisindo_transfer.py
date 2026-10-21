"""Transfer only the SIBI encoder to a new BISINDO classifier in two stages."""
import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.coordinate_model import CoordinateNet
from sibi_bisindo.coordinate_inference import CoordinatePredictor
from sibi_bisindo.coordinates import VERSION, augment_coordinates
from sibi_bisindo.npy_landmarks import sha256_file
from train_coordinates import epoch_order, forward_batches, metrics


def new_bisindo_model(source, classes):
    if source.selection['language'] != 'sibi' or source.selection['selected']['kind'] != 'cnn':
        raise ValueError('Transfer source must be the selected SIBI coordinate CNN')
    model = CoordinateNet(len(classes), 'cnn')
    model.encoder.load_state_dict(copy.deepcopy(source.model.encoder.state_dict()))
    # Fresh classifier: A in SIBI and A in BISINDO need not describe the same pose.
    return model


def set_stage(model, stage, config):
    if stage not in ('head', 'finetune'):
        raise ValueError('Unknown transfer stage')
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(stage == 'finetune')
    if stage == 'head':
        groups = [{'params': model.classifier.parameters(), 'lr': config['head_learning_rate']}]
    else:
        groups = [{'params': model.encoder.parameters(), 'lr': config['encoder_learning_rate']},
                  {'params': model.classifier.parameters(), 'lr': config['finetune_head_learning_rate']}]
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])


def train(args):
    config_path = args.config or ROOT/'configs/bisindo_transfer_v1.json'
    config = json.loads(config_path.read_text())
    source_dir = Path(args.source_run or ROOT/config['source_run']).resolve()
    features = Path(args.features_dir or ROOT/config['features_dir']).resolve()
    if args.head_epochs is not None:
        config['head_epochs'] = args.head_epochs
    if args.finetune_epochs is not None:
        config['finetune_epochs'] = args.finetune_epochs
    if min(config['head_epochs'], config['finetune_epochs'], config['patience'], config['batch_size'], args.threads) < 1:
        raise ValueError('Epochs, patience, batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dataset = json.loads((features/'dataset.json').read_text())
    if dataset['language'] != 'bisindo' or dataset['feature_version'] != VERSION:
        raise ValueError('Need prepared BISINDO coordinate data')
    required_policy = config.get('required_runtime_policy')
    policy = dataset.get('runtime_preprocessing')
    if required_policy and (not policy or policy.get('version') != required_policy):
        raise ValueError('This training config requires the recovered BISINDO dataset')
    if policy:
        import hashlib
        from sibi_bisindo.npy_landmarks import canonical
        if hashlib.sha256(canonical({k:v for k,v in policy.items() if k!='sha256'})).hexdigest() != policy['sha256']:
            raise ValueError('Dataset recovery policy identity mismatch')
        if any(dataset['extraction_metadata'][s].get('runtime_preprocessing') != policy for s in ('train', 'val')):
            raise ValueError('Dataset train/validation recovery differs')
    if sha256_file(ROOT/'src/sibi_bisindo/coordinates.py') != dataset['coordinate_implementation_sha256']:
        raise ValueError('Coordinate preprocessing changed since preparation')
    source = CoordinatePredictor(source_dir)
    source_hashes = {name: sha256_file(source_dir/name) for name in ['selection.json']+list(source.selection['artifact_sha256'])}
    arrays = {}
    for split in ('train', 'val'):
        path = features/f'{split}.npz'
        if sha256_file(path) != dataset['package_sha256'][path.name]:
            raise ValueError(f'{split} package changed')
        with np.load(path, allow_pickle=False) as archive:
            arrays[split] = {key: archive[key].copy() for key in ('X', 'labels', 'valid_mask', 'sources', 'paths')}
        sample = arrays[split]
        if sample['X'].shape[1:] != (42, 7) or not np.isfinite(sample['X']).all():
            raise ValueError('Invalid coordinate tensor')
    if set(arrays['train']['paths']) & set(arrays['val']['paths']):
        raise ValueError('Training/validation path overlap')
    t, v = arrays['train'], arrays['val']
    classes = dataset['classes']
    tx, ty = t['X'][t['valid_mask']], t['labels'][t['valid_mask']]
    vx, vy = v['X'][v['valid_mask']], v['labels'][v['valid_mask']]
    if set(ty) != set(classes) or set(vy) != set(classes):
        raise ValueError('Train/validation must contain all classes')
    target = np.asarray([classes.index(label) for label in ty], dtype=np.int64)
    device = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and device.type != 'cuda':
        raise ValueError('CUDA requested but unavailable')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    run = (args.output_dir or ROOT/'artifacts'/config['version']/f'bisindo_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    if run == source_dir or source_dir in run.parents:
        raise ValueError('Transfer output must be separate from source SIBI run')
    run.mkdir(parents=True, exist_ok=False)
    (run/'dataset.json').write_text(json.dumps(dataset, indent=2)+'\n')
    (run/'transfer_config.json').write_text(json.dumps(config, indent=2)+'\n')
    model = new_bisindo_model(source, classes).to(device)
    initial_encoder = {key: value.detach().cpu().clone() for key, value in model.encoder.state_dict().items()}
    criterion = torch.nn.CrossEntropyLoss()
    counts = np.bincount(target, minlength=len(classes))
    if not config['balanced_sampling']:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(target)/(len(classes)*counts), dtype=torch.float32, device=device))
    rng = np.random.default_rng(config['seed'])
    best_score, best_epoch, best_stage = -1., 0, None
    epoch, logs, stage_reports = 0, [], {}
    fields = ['epoch', 'stage', 'stage_epoch', 'loss', 'train_accuracy', 'val_accuracy_valid',
              'val_macro_f1_system', 'val_one_hand_accuracy', 'val_two_hands_accuracy', 'seconds', 'best_epoch']

    def checkpoint(path, stage):
        torch.save({'state_dict': {k: value.detach().cpu() for k, value in model.state_dict().items()},
                    'classes': classes, 'language': 'bisindo', 'kind': 'cnn', 'feature_version': VERSION,
                    'epoch': epoch, 'stage': stage}, path)

    print(f'BISINDO transfer: {len(tx)} train valid; {len(vx)}/{len(v["labels"])} validation valid; {device}', flush=True)
    with (run/'cnn_epochs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for stage, maximum in [('head', config['head_epochs']), ('finetune', config['finetune_epochs'])]:
            if stage == 'finetune':
                model.load_state_dict(torch.load(run/'head_best.pt', map_location=device, weights_only=True)['state_dict'])
            optimizer = set_stage(model, stage, config)
            stage_best, stale = -1., 0
            print(f'Stage {stage}: encoder {"frozen" if stage == "head" else "trainable"}', flush=True)
            for stage_epoch in range(1, maximum+1):
                epoch += 1; started = time.perf_counter()
                augmented = augment_coordinates(tx, rng, config['rotation_degrees'], config['coordinate_noise_std'])
                order = epoch_order(target, rng, config['balanced_sampling'])
                model.train(); loss_total = 0.
                for start in range(0, len(order), config['batch_size']):
                    indexes = order[start:start+config['batch_size']]
                    bx = torch.from_numpy(augmented[indexes]).to(device)
                    by = torch.from_numpy(target[indexes]).to(device)
                    optimizer.zero_grad(); loss = criterion(model(bx), by)
                    loss.backward(); optimizer.step(); loss_total += float(loss.detach())*len(indexes)
                train_prediction = np.asarray(classes)[forward_batches(model, tx, device).argmax(1)]
                prediction = np.asarray(classes)[forward_batches(model, vx, device).argmax(1)]
                full = np.full(len(v['labels']), '', dtype='<U1'); full[v['valid_mask']] = prediction
                score = float(f1_score(v['labels'], full, labels=classes, average='macro', zero_division=0))
                if score > stage_best+1e-8:
                    stage_best, stale = score, 0
                    checkpoint(run/f'{stage}_best.pt', stage)
                else:
                    stale += 1
                if score > best_score+1e-8:
                    best_score, best_epoch, best_stage = score, epoch, stage
                    checkpoint(run/'cnn_best.pt', stage)
                hand_counts = vx[:, 0, 6]+vx[:, 21, 6]
                hand_accuracy = lambda count: float(np.mean(prediction[hand_counts == count] == vy[hand_counts == count])) if np.any(hand_counts == count) else None
                row = dict(epoch=epoch, stage=stage, stage_epoch=stage_epoch, loss=loss_total/len(tx),
                           train_accuracy=float(np.mean(train_prediction == ty)), val_accuracy_valid=float(np.mean(prediction == vy)),
                           val_macro_f1_system=score, val_one_hand_accuracy=hand_accuracy(1), val_two_hands_accuracy=hand_accuracy(2),
                           seconds=time.perf_counter()-started, best_epoch=best_epoch)
                writer.writerow(row); stream.flush(); logs.append(row)
                print(f"{stage} {stage_epoch:03}/{maximum} | loss={row['loss']:.5f} train={row['train_accuracy']:.4f} "
                      f"val={row['val_accuracy_valid']:.4f} system macro-F1={score:.4f} best={best_epoch} | {row['seconds']:.1f}s", flush=True)
                if stale >= config['patience']:
                    print(f'Early stop {stage}', flush=True); break
            stage_reports[stage] = {'epochs_run': stage_epoch, 'best_system_macro_f1': stage_best}
            if stage == 'head':
                if not all(torch.equal(value.detach().cpu(), initial_encoder[key]) for key, value in model.encoder.state_dict().items()):
                    raise RuntimeError('Encoder changed during frozen head training')
                stage_reports[stage]['encoder_unchanged'] = True
    model.load_state_dict(torch.load(run/'cnn_best.pt', map_location=device, weights_only=True)['state_dict'])
    prediction = np.asarray(classes)[forward_batches(model, vx, device).argmax(1)]
    full = np.full(len(v['labels']), '', dtype='<U1'); full[v['valid_mask']] = prediction
    measured = metrics(v['labels'], full, v['valid_mask'], v['sources'], classes)
    for count in (1, 2):
        group = v['valid_mask'] & ((v['X'][:, 0, 6]+v['X'][:, 21, 6]) == count)
        measured.setdefault('by_detected_hand_count', {})[str(count)] = {
            'samples': int(group.sum()), 'accuracy': float(np.mean(full[group] == v['labels'][group])) if group.any() else None}
    timings = []
    for index in range(23):
        started = time.perf_counter()
        with torch.no_grad():
            torch.softmax(model(torch.from_numpy(vx[:1]).to(device)), dim=1).cpu().numpy()
        if index >= 3:
            timings.append((time.perf_counter()-started)*1000)
    latency = float(np.median(timings))
    measured.update(classifier_batch1_median_ms=latency, timing_note=f'Classifier only on {device}; no camera/MediaPipe/preprocessing.')
    (run/'cnn_softmax_validation.json').write_text(json.dumps(measured, indent=2)+'\n')
    if any(sha256_file(source_dir/name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError('SIBI source files changed during transfer')
    selected = {'name': 'cnn_softmax', 'kind': 'cnn', 'head': 'softmax', 'checkpoint': 'cnn_best.pt', 'svm': None,
                'macro_f1': best_score, 'accuracy': measured['system_all_images']['accuracy'],
                'best_epoch': best_epoch, 'best_stage': best_stage, 'classifier_batch1_median_ms': latency}
    selection = {'version': config['version'], 'language': 'bisindo', 'feature_version': VERSION, 'classes': classes,
                 'selected': selected, 'candidates': [selected], 'test_loaded': False,
                 'coordinate_implementation_sha256': dataset['coordinate_implementation_sha256'],
                 'model_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/coordinate_model.py'),
                 'training_script_sha256': sha256_file(Path(__file__)),
                 'train_package_sha256': dataset['package_sha256']['train.npz'], 'val_package_sha256': dataset['package_sha256']['val.npz'],
                 'artifact_sha256': {'cnn_best.pt': sha256_file(run/'cnn_best.pt')},
                 'initialization': {'mode': 'encoder_only_new_classifier', 'parent_run': str(source_dir), 'parent_sha256': source_hashes},
                 'stage_reports': stage_reports, 'torch_version': str(torch.__version__),
                 'limitations': ['BISINDO dataset split is not verified signer-independent.',
                                 'One-hand detections do not certify completeness of a two-hand sign.',
                                 'Transfer benefit over scratch not established by this run. Webcam performance unverified.',
                                 'Scores are uncalibrated; no unknown/background class.']}
    (run/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    if dataset.get('runtime_preprocessing'):
        selection['runtime_preprocessing_sha256'] = dataset['runtime_preprocessing']['sha256']
        selection['dataset_sha256'] = sha256_file(run/'dataset.json')
        (run/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([r['epoch'] for r in logs], [r['loss'] for r in logs]); axes[0].set_title('Train loss')
    axes[1].plot([r['epoch'] for r in logs], [r['val_macro_f1_system'] for r in logs]); axes[1].set_title('Validation system macro-F1')
    boundary = stage_reports['head']['epochs_run']+.5
    for axis in axes:
        axis.axvline(boundary, linestyle='--', color='grey', label='Unfreeze encoder'); axis.set_xlabel('Epoch'); axis.legend()
    fig.tight_layout(); fig.savefig(run/'learning_curves.png', dpi=150); plt.close(fig)
    print(f'Selected epoch {best_epoch} ({best_stage}); artifacts: {run}', flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--features-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--head-epochs', type=int)
    parser.add_argument('--finetune-epochs', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
