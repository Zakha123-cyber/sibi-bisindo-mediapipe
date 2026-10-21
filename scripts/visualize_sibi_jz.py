"""Render actual normalized J/Z/G/T coordinates and selected validation errors."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.coordinate_inference import CoordinatePredictor
from sibi_bisindo.npy_landmarks import sha256_file

FINGERS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
           (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
COLORS = ['#d97706', '#2563eb', '#9333ea', '#059669', '#dc2626']
LABELS = ['Ibu jari', 'Telunjuk', 'Tengah', 'Manis', 'Kelingking']


def draw(axis, points, bounds, projection=(0, 1), alpha=1., width=2., tips=True):
    for finger, color in zip(FINGERS, COLORS):
        values = points[list(finger)]
        axis.plot(values[:, projection[0]], values[:, projection[1]], color=color, alpha=alpha, linewidth=width)
        if tips:
            axis.scatter(values[:, projection[0]], values[:, projection[1]], color=color, s=9, alpha=alpha)
            tip = points[finger[-1]]
            axis.text(tip[projection[0]], tip[projection[1]], str(finger[-1]), fontsize=7, color=color)
    axis.set_xlim(bounds[projection[0]]); axis.set_ylim(bounds[projection[1]][::-1])
    axis.set_aspect('equal', adjustable='box'); axis.grid(alpha=.15)
    axis.tick_params(labelsize=7)


def representative_indices(points, count=6):
    flat = points.reshape(len(points), -1)
    chosen = [int(np.argmin(np.linalg.norm(flat-np.median(flat, axis=0), axis=1)))]
    while len(chosen) < min(count, len(flat)):
        distances = np.min([np.linalg.norm(flat-flat[i], axis=1) for i in chosen], axis=0)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'artifacts/coordinate-reset-v1/sibi_20261020_130607_889055')
    parser.add_argument('--features-dir', type=Path, default=ROOT/'data/features/coordinate-reset-v1/sibi')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/reports/sibi-jz-landmarks')
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictor = CoordinatePredictor(args.run_dir)
    if predictor.selection['language'] != 'sibi':
        raise ValueError('This review is for the SIBI experiment')
    arrays = {}
    for split in ('train', 'val'):
        path = args.features_dir/f'{split}.npz'
        if sha256_file(path) != predictor.selection[f'{split}_package_sha256']:
            raise ValueError('Data differs from selected training run')
        with np.load(path, allow_pickle=False) as archive:
            mask = archive['valid_mask'].copy()
            arrays[split] = {key: archive[key][mask].copy() for key in ('X', 'labels', 'paths', 'sources')}
    train, val = arrays['train'], arrays['val']
    # Single-hand normalized local xyz are exactly the first 21 points/channels.
    if any(not (a['X'][:, :21, 6] == 1).all() or not (a['X'][:, 21:, 6] == 0).all() for a in arrays.values()):
        raise ValueError('Expected single-hand SIBI coordinate tensors')
    with torch.no_grad():
        x = torch.from_numpy(val['X'])
        if predictor.svm is None:
            predictions = np.asarray(predictor.classes)[predictor.model(x).argmax(1).numpy()]
        else:
            predictions = predictor.svm.predict(predictor.model.embed(x).numpy())
    selected = np.isin(train['labels'], list('JZGT'))
    geometry = train['X'][selected, :21, :3]
    relevant_errors = np.flatnonzero(np.isin(val['labels'], ['J', 'Z']) & (predictions != val['labels']))
    combined = np.concatenate([geometry, val['X'][relevant_errors, :21, :3]])
    bounds = [(float(combined[:, :, i].min())-.25, float(combined[:, :, i].max())+.25) for i in range(3)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, representatives = [], {}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(6, 4, figsize=(13, 16))
    for column, label in enumerate(['J', 'G', 'T', 'Z']):
        indexes = np.flatnonzero(train['labels'] == label)
        examples = representative_indices(train['X'][indexes, :21, :3])
        representatives[label] = int(indexes[examples[0]])
        for row, local in enumerate(examples):
            index = int(indexes[local]); points = train['X'][index, :21, :3]
            draw(axes[row, column], points, bounds)
            source = 'ASL' if train['sources'][index] == 'asl_static_pose' else 'SIBI'
            axes[row, column].set_title(f'{label} | {source} | train #{index}\n' + ('Contoh dekat median' if row == 0 else 'Variasi koordinat'), fontsize=9)
            rows.append(dict(panel='training', position=f'{row+1},{column+1}', split='train', label=label,
                             prediction='', sample_index=index, source=str(train['sources'][index]), path=str(train['paths'][index])))
    fig.suptitle('J vs G/T dan Z: koordinat yang benar-benar dipakai training\nProyeksi XY, orientasi asli, skala plot sama. Angka = indeks ujung jari; bukan foto.', fontsize=13)
    fig.legend([Line2D([0], [0], color=c, lw=2) for c in COLORS], LABELS, loc='lower center', ncol=5)
    fig.tight_layout(rect=(0, .035, 1, .95)); fig.savefig(args.output_dir/'training_comparison.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for axis, label in zip(axes, ['J', 'G', 'T', 'Z']):
        draw(axis, train['X'][representatives[label], :21, :3], bounds)
        axis.set_title(f'{label} | '+('ASL' if label in ['J', 'Z'] else 'SIBI')+' | contoh training')
    fig.suptitle('Satu contoh nyata dekat median per kelas; orientasi asli dan skala plot sama\nOranye: ibu jari | biru: telunjuk | ungu: tengah | hijau: manis | merah: kelingking', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .85)); fig.savefig(args.output_dir/'key_poses.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    groups = [(train['X'][train['labels'] == 'J', :21, :3], 'J training'),
              (val['X'][(val['labels'] == 'J') & (predictions == 'J'), :21, :3], 'J validation benar'),
              (val['X'][(val['labels'] == 'J') & (predictions != 'J'), :21, :3], 'J validation salah')]
    for column, (points, title) in enumerate(groups):
        if not len(points):
            continue
        representative = points[representative_indices(points, 1)[0]]
        for row, projection in enumerate([(0, 1), (0, 2)]):
            for sample in points:
                draw(axes[row, column], sample, bounds, projection, alpha=.12, width=.7, tips=False)
            draw(axes[row, column], representative, bounds, projection)
            axes[row, column].set_title(f'{title}: {len(points)} sampel | '+('XY' if row == 0 else 'XZ'))
    fig.suptitle('Fokus J: variasi training vs validation yang dikenali dan tertukar\nTipis = semua sampel kelompok; tebal = satu sampel nyata dekat median, bukan pose rata-rata.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(args.output_dir/'j_train_vs_validation.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    for column, label in enumerate(['J', 'G', 'T', 'Z']):
        points = train['X'][train['labels'] == label, :21, :3]
        representative = train['X'][representatives[label], :21, :3]
        for row, projection in enumerate([(0, 1), (0, 2)]):
            for sample in points:
                draw(axes[row, column], sample, bounds, projection, alpha=.1, width=.7, tips=False)
            draw(axes[row, column], representative, bounds, projection)
            axes[row, column].set_title(f'{label}: {"XY" if row == 0 else "XZ"} | {len(points)} sampel training')
            axes[row, column].set_xlabel('x relatif'); axes[row, column].set_ylabel('y relatif' if row == 0 else 'z relatif')
    fig.suptitle('Sebaran pose per kelas: garis tipis = seluruh training, tebal = satu contoh nyata dekat median\nXY: y ke bawah. XZ: z lebih negatif ke atas; kedalaman MediaPipe bukan pengukuran metrik.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(args.output_dir/'pose_spread_xy_xz.png', dpi=160); plt.close(fig)

    if len(relevant_errors):
        fig, axes = plt.subplots(len(relevant_errors), 3, figsize=(12, len(relevant_errors)*2.8), squeeze=False)
        for row, index in enumerate(relevant_errors):
            label, prediction = str(val['labels'][index]), str(predictions[index])
            sample = val['X'][index, :21, :3]
            draw(axes[row, 0], sample, bounds)
            axes[row, 0].set_title(f'VAL #{index}: label {label} -> prediksi {prediction}', fontsize=10)
            rows.append(dict(panel='validation_error', position=f'{row+1},1', split='val', label=label,
                             prediction=prediction, sample_index=int(index), source=str(val['sources'][index]), path=str(val['paths'][index])))
            for column, candidate in enumerate([label, prediction], 1):
                indexes = np.flatnonzero(train['labels'] == candidate)
                distances = np.sqrt(np.mean((train['X'][indexes, :21, :3]-sample)**2, axis=(1, 2)))
                closest = int(indexes[distances.argmin()])
                draw(axes[row, column], train['X'][closest, :21, :3], bounds)
                axes[row, column].set_title(f'TRAIN {candidate} terdekat #{closest}\nRMS xyz={distances.min():.3f}', fontsize=9)
                rows.append(dict(panel='error_neighbor', position=f'{row+1},{column+1}', split='train', label=candidate,
                                 prediction='', sample_index=closest, source=str(train['sources'][closest]), path=str(train['paths'][closest])))
        fig.suptitle('Semua kesalahan validation J/Z + tetangga training dari kelas benar dan kelas prediksi\nTerdekat menurut RMS koordinat xyz; bukan jarak embedding CNN atau bukti pose berlabel salah.', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(args.output_dir/'validation_errors.png', dpi=140); plt.close(fig)
    with (args.output_dir/'samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(run=str(args.run_dir), selection_sha256=sha256_file(args.run_dir/'selection.json'),
                   feature_version=predictor.selection['feature_version'], training_counts={c: int(np.sum(train['labels'] == c)) for c in ['J', 'G', 'T', 'Z']},
                   validation={c: dict(total=int(np.sum(val['labels'] == c)), correct=int(np.sum((val['labels'] == c) & (predictions == c)))) for c in ['J', 'Z']},
                   errors=[dict(label=str(val['labels'][i]), prediction=str(predictions[i]), path=str(val['paths'][i])) for i in relevant_errors],
                   note='Actual model-input coordinates; no mirror/rotation alignment, no training, source images or held-out test data loaded. All source poses are labels supplied to the experiment, not linguistically certified.')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
