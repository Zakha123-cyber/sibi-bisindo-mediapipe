"""Report SIBI class balance and cached feature schema without retraining."""
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = ROOT/'data/features/landmarks-v1-final'
    output = ROOT/'data/reports/sibi-balance'
    output.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    for split in ('train', 'val', 'test'):
        with (cache/f'sibi_{split}_log.csv').open(encoding='utf-8', newline='') as stream:
            logs = list(csv.DictReader(stream))
        total = Counter(r['label'] for r in logs)
        valid = Counter(r['label'] for r in logs if r['feature_valid'] == '1')
        for label in sorted(total):
            rows.append(dict(split=split, label=label, total=total[label], valid=valid[label],
                             failed=total[label]-valid[label]))
        counts = np.array([valid[label] for label in sorted(total)])
        stats[split] = dict(total=sum(total.values()), valid=sum(valid.values()),
                            min_before=min(total.values()), max_before=max(total.values()),
                            min_valid=int(counts.min()), max_valid=int(counts.max()),
                            max_min_ratio=float(counts.max()/counts.min()),
                            majority_class_share_percent=round(float(counts.max()/counts.sum()*100), 2),
                            minimum_classes=[k for k in total if valid[k] == counts.min()])
    # Test class counts above are historical log metadata only. Inspect train arrays only.
    with np.load(cache/'sibi_train.npz', allow_pickle=False) as arrays:
        schema = {k: {'shape': list(arrays[k].shape), 'dtype': str(arrays[k].dtype)} for k in arrays.files}
        if not np.isfinite(arrays['X'][arrays['valid_mask']]).all():
            raise ValueError('Nonfinite training features marked valid')
        actual_counts = Counter(arrays['labels'][arrays['valid_mask']].tolist())
        if actual_counts != {r['label']: r['valid'] for r in rows if r['split'] == 'train'}:
            raise ValueError('Train NPZ and CSV disagree')
        stored = arrays['raw_landmarks'][arrays['valid_mask']]
        occupied = np.isfinite(stored).all(axis=(2, 3)).sum(axis=1)
        if not (occupied == 1).all():
            raise ValueError('Unexpected hand occupancy for valid SIBI train samples')
    selection = json.loads((ROOT/'artifacts/classification-v1/sibi/selection.json').read_text())
    candidates = [{k: r[k] for k in ('candidate', 'validation_system_accuracy', 'validation_system_macro_f1')}
                  for r in selection['candidates']]
    report = {'counts': stats, 'train_schema': schema, 'validation_candidates': candidates,
              'note': 'Historical cache audit, not a new training/evaluation run. Counts are not participant diversity.'}
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (output/'per_class.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for axis, splits in zip(axes, [('train',), ('val', 'test')]):
        labels = sorted({r['label'] for r in rows})
        positions = np.arange(len(labels))
        width = .6 if len(splits) == 1 else .35
        for index, split in enumerate(splits):
            selected = [r for r in rows if r['split'] == split]
            counts = [r['valid'] for r in selected]
            failed = [r['failed'] for r in selected]
            x = positions+(index-(len(splits)-1)/2)*width
            axis.bar(x, counts, width=width, label=f'{split}: valid', color='#1b9e77' if index == 0 else '#4682b4')
            axis.bar(x, failed, bottom=counts, width=width, label=f'{split}: gagal deteksi', color='#d95f02', alpha=1 if index == 0 else .5)
            for xpos, valid_count, failure in zip(x, counts, failed):
                axis.text(xpos, valid_count+failure+.3, str(valid_count), ha='center', fontsize=8)
        axis.set_xticks(positions, labels); axis.set_ylabel('Jumlah gambar')
        axis.set_ylim(0, 49 if splits == ('train',) else 12)
        axis.legend(loc='upper right', ncol=2, fontsize=8)
        axis.spines[['top', 'right']].set_visible(False)
    train = stats['train']
    axes[0].set_title(f"Training: {train['min_before']}–{train['max_before']} gambar/kelas sebelum ekstraksi; "
                      f"{train['min_valid']}–{train['max_valid']} fitur valid/kelas")
    axes[1].set_title('Validation dan test historis: masing-masing 9 gambar/kelas sebelum ekstraksi')
    fig.suptitle('Keseimbangan kelas SIBI — subset kurasi 24 huruf (tanpa J/Z)\nAngka di atas batang = jumlah fitur valid, bukan akurasi', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(output/'class_balance.png', dpi=160); plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
