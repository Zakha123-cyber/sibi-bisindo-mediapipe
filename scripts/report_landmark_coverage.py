"""Visualize existing extraction logs; never rerun detection or open source images."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ('SIBI Mendeley', 'sibi', 'landmarks-v1-final'),
    ('BISINDO Mendeley', 'bisindo', 'landmarks-v1-final'),
    ('BISINDO Kaggle', 'bisindo', 'landmarks-bisindo-kaggle-v2-conf03'),
]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/reports/landmark-coverage')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    original = read_csv(ROOT/'data/manifests/data-v1/all_samples.csv')
    kaggle = read_csv(ROOT/'data/manifests/bisindo-kaggle-v2/manifest.csv')
    results, split_rows, class_rows, inputs = [], [], [], {}
    for path in (ROOT/'data/manifests/data-v1/all_samples.csv', ROOT/'data/manifests/bisindo-kaggle-v2/manifest.csv'):
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for title, language, cache_name in DATASETS:
        cache = ROOT/'data/features'/cache_name
        summary_path = cache/'summary.json'
        summary = json.loads(summary_path.read_text())
        inputs[summary_path.relative_to(ROOT).as_posix()] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        inventory = kaggle if title == 'BISINDO Kaggle' else [r for r in original if r['language'] == language]
        selected = [r for r in inventory if r['include'] == '1']
        selected_paths = {r['path'] for r in selected}
        all_logs = []
        for split in ('train', 'val', 'test'):
            expected = {r['path']: r for r in selected if r['split'] == split}
            path = cache/f'{language}_{split}_log.csv'
            logs = read_csv(path) if path.exists() else []
            paths = [r['path'] for r in logs]
            if len(paths) != len(set(paths)) or not set(paths).issubset(expected):
                raise ValueError(f'Duplicate/unexpected rows: {path}')
            for row in logs:
                if row['label'] != expected[row['path']]['label'] or row['status'] == 'pending':
                    raise ValueError(f'Label mismatch/pending output: {path}')
            if logs:
                inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            statuses = Counter(r['status'] for r in logs)
            hands = Counter(int(r['hands']) for r in logs)
            valid = sum(r['feature_valid'] == '1' for r in logs)
            stored = summary['splits'].get(f'{language}/{split}')
            if stored and (stored['total'] != len(logs) or stored['feature_valid'] != valid
                           or stored['hand_counts'] != {str(k): v for k, v in hands.items()}
                           or stored['statuses'] != dict(statuses)):
                raise ValueError(f'Summary disagrees with CSV: {path}')
            item = dict(dataset=title, split=split, selected=len(expected), processed=len(logs),
                        unprocessed=len(expected)-len(logs), detected=sum(v for k, v in hands.items() if k > 0),
                        valid=valid, invalid=len(logs)-valid, no_hand=hands[0], one_hand=hands[1],
                        two_hands=hands[2], statuses=dict(statuses))
            item['valid_percent_processed'] = round(100*valid/len(logs), 2) if logs else None
            split_rows.append(item)
            all_logs.extend(logs)
        seen = {r['path'] for r in all_logs}
        if not seen.issubset(selected_paths):
            raise ValueError('Unexpected extraction source')
        valid = sum(r['feature_valid'] == '1' for r in all_logs)
        hands = Counter(int(r['hands']) for r in all_logs)
        result = dict(dataset=title, cache=cache_name, inventory_images=len(inventory), selected_images=len(selected),
                      excluded_from_protocol=len(inventory)-len(selected), processed=len(all_logs),
                      unprocessed_selected=len(selected)-len(all_logs), valid=valid, invalid=len(all_logs)-valid,
                      detected=sum(v for k, v in hands.items() if k > 0), no_hand=hands[0], one_hand=hands[1], two_hands=hands[2],
                      valid_percent_processed=round(100*valid/len(all_logs), 2))
        results.append(result)
        for label in sorted({r['label'] for r in selected}):
            for split in ('train', 'val', 'test'):
                subset = [r for r in all_logs if r['label'] == label and r['split'] == split]
                expected_count = sum(r['label'] == label and r['split'] == split for r in selected)
                count = sum(r['feature_valid'] == '1' for r in subset)
                class_rows.append(dict(dataset=title, label=label, split=split, selected=expected_count,
                                       processed=len(subset), valid=count, invalid=len(subset)-count,
                                       no_hand=sum(r['hands'] == '0' for r in subset),
                                       one_hand=sum(r['hands'] == '1' for r in subset),
                                       two_hands=sum(r['hands'] == '2' for r in subset),
                                       valid_percent_processed=round(100*count/len(subset), 2) if subset else None))

    # Absolute image counts prevent a percentage from hiding the unprocessed subset.
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = ['#1b9e77', '#d95f02', '#b8c1cc']
    y = np.arange(len(results))
    for axis, percent in zip(axes, (False, True)):
        left = np.zeros(len(results))
        for key, name, color in zip(('valid', 'invalid', 'unprocessed_selected'),
                                     ('Fitur valid', 'Diproses, fitur tidak valid', 'Belum diproses'), colors):
            values = np.array([r[key] for r in results], float)
            if percent:
                values = values/np.array([r['selected_images'] for r in results])*100
            axis.barh(y, values, left=left, color=color, label=name, height=.55)
            for i, (value, offset) in enumerate(zip(values, left)):
                if value > (3 if percent else 100):
                    axis.text(offset+value/2, i, f'{value:.1f}%' if percent else f'{int(value):,}',
                              ha='center', va='center', fontsize=9)
            left += values
        axis.set_yticks(y, [r['dataset'] for r in results]); axis.invert_yaxis()
        axis.set_xlabel('% gambar terpilih' if percent else 'Jumlah gambar terpilih')
    axes[0].set_title('Cakupan ekstraksi: gambar dalam protokol')
    axes[1].set_title('Valid / gagal / belum diproses (penyebut sama)')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=1, frameon=False)
    fig.suptitle('Hasil ekstraksi tersimpan — bukan akurasi pengenalan huruf', fontweight='bold')
    fig.tight_layout(rect=(0, .08, 1, .95)); fig.savefig(output/'overview.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for axis, (title, _, _) in zip(axes, DATASETS):
        data = np.full((26, 3), np.nan)
        for row in class_rows:
            if row['dataset'] == title and row['valid_percent_processed'] is not None:
                data[alphabet.index(row['label']), ('train', 'val', 'test').index(row['split'])] = row['valid_percent_processed']
        cmap = plt.get_cmap('RdYlGn').copy(); cmap.set_bad('#e4e7eb')
        axis.imshow(np.ma.masked_invalid(data), vmin=0, vmax=100, aspect='auto', cmap=cmap)
        for i in range(26):
            for j in range(3):
                axis.text(j, i, '—' if np.isnan(data[i, j]) else f'{data[i,j]:.1f}', ha='center', va='center', fontsize=8)
        axis.set_xticks(range(3), ['Train', 'Validation', 'Test'])
        axis.set_yticks(range(26), alphabet); axis.set_title(title)
    fig.suptitle('Fitur valid per huruf (%) dari gambar yang diproses\nAbu-abu: kelas tidak tersedia / split belum diekstrak; test hanya ringkasan historis', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(output/'per_letter.png', dpi=170); plt.close(fig)

    plotted = [r for r in split_rows if r['processed']]
    fig, axis = plt.subplots(figsize=(12, 6))
    left = np.zeros(len(plotted))
    for key, name, color in zip(('no_hand', 'one_hand', 'two_hands'), ('0 tangan', '1 tangan', '2 tangan'), ('#d95f02', '#66a5d6', '#4265a8')):
        values = np.array([r[key]/r['processed']*100 for r in plotted])
        axis.barh(range(len(plotted)), values, left=left, color=color, label=name)
        for i, (v, start) in enumerate(zip(values, left)):
            if v >= 4:
                axis.text(start+v/2, i, f'{v:.1f}%', ha='center', va='center', fontsize=9)
        left += values
    axis.set_yticks(range(len(plotted)), [r['dataset']+' / '+r['split'] for r in plotted]); axis.invert_yaxis()
    axis.set_xlim(0, 100); axis.set_xlabel('% gambar diproses')
    axis.set_title('Jumlah tangan terdeteksi — dua tangan belum menjamin pose lengkap/benar')
    axis.legend(loc='lower center', bbox_to_anchor=(.5, -.22), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(output/'hand_counts.png', dpi=160); plt.close(fig)

    comparisons = []
    for cache in sorted((ROOT/'data/features').glob('landmarks-bisindo-kaggle-v2*')):
        path = cache/'summary.json'
        if path.exists():
            for split, record in json.loads(path.read_text())['splits'].items():
                comparisons.append(dict(cache=cache.name, split=split, total=record['total'], valid=record['feature_valid'],
                                        valid_percent=round(record['feature_valid_rate']*100, 2)))
    for filename, rows in [('by_split.csv', split_rows), ('by_letter.csv', class_rows), ('previous_settings.csv', comparisons)]:
        with (output/filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(datasets=results, splits=split_rows, previous_settings=comparisons, source_sha256=inputs,
                  note='Reads historical CSV logs and summary metadata only. No new MediaPipe run, image decoding, test predictions, or model selection.')
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# Laporan cakupan ekstraksi landmark', '',
             'Dibuat dari log ekstraksi dan manifest tersimpan. Tidak menjalankan ulang MediaPipe atau membuka foto/test features.', '',
             '## Dataset dan ruang lingkup', '',
             '| Dataset | Inventaris manifest | Terpilih | Diproses | Fitur valid | Tidak valid | Belum diproses (terpilih) | % valid dari diproses |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['dataset']} | {r['inventory_images']:,} | {r['selected_images']:,} | {r['processed']:,} | {r['valid']:,} | {r['invalid']:,} | {r['unprocessed_selected']:,} | {r['valid_percent_processed']:.2f}% |")
    lines += ['', 'Inventaris adalah baris gambar pada manifest audit yang sudah ada, bukan penghitungan ulang semua file dataset saat ini. SIBI memakai subset kurasi 24 kelas tanpa J/Z; varian yang dikecualikan dari protokol tidak dihitung sebagai gagal deteksi. Kaggle memakai split v2 dan cache confidence 0,3; gambar test belum diekstrak pada cache ini.', '',
              '## Rincian split', '', '| Dataset | Split | Terpilih | Diproses | Valid | Gagal | Belum diproses | % valid |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in split_rows:
        pct = '—' if r['valid_percent_processed'] is None else f"{r['valid_percent_processed']:.2f}%"
        lines.append(f"| {r['dataset']} | {r['split']} | {r['selected']:,} | {r['processed']:,} | {r['valid']:,} | {r['invalid']:,} | {r['unprocessed']:,} | {pct} |")
    lines += ['', '## Visualisasi', '', '![Cakupan](overview.png)', '', '![Per huruf](per_letter.png)', '', '![Jumlah tangan](hand_counts.png)', '',
              '## Arti angka', '',
              '- Diproses: ada baris hasil pada log untuk gambar manifest. Penyebut persentase valid adalah gambar yang diproses.',
              '- Terdeteksi: setidaknya satu tangan. Fitur valid juga memerlukan geometri dan kebijakan jumlah tangan yang sesuai.',
              '- Pada cache ini semua fitur tidak valid berstatus no_hand; ringkasan dan log diverifikasi silang saat laporan dibuat.',
              '- BISINDO dengan satu tangan tetap dapat berstatus valid. Ini belum memastikan tangan kedua yang diperlukan suatu pose ikut terdeteksi; dua tangan pun belum membuktikan label benar.',
              '- Persentase ini bukan akurasi landmark atau classifier. Angka test hanya dilaporkan ulang, bukan dipakai memilih detektor/model baru.',
              '- Dataset ASL .npy sudah berupa koordinat dari sumber lain; bukan hasil ekstraksi gambar kita. Video J/Z dan koleksi webcam tidak dicampur ke penyebut gambar dataset ini.', '',
              '## Reproduksi', '', '`python scripts/report_landmark_coverage.py`', '',
              'Tabel lengkap: by_split.csv, by_letter.csv, previous_settings.csv; provenance hash: summary.json.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f'Report: {output.resolve()}')


if __name__ == '__main__':
    main()
