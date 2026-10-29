"""Export presentation-ready validation confusion matrices for the selected models."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'sibi': ROOT / 'artifacts/coordinate-reset-v1/sibi_20261020_130607_889055/cnn_softmax_validation.json',
    'bisindo': ROOT / 'artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637/cnn_softmax_validation.json',
}
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def plot(language: str, source: Path, destination: Path) -> None:
    metrics = json.loads(source.read_text(encoding='utf-8'))
    full = np.asarray(metrics['system_all_images']['confusion_matrix_with_reject_column'], dtype=int)
    if full.shape != (27, 27) or np.any(full[-1]) or np.any(full < 0):
        raise ValueError(f'Unexpected confusion matrix layout: {source}')
    counts = full[:-1, :]
    if counts.sum() != metrics['total'] or counts[:, -1].sum() != metrics['system_all_images']['rejected_due_to_invalid_features']:
        raise ValueError(f'Confusion matrix does not match reported totals: {source}')
    per_class = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(counts, per_class, out=np.zeros_like(counts, dtype=float), where=per_class != 0) * 100

    fig, ax = plt.subplots(figsize=(16, 14), constrained_layout=True)
    fig.patch.set_facecolor('white')
    chart = ax.imshow(percentages, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(27), [*LETTERS, 'Tolak'], fontsize=10)
    ax.set_yticks(range(26), LETTERS, fontsize=10)
    ax.tick_params(axis='x', labelrotation=90, length=0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Prediksi model (angka di sel = jumlah gambar)', fontsize=13, labelpad=12)
    ax.set_ylabel('Huruf sebenarnya', fontsize=13, labelpad=12)
    ax.axvline(25.5, color='#c44236', linewidth=2)
    for row, column in np.argwhere(counts > 0):
        ax.text(column, row, str(counts[row, column]), ha='center', va='center', fontsize=8,
                color='white' if percentages[row, column] >= 50 else '#24323c')
    colorbar = fig.colorbar(chart, ax=ax, fraction=.026, pad=.018)
    colorbar.set_label('% dari gambar pada huruf sebenarnya', fontsize=11)
    score = metrics['system_all_images']['accuracy'] * 100
    ax.set_title(f'Confusion matrix sistem {language.upper()} — validation offline\n'
                 f'n={metrics["total"]:,}; akurasi {score:.2f}%; tolak {counts[:, -1].sum()}',
                 fontsize=16, pad=24)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor='white')
    plt.close(fig)
    print(f'{language.upper()}: {destination} ({metrics["total"]} validation images)', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/presentation')
    args = parser.parse_args()
    for language, source in RUNS.items():
        plot(language, source, args.output_dir / f'confusion_matrix_{language}_validation.png')


if __name__ == '__main__':
    main()
