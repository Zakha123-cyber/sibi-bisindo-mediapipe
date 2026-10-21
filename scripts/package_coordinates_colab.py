"""Create a small, local Colab archive with code and prepared train/val tensors."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['sibi', 'bisindo'], default='sibi')
    parser.add_argument('--features-dir', type=Path)
    parser.add_argument('--init-run', type=Path)
    parser.add_argument('--transfer-config', type=Path, help='Include the staged BISINDO transfer trainer/config')
    args = parser.parse_args()
    if args.transfer_config and (args.language != 'bisindo' or not args.init_run):
        parser.error('BISINDO transfer packaging requires --language bisindo and --init-run')
    files = [ROOT/'scripts/train_coordinates.py', ROOT/'configs/coordinate_reset_v1.json',
             ROOT/'requirements-coordinate-training.txt']
    files += [ROOT/'src/sibi_bisindo'/name for name in ('__init__.py', 'coordinates.py', 'coordinate_model.py', 'coordinate_inference.py',
                                                       'npy_landmarks.py', 'training.py')]
    directory = (args.features_dir or ROOT/'data/features/coordinate-reset-v1'/args.language).resolve()
    files += [directory/name for name in ('train.npz', 'val.npz', 'dataset.json', 'manifest.csv')]
    if args.transfer_config:
        files += [ROOT/'scripts/train_bisindo_transfer.py', args.transfer_config.resolve()]
    if args.init_run:
        import json
        args.init_run = args.init_run.resolve()
        selection = json.loads((args.init_run/'selection.json').read_text())
        files += [args.init_run/'selection.json', args.init_run/selection['selected']['checkpoint']]
    if any(not path.exists() for path in files):
        raise FileNotFoundError('Prepare coordinate data before packaging')
    output = ROOT/'data/exports'; output.mkdir(parents=True, exist_ok=True)
    version = 'coordinate-jz-ft-v1' if args.init_run else 'coordinate-reset-v1'
    if args.transfer_config:
        version = json.loads(args.transfer_config.read_text())['version']
    target = output/f'{version}-{args.language}-colab.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'{target} ({target.stat().st_size/1024/1024:.2f} MiB); no raw images or test arrays included')


if __name__ == '__main__':
    main()
