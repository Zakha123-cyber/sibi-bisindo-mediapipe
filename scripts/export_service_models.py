"""Export only the two selected checkpoints and their exact inference metadata."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.npy_landmarks import sha256_file


def export(config, output):
    from sibi_bisindo.coordinate_inference import CoordinatePredictor
    if output.exists():
        raise ValueError('Output already exists; choose a new bundle directory')
    validated = {}
    for language, relative in config['models'].items():
        source = ROOT/relative
        predictor = CoordinatePredictor(source)
        if predictor.selection['language'] != language or predictor.svm is not None:
            raise ValueError('Selected service models must be the approved coordinate CNN softmax models')
        names = {'selection.json', 'dataset.json', *predictor.selection['artifact_sha256']}
        if any(Path(name).name != name or not (source/name).is_file() for name in names):
            raise ValueError('Invalid source model file')
        if predictor.selection.get('dataset_sha256') and sha256_file(source/'dataset.json') != predictor.selection['dataset_sha256']:
            raise ValueError('Run metadata changed after training')
        validated[language] = (source, names)
    if set(validated) != {'sibi', 'bisindo'}: raise ValueError('Need exactly SIBI and BISINDO')
    output.mkdir(parents=True)
    manifest = dict(bundle_version=config['bundle_version'], created_at_utc=datetime.now(timezone.utc).isoformat(), models={})
    for language, (source, names) in validated.items():
        target = output/language; target.mkdir()
        for name in sorted(names): shutil.copyfile(source/name, target/name)
        manifest['models'][language] = dict(source_run=source.relative_to(ROOT).as_posix(),
            files={name:sha256_file(target/name) for name in sorted(names)})
    (output/'bundle.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Exported {output}; upload this directory separately, do not commit it to Git')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/service_models.json')
    parser.add_argument('--output', type=Path, default=ROOT/'models/selected-sibi-bisindo-v1')
    args = parser.parse_args()
    export(json.loads(args.config.read_text()), args.output)


if __name__ == '__main__': main()
