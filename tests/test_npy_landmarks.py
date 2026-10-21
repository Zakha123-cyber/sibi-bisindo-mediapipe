import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from sibi_bisindo.npy_landmarks import frame_number, normalized_features, split_assignment
from prepare_npy_landmarks import prepare
from train_npy_landmarks import read_features


class Landmark63Tests(unittest.TestCase):
    def test_translation_and_uniform_scale_do_not_change_features(self):
        points = np.random.default_rng(3).normal(size=(21, 3))
        expected = normalized_features(points)
        np.testing.assert_allclose(expected, normalized_features(points * 3 + [2, 5, 7]), atol=1e-6)
        np.testing.assert_allclose(expected, normalized_features(points.reshape(63)))
        self.assertEqual(expected.dtype, np.float32)
        np.testing.assert_array_equal(expected[:3], 0)

    def test_orientation_is_retained_and_invalid_geometry_is_rejected(self):
        points = np.random.default_rng(4).normal(size=(21, 3))
        mirrored = points.copy()
        mirrored[:, 0] *= -1
        self.assertFalse(np.allclose(normalized_features(points), normalized_features(mirrored)))
        for invalid in (np.zeros((21, 3)), np.full((21, 3), np.nan), np.ones(62)):
            with self.assertRaises(ValueError):
                normalized_features(invalid)

    def test_numeric_order_and_boundary_gaps_are_explicit(self):
        paths = ['landmarks/A/frame_10.npy', 'landmarks/A/frame_2.npy']
        self.assertEqual(sorted(paths, key=frame_number)[0], paths[1])
        splits = [split_assignment(i, 400) for i in range(400)]
        self.assertEqual(splits.count('train'), 270)
        self.assertEqual(splits.count('val'), 40)
        self.assertEqual(splits.count('test'), 50)
        self.assertEqual(splits.count('gap'), 40)
        for path in ('../frame_1.npy', '/frame_1.npy', 'C:/frame_1.npy', 'frame_x.npy'):
            with self.assertRaises(ValueError):
                frame_number(path)


def make_fixture(root):
    config = json.loads((ROOT / 'configs/npy_landmarks_v1.json').read_text(encoding='utf-8'))
    config.update(classes=['A', 'B'], excluded_classes={'J': 'static_only'},
                  dataset_root='raw', audit_dir='audit', output_dir='features',
                  train_fraction=0.5, validation_fraction=0.25, gap_samples_each_side=1)
    rows = []
    rng = np.random.default_rng(5)
    for label in ('A', 'B', 'J'):
        count = 20 if label != 'J' else 2
        for i in range(count):
            a = rng.normal(size=(21, 3))
            a[4:9, 0] += 4 if label == 'A' else -4
            buffer = io.BytesIO()
            np.save(buffer, a.reshape(63))
            content = buffer.getvalue()
            relative = f'landmarks/{label}/frame_{i}.npy'
            rows.append({'path': relative, 'label': label, 'sha256': hashlib.sha256(content).hexdigest(), 'valid': '1'})
            # Test/gap/excluded arrays deliberately do not exist on disk. Preparing
            # training features must work without opening any of these files.
            if label != 'J' and split_assignment(i, count, 0.5, 0.25, 1) in ('train', 'val'):
                path = root / 'raw' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    rows.sort(key=lambda row: row['path'])
    snapshot = hashlib.sha256()
    for row in rows:
        snapshot.update((row['path'] + '\0' + row['sha256'] + '\n').encode('utf-8'))
    config['audit_snapshot_sha256'] = snapshot.hexdigest()
    audit = root / 'audit'
    audit.mkdir()
    (audit / 'summary.json').write_text(json.dumps({'snapshot_sha256': snapshot.hexdigest(),
                                                  'invalid_samples': [], 'file_count': len(rows)}))
    with (audit / 'inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return config


class PreparationAndTrainingTests(unittest.TestCase):
    def test_preparation_locks_split_and_never_requires_test_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_fixture(root)
            lock = prepare(config, root=root)
            self.assertFalse((root / 'features/test.npz').exists())
            self.assertFalse(lock['test_arrays_loaded'])
            self.assertEqual(lock, prepare(config, root=root))
            _, _, data = read_features(root / 'features')
            self.assertEqual(len(data['train'][1]), 18)
            self.assertEqual(len(data['val'][1]), 6)
            changed = copy.deepcopy(config)
            changed['gap_samples_each_side'] = 2
            with self.assertRaisesRegex(ValueError, 'Locked snapshot changed'):
                prepare(changed, root=root)

    def test_synthetic_training_saves_loadable_model_and_epoch_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare(make_fixture(root), root=root)
            output = root / 'run'
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/train_npy_landmarks.py'),
                '--features-dir', str(root / 'features'), '--output-dir', str(output),
                '--epochs', '2', '--families', 'svm', 'mlp',
            ], capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            selection = json.loads((output / 'selection.json').read_text())
            self.assertEqual(selection['source_language'], 'asl')
            self.assertEqual(selection['pose_mapping'], 'unverified')
            self.assertFalse(selection['test_accessed'])
            self.assertEqual(len(selection['candidates']), 7)
            self.assertEqual(len((output / 'mlp_128x64_epochs.csv').read_text().splitlines()), 3)
            bundle = joblib.load(output / selection['artifact'])
            _, _, data = read_features(root / 'features')
            self.assertEqual(bundle['model'].predict(data['val'][0]).shape, (6,))


if __name__ == '__main__':
    unittest.main()
