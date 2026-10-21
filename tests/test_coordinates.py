from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from sibi_bisindo.coordinates import coordinate_tensor, augment_coordinates
from prepare_coordinates import prepare


def hand():
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = .3+np.linspace(0, .2, 21)
    points[:, 1] = .3+np.linspace(0, .3, 21)
    points[:, 2] = np.linspace(0, .02, 21)
    return points


def dataset_fixture(root):
    config = json.loads((ROOT/'configs/coordinate_reset_v1.json').read_text())
    config.update(asl_gap=0, asl_train_fraction=.5, asl_val_fraction=.25, asl_train_limit_per_class=3)
    cache = root/config['sibi_cache']; cache.mkdir(parents=True)
    labels = np.array(list('ABCDEFGHIKLMNOPQRSTUVWXY'))
    for split in ('train', 'val'):
        raw = np.full((len(labels), 2, 21, 3), np.nan, dtype=np.float32)
        raw[:, 0] = hand()
        np.savez_compressed(cache/f'sibi_{split}.npz', raw_landmarks=raw, labels=labels,
                            valid_mask=np.ones(len(labels), bool), processed_count=len(labels),
                            paths=[f'{split}_{label}.jpg' for label in labels],
                            sha256=[f'{split}_{label}_digest' for label in labels],
                            source_groups=[f'{split}_{label}_group' for label in labels],
                            status=['ok']*len(labels), width=[640]*len(labels), height=[480]*len(labels),
                            metadata_json=json.dumps({'language': 'sibi', 'split': split}))
    for label in ('J', 'Z'):
        directory = root/config['asl_root']/label; directory.mkdir(parents=True)
        for i in range(12):
            sample = hand().copy(); sample[:, 0] += i*.001
            if label == 'Z':
                sample[:, 1] += .02
            np.save(directory/f'frame_{i}.npy', sample.reshape(63))
    return config


class CoordinateTests(unittest.TestCase):
    def test_detector_config_hash_ignores_formatting_but_rejects_changed_values(self):
        import hashlib
        from sibi_bisindo.coordinate_inference import verify_detector_config
        from sibi_bisindo.npy_landmarks import canonical
        config = {'detector': {'min_detection_confidence': .5}, 'version': 'fixture'}
        metadata = {'config_sha256': hashlib.sha256(canonical(config)).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(config, separators=(',', ':')))
            self.assertEqual(verify_detector_config(path, metadata), config)
            config['detector']['min_detection_confidence'] = .3
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'config changed'):
                verify_detector_config(path, metadata)

    def test_position_and_scale_invariance_keep_missing_slot_explicit(self):
        a = coordinate_tensor(hand(), .75)
        b = coordinate_tensor(hand()*2+[.1, .2, .3], .75)
        np.testing.assert_allclose(a, b, atol=2e-6)
        self.assertTrue((a[:21, 6] == 1).all())
        self.assertTrue((a[21:] == 0).all())
        noisy = augment_coordinates(a[None], np.random.default_rng(2))[0]
        self.assertTrue((noisy[21:] == 0).all())
        np.testing.assert_allclose(noisy[0, :3], 0, atol=1e-6)

    def test_two_hand_order_and_relative_position_are_preserved(self):
        first, second = hand(), hand()+[.4, 0, 0]
        a = coordinate_tensor([first, second], .75)
        np.testing.assert_allclose(a, coordinate_tensor([second, first], .75))
        b = coordinate_tensor([first, second+[.2, 0, 0]], .75)
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6)
        self.assertGreater(float(np.linalg.norm(a[:, 3:5]-b[:, 3:5])), .1)
        with self.assertRaises(ValueError):
            coordinate_tensor(np.zeros((1, 21, 3)))

    def test_prepare_never_loads_test_arrays_and_caps_asl_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = dataset_fixture(root)
            original = np.load
            loaded = []
            def load(path, *args, **kwargs):
                loaded.append(str(path))
                return original(path, *args, **kwargs)
            with patch('prepare_coordinates.np.load', side_effect=load), redirect_stdout(io.StringIO()):
                result = prepare(config, 'sibi', root)
            self.assertEqual(result['counts']['train']['classes_valid']['J'], 3)
            self.assertEqual(result['counts']['val']['classes_valid']['Z'], 3)
            self.assertFalse(any('_test.npz' in path for path in loaded))
            self.assertFalse(any(f'frame_{i}.npy' in path for path in loaded for i in (9, 10, 11)))
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare(config, 'sibi', root)

    def test_one_epoch_smoke_cnn_svm_artifact_and_inference(self):
        import torch
        from train_coordinates import train
        from sibi_bisindo.coordinate_inference import CoordinatePredictor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = dataset_fixture(root)
            with redirect_stdout(io.StringIO()):
                prepare(config, 'sibi', root)
                run = train(Namespace(threads=1, features_dir=root/config['output_dir']/'sibi',
                                      language='sibi', device='cpu', output_dir=root/'run', epochs=1, models=['cnn', 'mlp']))
            model = CoordinatePredictor(run)
            self.assertEqual(model.model.training, False)
            prediction = model.predict(np.array([hand()]), .75)
            self.assertIn(prediction['label'], list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            self.assertTrue(np.isfinite(prediction['score']))
            selection = json.loads((run/'selection.json').read_text())
            self.assertEqual(len(selection['candidates']), 6)
            self.assertFalse(selection['test_loaded'])
            self.assertTrue((run/'cnn_epochs.csv').exists())
            self.assertTrue((run/'learning_curves.png').exists())


if __name__ == '__main__':
    unittest.main()
