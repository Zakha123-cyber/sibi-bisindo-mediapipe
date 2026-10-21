from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_coordinates import dataset_fixture
from prepare_coordinates import prepare as prepare_base
from prepare_sibi_jz_finetune import prepare as expand
from train_coordinates import train, epoch_order


class FineTuneTests(unittest.TestCase):
    def test_sampling_balances_classes_without_double_loss_weighting(self):
        target = np.array([0]*10+[1]*1000)
        chosen = epoch_order(target, np.random.default_rng(42), balanced=True)
        self.assertEqual(len(chosen), len(target))
        self.assertGreater(float(np.mean(target[chosen] == 0)), .45)
        self.assertLess(float(np.mean(target[chosen] == 0)), .55)

    def test_expansion_loads_only_unused_train_and_preserves_validation_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = dataset_fixture(root)
            base = root/config['output_dir']/'sibi'
            output = root/'expanded'
            with redirect_stdout(io.StringIO()):
                before = prepare_base(config, 'sibi', root)
            original_load, loaded = np.load, []
            def load(path, *args, **kwargs):
                loaded.append(str(path))
                return original_load(path, *args, **kwargs)
            with patch('prepare_sibi_jz_finetune.np.load', side_effect=load), redirect_stdout(io.StringIO()):
                after = expand(base, output, root)
            self.assertEqual((base/'val.npz').read_bytes(), (output/'val.npz').read_bytes())
            self.assertEqual(before['counts']['train']['classes_valid']['J'], 3)
            self.assertEqual(after['counts']['train']['classes_valid']['J'], 6)
            self.assertEqual(after['counts']['train']['classes_valid']['Z'], 6)
            self.assertFalse(any('val.npz' in p or 'test.npz' in p for p in loaded))
            self.assertFalse(any(f'frame_{i}.npy' in p for p in loaded for i in (6, 7, 8, 9, 10, 11)))

    def test_epoch_zero_keeps_original_weights_when_no_improvement(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = dataset_fixture(root)
            base = root/config['output_dir']/'sibi'
            args = Namespace(threads=1, features_dir=base, language='sibi', device='cpu',
                             output_dir=root/'baseline', epochs=1, models=['cnn'], heads=['softmax'], init_run=None)
            with redirect_stdout(io.StringIO()):
                prepare_base(config, 'sibi', root)
                parent = train(args)
                expanded = expand(base, root/'expanded', root)
                expanded['config']['learning_rate'] = 0.
                (root/'expanded/dataset.json').write_text(json.dumps(expanded))
                args.features_dir = root/'expanded'; args.output_dir = root/'finetuned'; args.init_run = parent
                child = train(args)
            selection = json.loads((child/'selection.json').read_text())
            self.assertEqual(selection['selected']['best_epoch'], 0)
            self.assertTrue(selection['balanced_sampling'])
            self.assertIsNotNone(selection['initialization'])
            a = torch.load(parent/'cnn_best.pt', map_location='cpu', weights_only=True)
            b = torch.load(child/'cnn_best.pt', map_location='cpu', weights_only=True)
            for key in a['state_dict']:
                self.assertTrue(torch.equal(a['state_dict'][key], b['state_dict'][key]))


if __name__ == '__main__':
    unittest.main()
