from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_coordinates import ROOT, hand
from sibi_bisindo.coordinate_inference import CoordinatePredictor
from sibi_bisindo.coordinate_model import CoordinateNet
from sibi_bisindo.coordinates import VERSION, coordinate_tensor
from sibi_bisindo.npy_landmarks import sha256_file
from train_bisindo_transfer import new_bisindo_model, set_stage, train


def source_fixture(root):
    source = root/'source'; source.mkdir()
    model = CoordinateNet(26)
    with torch.no_grad():
        model.classifier.weight.fill_(.25); model.classifier.bias.fill_(.75)
    torch.save({'state_dict': model.state_dict(), 'kind': 'cnn', 'classes': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
                'language': 'sibi', 'feature_version': VERSION, 'epoch': 89}, source/'cnn_best.pt')
    selection = {'language': 'sibi', 'classes': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'feature_version': VERSION,
                 'selected': {'kind': 'cnn', 'checkpoint': 'cnn_best.pt', 'svm': None, 'head': 'softmax'},
                 'coordinate_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/coordinates.py'),
                 'model_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/coordinate_model.py'),
                 'artifact_sha256': {'cnn_best.pt': sha256_file(source/'cnn_best.pt')}}
    (source/'selection.json').write_text(json.dumps(selection))
    return source


class BisindoTransferTests(unittest.TestCase):
    def test_encoder_copied_classifier_new_and_freeze_then_unfreeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = CoordinatePredictor(source_fixture(root))
            model = new_bisindo_model(parent, list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            for key, value in parent.model.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, model.encoder.state_dict()[key]))
            self.assertFalse(torch.equal(parent.model.classifier.weight, model.classifier.weight))
            config = json.loads((ROOT/'configs/bisindo_transfer_v1.json').read_text())
            optimizer = set_stage(model, 'head', config)
            self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
            before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
            old_head = model.classifier.weight.detach().clone()
            x = torch.randn(4, 42, 7); y = torch.tensor([0, 1, 2, 3])
            loss = torch.nn.CrossEntropyLoss()(model(x), y); loss.backward(); optimizer.step()
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.encoder.state_dict().items()))
            self.assertFalse(torch.equal(old_head, model.classifier.weight))
            optimizer = set_stage(model, 'finetune', config)
            self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
            self.assertEqual(len(optimizer.param_groups), 2)

    def test_two_stage_smoke_exports_bisindo_without_changing_sibi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = source_fixture(root)
            digest = sha256_file(source/'cnn_best.pt')
            features = root/'features'; features.mkdir()
            classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            for split in ('train', 'val'):
                xs = [coordinate_tensor([hand()] if i % 2 else [hand(), hand()+[.4, 0, 0]], .75) for i in range(26)]
                labels, mask = classes.copy(), [True]*26
                if split == 'val':
                    xs.append(np.zeros((42, 7), np.float32)); labels.append('A'); mask.append(False)
                np.savez_compressed(features/f'{split}.npz', X=np.stack(xs), labels=labels, valid_mask=mask,
                                    sources=['bisindo_mediapipe']*len(labels), paths=[f'{split}/{i}' for i in range(len(labels))])
            dataset = {'language': 'bisindo', 'feature_version': VERSION, 'classes': classes,
                       'coordinate_implementation_sha256': sha256_file(ROOT/'src/sibi_bisindo/coordinates.py'),
                       'package_sha256': {f'{s}.npz': sha256_file(features/f'{s}.npz') for s in ('train', 'val')}}
            from test_recovery_runtime import policy_fixture
            policy = policy_fixture()
            dataset['runtime_preprocessing'] = policy
            dataset['extraction_metadata'] = {s:{'runtime_preprocessing':policy} for s in ('train','val')}
            (features/'dataset.json').write_text(json.dumps(dataset))
            args = Namespace(config=None, source_run=source, features_dir=features, output_dir=root/'run',
                             head_epochs=1, finetune_epochs=1, threads=1, device='cpu')
            with redirect_stdout(io.StringIO()):
                run = train(args)
            predictor = CoordinatePredictor(run)
            self.assertEqual(predictor.selection['language'], 'bisindo')
            self.assertEqual(sha256_file(source/'cnn_best.pt'), digest)
            self.assertTrue(predictor.selection['stage_reports']['head']['encoder_unchanged'])
            self.assertEqual(predictor.selection['stage_reports']['finetune']['epochs_run'], 1)
            self.assertFalse(predictor.selection['test_loaded'])
            self.assertEqual(predictor.selection['runtime_preprocessing_sha256'], policy['sha256'])
            self.assertEqual(predictor.selection['dataset_sha256'], sha256_file(run/'dataset.json'))
            result = predictor.predict([hand(), hand()+[.4, 0, 0]], .75)
            self.assertIn(result['label'], classes)
            metrics = json.loads((run/'cnn_softmax_validation.json').read_text())
            self.assertEqual(metrics['total'], 27)
            self.assertEqual(metrics['feature_valid'], 26)
            self.assertEqual(metrics['by_detected_hand_count']['2']['samples'], 13)
            self.assertTrue((run/'head_best.pt').exists())
            self.assertTrue((run/'finetune_best.pt').exists())
