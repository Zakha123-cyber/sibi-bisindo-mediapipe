import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.recovery_runtime import (make_policy, verify_policy, point_set_distance,
                                          suspicious_pair, RuntimeRecoveryExtractor)
from sibi_bisindo.recovery_quality import duplicate_distance
from sibi_bisindo.coordinate_detector import build_detector
from sibi_bisindo.npy_landmarks import canonical, sha256_file
from test_landmark_recovery import result
from test_landmarks import hand


def policy_fixture():
    config = {'recover_one_hand': True, 'views': [{'name':'first'}, {'name':'second'}],
              'minimum_wrist_separation_palms':.3, 'baseline_anchor_rms_palms':1.2,
              'maximum_landmark_outside_margin':.1}
    return make_policy({}, config, {'minimum_corresponding_distance_palms':.85,
                                   'minimum_point_set_distance_palms':.55})


class RuntimeRecoveryTests(unittest.TestCase):
    def test_reordered_duplicate_detected_without_corresponding_finger_numbers(self):
        a = hand(.3)
        b = a[::-1].copy()
        self.assertGreater(duplicate_distance([a,b], .75), .85)
        self.assertAlmostEqual(point_set_distance([a,b], .75), 0.)
        self.assertTrue(suspicious_pair([a,b], .75, policy_fixture()['quality']))
        self.assertFalse(suspicious_pair([hand(.2),hand(.7)], .75, policy_fixture()['quality']))

    def test_runtime_skips_bad_view_and_preserves_baseline_and_rgb_orientation(self):
        class Fake:
            calls = 0
            def extract(self, rgb, language, input_mirrored=False):
                self.calls += 1
                assert language == 'bisindo' and input_mirrored is False
                a = hand(.3)
                return result([a,a[::-1]]) if self.calls == 1 else result([hand(.2),hand(.7)])
        fake = Fake(); extractor = RuntimeRecoveryExtractor(fake, policy_fixture())
        base = result([]); original_time = base.process_ms
        recovered = extractor.extract(np.zeros((480,640,3), np.uint8), baseline=base)
        self.assertEqual(len(recovered.raw), 2)
        self.assertEqual(extractor.last_details['method'], 'second')
        self.assertEqual(fake.calls, 2)
        self.assertEqual(base.process_ms, original_time)
        complete = result([hand(.2),hand(.7)])
        extractor.extract(np.zeros((480,640,3), np.uint8), baseline=complete)
        self.assertEqual(fake.calls, 2)
        with self.assertRaises(ValueError):
            extractor.extract(np.zeros((480,640,3), np.uint8), input_mirrored=True)

    def test_policy_rejects_modified_threshold_or_implementation(self):
        policy = policy_fixture()
        verify_policy(policy)
        changed = copy.deepcopy(policy); changed['quality']['minimum_point_set_distance_palms'] = .1
        with self.assertRaisesRegex(ValueError, 'identity changed'): verify_policy(changed)
        changed = copy.deepcopy(policy); changed['implementations']['recovery_runtime.py'] = 'bad'
        changed['sha256'] = hashlib.sha256(canonical({k:v for k,v in changed.items() if k!='sha256'})).hexdigest()
        with self.assertRaisesRegex(ValueError, 'implementation changed'): verify_policy(changed)

    def test_model_policy_binding_prevents_silent_baseline_fallback(self):
        policy = policy_fixture()
        lock = {'runtime_preprocessing':policy, 'config_sha256':hashlib.sha256(canonical({})).hexdigest(),
                'feature_implementation_sha256':policy['implementations']['landmarks.py']}
        dataset = {'runtime_preprocessing':policy, 'extraction_metadata':{'train':lock,'val':copy.deepcopy(lock)}}
        selection = {'language':'bisindo','runtime_preprocessing_sha256':policy['sha256']}
        with patch('sibi_bisindo.coordinate_detector.HandExtractor') as factory:
            detector = build_detector(dataset, selection, ROOT, ROOT)
            self.assertIsInstance(detector, RuntimeRecoveryExtractor)
            factory.assert_called_once_with({})
        for broken in [dict(dataset, runtime_preprocessing=None),
                       dict(dataset, extraction_metadata={'train':lock,'val':{}})]:
            with self.assertRaises(ValueError): build_detector(broken, selection, ROOT, ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dataset.json'; path.write_text(json.dumps(dataset))
            locked_selection = dict(selection, dataset_sha256=sha256_file(path))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                build_detector(dataset, locked_selection, ROOT, directory)


if __name__ == '__main__': unittest.main()
