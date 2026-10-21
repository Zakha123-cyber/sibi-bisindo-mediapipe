import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from sibi_bisindo.pilot import TrialWriter, completed_counts, lock_collection, read_pose_review
from collect_webcam_pilot import main


def fixture(root):
    config = json.loads((ROOT / 'configs/webcam_pilot_v1.json').read_text(encoding='utf-8'))
    review = root / 'poses.csv'
    review.write_bytes((ROOT / 'docs/pilot_pose_review.csv').read_bytes())
    config.update(output_dir=str(root / 'collection'), pose_review_file=str(review))
    return config


class PilotDataTests(unittest.TestCase):
    def test_same_participant_has_same_split_across_languages_and_cancelled_frames_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            output = root / 'collection'
            for mode in ('sibi', 'bisindo'):
                pose, digest = read_pose_review(config['pose_review_file'], mode, 'A')
                trial = TrialWriter(output, config, mode, 'p003', 's01', 'A', 'positive',
                                    {'consent_recorded': True}, pose, digest)
                trial.add_frame(b'synthetic-image-bytes', 0.1)
                result = trial.finish(completed=False, reason='operator_cancel')
                self.assertEqual(result['split'], 'test')
                self.assertEqual(result['status'], 'aborted')
                self.assertEqual(result['frame_count'], 1)
                self.assertTrue((trial.directory / 'frames/frame_0000.jpg').exists())
                self.assertFalse(completed_counts(output / mode / 'p003' / 's01'))

    def test_protocol_cannot_be_changed_in_existing_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            output = root / 'collection'
            digest = lock_collection(output, config)
            self.assertEqual(digest, lock_collection(output, config))
            changed = copy.deepcopy(config)
            changed['participants'].update(p001='val', p002='train')
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                lock_collection(output, changed)

    def test_negative_trial_is_saved_without_requiring_a_hand_detector_or_pose_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            trial = TrialWriter(root / 'collection', config, 'bisindo', 'p001', 's02',
                                'no_hand', 'negative', {'consent_recorded': True})
            trial.add_frame(b'blank-frame', 0)
            result = trial.finish(completed=True)
            self.assertEqual(result['status'], 'completed')
            self.assertIsNone(result['pose_review'])
            with (trial.directory / 'frames.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            with self.assertRaises(ValueError):
                trial.add_frame(b'extra', 0.5)

    def test_unregistered_person_and_missing_consent_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            for person, context in [('../p001', {'consent_recorded': True}), ('p001', {})]:
                with self.assertRaises(ValueError):
                    TrialWriter(root / 'collection', config, 'sibi', person, 's01', 'no_hand', 'negative', context)
            self.assertFalse((root / 'collection').exists())

    def test_camera_loop_saves_raw_frames_after_countdown_without_overlays(self):
        import cv2

        class Camera:
            def isOpened(self):
                return True

            def set(self, prop, value):
                pass

            def get(self, prop):
                return 64

            def read(self):
                self.frame_index = getattr(self, 'frame_index', 0) + 1
                return True, np.full((64, 64, 3), self.frame_index, dtype=np.uint8)

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            config.update(countdown_seconds=0.2, capture_seconds=0.4, frame_interval_seconds=0.1)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            camera, tick = Camera(), [0.0]

            def clock():
                tick[0] += 0.1
                return tick[0]

            argv = ['collect_webcam_pilot.py', '--config', str(config_path), '--mode', 'sibi',
                    '--participant-id', 'p001', '--session-id', 's01', '--save', '--consent']
            with patch.object(sys, 'argv', argv), patch.object(cv2, 'VideoCapture', return_value=camera), \
                    patch.object(cv2, 'imshow'), patch.object(cv2, 'destroyAllWindows'), \
                    patch.object(cv2, 'waitKey', side_effect=[ord('a')]+[-1]*7+[27]), \
                    patch('collect_webcam_pilot.time.monotonic', side_effect=clock):
                main()
            paths = list((root / 'collection').glob('sibi/p001/s01/trials/*/trial.json'))
            self.assertEqual(len(paths), 1)
            trial = json.loads(paths[0].read_text())
            self.assertEqual(trial['status'], 'completed')
            self.assertEqual(trial['split'], 'train')
            self.assertEqual(trial['label'], 'A')
            images = list(paths[0].parent.glob('frames/*.jpg'))
            self.assertGreater(len(images), 0)
            saved = cv2.imread(str(sorted(images)[0]))
            self.assertGreaterEqual(float(saved.mean()), 3)  # No frame saved before countdown ends.
            self.assertEqual(float(saved.std()), 0)  # No text/landmark overlay in source images.
            self.assertTrue(camera.released)


if __name__ == '__main__':
    unittest.main()
