import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test_pilot import fixture
from sibi_bisindo.pilot import TrialWriter
from audit_webcam_pilot import audit


class PilotAuditTests(unittest.TestCase):
    def test_test_images_are_hashed_but_never_decoded_or_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            encoded = io.BytesIO()
            Image.new('RGB', (64, 48)).save(encoded, format='JPEG')
            for participant, content in [('p001', encoded.getvalue()), ('p003', b'not-decodable-test-image')]:
                trial = TrialWriter(root/'collection', config, 'sibi', participant, 's01',
                                    'no_hand', 'negative', {'consent_recorded': True, 'resolution': [64, 48]})
                trial.add_frame(content, 0)
                trial.finish(completed=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit(root/'collection', root/'report')
            self.assertEqual(result['integrity_issue_count'], 0)
            self.assertEqual(result['decoded_development_frames'], 1)
            self.assertEqual(result['decoded_test_frames'], 0)
            self.assertFalse(list((root/'report').glob('*p003*.jpg')))

    def test_changed_image_is_reported_and_not_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            trial = TrialWriter(root/'collection', config, 'sibi', 'p001', 's01',
                                'no_hand', 'negative', {'consent_recorded': True, 'resolution': [64, 48]})
            encoded = io.BytesIO()
            Image.new('RGB', (64, 48)).save(encoded, format='JPEG')
            trial.add_frame(encoded.getvalue(), 0)
            trial.finish(completed=True)
            (trial.directory/'frames/frame_0000.jpg').write_bytes(b'changed')
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit(root/'collection', root/'report')
            self.assertEqual(result['integrity_issue_count'], 1)
            self.assertIn('image_hash_mismatch', result['integrity_issues'][0]['issues'])
            self.assertFalse(list((root/'report').glob('*.jpg')))
