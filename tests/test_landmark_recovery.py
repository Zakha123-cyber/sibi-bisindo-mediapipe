from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sibi_bisindo.landmark_recovery import make_view, map_landmarks, acceptable, RecoveryExtractor
from sibi_bisindo.landmarks import Extraction, geometric_features
from sibi_bisindo.recovery_quality import duplicate_distance, QualityRecoveryExtractor
from test_landmarks import hand


def result(hands, width=640, height=480):
    hands = np.asarray(hands, np.float32).reshape(-1,21,3)
    feature, _ = geometric_features(hands, width, height)
    return Extraction(feature, hands, ['Right']*len(hands), [.9]*len(hands),
                      'ok' if len(hands) else 'no_hand', bool(len(hands)), 1., [], width, height)


class RecoveryTests(unittest.TestCase):
    def test_duplicate_filter_retries_next_view_without_inventing_or_dropping_baseline(self):
        self.assertLess(duplicate_distance(np.array([hand(.3),hand(.31)]), .75), .85)
        self.assertGreater(duplicate_distance(np.array([hand(.2),hand(.7)]), .75), .85)
        class Fake:
            calls=0
            def extract(self,*args,**kwargs):
                self.calls+=1
                return result([hand(.3),hand(.31)]) if self.calls==1 else result([hand(.2),hand(.7)])
        config=self.config();config['views']=[{'name':'first'},{'name':'second'}]
        fake=Fake(); extractor=QualityRecoveryExtractor(fake,config)
        recovered,details=extractor.extract(np.zeros((480,640,3),np.uint8),result([hand(.2)]))
        self.assertEqual(fake.calls,2)
        self.assertEqual(details['method'],'second')
        self.assertEqual(len(recovered.raw),2)

    def config(self):
        return {'recover_one_hand': True, 'views': [{'name': 'full'}],
                'minimum_wrist_separation_palms': .3, 'baseline_anchor_rms_palms': 1.2,
                'maximum_landmark_outside_margin': .1}

    def test_inverse_transform_and_depth_for_rectangular_crop_rotation_zoom(self):
        rgb = np.zeros((480,640,3), np.uint8)
        original = np.array([hand(.35,.6)], np.float32)
        for angle in [0,90,180,270,45,-45]:
            for scale in [1.,1.5]:
                view, inverse = make_view(rgb, {'roi':[.1,.2,.8,.9], 'angle':angle, 'scale':scale})
                forward = cv2.invertAffineTransform(inverse)
                points = original.copy()
                points[:,:,:2] = (original[:,:,:2]*[640,480])@forward[:,:2].T+forward[:,2]
                points[:,:,:2] /= [view.shape[1],view.shape[0]]
                points[:,:,2] = original[:,:,2]*640*scale/view.shape[1]
                restored = map_landmarks(points, view.shape, inverse, rgb.shape)
                np.testing.assert_allclose(restored, original, atol=2e-6)
                a,_=geometric_features(original,640,480); b,_=geometric_features(restored,640,480)
                np.testing.assert_allclose(a,b,atol=2e-5)

    def test_fake_duplicate_or_unanchored_two_hand_candidate_rejected(self):
        config = self.config()
        self.assertFalse(acceptable(np.array([hand(),hand()]), (480,640,3), [], config))
        self.assertFalse(acceptable(np.array([hand(.7),hand(.9)]), (480,640,3), np.array([hand(.1)]), config))
        self.assertTrue(acceptable(np.array([hand(.2),hand(.7)]), (480,640,3), np.array([hand(.2)]), config))

    def test_failed_fallback_preserves_baseline_without_inventing_second_hand(self):
        class Fake:
            def extract(self, *args, **kwargs): return result([])
        base = result([hand()])
        recovered, details = RecoveryExtractor(Fake(), self.config()).extract(np.zeros((480,640,3),np.uint8), base)
        self.assertIs(recovered, base)
        self.assertEqual(details['attempts'],1)
        self.assertFalse(details['recovered'])

    def test_recovers_without_label_and_does_not_rerun_complete_baseline(self):
        class Fake:
            calls=0
            def extract(self, *args, **kwargs):
                self.calls+=1; return result([hand(.2),hand(.7)])
        fake=Fake(); extractor=RecoveryExtractor(fake,self.config()); rgb=np.zeros((480,640,3),np.uint8)
        recovered, details=extractor.extract(rgb,result([]))
        self.assertEqual(len(recovered.raw),2)
        self.assertTrue(details['recovered'])
        base=result([hand(.2),hand(.7)])
        same, details=extractor.extract(rgb,base)
        self.assertIs(same,base)
        self.assertEqual(fake.calls,1)
