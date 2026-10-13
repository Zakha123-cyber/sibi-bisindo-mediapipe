"""Behavioral checks for geometry, mirroring, and missing-hand handling."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sibi_bisindo.landmarks import geometric_features,feature_names,policy_status,preprocess_rgb


def hand(x=.3,y=.6):
    points=np.zeros((21,3))
    points[0]=[x,y,0]
    for i in range(1,21):
        points[i]=[x+.01*(i%4),y-.015*i,-.001*i]
    return points


class GeometryTests(unittest.TestCase):
    def test_dimensions_and_missing_slot_mask(self):
        features,_=geometric_features([hand()],640,480)
        self.assertEqual(len(features),132)
        self.assertEqual(len(feature_names()),132)
        np.testing.assert_array_equal(features[:2],[1,0])
        np.testing.assert_array_equal(features[65:128],np.zeros(63))

    def test_no_hand_has_no_feature_vector(self):
        features,order=geometric_features([],640,480)
        self.assertIsNone(features)
        self.assertEqual(order,[])
        self.assertEqual(policy_status('sibi',0,False),('no_hand',False))

    def test_detector_order_does_not_change_features(self):
        a,b=hand(.2),hand(.7)
        first,order=geometric_features([b,a],640,480)
        second,_=geometric_features([a,b],640,480)
        np.testing.assert_allclose(first,second)
        self.assertEqual(order,[1,0])

    def test_global_translation_and_uniform_scale_invariance(self):
        hands=np.array([hand(.2),hand(.7)])
        original,_=geometric_features(hands,640,480)
        transformed=hands*1.2+np.array([.05,-.1,.02])
        actual,_=geometric_features(transformed,640,480)
        np.testing.assert_allclose(original,actual,atol=1e-5)

    def test_aspect_correction_preserves_pixel_geometry(self):
        hands=np.array([hand(.2),hand(.7)])
        original,_=geometric_features(hands,640,480)
        padded=hands*np.array([640/1280,480/720,640/1280])
        actual,_=geometric_features(padded,1280,720)
        np.testing.assert_allclose(original,actual,atol=1e-5)

    def test_relative_hand_position_is_retained(self):
        original,_=geometric_features([hand(.2),hand(.7)],640,480)
        moved,_=geometric_features([hand(.2),hand(.9)],640,480)
        self.assertNotEqual(original[128],moved[128])

    def test_orientation_is_retained(self):
        points=hand()
        original,_=geometric_features([points],640,640)
        flipped=points.copy();flipped[:,0]=1-flipped[:,0]
        changed,_=geometric_features([flipped],640,640)
        self.assertFalse(np.allclose(original,changed))

    def test_invalid_geometry_fails_explicitly(self):
        for points in [np.zeros((1,21,3)),np.full((1,21,3),np.nan),np.zeros((3,21,3))]:
            with self.assertRaises(ValueError):geometric_features(points,640,480)

    def test_extra_sibi_hand_is_not_silently_selected(self):
        self.assertEqual(policy_status('sibi',2,True),('unexpected_extra_hand',False))

    def test_bisindo_completeness_is_not_inferred_from_count_alone(self):
        status,valid=policy_status('bisindo',1,True)
        self.assertTrue(valid)
        self.assertIn('unverified',status)
        status,_=policy_status('bisindo',2,True)
        self.assertIn('actor_unverified',status)

    def test_mirrored_input_is_unmirrored_once(self):
        image=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        cfg={'preprocessing':{'contrast':'none'}}
        actual=preprocess_rgb(image[:,::-1],cfg,input_mirrored=True)
        np.testing.assert_array_equal(image,actual)
        self.assertTrue(actual.flags.c_contiguous)

    def test_non_rgb_input_is_rejected(self):
        cfg={'preprocessing':{'contrast':'none'}}
        with self.assertRaises(ValueError):preprocess_rgb(np.zeros((3,3),dtype=np.uint8),cfg)


if __name__=='__main__':unittest.main()
