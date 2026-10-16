"""Analytic checks distinguish sliding, airborne motion and floor penetration."""
import json
from pathlib import Path
import unittest
import numpy as np
from physics_metrics import measure


class PhysicsChecks(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((Path(__file__).parent/'physics_protocol.json').read_text(encoding='utf-8'))
        rng=np.random.RandomState(1)
        self.pose=rng.uniform(-.5,.5,(20,22,3))
        self.pose[:]=self.pose[0]
        self.pose[...,1]+=1
        self.pose[:,[7,8,10,11],1]=.02

    def test_static(self):
        values=measure(self.pose,self.protocol)
        self.assertEqual(values['near_floor_speed_mps'],0)
        self.assertEqual(values['skating_frame_ratio'],0)
        self.assertEqual(values['near_floor_toe_interval_fraction'],1)
        self.assertEqual(values['foot_penetration_mean_m'],0)
        self.assertEqual(values['relative_jerk_rms_mps3'],0)

    def test_known_grounded_slide(self):
        self.pose[...,0]+=np.arange(20)[:,None]/20
        values=measure(self.pose,self.protocol)
        self.assertAlmostEqual(values['near_floor_speed_mps'],1)
        self.assertEqual(values['skating_frame_ratio'],1)
        self.assertEqual(values['skating_given_contact_ratio'],1)
        self.assertLess(values['bone_length_cv'],1e-12)
        self.assertLess(values['relative_jerk_rms_mps3'],1e-10)

    def test_airborne_does_not_masquerade_as_good_contact(self):
        self.pose[...,1]+=.3
        values=measure(self.pose,self.protocol)
        self.assertIsNone(values['near_floor_speed_mps'])
        self.assertIsNone(values['skating_given_contact_ratio'])
        self.assertEqual(values['near_floor_intervals'],0)
        self.assertEqual(values['airborne_frame_ratio'],1)

    def test_penetration_and_tolerance(self):
        self.pose[:,[7,8,10,11],1]=-.03
        values=measure(self.pose,self.protocol)
        self.assertAlmostEqual(values['foot_penetration_mean_m'],.03)
        self.assertEqual(values['penetration_frame_ratio'],1)
        self.pose[:,[7,8,10,11],1]=-.01
        self.assertEqual(measure(self.pose,self.protocol)['penetration_frame_ratio'],0)

    def test_endpoint_contact_does_not_use_speed(self):
        self.pose[1:,[10,11],1]=.2
        self.pose[...,0]+=np.arange(20)[:,None]
        self.assertEqual(measure(self.pose,self.protocol)['near_floor_intervals'],0)

    def test_horizontal_rigid_transform_invariant(self):
        self.pose[...,0]+=np.arange(20)[:,None]/20
        before=measure(self.pose,self.protocol)
        rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        transformed=self.pose@rotation+np.array([10,0,-3])
        after=measure(transformed,self.protocol)
        for key in ['near_floor_speed_mps','skating_frame_ratio','foot_penetration_mean_m','root_speed_mps']:
            self.assertAlmostEqual(before[key],after[key])

    def test_fps_changes_speed(self):
        self.pose[...,0]+=np.arange(20)[:,None]/20
        self.protocol['fps']=40
        self.assertAlmostEqual(measure(self.pose,self.protocol)['near_floor_speed_mps'],2)

    def test_invalid_input_rejected(self):
        self.pose[0,0,0]=np.nan
        with self.assertRaises(ValueError):measure(self.pose,self.protocol)


if __name__=='__main__':unittest.main()
