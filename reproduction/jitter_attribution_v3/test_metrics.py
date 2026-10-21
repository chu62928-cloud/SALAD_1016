"""Analytic metric checks; these are not model quality pass/fail thresholds."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from metrics import PARENTS,measures,fixed_directions,paired_error

PROTOCOL = dict(fps=20,contact_height_m=.05,slide_speed_mps=.2,penetration_tolerance_m=.02,
                sensitivity_heights_m=[.03,.05,.1],sensitivity_speeds_mps=[.1,.2,.3])


class MetricsTests(unittest.TestCase):
    def base(self):
        offsets = np.zeros((22,3))
        offsets[1:,0]=.2
        p = np.zeros((120,22,3))
        for j in range(1,22): p[:,j]=p[:,PARENTS[j]]+offsets[j]
        p[:,:,1]+=1
        return p,offsets

    def test_constant_and_translation(self):
        p,_ = self.base()
        result,_,_,_ = measures(p,PROTOCOL)
        self.assertLess(result['bone_length_cv'],1e-12)
        self.assertEqual(result['world_jerk_rms_mps3'],0)
        p[:,:,2] = np.arange(len(p))[:,None]/20
        result,_,_,_ = measures(p,PROTOCOL)
        self.assertAlmostEqual(result['world_speed_rms_mps'],1,places=12)
        self.assertLess(result['local_jerk_rms_mps3'],1e-8)

    def test_polynomial_units(self):
        p,_ = self.base()
        time = np.arange(len(p))/20
        p[:,:,2] = time[:,None]**3
        result,_,_,_ = measures(p,PROTOCOL)
        self.assertAlmostEqual(result['world_jerk_rms_mps3'],6,places=8)
        self.assertAlmostEqual(result['root_jerk_rms_mps3'],6,places=8)

    def test_bone_oscillation_and_projection(self):
        p,offsets = self.base()
        p[:,1,0] += .02*np.sin(2*np.pi*4*np.arange(len(p))/20)
        result,_,_,_ = measures(p,PROTOCOL)
        self.assertGreater(result['bone_hf3_cv'],.001)
        corrected = fixed_directions(p,offsets)
        fixed,_,_,_ = measures(corrected,PROTOCOL)
        self.assertLess(fixed['bone_length_cv'],1e-12)
        np.testing.assert_array_equal(corrected[:,0],p[:,0])

    def test_time_units_and_vector_rms(self):
        p,_ = self.base()
        p[:,1,1] += .01*np.sin(2*np.pi*4*np.arange(len(p))/20)
        result,_,_,_ = measures(p,PROTOCOL)
        doubled,*_ = measures(p,dict(PROTOCOL,fps=40))
        self.assertAlmostEqual(doubled['local_jerk_rms_mps3']/result['local_jerk_rms_mps3'],8)
        self.assertAlmostEqual(result['local_jerk_rms_mps3'],result['relative_jerk_rms_mps3'])
        error = paired_error(p,p)
        self.assertTrue(all(value==0 for value in error.values()))


if __name__ == '__main__':
    unittest.main()
