import unittest
import numpy as np
from analyzer_IMD import compute_IMD_smpte
from generator import synth_two_tones, apply_nonlinear_distortion, soft_clip_tanh

class TestIMDSMPTE(unittest.TestCase):

    def setUp(self):
        self.fs_test = 48000
        self.f1_test = 60.0  # SMPTE f1
        self.f2_test = 7000.0 # SMPTE f2
        self.clean_signal = synth_two_tones(fs=self.fs_test, f1=self.f1_test, f2=self.f2_test, amp1=0.8, amp2=0.2)

    def test_clean_signal_imd(self):
        """Test 1: Senyal neta sense distorsió (esperem IMD = 0%)"""
        imd_clean = compute_IMD_smpte(self.clean_signal, self.fs_test, self.f1_test, self.f2_test)
        print(f"\nIMD (SMPTE) de la senyal neta: {imd_clean:.2f} %")
        self.assertAlmostEqual(imd_clean, 0.0, places=1, msg=f"S'esperava IMD ~= 0, però es va obtenir {imd_clean:.2f}%")
        print("PASS: La senyal neta té IMD proper a 0%.")

    def test_quadratic_distortion_imd(self):
        """Test 2: Senyal amb distorsió quadràtica (esperem IMD > 0%)"""
        alpha_test = 0.1
        distorted_quad_test_signal = apply_nonlinear_distortion(self.clean_signal, alpha=alpha_test)
        imd_quad_test = compute_IMD_smpte(distorted_quad_test_signal, self.fs_test, self.f1_test, self.f2_test)
        print(f"\nIMD (SMPTE) amb Distorsió Quadràtica (alpha={alpha_test}): {imd_quad_test:.2f} %")
        self.assertTrue(imd_quad_test > 1.0, f"S'esperava IMD > 1% per distorsió quadràtica, però es va obtenir {imd_quad_test:.2f}%")
        print("PASS: La senyal amb distorsió quadràtica té IMD significatiu.")

    def test_soft_clipper_tanh_imd(self):
        """Test 3: Senyal amb Soft Clipper (esperem IMD baix per a no-linealitat senar)"""
        dist_tanh_test = 5.0
        distorted_tanh_test_signal = soft_clip_tanh(self.clean_signal, dist=dist_tanh_test)
        imd_tanh_test = compute_IMD_smpte(distorted_tanh_test_signal, self.fs_test, self.f1_test, self.f2_test)
        print(f"\nIMD (SMPTE) amb Soft Clipper (dist={dist_tanh_test}): {imd_tanh_test:.2f} %")
        self.assertTrue(imd_tanh_test < 1.0, f"S'esperava IMD < 1% per soft clipper (tanh), però es va obtenir {imd_tanh_test:.2f}%")
        print("PASS: La senyal amb soft clipper (tanh) té IMD SMPTE baix, com s'espera per a una no-linealitat senar.")

# Per executar els tests en un entorn de Colab/Jupyter
if __name__ == '__main__':
    # Per a que els prints dels tests siguin visibles directament
    # sense la necessitat de generar un informe de test complet
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestIMDSMPTE))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
