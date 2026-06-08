import unittest
import numpy as np
from analyzer_IMD import compute_IMD_ccif
from generator import synth_two_tones, apply_nonlinear_distortion, soft_clip_tanh, generate_signal_with_target_imd_senar


class TestIMDCCIF(unittest.TestCase):
    """Test cases for CCIF/DIM IMD computation"""

    @classmethod
    def setUpClass(cls):
        """Set up test parameters segons estàndard CCIF (19 kHz i 20 kHz)"""
        cls.fs_test = 48000
        cls.f1_test = 19000.0  # CCIF estàndard
        cls.f2_test = 20000.0  # CCIF estàndard
        cls.clean_signal = synth_two_tones(fs=cls.fs_test, f1=cls.f1_test, f2=cls.f2_test)

    def test_clean_signal_imd_zero(self):
        """Test 1: Senyal neta sense distorsió (esperem IMD = 0%)"""
        imd_clean = compute_IMD_ccif(self.clean_signal, self.fs_test, self.f1_test, self.f2_test)
        self.assertAlmostEqual(imd_clean, 0.0, delta=0.01,
                               msg=f"S'esperava IMD ~= 0, però es va obtenir {imd_clean:.2f}%")

    def test_quadratic_distortion_imd(self):
        """Test 2: Senyal amb distorsió quadràtica (esperem IMD > 0%)"""
        alpha_test = 0.1
        distorted_quad_signal = apply_nonlinear_distortion(self.clean_signal, alpha=alpha_test)
        imd_quad = compute_IMD_ccif(distorted_quad_signal, self.fs_test, self.f1_test, self.f2_test)
        print(imd_quad)
        self.assertGreater(imd_quad, 1.0,
                           msg=f"S'esperava IMD > 1% per distorsió quadràtica, però es va obtenir {imd_quad:.2f}%")

    def test_soft_clipper_imd(self):
        """Test 3: Senyal amb Soft Clipper (esperem IMD > 0% i possiblement més alt)"""
        dist_tanh_test = 5.0
        distorted_tanh_signal = soft_clip_tanh(self.clean_signal, dist=dist_tanh_test)
        imd_tanh = compute_IMD_ccif(distorted_tanh_signal, self.fs_test, self.f1_test, self.f2_test)
        print(imd_tanh)
        self.assertGreater(imd_tanh, 10.0,
                           msg=f"S'esperava IMD > 10% per soft clipper, però es va obtenir {imd_tanh:.2f}%")
    def test_target_imd_generator_ccif(self):
        """Test 4: Senyal generat amb IMD CCIF objectiu del 10% (distorsió senar)"""
        target_imd = 10.0
        distorted_signal, final_dist, actual_imd, iterations = generate_signal_with_target_imd_senar(
            target_imd=target_imd,
            fs=self.fs_test, f1=self.f1_test, f2=self.f2_test,
            amp1=0.5, amp2=0.5,
            tolerance=0.05
        )
        imd_measured = compute_IMD_ccif(distorted_signal, self.fs_test, self.f1_test, self.f2_test)
        print(f"\nIMD objectiu: {target_imd:.2f}%, IMD real del generador: {actual_imd:.2f}%, IMD mesurada: {imd_measured:.2f}%")
        self.assertAlmostEqual(imd_measured, actual_imd, delta=0.5,
                            msg=f"S'esperava IMD ~= {actual_imd:.2f}%, però es va obtenir {imd_measured:.2f}%")


if __name__ == '__main__':
    unittest.main()