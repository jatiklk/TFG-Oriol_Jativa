import unittest
from analyzer_IMD import compute_IMD_smpte, compute_IMD_ccif
from generator import generate_signal_with_target_imd_QUADRARTIC, generate_signal_with_target_imd_senar

class TestIMDEspecific(unittest.TestCase):

    def test_generator_smpte(self):
        """Test 1: Generador quadràtic amb IMD SMPTE objectiu del 5%"""
        target_imd = 5.0
        distorted_signal, final_alpha, actual_imd, iterations = generate_signal_with_target_imd_QUADRARTIC(
            target_imd=target_imd,
            fs=48000, f1=60.0, f2=7000.0,
            amp1=0.8, amp2=0.2,
            tolerance=0.05
        )
        imd_measured = compute_IMD_smpte(distorted_signal, 48000, 60.0, 7000.0)
        print(f"\nIMD objectiu: {target_imd:.2f}%, IMD real del generador: {actual_imd:.2f}%, IMD mesurada: {imd_measured:.2f}%")
        self.assertAlmostEqual(imd_measured, actual_imd, delta=0.5,
                               msg=f"S'esperava IMD ~= {actual_imd:.2f}%, però es va obtenir {imd_measured:.2f}%")

    def test_generator_ccif(self):
        """Test 2: Generador senar amb IMD CCIF objectiu del 10%"""
        target_imd = 10.0
        distorted_signal, final_dist, actual_imd, iterations = generate_signal_with_target_imd_senar(
            target_imd=target_imd,
            fs=48000, f1=19000.0, f2=20000.0,
            amp1=1.0, amp2=1.0,
            tolerance=0.05
        )
        imd_measured = compute_IMD_ccif(distorted_signal, 48000, 19000.0, 20000.0)
        print(f"\nIMD objectiu: {target_imd:.2f}%, IMD real del generador: {actual_imd:.2f}%, IMD mesurada: {imd_measured:.2f}%")
        self.assertAlmostEqual(imd_measured, actual_imd, delta=0.5,
                               msg=f"S'esperava IMD ~= {actual_imd:.2f}%, però es va obtenir {imd_measured:.2f}%")

if __name__ == '__main__':
    unittest.main()