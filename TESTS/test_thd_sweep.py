import unittest
import numpy as np
from scipy.io.wavfile import write
from analyzer_THD import compute_THD_F
from generator import synth_tone_with_thd 

class TestTHD(unittest.TestCase):

    def test_sweep_THD(self):
        fs = 48000
        f0 = 1000
        # generar sine amb thd de 0.03
        x = synth_tone_with_thd(fs=fs, dur=1.0, f0=f0, thd_target=0.03, n_harm=5)
        # calcula la THD 
        N = len(x)
        thd_measured = compute_THD_F(x, fs, N, H=6)/100
        # comprova que està dins d’un marge raonable (±0.002 = ±0.2%)
        print(f"THD mesurat: {thd_measured:.4f}")
        self.assertAlmostEqual(thd_measured, 0.03, delta=0.002)



if __name__ == '__main__':
    unittest.main()