import unittest
import numpy as np
from scipy.io.wavfile import write
from analyzer_THD import compute_THD_F, compute_THD_rms
from generator import synth_tone_with_thd 

class TestTHD(unittest.TestCase):

    def test_tone_thd_accuracy(self):
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

    def test_pure_sine_zero_thd(self):
        fs = 48000
        f0 = 1000
        t = np.arange(int(fs*1.0))/fs
        x = np.sin(2*np.pi*f0*t)
        N = len(x)
        thd_measured = compute_THD_F(x, fs, N, H=5)/100 # tot i que sabem que es 0
        print(thd_measured)
        # una sinusoidal pura hauria de tenir THD ≈ 0
        self.assertLess(thd_measured, 1e-4)


if __name__ == '__main__':
    unittest.main()