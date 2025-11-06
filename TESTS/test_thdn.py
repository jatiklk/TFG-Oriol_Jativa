import unittest
from analyzer_THD import compute_THDN
from generator import synth_tone_with_thd_and_noise

class TestTHDNWithSynthTone(unittest.TestCase):

    def test_thdn_with_pure_tone(self):
        # Test with a tone that should have very low THD (set thd_target to a very small value)
        fs = 48000
        dur = 1.0
        f0 = 1000.0
        thd_target = 1e-9 # Target a very low THD
        noise_level = 0.0 # No noise
        tone = synth_tone_with_thd_and_noise(fs=fs, dur=dur, f0=f0, thd_target=thd_target, noise_level=noise_level)

        # Compute THD+N and assert it's close to the target (considering the small target)
        thdn_computed = compute_THDN(tone, fs, N=len(tone))
        self.assertAlmostEqual(thdn_computed, thd_target * 100, delta=0.1, msg="THD+N for near-pure tone is incorrect")

    def test_thdn_with_harmonics(self):
        # Test with a tone that has a specific THD target due to harmonics
        fs = 48000
        dur = 1.0
        f0 = 1000.0
        thd_target = 0.05 # 5% THD target
        noise_level = 0.0 # No noise
        tone = synth_tone_with_thd_and_noise(fs=fs, dur=dur, f0=f0, thd_target=thd_target, noise_level=noise_level)

        # Compute THD+N and assert it's close to the target
        thdn_computed = compute_THDN(tone, fs, N=len(tone))
        # The actual THD+N might be slightly different from the thd_target due to the decay and normalization,
        # so we allow a larger delta.
        self.assertAlmostEqual(thdn_computed, thd_target * 100, delta=1.0, msg="THD+N with harmonics is incorrect")

    def test_thdn_with_noise(self):
        # Test with a tone that includes noise
        fs = 48000
        dur = 1.0
        f0 = 1000.0
        thd_target = 0.0 # No harmonics
        noise_level = 0.01 # Add noise
        tone = synth_tone_with_thd_and_noise(fs=fs, dur=dur, f0=f0, thd_target=thd_target, noise_level=noise_level)

        # Compute THD+N and assert it's greater than 0 due to noise
        thdn_computed = compute_THDN(tone, fs, N=len(tone))
        self.assertGreater(thdn_computed, 0.0, msg="THD+N with noise should be greater than 0")


# Para correr el test
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)