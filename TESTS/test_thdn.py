import unittest
import numpy as np
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
    
    def test_thdn_with_harmonics_and_noise(self):
        # Test amb harmònics i soroll simultanis
        fs = 48000
        dur = 1.0
        f0 = 1000.0
        thd_target = 0.05  # 5% THD
        noise_level = 0.01  # 1% soroll

        tone = synth_tone_with_thd_and_noise(fs=fs, dur=dur, f0=f0,
                                            thd_target=thd_target,
                                            noise_level=noise_level)

        thdn_computed = compute_THDN(tone, fs, N=len(tone))

        # El THD+N ha de ser estrictament superior al THD sol (5%)
        # ja que el soroll afegeix contribució addicional al residu
        self.assertGreater(thdn_computed, thd_target * 100,
                        msg="THD+N amb soroll ha de ser superior al THD sol")

# ──────────────────────────────────────────────────────────
    # TEST DIAGNÒSTIC: escombrat de nivells de soroll
    # ──────────────────────────────────────────────────────────
    def test_thdn_noise_sweep_diagnostic(self):
        """
        Escombrat de nivells de soroll per identificar a partir de quin
        SNR es degrada l'estimació de f0 i, per tant, el THD+N.
        Compara f0 auto-estimada vs f0 coneguda (hint) per aïllar
        l'error d'estimació de freqüència del THD+N real.
        """
        fs = 48000
        dur = 1.0
        f0 = 1000.0
        thd_target = 0.03  # 3% THD fix, només variem soroll
 
        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
 
        print("\n" + "=" * 90)
        print(f"{'Noise':>8} | {'SNR aprox':>10} | {'THD+N (auto f0)':>16} | {'THD+N (f0 hint)':>16} | {'Diferència':>11}")
        print("-" * 90)
 
        results = []
        for noise_level in noise_levels:
            tone = synth_tone_with_thd_and_noise(fs=fs, dur=dur, f0=f0,
                                                  thd_target=thd_target,
                                                  noise_level=noise_level)
 
            thdn_auto = compute_THDN(tone, fs, N=len(tone))
            thdn_hint = compute_THDN(tone, fs, N=len(tone), f0_hint=f0)
 
            diff = thdn_auto - thdn_hint
            # SNR aproximat (relatiu a amplitud senyal ~1.0, només indicatiu)
            snr_approx = 20 * np.log10(1.0 / noise_level) if noise_level > 0 else float('inf')
 
            print(f"{noise_level:>8.3f} | {snr_approx:>9.1f}dB | {thdn_auto:>15.4f}% | {thdn_hint:>15.4f}% | {diff:>10.4f}%")
 
            results.append({
                'noise_level': noise_level,
                'snr_db': snr_approx,
                'thdn_auto': thdn_auto,
                'thdn_hint': thdn_hint,
                'diff': diff,
            })
 
        print("=" * 90)
 
        # Identifiquem el punt on la diferència supera un llindar significatiu (>5% relatiu del target)
        threshold = thd_target * 100 * 0.5  # 50% relatiu al target com a llindar d'alerta
        degraded_points = [r for r in results if abs(r['diff']) > threshold]
 
        if degraded_points:
            first_degraded = degraded_points[0]
            print(f"\n⚠ Degradació significativa detectada a partir de "
                  f"noise_level={first_degraded['noise_level']} "
                  f"(SNR≈{first_degraded['snr_db']:.1f}dB)")
        else:
            print("\n✓ Sense degradació significativa en el rang testejat")
 
        # Guardem els resultats per inspecció (no fem assert estricte, és diagnòstic)
        self.results = results
 
 
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)