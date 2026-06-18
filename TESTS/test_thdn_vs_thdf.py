import unittest
import numpy as np
from analyzer_THD import compute_THDN, compute_THD_F
from generator import synth_tone_with_thd_and_noise


class TestTHDComparisonNoiseRobustness(unittest.TestCase):

    def test_thdf_vs_thdn_noise_sweep(self):
        """
        Compara THD_F (ignora soroll, mira només bins d'harmònics) vs
        THD+N (inclou tot el residu, soroll inclòs) sota el mateix
        escombrat de nivells de soroll.

        Hipòtesi: THD_F s'ha de mantenir proper al thd_target
        independentment del soroll. THD+N ha de pujar amb el soroll.
        """
        fs = 48000
        dur = 1.0
        f0 = 1000.0
        thd_target = 0.03  # 3% THD fix

        noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

        print("\n" + "=" * 100)
        print(f"Comparativa THD_F vs THD+N  ·  thd_target = {thd_target*100:.1f}%")
        print("=" * 100)
        print(f"{'Noise':>8} | {'SNR aprox':>10} | {'THD_F':>10} | {'THD+N':>10} | {'THD_F desv.':>12} | {'THD+N desv.':>12}")
        print("-" * 100)

        results = []
        for noise_level in noise_levels:
            tone = synth_tone_with_thd_and_noise(fs=fs, dur=dur, f0=f0,
                                                  thd_target=thd_target,
                                                  noise_level=noise_level)

            thd_f = compute_THD_F(tone, fs, N=len(tone), H=6)
            thdn = compute_THDN(tone, fs, N=len(tone))

            snr_approx = 20 * np.log10(1.0 / noise_level) if noise_level > 0 else float('inf')
            target_pct = thd_target * 100

            dev_thdf = thd_f - target_pct
            dev_thdn = thdn - target_pct

            print(f"{noise_level:>8.3f} | {snr_approx:>9.1f}dB | {thd_f:>9.3f}% | {thdn:>9.3f}% | "
                  f"{dev_thdf:>+11.3f}% | {dev_thdn:>+11.3f}%")

            results.append({
                'noise_level': noise_level,
                'snr_db': snr_approx,
                'thd_f': thd_f,
                'thdn': thdn,
                'dev_thdf': dev_thdf,
                'dev_thdn': dev_thdn,
            })

        print("=" * 100)

        # Conclusió automàtica
        max_dev_thdf = max(abs(r['dev_thdf']) for r in results)
        max_dev_thdn = max(abs(r['dev_thdn']) for r in results)

        print(f"\nDesviació màxima THD_F respecte target: {max_dev_thdf:.3f}%")
        print(f"Desviació màxima THD+N respecte target: {max_dev_thdn:.3f}%")

        if max_dev_thdf < max_dev_thdn:
            print("\n✓ Confirmat: THD_F és molt més robust al soroll que THD+N.")
            print("  THD_F ignora el soroll de fons (només mira bins d'harmònics).")
            print("  THD+N inclou tot el residu (harmònics + soroll) per disseny.")
        else:
            print("\n⚠ Resultat inesperat: revisar implementació.")

        self.results = results

        # Assert suau: THD_F s'ha de mantenir molt més proper al target que THD+N
        self.assertLess(max_dev_thdf, max_dev_thdn,
                         msg="THD_F hauria de ser més robust al soroll que THD+N")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)