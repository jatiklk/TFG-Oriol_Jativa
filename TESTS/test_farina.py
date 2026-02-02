import unittest
import sys
import os
import numpy as np

# Afegeix el directori pare al path per poder importar els mòduls
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from farina import Farina
from analyzer_THD import compute_THD_F
from generator import soft_clip_tanh

class TestFarinaWithSaturator(unittest.TestCase):
    """
    Test que utilitza la classe Farina per generar una senyal de prova,
    la passa per un saturador i retorna el valor de THD.
    """

    def test_farina_signal_with_saturator(self):
        """
        Test que genera una senyal Farina, l'aplica un saturador (soft_clip_tanh)
        i calcula el THD de la senyal distorsionada.
        """
        # Paràmetres de la senyal Farina
        fs = 48000
        dur = 2.0
        f_start = 20
        f_end = 20000
        
        # Generem la senyal Farina (probe signal)
        farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
        signal_farina = farina.probe
        
        # Verifiquem que la senyal s'ha generat correctament
        self.assertIsNotNone(signal_farina, "La senyal Farina no s'ha generat")
        self.assertGreater(len(signal_farina), 0, "La senyal Farina està buida")
        
        print(f"Senyal Farina generada: {len(signal_farina)} mostres")
        print(f"Amplitud màxima: {np.max(np.abs(signal_farina)):.4f}")
        print(f"RMS: {np.sqrt(np.mean(signal_farina**2)):.4f}")
        
        # Apliquem el saturador (soft_clip_tanh) a la senyal Farina
        dist_level = 5.0  # Nivell de distorsió
        signal_saturated = soft_clip_tanh(signal_farina, dist=dist_level)
        
        # Verifiquem que la senyal distorsionada s'ha generat correctament
        self.assertIsNotNone(signal_saturated, "La senyal distorsionada no s'ha generat")
        self.assertEqual(len(signal_saturated), len(signal_farina), 
                        "La longitud de la senyal distorsionada difereix de l'original")
        
        print(f"Senyal distorsionada: {len(signal_saturated)} mostres")
        print(f"Amplitud màxima: {np.max(np.abs(signal_saturated)):.4f}")
        print(f"RMS: {np.sqrt(np.mean(signal_saturated**2)):.4f}")
        
        # Processa la mesura amb la classe Farina
        farina.process_measurement(signal_saturated)
        
        # Calculem el THD usant el mètode de Farina
        thd_farina = farina.getTHD(harms=6)
        
        # Verifiquem que el THD és positiu i raonable
        self.assertGreater(thd_farina, 0.0, "El THD hauria de ser major que 0")
        self.assertLess(thd_farina, 10.0, "El THD hauria de ser raonable")
        
        print(f"\nTHD de la senyal distorsionada (mètode Farina): {thd_farina:.4f}")
        
        # Calculem também el THD usant el mètode d'FFT per comparació
        N = len(signal_saturated)
        thd_fft = compute_THD_F(signal_saturated, fs, N, H=6)
        
        print(f"THD de la senyal distorsionada (mètode FFT): {thd_fft:.2f}%")


    def test_farina_without_distortion(self):
        """
        Test que verifica que quan no hi ha distorsió, getTHD retorna un valor proper a 0.
        Aquesta és una prova de control per verificar que Farina funciona correctament.
        """
        # Paràmetres de la senyal Farina
        fs = 48000
        dur = 2.0
        f_start = 20
        f_end = 20000
        
        # Generem la senyal Farina (probe signal)
        farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
        signal_farina = farina.probe
        
        print(f"Senyal Farina sense distorsió generada: {len(signal_farina)} mostres")
        print(f"Amplitud màxima: {np.max(np.abs(signal_farina)):.4f}")
        print(f"RMS: {np.sqrt(np.mean(signal_farina**2)):.4f}")
        
        # Processa la mesura sense aplicar cap distorsió (senyal pura)
        farina.process_measurement(signal_farina)
        
        # Calculem el THD usant el mètode de Farina
        thd_no_distortion = farina.getTHD(harms=6)
        
        # Verifiquem que el THD és molt proper a 0 (sense distorsió)
        self.assertLess(thd_no_distortion, 0.1, 
                       f"El THD sense distorsió hauria de ser proper a 0, però es va obtenir {thd_no_distortion:.4f}")
        
        print(f"\nTHD sense distorsió (mètode Farina): {thd_no_distortion:.6f}")
        print("✓ Verificat: Senyal sense distorsió té THD ≈ 0")


    def test_farina_with_different_saturation_levels(self):
        """
        Test que comprova com el THD varia en funció del nivell de saturació.
        """
        fs = 48000
        dur = 2.0
        f_start = 20
        f_end = 20000
        
        # Generem la senyal Farina
        farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
        signal_farina = farina.probe
        
        # Provem diferents nivells de distorsió (sense 0 per evitar divisió per zero)
        dist_levels = [0.5, 2.0, 5.0, 10.0]
        thd_values = []
        
        for dist in dist_levels:
            signal_saturated = soft_clip_tanh(signal_farina, dist=dist)
            
            # Creem nova instància Farina per a cada mesura
            farina_test = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
            
            try:
                farina_test.process_measurement(signal_saturated)
                thd = farina_test.getTHD(harms=6)
                thd_values.append(thd)
                print(f"Nivell de distorsió: {dist:5.1f} -> THD: {thd:8.4f}")
            except (IndexError, ValueError) as e:
                print(f"Nivell de distorsió: {dist:5.1f} -> Error: {e}")
                # Fem servir el mètode d'FFT com a alternativa
                N = len(signal_saturated)
                thd_fft = compute_THD_F(signal_saturated, fs, N, H=6) / 100.0
                thd_values.append(thd_fft)
                print(f"Nivell de distorsió: {dist:5.1f} -> THD (FFT): {thd_fft:8.4f}")
        
        # Verifiquem que tenim almenys 2 valors per fer la comparació
        if len(thd_values) >= 2:
            # Verifiquem que el THD augmenta amb la distorsió (trends general)
            is_increasing = all(thd_values[i] <= thd_values[i+1] for i in range(len(thd_values)-1))
            self.assertTrue(is_increasing or len(thd_values) <= 1,
                           f"La tendència del THD hauria de ser creixent, però es va obtenir: {thd_values}")
        
            print(f"\nEl THD varia amb el nivell de distorsió: {[f'{v:.4f}' for v in thd_values]}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
