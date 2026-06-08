import unittest
import sys
import os
import numpy as np
import scipy.signal as signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from farina import Farina
from generator import soft_clip_tanh

class TestFarinaWithSaturator(unittest.TestCase):

    def setUp(self):
        self.fs = 48000
        self.dur = 5.0
        self.f_start = 20
        self.f_end = 20000
        self.farina = Farina(duration=self.dur, fs=self.fs, f0=self.f_start, f1=self.f_end)

    def test_farina_with_distortion(self):
        """Test 1: Senyal amb distorsió (soft_clip_tanh) - THD ha de ser > 0%"""
        signal_dist = soft_clip_tanh(self.farina.probe, dist=5.0)
        self.farina.process_measurement(signal_dist)
        thd = self.farina.getTHD(harms=8)
        print(f"\nTHD senyal distorsionat (Farina): {thd*100:.4f}%")
        self.assertGreater(thd, 0.0,
                           msg=f"El THD hauria de ser > 0% amb distorsió, però s'ha obtingut {thd*100:.4f}%")

    def test_farina_without_distortion(self):
        """Test 2: Senyal pur sense distorsió - THD hauria de ser ~0% (limitació coneguda)"""
        self.farina.process_measurement(self.farina.probe)
        thd = self.farina.getTHD(harms=8)
        print(f"\nTHD senyal pur (Farina): {thd*100:.4f}%")
        # Aquest test documenta la limitació: el mètode retorna un valor espuri
        # fins i tot sense distorsió, degut a artefactes de la deconvolució
        self.assertLess(thd * 100, 1.0,
                        msg=f"LIMITACIÓ: El THD sense distorsió hauria de ser ~0%, "
                            f"però el mètode retorna {thd*100:.4f}% degut a artefactes de segmentació")

if __name__ == '__main__':
    unittest.main()