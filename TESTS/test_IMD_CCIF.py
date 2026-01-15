from analyzer_IMD import compute_IMD_ccif
from generator import synth_two_tones, apply_nonlinear_distortion, soft_clip_tanh
import numpy as np

fs_test = 48000
f1_test = 1000.0
f2_test = 1200.0

# Test 1: Senyal neta sense distorsió (esperem IMD = 0%)
clean_signal = synth_two_tones(fs=fs_test, f1=f1_test, f2=f2_test)
imd_clean = compute_IMD_ccif(clean_signal, fs_test, f1_test, f2_test)
print(f"IMD (CCIF/DIM) de la senyal neta: {imd_clean:.2f} %")
assert np.isclose(imd_clean, 0.0, atol=0.01), f"S'esperava IMD ~= 0, però es va obtenir {imd_clean:.2f}%"
print("PASS: La senyal neta té IMD proper a 0%.")

print("\n" + "-"*50 + "\n")

# Test 2: Senyal amb distorsió quadràtica (esperem IMD > 0%)
alpha_test = 0.1
distorted_quad_test_signal = apply_nonlinear_distortion(clean_signal, alpha=alpha_test)
imd_quad_test = compute_IMD_ccif(distorted_quad_test_signal, fs_test, f1_test, f2_test)
print(f"IMD (CCIF/DIM) amb Distorsió Quadràtica (alpha={alpha_test}): {imd_quad_test:.2f} %")
assert imd_quad_test > 1.0, f"S'esperava IMD > 1% per distorsió quadràtica, però es va obtenir {imd_quad_test:.2f}%"
print("PASS: La senyal amb distorsió quadràtica té IMD significatiu.")

print("\n" + "-"*50 + "\n")

# Test 3: Senyal amb Soft Clipper (esperem IMD > 0% i possiblement més alt)
dist_tanh_test = 5.0
distorted_tanh_test_signal = soft_clip_tanh(clean_signal, dist=dist_tanh_test)
imd_tanh_test = compute_IMD_ccif(distorted_tanh_test_signal, fs_test, f1_test, f2_test)
print(f"IMD (CCIF/DIM) amb Soft Clipper (dist={dist_tanh_test}): {imd_tanh_test:.2f} %")
assert imd_tanh_test > 10.0, f"S'esperava IMD > 10% per soft clipper, però es va obtenir {imd_tanh_test:.2f}%"
print("PASS: La senyal amb soft clipper té IMD significatiu.")