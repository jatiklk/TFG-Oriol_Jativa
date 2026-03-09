"""
Debug script to investigate the high THD value (0.097734) 
returned by Farina.getTHD() when testing without distortion.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from farina import Farina
from generator import soft_clip_tanh
import scipy.signal as signal

def debug_farina_harmonic_separation():
    """
    Debug the harmonic separation logic to see if there's an issue
    with how harmonics are being extracted.
    """
    print("=" * 80)
    print("DEBUG: Harmonic Separation Analysis")
    print("=" * 80)
    
    # Setup
    fs = 48000
    dur = 2.0
    f_start = 20
    f_end = 20000
    
    # Create Farina object
    farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
    signal_farina = farina.probe
    
    print(f"\nProbe signal stats:")
    print(f"  Length: {len(signal_farina)} samples")
    print(f"  Max amplitude: {np.max(np.abs(signal_farina)):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(signal_farina**2)):.6f}")
    print(f"  Peak-to-peak: {np.max(signal_farina) - np.min(signal_farina):.6f}")
    
    # Process without distortion
    farina.process_measurement(signal_farina)
    
    print(f"\nFar response (after convolution with inverse probe):")
    print(f"  Max amplitude: {np.max(np.abs(farina.far_response)):.6f}")
    print(f"  RMS: {np.sqrt(np.mean(farina.far_response**2)):.6f}")
    print(f"  Mean: {np.mean(farina.far_response):.6f}")
    
    print(f"\nHarmonic times (sample indices): {farina.harm_times}")
    print(f"Number of harmonics detected: {len(farina.harm_responses)}")
    
    print(f"\nIndividual harmonic response stats:")
    for i, harm_resp in enumerate(farina.harm_responses, start=1):
        rms_val = np.sqrt(np.mean(harm_resp**2))
        max_val = np.max(np.abs(harm_resp))
        print(f"  H{i}: length={len(harm_resp)}, max={max_val:.6f}, RMS={rms_val:.6f}")
    
    # Check what getTHD is actually computing
    print(f"\nDEBUG getTHD computation:")
    harms = 9
    rms_vals = np.zeros(harms)
    
    for idx, response in enumerate(farina.harm_responses[:harms]):
        r_corr = signal.convolve(response, farina.get_IR())
        rms_vals[idx] = np.sqrt(np.mean(r_corr**2))
        print(f"  H{idx+1}: correlation length={len(r_corr)}, RMS={rms_vals[idx]:.6f}")
    
    # Normalize
    rms_vals /= rms_vals[0]
    print(f"\nNormalized RMS values: {rms_vals}")
    
    # Compute THD
    thd = np.sqrt(np.sum(rms_vals[1:]**2)) / rms_vals[0]
    print(f"\nFinal THD: {thd:.6f}")
    

def debug_with_known_distortion():
    """
    Test with a known distortion level to see if the method correctly
    identifies it.
    """
    print("\n" + "=" * 80)
    print("DEBUG: Testing with Known Distortion Level")
    print("=" * 80)
    
    fs = 48000
    dur = 2.0
    f_start = 20
    f_end = 20000
    
    # Test with different distortion levels
    dist_levels = [0.0, 2.0, 5.0, 10.0]
    
    for dist_level in dist_levels:
        print(f"\n--- Distortion level: {dist_level} ---")
        
        farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
        signal_farina = farina.probe
        
        if dist_level > 0:
            signal_test = soft_clip_tanh(signal_farina, dist=dist_level)
        else:
            signal_test = signal_farina
        
        # Process
        farina.process_measurement(signal_test)
        thd = farina.getTHD(harms=6)
        
        print(f"  THD: {thd:.6f} ({thd*100:.4f}%)")


def analyze_noise_floor():
    """
    Analyze the noise floor and energy distribution in the responses.
    """
    print("\n" + "=" * 80)
    print("DEBUG: Noise Floor Analysis")
    print("=" * 80)
    
    fs = 48000
    dur = 2.0
    f_start = 20
    f_end = 20000
    
    farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
    signal_farina = farina.probe
    
    # Get a clean measurement
    farina.process_measurement(signal_farina)
    
    # Analyze the far_response spectrum
    N = len(farina.far_response)
    spectrum = np.abs(np.fft.fft(farina.far_response))[:N//2]
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spectrum, height=np.max(spectrum)*0.1)
    
    print(f"\nSpectrum analysis of far_response:")
    print(f"  Total energy (sum): {np.sum(spectrum):.6f}")
    print(f"  Max peak: {np.max(spectrum):.6f}")
    print(f"  Number of significant peaks (>10% of max): {len(peaks)}")
    print(f"  Peak frequencies: {freqs[peaks[:min(10, len(peaks))]]}")
    
    # Check if there's linearity in the response
    print(f"\nLinear response check:")
    IR = farina.get_IR()
    print(f"  IR length: {len(IR)}")
    print(f"  IR max: {np.max(np.abs(IR)):.6f}")
    print(f"  IR RMS: {np.sqrt(np.mean(IR**2)):.6f}")


def plot_harmonic_responses():
    """
    Plot the individual harmonic responses for visual inspection.
    """
    print("\n" + "=" * 80)
    print("Creating plots of harmonic responses...")
    print("=" * 80)
    
    fs = 48000
    dur = 2.0
    f_start = 20
    f_end = 20000
    
    farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
    signal_farina = farina.probe
    farina.process_measurement(signal_farina)
    
    n_harms = min(4, len(farina.harm_responses))
    
    fig, axes = plt.subplots(n_harms, 1, figsize=(12, 3*n_harms))
    if n_harms == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(farina.harm_responses):
            resp = farina.harm_responses[i]
            ax.plot(resp, alpha=0.7)
            ax.set_title(f"Harmonic {i+1} Response (Clean Signal)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'debug_harmonic_responses.png'), dpi=100)
    print("  Saved: debug_harmonic_responses.png")
    
    # Also plot distorted version
    fig, axes = plt.subplots(n_harms, 1, figsize=(12, 3*n_harms))
    if n_harms == 1:
        axes = [axes]
    
    signal_dist = soft_clip_tanh(signal_farina, dist=5.0)
    farina2 = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
    farina2.process_measurement(signal_dist)
    
    for i, ax in enumerate(axes):
        if i < len(farina2.harm_responses):
            resp = farina2.harm_responses[i]
            ax.plot(resp, alpha=0.7, color='red')
            ax.set_title(f"Harmonic {i+1} Response (Distorted Signal, dist=5.0)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'debug_harmonic_responses_distorted.png'), dpi=100)
    print("  Saved: debug_harmonic_responses_distorted.png")


if __name__ == "__main__":
    debug_farina_harmonic_separation()
    debug_with_known_distortion()
    analyze_noise_floor()
    plot_harmonic_responses()
    
    print("\n" + "=" * 80)
    print("Debug analysis complete. Check generated PNG files.")
    print("=" * 80)
