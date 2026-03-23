"""
Visualization script to show harmonic window extraction with guard bands.
Shows how harmonics are split and extracted from the impulse response.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from custom_farina import Farina
from generator import soft_clip_tanh

def plot_harmonic_extraction():
    """
    Plot the harmonic extraction process showing guard bands.
    """
    print("Creating visualization of harmonic extraction with guard bands...")

    # Setup
    fs = 48000
    dur = 2.0
    f_start = 20
    f_end = 20000

    # Create Farina object and process clean signal
    farina = Farina(duration=dur, fs=fs, f0=f_start, f1=f_end)
    signal_farina = farina.probe
    farina.process_measurement(signal_farina)

    # Get the key parameters
    amax = np.argmax(farina.far_response)
    level = np.array([0.1] * len(farina.far_response))  # Simplified for visualization
    off = int(fs/10)
    end = amax + 5000  # Simplified end point

    # Calculate harmonic windows (simplified for visualization)
    harm_windows = []
    harm_windows.append((amax, end))  # Fundamental

    for i in range(1, len(farina.harm_times)):
        start = amax - farina.harm_times[i]
        end_win = amax - farina.harm_times[i-1]

        # Apply guard band
        guard = int((end_win - start) * 0.15)
        start_guarded = start + guard
        end_guarded = end_win - guard

        harm_windows.append((start_guarded, end_guarded))

    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: Full impulse response with harmonic windows
    time_axis = np.arange(len(farina.far_response)) / fs * 1000  # Convert to milliseconds

    axes[0].plot(time_axis, farina.far_response, 'b-', alpha=0.7, linewidth=1)
    axes[0].axvline(amax/fs*1000, color='red', linestyle='--', alpha=0.7, label=f'Fundamental peak (t={amax/fs*1000:.1f}ms)')

    # Plot harmonic windows
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (start, end) in enumerate(harm_windows[:5]):  # Show first 5 harmonics
        if start < len(farina.far_response) and end < len(farina.far_response) and start >= 0:
            start_ms = start / fs * 1000
            end_ms = end / fs * 1000
            axes[0].axvspan(start_ms, end_ms, alpha=0.2, color=colors[i%len(colors)],
                          label=f'H{i+1} window ({start_ms:.1f}-{end_ms:.1f}ms)')

    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Impulse Response with Harmonic Extraction Windows (15% Guard Bands)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(amax/fs*1000 - 100, amax/fs*1000 + 200)  # Focus on the region of interest

    # Plot 2: Zoomed view showing guard band effect
    zoom_start = max(0, amax - 20000)  # Start 20000 samples before peak
    zoom_end = min(len(farina.far_response), amax + 10000)  # End 10000 samples after peak
    zoom_time = time_axis[zoom_start:zoom_end]
    zoom_signal = farina.far_response[zoom_start:zoom_end]

    axes[1].plot(zoom_time, zoom_signal, 'b-', alpha=0.8, linewidth=1.5)

    # Show theoretical vs actual windows for H2
    if len(harm_windows) > 1:
        h2_start_theoretical = amax - farina.harm_times[1]
        h2_end_theoretical = amax - farina.harm_times[0]

        h2_start_actual = harm_windows[1][0]
        h2_end_actual = harm_windows[1][1]

        # Theoretical window (before guard bands)
        axes[1].axvspan(h2_start_theoretical/fs*1000, h2_end_theoretical/fs*1000,
                       alpha=0.1, color='red', label='H2 theoretical window')

        # Actual window (after guard bands)
        axes[1].axvspan(h2_start_actual/fs*1000, h2_end_actual/fs*1000,
                       alpha=0.3, color='orange', label='H2 actual window (15% guard)')

        # Show guard band regions
        guard_size = int((h2_end_theoretical - h2_start_theoretical) * 0.15)
        axes[1].axvspan(h2_start_theoretical/fs*1000, (h2_start_theoretical + guard_size)/fs*1000,
                       alpha=0.2, color='gray', label='Guard band (removed)')
        axes[1].axvspan((h2_end_theoretical - guard_size)/fs*1000, h2_end_theoretical/fs*1000,
                       alpha=0.2, color='gray')

    axes[1].axvline(amax/fs*1000, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Zoomed View: Guard Band Effect on H2 Extraction')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'harmonic_extraction_visualization.png'), dpi=150, bbox_inches='tight')
    print("Saved: harmonic_extraction_visualization.png")

    # Plot 3: Individual harmonic responses
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot each harmonic response
    harmonic_names = ['Fundamental (H1)', '2nd Harmonic (H2)', '3rd Harmonic (H3)', '4th Harmonic (H4)']
    for i in range(4):
        if i < len(farina.harm_responses):
            response = farina.harm_responses[i]
            time_resp = np.arange(len(response)) / fs * 1000

            axes[i//2, i%2].plot(time_resp, response, 'b-', linewidth=1.5)
            axes[i//2, i%2].set_title(f'{harmonic_names[i]} - {len(response)} samples')
            axes[i//2, i%2].set_xlabel('Time (ms)')
            axes[i//2, i%2].set_ylabel('Amplitude')
            axes[i//2, i%2].grid(True, alpha=0.3)

            # Add RMS value
            rms_val = np.sqrt(np.mean(response**2))
            axes[i//2, i%2].text(0.02, 0.98, f'RMS: {rms_val:.2e}',
                               transform=axes[i//2, i%2].transAxes,
                               verticalalignment='top', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'individual_harmonic_responses.png'), dpi=150, bbox_inches='tight')
    print("Saved: individual_harmonic_responses.png")

    return farina

def compare_clean_vs_distorted():
    """
    Compare harmonic extraction for clean vs distorted signals.
    """
    print("\nComparing clean vs distorted harmonic extraction...")

    fs = 48000
    dur = 2.0

    # Clean signal
    farina_clean = Farina(duration=dur, fs=fs, f0=20, f1=20000)
    signal_clean = farina_clean.probe
    farina_clean.process_measurement(signal_clean)

    # Distorted signal
    signal_dist = soft_clip_tanh(signal_clean, dist=5.0)
    farina_dist = Farina(duration=dur, fs=fs, f0=20, f1=20000)
    farina_dist.process_measurement(signal_dist)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    harmonic_names = ['H1 (Fundamental)', 'H2 (2nd Harmonic)']
    colors = ['blue', 'red']

    for h_idx in range(2):
        # Clean signal
        if h_idx < len(farina_clean.harm_responses):
            resp_clean = farina_clean.harm_responses[h_idx]
            time_clean = np.arange(len(resp_clean)) / fs * 1000
            axes[h_idx, 0].plot(time_clean, resp_clean, colors[0], linewidth=1.5, label='Clean')

        # Distorted signal
        if h_idx < len(farina_dist.harm_responses):
            resp_dist = farina_dist.harm_responses[h_idx]
            time_dist = np.arange(len(resp_dist)) / fs * 1000
            axes[h_idx, 0].plot(time_dist, resp_dist, colors[1], linewidth=1.5, label='Distorted (tanh, dist=5.0)')

        axes[h_idx, 0].set_title(f'{harmonic_names[h_idx]} Response Comparison')
        axes[h_idx, 0].set_xlabel('Time (ms)')
        axes[h_idx, 0].set_ylabel('Amplitude')
        axes[h_idx, 0].legend()
        axes[h_idx, 0].grid(True, alpha=0.3)

        # RMS comparison bar chart
        labels = ['Clean', 'Distorted']
        rms_clean = np.sqrt(np.mean(farina_clean.harm_responses[h_idx]**2)) if h_idx < len(farina_clean.harm_responses) else 0
        rms_dist = np.sqrt(np.mean(farina_dist.harm_responses[h_idx]**2)) if h_idx < len(farina_dist.harm_responses) else 0
        rms_vals = [rms_clean, rms_dist]

        bars = axes[h_idx, 1].bar(labels, rms_vals, color=colors, alpha=0.7)
        axes[h_idx, 1].set_title(f'{harmonic_names[h_idx]} RMS Energy')
        axes[h_idx, 1].set_ylabel('RMS Amplitude')
        axes[h_idx, 1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, rms_vals):
            height = bar.get_height()
            axes[h_idx, 1].text(bar.get_x() + bar.get_width()/2., height + max(rms_vals)*0.02,
                              f'{val:.2e}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'clean_vs_distorted_comparison.png'), dpi=150, bbox_inches='tight')
    print("Saved: clean_vs_distorted_comparison.png")

    # Print THD comparison
    thd_clean = farina_clean.getTHD(harms=6)
    thd_dist = farina_dist.getTHD(harms=6)
    print(".1f")
    print(".1f")

if __name__ == "__main__":
    farina_obj = plot_harmonic_extraction()
    compare_clean_vs_distorted()
    print("\nVisualization complete. Check the generated PNG files.")
