import numpy as np
import matplotlib.pyplot as plt

def plot_fft(signal, fs):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft = np.abs(np.fft.rfft(signal)) / N

    plt.figure()
    plt.plot(freqs, 20 * np.log10(fft))
    plt.title("Espectre de la senyal")
    plt.xlabel("Freqüència (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.grid()
    plt.show()

def extract_fundamental_note(signal, fs):
    """
    Extreu la freqüència fonamental i el nom de la nota.
    """
    # FFT
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(spectrum[:N//2])
    freqs = freqs[:N//2]

    # Troba el pic màxim (ignora DC)
    idx_peak = np.argmax(magnitude[1:]) + 1
    fundamental_freq = freqs[idx_peak]

    # Converteix la freqüència a nom de nota
    note_name = freq_to_note(fundamental_freq)
    return fundamental_freq, note_name

def freq_to_note(freq):
    """
    Converteix una freqüència a nom de nota (A4 = 440 Hz).
    """
    if freq <= 0:
        return "Unknown"
    A4 = 440.0
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    n = int(round(12 * np.log2(freq / A4)))
    note_index = (n + 9) % 12  # A4 és el 9è index
    octave = 4 + ((n + 9) // 12)
    return f"{notes[note_index]}{octave}"

def extract_impulse_response(signal, fs):
    """
    Calcula una resposta impulsiva simple (autocorrelació normalitzada).
    """
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr /= np.max(np.abs(autocorr))
    return autocorr
