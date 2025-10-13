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
    Extreu la freqüència fonamental.
    """
    # FFT
    N = len(signal)
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(spectrum[:N//2])
    freqs = freqs[:N//2]

    # troba el pic màxim (ignora DC)
    idx_peak = np.argmax(magnitude[1:]) + 1
    fundamental_freq = freqs[idx_peak]

    return fundamental_freq, idx_peak

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


def compute_THD(signal, fs, N, H):
    """
    Calcula la distorsió harmònica total (THD) de la senyal.
    Retorna el valor THD en percentatge.
    """
    # N = len(signal) li pasem per parametre ara
    spectrum = np.fft.fft(signal) 
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(spectrum[:N//2])
    freqs = freqs[:N//2]

    # troba la freqüència fonamental idx_fund
    f0, idx_peack = extract_fundamental_note(signal, fs)

    # busca harmònics (2a a H), amb el paràmetre H triem fins a quants harmónics volem agafar
    thd_numerador = 0
    for h in range(2, H):
        harmonic_freq = f0 * h
        idx_harm = np.argmin(np.abs(freqs - harmonic_freq))
        thd_numerador += magnitude[idx_harm] ** 2

    thd_denominador = magnitude[idx_peack] ** 2
    thd = np.sqrt(thd_numerador) / magnitude[idx_peack]
    thd_percent = thd * 100
    return thd_percent   

def compute_THDN(signal, fs, N, f0_hint=None):
    """
    Calcula la distorsió harmònica total amb soroll (THD + N) de la senyal.
    - Extreu la fonamental (estimada) per projecció sin/cos.
    - Residual = senyal - fonamental -> (distorsió + soroll).
    - THD+N = RMS(residual) / RMS(fonamental).
    """
    EPS = 1e-15
    x = np.asarray(signal, dtype=np.float64)
    if N is None:
        N = len(x)
    x = x[:N]
    # treu DC
    x = x - np.mean(x)

    # esstima la fonamental si no li passem
    if f0_hint is not None:
        f0_est = float(f0_hint)
    else:
        f0_est, _ = extract_fundamental_note(x, fs) 

    # reconstrueix la component fonamental per projecció sin/cos i fem una projecció lineal
    t = np.arange(N) / fs
    s = np.sin(2*np.pi*f0_est*t)
    c = np.cos(2*np.pi*f0_est*t)
    A = np.vstack([s, c]).T  # [sin, cos]
    coeff, *_ = np.linalg.lstsq(A, x, rcond=None)
    x_fund = A @ coeff # component fonamental estimada

    # RMS fonamental i RMS residual (distorsió + soroll)
    rms_fund = np.sqrt(np.mean(x_fund**2)) + EPS
    resid = x - x_fund
    rms_resid = np.sqrt(np.mean(resid**2))

    # finalment calculem THD+N
    thdn_percent = (rms_resid / rms_fund) * 100.0

    return thdn_percent
