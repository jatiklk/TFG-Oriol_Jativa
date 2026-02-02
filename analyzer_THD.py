import numpy as np
#import audio_dspy as dsp
import matplotlib.pyplot as plt
import audio_dspy as adspy

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

def extract_impulse_response(signal, fs):
    """
    Calcula una resposta impulsiva simple (autocorrelació normalitzada).
    """
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr /= np.max(np.abs(autocorr))
    return autocorr

def compute_THD_F(signal, fs, N, H=6):
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
    idx_peack = np.argmax(magnitude[1:]) + 1
    f0 = freqs[idx_peack]

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

def compute_THD_rms(signal, fs, N, H=6):
    """
    Calcula la distorsió harmònica total (THD-R) d'una senyal, amb el RMS i no amb la fonamenal.
    Retornem el THD-R en %.
    """
    # N = len(signal) li pasem per parametre ara
    spectrum = np.fft.fft(signal) 
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(spectrum[:N//2])
    freqs = freqs[:N//2]

    # troba la freqüència fonamental
    idx_fund = np.argmax(magnitude[1:]) + 1
    f0 = freqs[idx_fund]

    # suma dels quadrats de les magnituds dels harmònics (2a a 5a)
    thd_numerador_sq = 0
    for h in range(2, H):
        harmonic_freq = f0 * h
        idx_harm = np.argmin(np.abs(freqs - harmonic_freq))
        thd_numerador_sq += magnitude[idx_harm] ** 2

    # Suma dels quadrats de les magnituds de totes les components AC (excloent DC)
    # magnitude[1:] conté totes les components amb freqüència > 0
    thd_denominador_sq = np.sum(magnitude[1:] ** 2)

    # Evitar divisió per zero si no hi ha components AC
    if thd_denominador_sq == 0:
        return 0.0

    thd = np.sqrt(thd_numerador_sq) / np.sqrt(thd_denominador_sq)
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

    # reconstrueix la component fonamental per projecció sin/cos i fem una projecció lineal (Fourier)
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

#per treballar amb sweep ara

def segment_signal(signal, num_partitions):
    """
    Divideixo la senyal en segments per poder calcular posteriorment la THD per parts.
    Retorna una array de segments del senyal.
    """
    segment_length = len(signal) // num_partitions
    segments = []
    for i in range(num_partitions):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segments.append(signal[start_index:end_index])
    return segments

def compute_THD_sweep(signal, num_partitions, segment_length=2048):
    """
    Calculem el THD de cada segment per poder plotejar el resultat de manera gràfica.
    Retornem una array de valors i freqs.
    """
    signal_segments = segment_signal(signal, num_partitions)
    thd_values = []
    dominant_frequencies = []
    fs = 48000

    for segment in signal_segments:
        #calculem la thd de cada segment
        thd = compute_THD_F(segment, fs, len(segment))
        thd_values.append(thd)
        #calculem la freq dominant de cada segment
        spectrum = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1/fs)
        magnitude = np.abs(spectrum[:len(segment)//2])
        freqs = freqs[:len(segment)//2]
        #trobem el pic màxim (ignora DC)
        idx_peak = np.argmax(magnitude[1:]) + 1
        dominant_freq = freqs[idx_peak]
        dominant_frequencies.append(dominant_freq)
    
    return thd_values, dominant_frequencies

def plot_THDs(thd_values, dominant_frequencies):
    """
    Plotejem la gràfica. 
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(dominant_frequencies, thd_values, s=10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("THD (%)")
    plt.title("THD vs. Frequency")
    plt.grid(True)
    plt.xscale('log') # Set the x-axis scale to logarithmic
    plt.xlim([20, 20000]) # Set the x-axis limits from 20 Hz to 20 kHz
    plt.text(0.7, 0.85, f'Mean THD: {np.mean(thd_values):.2f}%', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.show()

def compute_THD_via_farina():
    """
    Utilitzem el metode de Farina per calcular el THD.
    """
    f = adspy.Farina(20, 20, 20000)
    return f.getTHD()

# mirar q farina sense distorsio dona distorsio i testejarho (sha de pulir farina)
    