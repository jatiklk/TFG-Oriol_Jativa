import numpy as np
import matplotlib.pyplot as plt

def sine(fs, dur, f0, amp=1.0):
    """
    Generem un to pur.
    """
    t = np.arange(int(fs*dur))/fs
    return (amp*np.sin(2*np.pi*f0*t)).astype(np.float64)

def synth_tone_with_thd(fs=48000, dur=2.0, f0=1000.0, thd_target=0.05,  n_harm=5, decay=0.6, peak=0.9):
    """
    Generem un to amb una THD específica.
    """
    t = np.arange(int(fs*dur))/fs
    # repartim els harmónics de manera geomètrica
    rel = np.array([decay**(n-2) for n in range(2, n_harm+1)])
    rel = rel / np.linalg.norm(rel)

    # inicialitzem amplituds
    A1 = 1.0
    Ahs = thd_target * rel  # vector d'amplituds

    # senyal i normalitzem
    x = A1*np.sin(2*np.pi*f0*t)
    for n, Ah in enumerate(Ahs, start=2):
        x += Ah*np.sin(2*np.pi*(n*f0)*t)

    # normalitzar a 'peak'
    x = x / np.max(np.abs(x)) * peak
    return x.astype(np.float32)

def synth_tone_with_thd_and_noise(fs=48000, dur=2.0, f0=1000.0, thd_target=0.05,  n_harm=5, decay=0.6, peak=0.9, noise_level=0.01):
    """
    Generem un to amb una THD específica i afegim soroll gaussià.
    """
    t = np.arange(int(fs*dur))/fs
    # repartim els harmónics de manera geométrica
    rel = np.array([decay**(n-2) for n in range(2, n_harm+1)])
    rel = rel / np.linalg.norm(rel)

    # inicialitzem amplituds
    A1 = 1.0
    Ahs = thd_target * rel  # vector d'amplituds

    # senyal i normalitzem
    x = A1*np.sin(2*np.pi*f0*t)
    for n, Ah in enumerate(Ahs, start=2):
        x += Ah*np.sin(2*np.pi*(n*f0)*t)

    # afegim soroll gaussià
    noise = np.random.normal(0, noise_level, len(x))
    x += noise

    # normalitzar a 'peak'
    x = x / np.max(np.abs(x)) * peak
    return x.astype(np.float32)

def soft_clip_tanh(x, dist=5.0):
    """
    Generem un to amb una distorsió suau tipus soft cliper, regulem el nivell de distorsió amb 'dist'
    i normalitzem el valor de sortida a 1.
    """
    y = np.tanh(dist * x)
    return y / (np.max(np.abs(y)) + 1e-12)

def apply_nonlinear_distortion(signal, alpha=0.1):
    """
    Apliquem una distorssió no lineal a una senyal utilitzant un terme quadràtic amb un paràmetre alpha que regulta
    aquesta distorssió.
    """
    #  y = x + alpha * x^2
    # afegeix harmònics parells
    distorted_signal = signal + alpha * (signal**2)

    # normalitzem la senyal 
    max_val = np.max(np.abs(distorted_signal))
    if max_val > 1.0:
        distorted_signal = distorted_signal / max_val * 0.9
    return distorted_signal

def sweep_generator(fs=48000, dur=5.0, f_start=20.0, f_end=20000.0):
    """
    Generem un sweep logarítmic.
    """
    t = np.arange(int(fs*dur))/fs
    K = dur / np.log(f_end/f_start)
    L = f_start * K
    sweep = np.sin(2*np.pi*L*(np.exp(t/K)-1))
    return sweep.astype(np.float32)

def sweep_generator_linear(fs=48000, dur=5.0, f_start=20.0, f_end=20000.0):
    """
    Generem un sweep lineal.
    """
    t = np.arange(int(fs*dur))/fs
    sweep = np.sin(2*np.pi * ( (f_start * t) + ( (f_end - f_start) / (2 * dur) ) * t**2 ))
    return sweep.astype(np.float32)

def sweep_generator_exponential(fs=48000, dur=5.0, f_start=20.0, f_end=20000.0):
    """
    Generem un sweep exponencial.
    """
    t = np.arange(int(fs*dur))/fs
    beta = np.log(f_end / f_start) / dur
    sweep = np.sin(2 * np.pi * (f_start / beta) * (np.exp(beta * t) - 1))
    return sweep.astype(np.float32)

# ara afegire les funcions per generar senyals per a l'anàñisi de IMD

def synth_two_tones(fs=48000, dur=2.0, f1=19000.0, f2=20000.0, amplitude=0.5):
    """
    Genera una senyal amb dos tons. 
    """
    t = np.arange(int(fs*dur))/fs
    signal = amplitude * (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t))
    # Normalizar para evitar clipping si la suma excede 1.0
    signal = signal / np.max(np.abs(signal)) * 0.9 # Normalizamos a 0.9 para dejar margen
    return signal.astype(np.float32)


