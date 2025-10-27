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

def soft_clip_tanh(x, dist=5.0):
    """
    Generem un to amb una distorsió suau tipus soft cliper, regulem el nivell de distorsió amb 'dist'
    i normalitzem el valor de sortida a 1.
    """
    y = np.tanh(dist * x)
    return y / (np.max(np.abs(y)) + 1e-12)

def sweep_generator(fs=48000, dur=5.0, f_start=20.0, f_end=20000.0):
    """
    Generem un sweep logarítmic.
    """
    t = np.arange(int(fs*dur))/fs
    K = dur / np.log(f_end/f_start)
    L = f_start * K
    sweep = np.sin(2*np.pi*L*(np.exp(t/K)-1))
    return sweep.astype(np.float32)


