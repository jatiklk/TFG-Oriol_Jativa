import numpy as np
import matplotlib.pyplot as plt

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