import numpy as np
import matplotlib.pyplot as plt
from analyzer_IMD import compute_IMD_smpte, compute_IMD_ccif 

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

def synth_two_tones(fs=48000, dur=2.0, f1=60.0, f2=7000.0, amp1=0.8, amp2=0.2):
    """
    Generem un senyal amb dos tons sinusoidals per provar IMD, amb amplituds i freqüències regulables.
    Per defecte, utilitzem les freqüències i una relació d'amplitud típiques de SMPTE (f1:f2 = 4:1).
    """
    t = np.arange(int(fs*dur))/fs
    signal = amp1 * np.sin(2*np.pi*f1*t) + amp2 * np.sin(2*np.pi*f2*t)
    # normalitzem la senyal a 0.9 per deixar marge i evitar clipping, mantenint la relació d'amplitud entre els dos tons.
    signal = signal / np.max(np.abs(signal)) * 0.9 
    return signal.astype(np.float32)

# metode generat 100% amb IA, es un metode que el que fa es generar una senyal amb un IMD especific, 
# l'utilitzo per comprovar que el meu metode de mesura d'IMD és correcte i ens dona el valor de IMD
# esperat. 


def generate_signal_with_target_imd_QUADRARTIC(target_imd,
                                      fs=48000, f1=60.0, f2=7000.0, amp1=0.8, amp2=0.2,
                                      dur=2.0,
                                      initial_alpha=0.01, tolerance=0.1, max_iterations=100,
                                      alpha_step_factor=1.2, alpha_decay_factor=0.8):
    """
    Generates a two-tone signal and iteratively adjusts the 'alpha' parameter of a quadratic distortion
    to achieve a target Intermodulation Distortion (IMD) using the SMPTE method.

    Args:
        target_imd (float): The desired IMD value in percentage.
        fs (int): Sampling frequency in Hz.
        f1 (float): Frequency of the first tone in Hz.
        f2 (float): Frequency of the second tone in Hz.
        amp1 (float): Amplitude of the first tone.
        amp2 (float): Amplitude of the second tone.
        dur (float): Duration of the signal in seconds.
        initial_alpha (float): Starting value for the quadratic distortion coefficient.
        tolerance (float): Acceptable deviation from the target IMD (in percentage).
        max_iterations (int): Maximum number of iterations for alpha adjustment.
        alpha_step_factor (float): Factor to multiply alpha by if IMD is too low.
        alpha_decay_factor (float): Factor to multiply alpha by if IMD is too high (should be < 1).

    Returns:
        tuple: A tuple containing:
            - distorted_signal (np.ndarray): The signal with the achieved target IMD.
            - final_alpha (float): The alpha coefficient that resulted in the target IMD.
            - actual_imd (float): The actual IMD achieved.
            - iterations (int): The number of iterations taken.
    """
    # Generate the clean two-tone signal once
    clean_signal = synth_two_tones(fs=fs, dur=dur, f1=f1, f2=f2, amp1=amp1, amp2=amp2)

    current_alpha = initial_alpha
    actual_imd = 0.0

    for i in range(max_iterations):
        # Ensure alpha does not go below a very small positive number to avoid issues
        current_alpha = max(current_alpha, 1e-8)

        # Apply distortion with the current alpha
        distorted_signal = apply_nonlinear_distortion(clean_signal, alpha=current_alpha)

        # Calculate IMD
        actual_imd = compute_IMD_smpte(distorted_signal, fs, f1, f2)

        # Check for convergence
        if abs(actual_imd - target_imd) <= tolerance:
            print(f"Converged in {i+1} iterations. Target IMD: {target_imd:.2f}%, Actual IMD: {actual_imd:.2f}%, Alpha: {current_alpha:.6f}")
            return distorted_signal, current_alpha, actual_imd, i + 1

        # Adjust alpha based on IMD comparison
        if actual_imd < target_imd:
            current_alpha *= alpha_step_factor # Increase alpha to get more distortion
        else:
            current_alpha *= alpha_decay_factor # Decrease alpha to get less distortion

    print(f"Max iterations reached. Target IMD: {target_imd:.2f}%, Achieved IMD: {actual_imd:.2f}%, Final Alpha: {current_alpha:.6f}")
    return distorted_signal, current_alpha, actual_imd, max_iterations